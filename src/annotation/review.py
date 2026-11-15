"""Run a local browser-based, frame-by-frame handover annotation tool."""

import argparse
import csv
import json
import math
import secrets
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from urllib.parse import parse_qs, urlparse

import cv2

ROOT = Path(__file__).resolve().parents[2]
FIELDS = [
    "video_id", "event_id", "object_id", "giver_id", "receiver_id",
    "start_frame", "end_frame", "transfer_frame",
]


def load_videos(manifest: Path, video_dir: Path) -> dict[str, dict]:
    videos = {}
    with manifest.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            video_id = row["video_id"]
            filename = row["file_name"]
            if not video_id or video_id in videos or Path(filename).name != filename:
                raise ValueError(f"Invalid or duplicate video entry: {video_id!r}")
            path = video_dir / filename
            capture = cv2.VideoCapture(str(path))
            try:
                if not capture.isOpened():
                    raise ValueError(f"Cannot open video: {path}")
                fps = capture.get(cv2.CAP_PROP_FPS)
                frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                if not math.isfinite(fps) or fps <= 0 or frames <= 0:
                    raise ValueError(f"Invalid frame rate or count: {path}")
                videos[video_id] = {
                    "id": video_id, "name": filename, "fps": fps,
                    "frames": frames, "path": path,
                    "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                }
            finally:
                capture.release()
    if not videos:
        raise ValueError("The video manifest has no videos.")
    return videos


def encode_frame(video: dict, frame_no: int) -> bytes:
    """Decode one frame of a manifest video and return it as JPEG bytes."""
    capture = cv2.VideoCapture(str(video["path"]))
    try:
        if not capture.isOpened() or not capture.set(cv2.CAP_PROP_POS_FRAMES, frame_no):
            raise ValueError("Unable to seek to frame.")
        ok, frame = capture.read()
        if not ok or int(capture.get(cv2.CAP_PROP_POS_FRAMES)) != frame_no + 1:
            raise ValueError("Unable to decode the requested frame.")
        ok, image = cv2.imencode(".jpg", frame)
        if not ok:
            raise ValueError("Unable to encode frame.")
        return image.tobytes()
    finally:
        capture.release()


def read_events(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != FIELDS:
            raise ValueError(f"Unexpected annotation headers in {path}")
        return list(reader)


def save_event(path: Path, videos: dict[str, dict], event: dict) -> dict[str, str]:
    if not isinstance(event, dict):
        raise ValueError("Event must be an object.")
    if set(event) != set(FIELDS):
        raise ValueError("Event must contain exactly the required fields.")
    if not all(isinstance(event[field], str) for field in FIELDS):
        raise ValueError("Event fields must be strings.")
    row = {field: event[field].strip() for field in FIELDS}
    if any(not row[field] for field in FIELDS if field != "transfer_frame"):
        raise ValueError("All fields except transfer frame are required.")
    if any("\n" in value or "\r" in value for value in row.values()):
        raise ValueError("Fields must not contain line breaks.")
    if row["video_id"] not in videos:
        raise ValueError("Unknown video.")
    if row["giver_id"] == row["receiver_id"]:
        raise ValueError("Giver and receiver must have different IDs.")
    for field in ("start_frame", "end_frame", "transfer_frame"):
        if row[field] and (not row[field].isascii() or not row[field].isdigit()):
            raise ValueError(f"{field} must be a non-negative whole number.")
    start, end = int(row["start_frame"]), int(row["end_frame"])
    if start > end or end >= videos[row["video_id"]]["frames"]:
        raise ValueError("The event window must be ordered and within the video.")
    if row["transfer_frame"] and not start <= int(row["transfer_frame"]) <= end:
        raise ValueError("Transfer frame must be within the event window.")
    events = read_events(path)
    if any(e["video_id"] == row["video_id"] and e["event_id"] == row["event_id"]
           for e in events):
        raise ValueError("That event ID is already used in this video.")
    with path.open("a", newline="", encoding="utf-8") as file:
        csv.DictWriter(file, fieldnames=FIELDS).writerow(row)
    return row


def make_handler(videos: dict[str, dict], annotations: Path, token: str):
    lock = Lock()

    class Handler(BaseHTTPRequestHandler):
        def send_bytes(self, status: int, content_type: str, data: bytes):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(data)

        def send_json(self, status: int, value):
            self.send_bytes(status, "application/json", json.dumps(value).encode("utf-8"))

        def do_GET(self):
            url = urlparse(self.path)
            if url.path == "/":
                page = (Path(__file__).parent / "review.html").read_bytes()
                self.send_bytes(200, "text/html; charset=utf-8", page)
            elif url.path == "/api/videos":
                self.send_json(200, [
                    {key: value for key, value in video.items() if key != "path"}
                    for video in videos.values()
                ])
            elif url.path == "/api/events":
                try:
                    with lock:
                        events = read_events(annotations)
                    self.send_json(200, events)
                except (OSError, ValueError) as exc:
                    self.send_json(500, {"error": str(exc)})
            elif url.path == "/api/frame":
                query = parse_qs(url.query)
                try:
                    video = videos[query["video"][0]]
                    frame_no = int(query["frame"][0])
                    if not 0 <= frame_no < video["frames"]:
                        raise ValueError("Frame is out of range.")
                    image = encode_frame(video, frame_no)
                    self.send_bytes(200, "image/jpeg", image)
                except (KeyError, IndexError, ValueError) as exc:
                    self.send_json(400, {"error": str(exc)})
            else:
                self.send_json(404, {"error": "Not found."})

        def do_POST(self):
            if self.path != "/api/events" or self.headers.get("X-Review-Token") != token:
                self.send_json(403, {"error": "Unauthorized request."})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 8192:
                    raise ValueError("Invalid request size.")
                event = json.loads(self.rfile.read(length))
                with lock:
                    saved = save_event(annotations, videos, event)
                self.send_json(201, saved)
            except (ValueError, UnicodeError) as exc:
                self.send_json(400, {"error": str(exc)})
            except OSError as exc:
                self.send_json(500, {"error": str(exc)})

        def log_message(self, format, *args):
            pass

    return Handler


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("Port must be between 1 and 65535.")
    try:
        videos = load_videos(ROOT / "data/annotations/videos.csv", ROOT / "data/raw_videos")
        annotations = ROOT / "data/annotations/transfers.csv"
        read_events(annotations)
        token = secrets.token_urlsafe(32)
        server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(videos, annotations, token))
    except (OSError, ValueError, KeyError) as exc:
        print(f"Cannot start annotation tool: {exc}", file=sys.stderr)
        return 1
    print(f"Open http://127.0.0.1:{args.port}/#{token}", flush=True)
    print("Press Ctrl+C to stop the server.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
