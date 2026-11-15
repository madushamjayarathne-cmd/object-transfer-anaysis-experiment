"""Run a local browser tool to review and correct bounding boxes for detector training.

Model predictions are shown only as unverified suggestions. A frame is recorded as reviewed
only when the user explicitly saves it.
"""

import argparse
import csv
import json
import os
import re
import secrets
import sys
import tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from urllib.parse import parse_qs, urlparse

from src.annotation.review import ROOT, encode_frame, load_videos

BOX_FIELDS = [
    "video_id", "frame_index", "entity_id", "class_name",
    "x_min", "y_min", "x_max", "y_max", "occluded",
]
FRAME_FIELDS = ["video_id", "frame_index"]
CLASSES = ("person", "bottle", "laptop")
ENTITY_PATTERN = re.compile(r"[A-Za-z0-9_-]{1,40}")
MIN_CONFIDENCE = {"person": 0.3, "bottle": 0.1, "laptop": 0.1}


def read_table(path: Path, fields: list[str]) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != fields:
            raise ValueError(f"Unexpected headers in {path}")
        return list(reader)


def write_table(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    """Replace a CSV atomically so an interrupted save cannot truncate annotations."""
    handle, temp_name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(handle, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temp_name, path)
    except BaseException:
        Path(temp_name).unlink(missing_ok=True)
        raise


def sample_frames(frames: int, events: list[dict[str, str]], step: int) -> list[int]:
    chosen = set(range(0, frames, step)) | {frames - 1}
    for event in events:
        for field in ("start_frame", "end_frame", "transfer_frame"):
            if event[field]:
                chosen.add(int(event[field]))
    return sorted(frame for frame in chosen if 0 <= frame < frames)


def object_defaults(events: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    names = {"b": "bottle", "l": "laptop"}
    defaults: dict[str, dict[str, str]] = {}
    for event in events:
        name = names.get(event["object_id"][:1].lower())
        if name:
            defaults.setdefault(event["video_id"], {})[name] = event["object_id"]
    return defaults


def load_prelabels(directory: Path | None, videos: dict[str, dict]) -> dict[str, dict[int, list]]:
    result: dict[str, dict[int, list]] = {}
    if directory is None:
        return result
    for video_id, video in videos.items():
        path = directory / f"{Path(video['name']).stem}_tracks.csv"
        if not path.is_file():
            continue
        frames: dict[int, list] = {}
        with path.open(newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                name = row["class_name"]
                if name in MIN_CONFIDENCE and float(row["confidence"]) >= MIN_CONFIDENCE[name]:
                    frames.setdefault(int(row["frame_index"]), []).append({
                        "class_name": name, "confidence": float(row["confidence"]),
                        **{k: float(row[k]) for k in ("x_min", "y_min", "x_max", "y_max")},
                    })
        result[video_id] = frames
    return result


def validate_boxes(video: dict, boxes) -> list[dict[str, str]]:
    if not isinstance(boxes, list) or len(boxes) > 50:
        raise ValueError("Boxes must be a list of at most 50 items.")
    rows, seen = [], set()
    for box in boxes:
        if not isinstance(box, dict) or set(box) != {
            "entity_id", "class_name", "x_min", "y_min", "x_max", "y_max", "occluded"
        }:
            raise ValueError("Each box must contain exactly the required fields.")
        entity, name = box["entity_id"], box["class_name"]
        if not isinstance(entity, str) or not ENTITY_PATTERN.fullmatch(entity):
            raise ValueError("Every box needs an entity ID of letters, digits, _ or -.")
        if name not in CLASSES:
            raise ValueError(f"Class must be one of {', '.join(CLASSES)}.")
        if entity in seen:
            raise ValueError(f"Entity ID {entity} is used twice in this frame.")
        seen.add(entity)
        if not isinstance(box["occluded"], bool):
            raise ValueError("Occluded must be true or false.")
        coords = []
        for key in ("x_min", "y_min", "x_max", "y_max"):
            value = box[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value != value:
                raise ValueError(f"{key} must be a number.")
            coords.append(round(float(value), 1))
        x0, y0, x1, y1 = coords
        if not (0 <= x0 < x1 <= video["width"] and 0 <= y0 < y1 <= video["height"]):
            raise ValueError("Boxes must satisfy 0 <= min < max <= image size.")
        rows.append({
            "entity_id": entity, "class_name": name,
            "x_min": f"{x0:g}", "y_min": f"{y0:g}", "x_max": f"{x1:g}", "y_max": f"{y1:g}",
            "occluded": "1" if box["occluded"] else "0",
        })
    return rows


def save_frame(boxes_path: Path, frames_path: Path, videos: dict, video_id, frame_index, boxes):
    if video_id not in videos:
        raise ValueError("Unknown video.")
    if isinstance(frame_index, bool) or not isinstance(frame_index, int):
        raise ValueError("Frame index must be a whole number.")
    if not 0 <= frame_index < videos[video_id]["frames"]:
        raise ValueError("Frame is out of range.")
    rows = validate_boxes(videos[video_id], boxes)
    key = (video_id, str(frame_index))
    all_boxes = [r for r in read_table(boxes_path, BOX_FIELDS)
                 if (r["video_id"], r["frame_index"]) != key]
    all_boxes += [{"video_id": video_id, "frame_index": str(frame_index), **row} for row in rows]
    reviewed = {(r["video_id"], r["frame_index"]) for r in read_table(frames_path, FRAME_FIELDS)}
    reviewed.add(key)
    all_boxes.sort(key=lambda r: (r["video_id"], int(r["frame_index"]), r["entity_id"]))
    # Write boxes first: a reviewed frame must never exist without its boxes.
    write_table(boxes_path, BOX_FIELDS, all_boxes)
    write_table(frames_path, FRAME_FIELDS, [
        {"video_id": v, "frame_index": f} for v, f in sorted(reviewed, key=lambda x: (x[0], int(x[1])))
    ])
    return len(rows)


def make_handler(videos, events, prelabels, boxes_path, frames_path, token, step):
    lock = Lock()
    defaults = object_defaults(events)

    class Handler(BaseHTTPRequestHandler):
        def send_bytes(self, status, content_type, data):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(data)

        def send_json(self, status, value):
            self.send_bytes(status, "application/json", json.dumps(value).encode("utf-8"))

        def do_GET(self):
            url = urlparse(self.path)
            query = parse_qs(url.query)
            try:
                if url.path == "/":
                    page = (Path(__file__).parent / "boxes.html").read_bytes()
                    self.send_bytes(200, "text/html; charset=utf-8", page)
                elif url.path == "/api/videos":
                    self.send_json(200, [{
                        "id": v["id"], "name": v["name"], "frames": v["frames"],
                        "width": v["width"], "height": v["height"],
                        "samples": sample_frames(
                            v["frames"], [e for e in events if e["video_id"] == v["id"]], step),
                        "defaults": defaults.get(v["id"], {}),
                    } for v in videos.values()])
                elif url.path == "/api/saved":
                    video_id = query["video"][0]
                    with lock:
                        rows = [r for r in read_table(boxes_path, BOX_FIELDS) if r["video_id"] == video_id]
                        reviewed = [int(r["frame_index"]) for r in read_table(frames_path, FRAME_FIELDS)
                                    if r["video_id"] == video_id]
                    frames = {f: [] for f in reviewed}
                    for r in rows:
                        frames.setdefault(int(r["frame_index"]), []).append({
                            "entity_id": r["entity_id"], "class_name": r["class_name"],
                            "occluded": r["occluded"] == "1",
                            **{k: float(r[k]) for k in ("x_min", "y_min", "x_max", "y_max")},
                        })
                    self.send_json(200, frames)
                elif url.path == "/api/prelabels":
                    video_id, frame = query["video"][0], int(query["frame"][0])
                    self.send_json(200, prelabels.get(video_id, {}).get(frame, []))
                elif url.path == "/api/frame":
                    video = videos[query["video"][0]]
                    frame_no = int(query["frame"][0])
                    if not 0 <= frame_no < video["frames"]:
                        raise ValueError("Frame is out of range.")
                    self.send_bytes(200, "image/jpeg", encode_frame(video, frame_no))
                else:
                    self.send_json(404, {"error": "Not found."})
            except (KeyError, IndexError, ValueError) as exc:
                self.send_json(400, {"error": str(exc) or "Bad request."})
            except OSError as exc:
                self.send_json(500, {"error": str(exc)})

        def do_POST(self):
            if self.path != "/api/boxes" or self.headers.get("X-Review-Token") != token:
                self.send_json(403, {"error": "Unauthorized request."})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 65536:
                    raise ValueError("Invalid request size.")
                data = json.loads(self.rfile.read(length))
                if not isinstance(data, dict) or set(data) != {"video_id", "frame_index", "boxes"}:
                    raise ValueError("Invalid request.")
                with lock:
                    count = save_frame(boxes_path, frames_path, videos, data["video_id"],
                                       data["frame_index"], data["boxes"])
                self.send_json(201, {"saved_boxes": count})
            except (ValueError, UnicodeError) as exc:
                self.send_json(400, {"error": str(exc)})
            except OSError as exc:
                self.send_json(500, {"error": str(exc)})

        def log_message(self, format, *args):
            pass

    return Handler


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--step", type=int, default=5, help="Suggest every Nth frame (default 5).")
    parser.add_argument("--prelabels", type=Path, default=None,
                        help="Directory of *_tracks.csv files used as unverified suggestions.")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535 or args.step < 1:
        parser.error("Port must be 1-65535 and step must be at least 1.")
    ann = ROOT / "data/annotations"
    try:
        videos = load_videos(ann / "videos.csv", ROOT / "data/raw_videos")
        events = list(csv.DictReader((ann / "transfers.csv").open(newline="", encoding="utf-8")))
        read_table(ann / "boxes.csv", BOX_FIELDS)
        read_table(ann / "reviewed_frames.csv", FRAME_FIELDS)
        prelabels = load_prelabels(args.prelabels, videos)
        token = secrets.token_urlsafe(32)
        handler = make_handler(videos, events, prelabels, ann / "boxes.csv",
                               ann / "reviewed_frames.csv", token, args.step)
        server = ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    except (OSError, ValueError, KeyError) as exc:
        print(f"Cannot start box labeling tool: {exc}", file=sys.stderr)
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
