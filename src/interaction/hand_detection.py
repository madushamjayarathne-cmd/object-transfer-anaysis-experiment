"""Track people and estimate their wrist positions with a pretrained YOLO pose model.

Wrists are used as a proxy for hands. They can be wrong or missing when a hand is hidden,
so every wrist carries the pose model's own confidence.
"""

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
from ultralytics import YOLO

LEFT_WRIST, RIGHT_WRIST = 9, 10  # COCO keypoint indices
FIELDS = [
    "frame_index", "track_id", "confidence", "x_min", "y_min", "x_max", "y_max",
    "lw_x", "lw_y", "lw_conf", "rw_x", "rw_y", "rw_conf",
]


@dataclass(frozen=True)
class Person:
    track_id: int
    box: tuple[float, float, float, float]
    confidence: float
    wrists: tuple[tuple[float, float, float], tuple[float, float, float]]  # (x, y, conf)


def load_persons(path: Path) -> dict[int, list[Person]]:
    """Read a persons CSV into a mapping from frame index to the people in that frame."""
    frames: dict[int, list[Person]] = {}
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != FIELDS:
            raise ValueError(f"Unexpected headers in {path}")
        for row in reader:
            box = (float(row["x_min"]), float(row["y_min"]), float(row["x_max"]), float(row["y_max"]))
            left = (float(row["lw_x"]), float(row["lw_y"]), float(row["lw_conf"]))
            right = (float(row["rw_x"]), float(row["rw_y"]), float(row["rw_conf"]))
            frames.setdefault(int(row["frame_index"]), []).append(
                Person(int(row["track_id"]), box, float(row["confidence"]), (left, right)))
    return frames


def process_video(source: Path, output: Path, weights: str, confidence: float, image_size: int) -> int:
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Cannot open video: {source}")
    try:
        model = YOLO(weights)
        count = 0
        with output.open("x", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(FIELDS)
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                result = model.track(
                    source=frame, persist=True, tracker="bytetrack.yaml", classes=[0],
                    conf=confidence, imgsz=image_size, verbose=False,
                )[0]
                if result.boxes is not None and result.boxes.id is not None and result.keypoints is not None:
                    boxes, ids = result.boxes.cpu(), result.boxes.id.cpu()
                    points = result.keypoints.data.cpu()
                    for box, track_id, kpts in zip(boxes, ids, points):
                        row = [count, int(track_id.item()), round(float(box.conf.item()), 4)]
                        row += [round(float(v), 2) for v in box.xyxy[0].tolist()]
                        for index in (LEFT_WRIST, RIGHT_WRIST):
                            x, y, conf = (float(v) for v in kpts[index].tolist())
                            row += [round(x, 2), round(y, 2), round(conf, 4)]
                        writer.writerow(row)
                count += 1
        if count == 0:
            raise RuntimeError(f"No frames decoded from {source}")
        return count
    finally:
        capture.release()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("videos", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("results/persons"))
    parser.add_argument("--weights", default="yolo11s-pose.pt")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--image-size", type=int, default=1280)
    args = parser.parse_args()
    if not 0 < args.confidence <= 1 or args.image_size <= 0:
        parser.error("Confidence must be in (0, 1] and image size must be positive.")
    if len({v.stem for v in args.videos}) != len(args.videos) or not all(v.is_file() for v in args.videos):
        parser.error("Videos must exist and have distinct names.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for video in args.videos:
        print(f"Processing {video}...", flush=True)
        try:
            frames = process_video(
                video, args.output_dir / f"{video.stem}_persons.csv",
                args.weights, args.confidence, args.image_size)
        except (OSError, RuntimeError, ValueError) as exc:
            print(f"Failed to process {video}: {exc}", file=sys.stderr)
            return 1
        print(f"Processed {frames} frames.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
