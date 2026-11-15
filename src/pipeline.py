"""Generate annotated videos and detection tracks for the pilot recordings."""

import argparse
import csv
import math
import sys
from pathlib import Path

import cv2

from src.detection.detector import load_detector
from src.tracking.tracker import VideoTracker


def process_video(
    source: Path,
    output_dir: Path,
    weights: str,
    confidence: float,
    classes: list[int] | None,
    image_size: int = 640,
) -> int:
    video_path = output_dir / f"{source.stem}_tracked.mp4"
    csv_path = output_dir / f"{source.stem}_tracks.csv"
    if video_path.exists() or csv_path.exists():
        raise FileExistsError(f"Output already exists for {source.name}: {output_dir}")

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Cannot open video: {source}")

    writer = None
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError(f"Video has invalid frame rate: {source}")
        model = load_detector(weights)
        tracker = VideoTracker(
            model, confidence=confidence, classes=classes, image_size=image_size
        )
        count = 0
        with csv_path.open("x", newline="", encoding="utf-8") as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow(
                ["frame_index", "time_seconds_approx", "track_id", "class_id",
                 "class_name", "confidence", "x_min", "y_min", "x_max", "y_max"]
            )
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                result = tracker.track(frame)
                annotated = result.plot()
                if writer is None:
                    height, width = annotated.shape[:2]
                    writer = cv2.VideoWriter(
                        str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
                    )
                    if not writer.isOpened():
                        raise RuntimeError(f"Cannot create output video: {video_path}")
                writer.write(annotated)
                if result.boxes is not None:
                    boxes = result.boxes.cpu()
                    for box in boxes:
                        class_id = int(box.cls.item())
                        track_id = int(box.id.item()) if box.id is not None else ""
                        coords = [round(float(value), 2) for value in box.xyxy[0].tolist()]
                        csv_writer.writerow(
                            [count, round(count / fps, 4), track_id, class_id,
                             result.names[class_id], round(float(box.conf.item()), 4), *coords]
                        )
                count += 1
        if count == 0:
            raise RuntimeError(f"No frames decoded from {source}")
        reported_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if reported_count > 0 and count != reported_count:
            print(
                f"Warning: {source.name}: decoded {count} frames, "
                f"but the container reports {reported_count}; inspect for truncation.",
                file=sys.stderr,
            )
        return count
    finally:
        capture.release()
        if writer is not None:
            writer.release()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("videos", nargs="+", type=Path, help="Input video paths")
    parser.add_argument("--output-dir", type=Path, default=Path("results/tracking"))
    parser.add_argument("--weights", default="yolo11n.pt", help="YOLO weights or path")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument(
        "--image-size", type=int, default=640,
        help="YOLO inference size in pixels (default: 640; try 1280 for small objects)",
    )
    parser.add_argument(
        "--classes", type=int, nargs="+", default=None,
        help="Optional COCO class IDs, e.g. 0 39 63 for person, bottle, laptop",
    )
    args = parser.parse_args()
    if not 0 < args.confidence <= 1:
        parser.error("--confidence must be greater than 0 and at most 1")
    if args.image_size <= 0:
        parser.error("--image-size must be positive")
    if any(class_id < 0 for class_id in args.classes or []):
        parser.error("--classes must contain non-negative class IDs")
    if len({path.stem for path in args.videos}) != len(args.videos):
        parser.error("Input videos must have distinct filenames without extensions")
    for path in args.videos:
        if not path.is_file():
            parser.error(f"Input video not found: {path}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for video in args.videos:
        print(f"Processing {video}...", flush=True)
        try:
            frames = process_video(
                video, args.output_dir, args.weights, args.confidence, args.classes,
                args.image_size,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            print(f"Failed to process {video}: {exc}", file=sys.stderr)
            print("Incomplete output files may need to be removed before retrying.", file=sys.stderr)
            return 1
        print(f"Processed {frames} frames from {video.name}.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())