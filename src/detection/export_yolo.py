"""Export reviewed box annotations as an Ultralytics YOLO detection dataset.

Only frames listed in reviewed_frames.csv are exported, so an unreviewed frame is never treated as
"no objects". Splits come from data/splits/videos.csv and are applied per video.
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import cv2

CLASSES = ["person", "bottle", "laptop"]
SPLITS = ("train", "val")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def yolo_line(row: dict[str, str], width: int, height: int) -> str:
    x0, y0, x1, y1 = (float(row[k]) for k in ("x_min", "y_min", "x_max", "y_max"))
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError(f"Box outside image in {row['video_id']} frame {row['frame_index']}")
    return " ".join([
        str(CLASSES.index(row["class_name"])),
        f"{(x0 + x1) / 2 / width:.6f}", f"{(y0 + y1) / 2 / height:.6f}",
        f"{(x1 - x0) / width:.6f}", f"{(y1 - y0) / height:.6f}",
    ])


def export(annotations: Path, splits: Path, videos_dir: Path, output: Path) -> dict[str, int]:
    videos = {r["video_id"]: r for r in read_csv(annotations / "videos.csv")}
    split_of = {r["video_id"]: r["split"] for r in read_csv(splits)}
    for video_id, split in split_of.items():
        if video_id not in videos or split not in SPLITS:
            raise ValueError(f"Invalid split entry: {video_id!r} -> {split!r}")
    boxes = defaultdict(list)
    for row in read_csv(annotations / "boxes.csv"):
        if row["class_name"] not in CLASSES:
            raise ValueError(f"Unknown class {row['class_name']!r}")
        boxes[(row["video_id"], int(row["frame_index"]))].append(row)
    reviewed = defaultdict(set)
    for row in read_csv(annotations / "reviewed_frames.csv"):
        reviewed[row["video_id"]].add(int(row["frame_index"]))
    if any(key[1] not in reviewed[key[0]] for key in boxes):
        raise ValueError("boxes.csv contains boxes for frames not listed as reviewed.")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")

    counts = {split: 0 for split in SPLITS}
    for video_id, frames in sorted(reviewed.items()):
        if video_id not in split_of:
            raise ValueError(f"Reviewed video {video_id} has no split.")
        split = split_of[video_id]
        info = videos[video_id]
        capture = cv2.VideoCapture(str(videos_dir / info["file_name"]))
        try:
            if not capture.isOpened():
                raise ValueError(f"Cannot open {info['file_name']}")
            for frame_index in sorted(frames):
                if not capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index):
                    raise ValueError(f"Cannot seek to frame {frame_index} of {video_id}")
                ok, image = capture.read()
                if not ok:
                    raise ValueError(f"Cannot decode frame {frame_index} of {video_id}")
                height, width = image.shape[:2]
                if (width, height) != (int(info["width"]), int(info["height"])):
                    raise ValueError(f"{video_id} size differs from videos.csv")
                stem = f"{video_id}_{frame_index:05d}"
                image_dir, label_dir = (output / kind / split for kind in ("images", "labels"))
                image_dir.mkdir(parents=True, exist_ok=True)
                label_dir.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(image_dir / f"{stem}.jpg"), image, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                    raise ValueError(f"Cannot write image {stem}")
                lines = [yolo_line(r, width, height) for r in boxes.get((video_id, frame_index), [])]
                (label_dir / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
                counts[split] += 1
        finally:
            capture.release()
    names = "\n".join(f"  {i}: {name}" for i, name in enumerate(CLASSES))
    (output / "dataset.yaml").write_text(
        f"path: {output.resolve()}\ntrain: images/train\nval: images/val\nnames:\n{names}\n")
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=Path("data/annotations"))
    parser.add_argument("--splits", type=Path, default=Path("data/splits/videos.csv"))
    parser.add_argument("--videos", type=Path, default=Path("data/raw_videos"))
    parser.add_argument("--output", type=Path, default=Path("data/yolo"))
    args = parser.parse_args()
    try:
        counts = export(args.annotations, args.splits, args.videos, args.output)
    except (OSError, ValueError) as exc:
        print(f"Export failed: {exc}", file=sys.stderr)
        return 1
    print(f"Exported {counts['train']} train and {counts['val']} val frames to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
