"""Measure how often the labeled transferred object is detected inside each labeled event window.

Coverage is only a necessary condition: a detection of the right class may still be the wrong
object or in the wrong place. It is not a detector accuracy metric.
"""

import argparse
import csv
import sys
from pathlib import Path

OBJECT_CLASSES = {"b": "bottle", "l": "laptop"}


def object_class(object_id: str) -> str:
    try:
        return OBJECT_CLASSES[object_id[0].lower()]
    except (IndexError, KeyError):
        raise ValueError(f"Cannot infer object class from object ID {object_id!r}") from None


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def coverage(tracks_dir: Path, videos: Path, transfers: Path, min_confidence: float):
    stems = {row["video_id"]: Path(row["file_name"]).stem for row in read_csv(videos)}
    detected: dict[str, dict[str, set[int]]] = {}
    results = []
    for event in read_csv(transfers):
        video = event["video_id"]
        if video not in detected:
            path = tracks_dir / f"{stems[video]}_tracks.csv"
            frames: dict[str, set[int]] = {}
            for row in read_csv(path):
                if float(row["confidence"]) >= min_confidence:
                    frames.setdefault(row["class_name"], set()).add(int(row["frame_index"]))
            detected[video] = frames
        start, end = int(event["start_frame"]), int(event["end_frame"])
        name = object_class(event["object_id"])
        hit = {f for f in detected[video].get(name, set()) if start <= f <= end}
        results.append((video, event["event_id"], name, start, end, len(hit), end - start + 1))
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tracks_dir", type=Path)
    parser.add_argument("--videos", type=Path, default=Path("data/annotations/videos.csv"))
    parser.add_argument("--transfers", type=Path, default=Path("data/annotations/transfers.csv"))
    parser.add_argument("--min-confidence", type=float, default=0.0)
    args = parser.parse_args()
    try:
        results = coverage(args.tracks_dir, args.videos, args.transfers, args.min_confidence)
    except (OSError, ValueError, KeyError) as exc:
        print(f"Cannot compute coverage: {exc}", file=sys.stderr)
        return 1
    print("video      event     object  window     frames_with_object  coverage")
    hits = total = 0
    for video, event, name, start, end, hit, length in results:
        hits, total = hits + hit, total + length
        print(f"{video:10} {event:9} {name:7} {start:3}-{end:<3}    {hit:3}/{length:<3}             {hit / length:6.1%}")
    print(f"overall: {hits}/{total} = {hits / total:.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
