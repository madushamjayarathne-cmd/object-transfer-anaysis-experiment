"""Decide, frame by frame, which person is holding a detected object."""

import csv
import math
from pathlib import Path

from src.interaction.hand_detection import Person
from src.transfer.rules import TransferParams, holder_for_frame

Box = tuple[float, float, float, float]
OBJECT_CLASSES = ("bottle", "laptop")


def load_objects(path: Path, params: TransferParams) -> dict[str, dict[int, Box]]:
    """Return the most confident box per frame for each object class in a tracks CSV."""
    best: dict[str, dict[int, tuple[float, Box]]] = {name: {} for name in OBJECT_CLASSES}
    with path.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            name, conf = row["class_name"], float(row["confidence"])
            if name not in best or conf < params.min_object_confidence:
                continue
            frame = int(row["frame_index"])
            box = (float(row["x_min"]), float(row["y_min"]), float(row["x_max"]), float(row["y_max"]))
            if frame not in best[name] or conf > best[name][frame][0]:
                best[name][frame] = (conf, box)
    return {
        name: {frame: box for frame, (_, box) in frames.items()}
        for name, frames in best.items()
        if len(frames) >= params.min_object_frames
    }


def wrist_distance(person: Person, box: Box, params: TransferParams) -> float | None:
    """Distance from the nearest reliable wrist to the object, in person-heights."""
    height = person.box[3] - person.box[1]
    if height <= 0:
        return None
    x0, y0, x1, y1 = box
    distances = []
    for x, y, conf in person.wrists:
        if conf >= params.min_wrist_confidence:
            if params.distance_to == "center":
                distances.append(math.hypot(x - (x0 + x1) / 2, y - (y0 + y1) / 2) / height)
            else:
                distances.append(math.hypot(max(x0 - x, 0.0, x - x1), max(y0 - y, 0.0, y - y1)) / height)
    return min(distances) if distances else None


def frame_holders(
    persons: dict[int, list[Person]], objects: dict[int, Box], num_frames: int, params: TransferParams,
) -> list[int | None]:
    """Return the holder (person track ID) for every frame, or None when unclear or unseen."""
    holders: list[int | None] = []
    for frame in range(num_frames):
        box = objects.get(frame)
        if box is None:
            holders.append(None)
            continue
        distances = {}
        for person in persons.get(frame, []):
            distance = wrist_distance(person, box, params)
            if distance is not None:
                distances[person.track_id] = distance
        holders.append(holder_for_frame(distances, params))
    return holders
