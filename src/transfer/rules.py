"""Rule parameters and the per-frame holder rule for object transfer detection.

The defaults were chosen by grid search on the pilot training videos only (see
experiments/transfer_evaluation.py); they are not tuned for other footage."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TransferParams:
    # A wrist counts as touching the object when it is within this distance of the object
    # box, as a fraction of that person's box height.
    contact_threshold: float = 0.12
    # If the two nearest people are closer than this (same units), the frame is contested
    # and assigns no holder.
    contest_margin: float = 0.0
    # "box": distance to the nearest point of the object box; "center": distance to its centre.
    distance_to: str = "center"
    min_wrist_confidence: float = 0.3
    min_object_confidence: float = 0.15
    # Consecutive-ish frames a person must hold the object to become the confirmed holder.
    min_hold_frames: int = 8
    # Largest gap in frames tolerated inside one holding run.
    max_gap_frames: int = 4
    # An object class seen in fewer frames than this is ignored as noise.
    min_object_frames: int = 15

    def __post_init__(self) -> None:
        if self.contact_threshold <= 0 or self.contest_margin < 0:
            raise ValueError("Contact threshold must be positive and margin non-negative.")
        if self.distance_to not in ("box", "center"):
            raise ValueError("distance_to must be 'box' or 'center'.")
        if not 0 <= self.min_wrist_confidence <= 1 or not 0 <= self.min_object_confidence <= 1:
            raise ValueError("Confidence thresholds must be between 0 and 1.")
        if self.min_hold_frames < 1 or self.max_gap_frames < 0 or self.min_object_frames < 1:
            raise ValueError("Frame counts must be positive (gap may be zero).")


def holder_for_frame(distances: dict[int, float], params: TransferParams) -> int | None:
    """Choose the holder from normalized wrist-to-object distances, or None if unclear."""
    near = sorted((d, person) for person, d in distances.items() if d <= params.contact_threshold)
    if not near:
        return None
    if len(near) > 1 and near[1][0] - near[0][0] < params.contest_margin:
        return None
    return near[0][1]
