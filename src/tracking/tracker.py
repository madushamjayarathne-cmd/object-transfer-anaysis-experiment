"""Track detections across consecutive frames of one video."""

from ultralytics import YOLO
from ultralytics.engine.results import Results
import numpy as np


class VideoTracker:
    """A tracker instance must be used for only one video."""

    def __init__(
        self, model: YOLO, confidence: float = 0.25, classes: list[int] | None = None,
        image_size: int = 640,
    ) -> None:
        self.model = model
        self.confidence = confidence
        self.classes = classes
        self.image_size = image_size

    def track(self, frame: np.ndarray) -> Results:
        results = self.model.track(
            source=frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=self.confidence,
            classes=self.classes,
            imgsz=self.image_size,
            verbose=False,
        )
        if len(results) != 1:
            raise RuntimeError(f"Expected one result per frame, got {len(results)}")
        return results[0]