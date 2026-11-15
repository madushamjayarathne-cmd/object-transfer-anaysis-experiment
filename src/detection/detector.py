"""Load a pretrained Ultralytics detection model."""

from ultralytics import YOLO


def load_detector(weights: str = "yolo11n.pt") -> YOLO:
    """Load local weights, or download official weights on first use."""
    return YOLO(weights)