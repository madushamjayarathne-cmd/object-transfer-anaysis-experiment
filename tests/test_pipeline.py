import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from src.pipeline import process_video
from src.tracking.tracker import VideoTracker


class EmptyResult:
    boxes = None

    def __init__(self, frame):
        self.frame = frame

    def plot(self):
        return self.frame.copy()


class FakeModel:
    def __init__(self):
        self.calls = []

    def track(self, **kwargs):
        self.calls.append(kwargs)
        return [EmptyResult(kwargs["source"])]


class FakeBox:
    def __init__(self, track_id):
        self.cls = np.array([41])
        self.conf = np.array([0.85])
        self.id = None if track_id is None else np.array([track_id])
        self.xyxy = np.array([[1.0, 2.0, 20.0, 22.0]])


class FakeBoxes(list):
    def cpu(self):
        return self


class DetectionModel(FakeModel):
    def track(self, **kwargs):
        result = EmptyResult(kwargs["source"])
        result.boxes = FakeBoxes([FakeBox(7), FakeBox(None)])
        result.names = {41: "cup"}
        return [result]


class PipelineTests(unittest.TestCase):
    def test_tracker_passes_bytetrack_and_preserves_state(self):
        model = FakeModel()
        tracker = VideoTracker(model, confidence=0.4, classes=[0, 41], image_size=1280)
        tracker.track(np.zeros((24, 32, 3), dtype=np.uint8))
        tracker.track(np.zeros((24, 32, 3), dtype=np.uint8))
        self.assertEqual(len(model.calls), 2)
        self.assertEqual(model.calls[0]["tracker"], "bytetrack.yaml")
        self.assertTrue(model.calls[1]["persist"])
        self.assertEqual(model.calls[0]["classes"], [0, 41])
        self.assertEqual(model.calls[0]["imgsz"], 1280)

    def test_video_output_and_existing_files_are_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "input.avi"
            writer = cv2.VideoWriter(
                str(source), cv2.VideoWriter_fourcc(*"MJPG"), 10, (32, 24)
            )
            self.assertTrue(writer.isOpened())
            for value in (0, 100, 200):
                writer.write(np.full((24, 32, 3), value, dtype=np.uint8))
            writer.release()
            output = root / "output"
            output.mkdir()
            with patch("src.pipeline.load_detector", return_value=FakeModel()):
                self.assertEqual(process_video(source, output, "unused", 0.25, None), 3)
            with (output / "input_tracks.csv").open(newline="", encoding="utf-8") as file:
                self.assertEqual(len(list(csv.reader(file))), 1)
            capture = cv2.VideoCapture(str(output / "input_tracked.mp4"))
            self.assertTrue(capture.isOpened())
            count = 0
            while capture.read()[0]:
                count += 1
            capture.release()
            self.assertEqual(count, 3)
            with self.assertRaises(FileExistsError):
                process_video(source, output, "unused", 0.25, None)

    def test_missing_video_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(RuntimeError):
                process_video(Path(tmp) / "missing.mov", Path(tmp), "unused", 0.25, None)

    def test_csv_contains_tracked_and_untracked_boxes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "input.avi"
            writer = cv2.VideoWriter(
                str(source), cv2.VideoWriter_fourcc(*"MJPG"), 10, (32, 24)
            )
            self.assertTrue(writer.isOpened())
            writer.write(np.zeros((24, 32, 3), dtype=np.uint8))
            writer.release()
            with patch("src.pipeline.load_detector", return_value=DetectionModel()):
                self.assertEqual(process_video(source, root, "unused", 0.25, None), 1)
            with (root / "input_tracks.csv").open(newline="", encoding="utf-8") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["track_id"], "7")
            self.assertEqual(rows[1]["track_id"], "")
            self.assertEqual(rows[0]["frame_index"], "0")
            self.assertEqual(rows[0]["class_name"], "cup")
            self.assertEqual(rows[0]["x_max"], "20.0")


if __name__ == "__main__":
    unittest.main()
