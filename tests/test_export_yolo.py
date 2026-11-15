import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.detection.export_yolo import export, yolo_line


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "ann").mkdir()
        for name in ("a.avi", "b.avi"):
            writer = cv2.VideoWriter(str(self.root / name), cv2.VideoWriter_fourcc(*"MJPG"), 10, (40, 20))
            for value in (0, 100, 200):
                writer.write(np.full((20, 40, 3), value, dtype=np.uint8))
            writer.release()
        header = "video_id,file_name,session_id,fps,width,height,frame_count\n"
        (self.root / "ann/videos.csv").write_text(
            header + "va,a.avi,s,10,40,20,3\nvb,b.avi,s,10,40,20,3\n")
        (self.root / "ann/reviewed_frames.csv").write_text("video_id,frame_index\nva,0\nva,2\nvb,1\n")
        (self.root / "ann/boxes.csv").write_text(
            "video_id,frame_index,entity_id,class_name,x_min,y_min,x_max,y_max,occluded\n"
            "va,0,b1,bottle,10,5,20,15,0\nva,0,p1,person,0,0,40,20,0\nvb,1,l1,laptop,4,2,8,6,1\n")
        (self.root / "splits.csv").write_text("video_id,split\nva,train\nvb,val\n")

    def tearDown(self):
        self.tmp.cleanup()

    def run_export(self, out="out"):
        return export(self.root / "ann", self.root / "splits.csv", self.root, self.root / out)

    def test_normalized_line(self):
        row = {"video_id": "v", "frame_index": "0", "class_name": "bottle",
               "x_min": "10", "y_min": "5", "x_max": "20", "y_max": "15"}
        self.assertEqual(yolo_line(row, 40, 20), "1 0.375000 0.500000 0.250000 0.500000")
        with self.assertRaises(ValueError):
            yolo_line(row | {"x_max": "41"}, 40, 20)

    def test_export_splits_by_video_and_keeps_reviewed_empty_frames(self):
        self.assertEqual(self.run_export(), {"train": 2, "val": 1})
        out = self.root / "out"
        self.assertEqual(len((out / "labels/train/va_00000.txt").read_text().splitlines()), 2)
        self.assertEqual((out / "labels/train/va_00002.txt").read_text(), "")
        self.assertEqual((out / "labels/val/vb_00001.txt").read_text().split()[0], "2")
        self.assertFalse(any(out.glob("images/*/va_00001.jpg")))
        self.assertIn("2: laptop", (out / "dataset.yaml").read_text())

    def test_refuses_nonempty_output_and_bad_data(self):
        self.run_export()
        with self.assertRaises(FileExistsError):
            self.run_export()
        (self.root / "splits.csv").write_text("video_id,split\nva,train\n")
        with self.assertRaisesRegex(ValueError, "no split"):
            self.run_export("other")
        (self.root / "splits.csv").write_text("video_id,split\nva,train\nvb,test\n")
        with self.assertRaisesRegex(ValueError, "Invalid split"):
            self.run_export("third")


if __name__ == "__main__":
    unittest.main()
