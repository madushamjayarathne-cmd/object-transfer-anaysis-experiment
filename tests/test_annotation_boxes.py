import csv
import json
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import cv2
import numpy as np

from src.annotation.boxes import (
    BOX_FIELDS, FRAME_FIELDS, load_prelabels, make_handler, object_defaults,
    read_table, sample_frames, save_frame, write_table,
)
from src.annotation.review import load_videos


def box(entity="p1", name="person", x0=2, y0=2, x1=20, y1=20, occluded=False):
    return {"entity_id": entity, "class_name": name, "x_min": x0, "y_min": y0,
            "x_max": x1, "y_max": y1, "occluded": occluded}


class BoxTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        writer = cv2.VideoWriter(str(self.root / "s.avi"), cv2.VideoWriter_fourcc(*"MJPG"), 10, (32, 24))
        for value in (0, 100, 200):
            writer.write(np.full((24, 32, 3), value, dtype=np.uint8))
        writer.release()
        (self.root / "videos.csv").write_text("video_id,file_name\nv1,s.avi\n", encoding="utf-8")
        self.boxes = self.root / "boxes.csv"
        self.frames = self.root / "frames.csv"
        write_table(self.boxes, BOX_FIELDS, [])
        write_table(self.frames, FRAME_FIELDS, [])
        self.videos = load_videos(self.root / "videos.csv", self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def test_video_dimensions_loaded(self):
        self.assertEqual((self.videos["v1"]["width"], self.videos["v1"]["height"]), (32, 24))

    def test_save_replaces_only_that_frame_and_records_review(self):
        save_frame(self.boxes, self.frames, self.videos, "v1", 1, [box("p1"), box("b1", "bottle", 5, 5, 9, 9, True)])
        save_frame(self.boxes, self.frames, self.videos, "v1", 2, [box("p2")])
        save_frame(self.boxes, self.frames, self.videos, "v1", 1, [box("p1", x1=22)])
        rows = read_table(self.boxes, BOX_FIELDS)
        self.assertEqual([(r["frame_index"], r["entity_id"], r["x_max"]) for r in rows],
                         [("1", "p1", "22"), ("2", "p2", "20")])
        self.assertEqual([r["frame_index"] for r in read_table(self.frames, FRAME_FIELDS)], ["1", "2"])

    def test_empty_frame_is_reviewed_without_boxes(self):
        self.assertEqual(save_frame(self.boxes, self.frames, self.videos, "v1", 0, []), 0)
        self.assertEqual(read_table(self.boxes, BOX_FIELDS), [])
        self.assertEqual(len(read_table(self.frames, FRAME_FIELDS)), 1)

    def test_invalid_input_rejected_and_files_unchanged(self):
        save_frame(self.boxes, self.frames, self.videos, "v1", 0, [box()])
        before = (self.boxes.read_bytes(), self.frames.read_bytes())
        bad = [
            ("v1", 3, [box()]), ("v1", -1, [box()]), ("v1", True, [box()]), ("nope", 0, [box()]),
            ("v1", 1, [box(x1=33)]), ("v1", 1, [box(x0=5, x1=5)]), ("v1", 1, [box(y1=25)]),
            ("v1", 1, [box(entity="")]), ("v1", 1, [box(entity="a,b")]),
            ("v1", 1, [box(name="cup")]), ("v1", 1, [box(), box()]),
            ("v1", 1, [box(occluded="yes")]), ("v1", 1, [box(x0=float("nan"))]),
            ("v1", 1, [{"entity_id": "p1"}]), ("v1", 1, "x"),
        ]
        for video, frame, boxes in bad:
            with self.subTest(frame=frame, boxes=boxes), self.assertRaises(ValueError):
                save_frame(self.boxes, self.frames, self.videos, video, frame, boxes)
        self.assertEqual((self.boxes.read_bytes(), self.frames.read_bytes()), before)

    def test_sampling_and_defaults(self):
        events = [{"video_id": "v1", "object_id": "b1", "start_frame": "3", "end_frame": "7", "transfer_frame": ""}]
        self.assertEqual(sample_frames(10, events, 4), [0, 3, 4, 7, 8, 9])
        self.assertEqual(object_defaults(events), {"v1": {"bottle": "b1"}})

    def test_prelabels_apply_thresholds(self):
        directory = self.root / "tracks"
        directory.mkdir()
        header = "frame_index,time_seconds_approx,track_id,class_id,class_name,confidence,x_min,y_min,x_max,y_max\n"
        (directory / "s_tracks.csv").write_text(
            header + "0,0,1,0,person,0.5,1,1,9,9\n0,0,2,0,person,0.1,1,1,9,9\n0,0,3,39,bottle,0.12,1,1,9,9\n",
            encoding="utf-8")
        result = load_prelabels(directory, self.videos)["v1"][0]
        self.assertEqual([r["class_name"] for r in result], ["person", "bottle"])

    def test_http_flow(self):
        handler = make_handler(self.videos, [], {}, self.boxes, self.frames, "tok", 1)
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        base = f"http://127.0.0.1:{server.server_port}"
        try:
            with urlopen(f"{base}/api/videos") as r:
                info = json.load(r)
            self.assertEqual(info[0]["samples"], [0, 1, 2])
            payload = json.dumps({"video_id": "v1", "frame_index": 1, "boxes": [box()]}).encode()
            request = Request(f"{base}/api/boxes", data=payload, method="POST")
            with self.assertRaises(HTTPError) as error:
                urlopen(request)
            self.assertEqual(error.exception.code, 403)
            error.exception.close()
            request.add_header("X-Review-Token", "tok")
            with urlopen(request) as r:
                self.assertEqual(r.status, 201)
            with urlopen(f"{base}/api/saved?video=v1") as r:
                saved = json.load(r)
            self.assertEqual(saved["1"][0]["entity_id"], "p1")
            bad = Request(f"{base}/api/boxes", method="POST", headers={"X-Review-Token": "tok"},
                          data=json.dumps({"video_id": "v1", "frame_index": 1, "boxes": [box(x1=99)]}).encode())
            with self.assertRaises(HTTPError) as error:
                urlopen(bad)
            self.assertEqual(error.exception.code, 400)
            error.exception.close()
            self.assertEqual(len(read_table(self.boxes, BOX_FIELDS)), 1)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == "__main__":
    unittest.main()
