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

from src.annotation.review import FIELDS, load_videos, make_handler, read_events, save_event


class ReviewTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        writer = cv2.VideoWriter(
            str(self.root / "sample.avi"), cv2.VideoWriter_fourcc(*"MJPG"), 10, (32, 24)
        )
        self.assertTrue(writer.isOpened())
        for value in (0, 100, 200):
            writer.write(np.full((24, 32, 3), value, dtype=np.uint8))
        writer.release()
        self.manifest = self.root / "videos.csv"
        self.manifest.write_text("video_id,file_name\nv1,sample.avi\n", encoding="utf-8")
        self.annotations = self.root / "transfers.csv"
        with self.annotations.open("w", newline="", encoding="utf-8") as file:
            csv.DictWriter(file, fieldnames=FIELDS).writeheader()
        self.videos = load_videos(self.manifest, self.root)
        self.event = {
            "video_id": "v1", "event_id": "event_01", "object_id": "bottle_01",
            "giver_id": "person_01", "receiver_id": "person_02",
            "start_frame": "0", "end_frame": "2", "transfer_frame": "1",
        }

    def tearDown(self):
        self.tmp.cleanup()

    def test_saves_and_rejects_duplicate_without_changing_original(self):
        save_event(self.annotations, self.videos, self.event)
        before = self.annotations.read_bytes()
        with self.assertRaisesRegex(ValueError, "already used"):
            save_event(self.annotations, self.videos, self.event)
        self.assertEqual(self.annotations.read_bytes(), before)
        self.assertEqual(read_events(self.annotations), [self.event])

    def test_invalid_frame_window_and_receiver_rejected(self):
        for changes in (
            {"start_frame": "2", "end_frame": "1"},
            {"end_frame": "3"},
            {"transfer_frame": "3"},
            {"giver_id": "person_02"},
            {"start_frame": "-1"},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                save_event(self.annotations, self.videos, self.event | changes)
        self.assertEqual(read_events(self.annotations), [])

    def test_optional_transfer_frame(self):
        event = self.event | {"transfer_frame": ""}
        save_event(self.annotations, self.videos, event)
        self.assertEqual(read_events(self.annotations), [event])

    def test_http_frames_and_authenticated_save(self):
        server = ThreadingHTTPServer(
            ("127.0.0.1", 0), make_handler(self.videos, self.annotations, "test-token")
        )
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        try:
            base = f"http://127.0.0.1:{server.server_port}"
            with urlopen(f"{base}/api/frame?video=v1&frame=2") as response:
                self.assertEqual(response.headers.get_content_type(), "image/jpeg")
                image = cv2.imdecode(
                    np.frombuffer(response.read(), dtype=np.uint8), cv2.IMREAD_COLOR
                )
                self.assertIsNotNone(image)
                self.assertAlmostEqual(float(image.mean()), 200.0, delta=5.0)
            with self.assertRaises(HTTPError) as error:
                urlopen(f"{base}/api/frame?video=v1&frame=3")
            self.assertEqual(error.exception.code, 400)
            error.exception.close()
            body = json.dumps(self.event).encode()
            request = Request(f"{base}/api/events", data=body, method="POST")
            with self.assertRaises(HTTPError) as error:
                urlopen(request)
            self.assertEqual(error.exception.code, 403)
            error.exception.close()
            request.add_header("X-Review-Token", "test-token")
            with urlopen(request) as response:
                self.assertEqual(response.status, 201)
            self.assertEqual(read_events(self.annotations), [self.event])
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == "__main__":
    unittest.main()
