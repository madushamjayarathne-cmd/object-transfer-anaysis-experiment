import tempfile
import unittest
from pathlib import Path

from src.history.event_store import EventStore, TransferEvent
from src.interaction.hand_detection import Person
from src.interaction.possession import frame_holders, load_objects, wrist_distance
from src.transfer.analyze import detect_events
from src.transfer.rules import TransferParams, holder_for_frame
from src.transfer.state_machine import TransferStateMachine
from experiments.transfer_evaluation import Gold, iou, map_tracks, score_video


def person(track, wrist_x, wrist_y=100.0, conf=0.9, box=(0, 0, 50, 200)):
    return Person(track, box, 0.9, ((wrist_x, wrist_y, conf), (0.0, 0.0, 0.0)))


class RuleTests(unittest.TestCase):
    def test_holder_rules(self):
        p = TransferParams(contact_threshold=0.1, contest_margin=0.02)
        self.assertIsNone(holder_for_frame({1: 0.5}, p))
        self.assertEqual(holder_for_frame({1: 0.05, 2: 0.5}, p), 1)
        self.assertIsNone(holder_for_frame({1: 0.05, 2: 0.06}, p))
        self.assertEqual(holder_for_frame({1: 0.02, 2: 0.08}, p), 1)

    def test_invalid_params_rejected(self):
        for kwargs in ({"contact_threshold": 0}, {"distance_to": "x"}, {"min_hold_frames": 0},
                       {"min_object_confidence": 2}):
            with self.assertRaises(ValueError):
                TransferParams(**kwargs)

    def test_wrist_distance_modes_and_confidence(self):
        box = (100.0, 90.0, 120.0, 110.0)
        box_mode = TransferParams(distance_to="box")
        center_mode = TransferParams(distance_to="center")
        p = person(1, 90.0)  # 10 px left of the box, person height 200
        self.assertAlmostEqual(wrist_distance(p, box, box_mode), 10 / 200)
        self.assertAlmostEqual(wrist_distance(p, box, center_mode), 20 / 200)
        self.assertEqual(wrist_distance(person(1, 110.0), box, box_mode), 0.0)
        self.assertIsNone(wrist_distance(person(1, 90.0, conf=0.1), box, box_mode))


class StateMachineTests(unittest.TestCase):
    params = TransferParams(min_hold_frames=3, max_gap_frames=2)

    def run_sequence(self, holders):
        machine = TransferStateMachine("v", "bottle", self.params)
        return [e for f, h in enumerate(holders) if (e := machine.update(f, h))]

    def test_transfer_confirmed_after_min_hold(self):
        events = self.run_sequence([1] * 5 + [None] * 2 + [2] * 4)
        self.assertEqual(len(events), 1)
        event = events[0]
        self.assertEqual((event.giver, event.receiver), (1, 2))
        self.assertEqual((event.start_frame, event.transfer_frame, event.end_frame), (4, 7, 9))

    def test_first_holder_is_not_an_event_and_brief_contact_ignored(self):
        self.assertEqual(self.run_sequence([1] * 6), [])
        self.assertEqual(self.run_sequence([1] * 5 + [2, 2] + [1] * 5), [])

    def test_long_gap_restarts_candidate(self):
        self.assertEqual(self.run_sequence([1] * 4 + [2, 2, None, None, None, None, 2, 2]), [])

    def test_returning_to_first_holder_makes_second_event(self):
        events = self.run_sequence([1] * 4 + [2] * 4 + [1] * 4)
        self.assertEqual([(e.giver, e.receiver) for e in events], [(1, 2), (2, 1)])


class StoreTests(unittest.TestCase):
    def test_history_and_csv_round_trip(self):
        store = EventStore([
            TransferEvent("v", "bottle", 1, 2, 10, 12, 15),
            TransferEvent("v", "bottle", 2, 3, 40, 42, 45),
            TransferEvent("v", "laptop", 3, 1, 5, 6, 9),
        ])
        self.assertEqual(store.object_history("v", "bottle"), [
            {"holder": 1, "from_frame": 0, "to_frame": 12},
            {"holder": 2, "from_frame": 12, "to_frame": 42},
            {"holder": 3, "from_frame": 42, "to_frame": None},
        ])
        self.assertEqual(store.object_history("v", "cup"), [])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.csv"
            store.to_csv(path)
            self.assertEqual(EventStore.from_csv(path).events, store.events)
        with self.assertRaises(ValueError):
            TransferEvent("v", "bottle", 1, 1, 0, 0, 0)
        with self.assertRaises(ValueError):
            TransferEvent("v", "bottle", 1, 2, 5, 3, 9)


class PipelineTests(unittest.TestCase):
    def test_end_to_end_detect_events_and_object_loading(self):
        params = TransferParams(min_hold_frames=3, max_gap_frames=2, min_object_frames=3, contact_threshold=0.2)
        box = (100.0, 90.0, 120.0, 110.0)
        persons = {f: [person(1, 100.0, box=(0, 0, 50, 200)), person(2, 500.0, box=(400, 0, 450, 200))]
                   for f in range(12)}
        for f in range(6, 12):  # person 2's wrist moves to the object, person 1's moves away
            persons[f] = [person(1, 0.0), person(2, 110.0, box=(400, 0, 450, 200))]
        objects = {"bottle": {f: box for f in range(12)}}
        events = detect_events("v", persons, objects, params)
        self.assertEqual([(e.giver, e.receiver) for e in events], [(1, 2)])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "t.csv"
            path.write_text(
                "frame_index,time_seconds_approx,track_id,class_id,class_name,confidence,x_min,y_min,x_max,y_max\n"
                + "".join(f"{f},0,,1,bottle,0.{5 + f % 2},1,2,3,4\n" for f in range(4))
                + "0,0,,1,bottle,0.05,9,9,9,9\n0,0,,0,person,0.9,1,1,2,2\n")
            loaded = load_objects(path, params)
            self.assertEqual(sorted(loaded), ["bottle"])
            self.assertEqual(len(loaded["bottle"]), 4)
            self.assertEqual(frame_holders({}, loaded["bottle"], 5, params), [None] * 5)


class EvaluationTests(unittest.TestCase):
    def test_iou_and_mapping(self):
        self.assertAlmostEqual(iou((0, 0, 10, 10), (5, 0, 15, 10)), 1 / 3)
        persons = {0: [Person(7, (0, 0, 10, 10), 0.9, ((0, 0, 0), (0, 0, 0)))],
                   1: [Person(7, (0, 0, 10, 10), 0.9, ((0, 0, 0), (0, 0, 0)))]}
        labeled = {0: {"p1": (0, 0, 10, 9)}, 1: {"p1": (1, 0, 10, 10), "p2": (50, 50, 60, 60)}}
        mapping, purity = map_tracks(persons, labeled)
        self.assertEqual(mapping, {7: "p1"})
        self.assertEqual(purity, {7: 1.0})

    def test_score_requires_identity_and_window(self):
        gold = [Gold("v", "e1", "bottle", "p1", "p2", 10, 30, 20), Gold("v", "e2", "bottle", "p2", "p3", 50, 70, None)]
        mapping = {1: "p1", 2: "p2", 3: "p3"}
        good = TransferEvent("v", "bottle", 1, 2, 10, 24, 30)
        reversed_ = TransferEvent("v", "bottle", 2, 3, 40, 100, 105)  # right people, outside window
        wrong_class = TransferEvent("v", "laptop", 2, 3, 50, 55, 60)
        score = score_video([good, reversed_, wrong_class], gold, mapping, tolerance=5)
        self.assertEqual((score.tp, score.fp, score.fn), (1, 2, 1))
        self.assertEqual(score.errors, (4,))
        self.assertEqual(score.timing_only, 1)
        self.assertAlmostEqual(score.f1, 0.4)


if __name__ == "__main__":
    unittest.main()
