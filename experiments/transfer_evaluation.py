"""Evaluate detected transfer events against the labeled handovers.

Tracker person IDs are mapped to the annotated person IDs (p1, p2, ...) by box overlap on the
reviewed frames. The mapping uses person boxes only, never the transfer labels. A predicted event
is correct if the object class, giver and receiver match and its transfer frame lies inside the
labeled event window (plus a tolerance).

Parameters are chosen on the training videos only and then scored on the validation videos.
"""

import argparse
import csv
import itertools
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path

from src.history.event_store import TransferEvent
from src.interaction.hand_detection import Person, load_persons
from src.interaction.possession import load_objects
from src.transfer.analyze import detect_events
from src.transfer.rules import TransferParams

OBJECT_PREFIX = {"b": "bottle", "l": "laptop"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def iou(a: tuple, b: tuple) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    inter = w * h
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def map_tracks(
    persons: dict[int, list[Person]], labeled: dict[int, dict[str, tuple]], min_iou: float = 0.5,
) -> tuple[dict[int, str], dict[int, float]]:
    """Map tracker IDs to annotated person IDs by majority vote; also return vote purity."""
    votes: dict[int, Counter] = defaultdict(Counter)
    for frame, boxes in labeled.items():
        for person in persons.get(frame, []):
            scored = [(iou(person.box, box), pid) for pid, box in boxes.items()]
            best = max(scored, default=(0.0, ""))
            if best[0] >= min_iou:
                votes[person.track_id][best[1]] += 1
    mapping = {t: c.most_common(1)[0][0] for t, c in votes.items()}
    purity = {t: c.most_common(1)[0][1] / sum(c.values()) for t, c in votes.items()}
    return mapping, purity


@dataclass(frozen=True)
class Gold:
    video: str
    event: str
    object_class: str
    giver: str
    receiver: str
    start: int
    end: int
    transfer: int | None


@dataclass
class Score:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    timing_only: int = 0
    errors: tuple = ()

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0


def score_video(
    predicted: list[TransferEvent], gold: list[Gold], mapping: dict[int, str], tolerance: int,
) -> Score:
    """Greedy one-to-one matching of predictions to labeled events for one video."""
    def in_window(event: TransferEvent, g: Gold) -> bool:
        return event.object_class == g.object_class and g.start - tolerance <= event.transfer_frame <= g.end + tolerance

    def centre(g: Gold) -> int:
        return g.transfer if g.transfer is not None else (g.start + g.end) // 2

    pairs = sorted(
        (abs(e.transfer_frame - centre(g)), i, j)
        for i, e in enumerate(predicted) for j, g in enumerate(gold)
        if in_window(e, g) and mapping.get(e.giver) == g.giver and mapping.get(e.receiver) == g.receiver)
    used_p, used_g, errors = set(), set(), []
    for error, i, j in pairs:
        if i in used_p or j in used_g:
            continue
        used_p.add(i)
        used_g.add(j)
        if gold[j].transfer is not None:
            errors.append(predicted[i].transfer_frame - gold[j].transfer)
    timing = {j for j, g in enumerate(gold) if any(in_window(e, g) for e in predicted)}
    return Score(len(used_p), len(predicted) - len(used_p), len(gold) - len(used_g), len(timing), tuple(errors))


def combine(scores: list[Score]) -> Score:
    return Score(
        sum(s.tp for s in scores), sum(s.fp for s in scores), sum(s.fn for s in scores),
        sum(s.timing_only for s in scores), tuple(itertools.chain.from_iterable(s.errors for s in scores)))


class Evaluator:
    def __init__(self, annotations: Path, splits: Path, persons_dir: Path, objects_dir: Path, tolerance: int):
        self.tolerance, self.objects_dir = tolerance, objects_dir
        self.videos = {r["video_id"]: Path(r["file_name"]).stem for r in read_csv(annotations / "videos.csv")}
        self.split = {r["video_id"]: r["split"] for r in read_csv(splits)}
        self.persons = {v: load_persons(persons_dir / f"{s}_persons.csv") for v, s in self.videos.items()}
        self.gold: dict[str, list[Gold]] = defaultdict(list)
        for r in read_csv(annotations / "transfers.csv"):
            name = OBJECT_PREFIX[r["object_id"][0].lower()]
            transfer = int(r["transfer_frame"]) if r["transfer_frame"] else None
            self.gold[r["video_id"]].append(Gold(
                r["video_id"], r["event_id"], name, r["giver_id"], r["receiver_id"],
                int(r["start_frame"]), int(r["end_frame"]), transfer))
        labeled: dict[str, dict[int, dict[str, tuple]]] = defaultdict(lambda: defaultdict(dict))
        for r in read_csv(annotations / "boxes.csv"):
            if r["class_name"] == "person":
                labeled[r["video_id"]][int(r["frame_index"])][r["entity_id"]] = tuple(
                    float(r[k]) for k in ("x_min", "y_min", "x_max", "y_max"))
        self.mapping, self.purity = {}, {}
        for v in self.videos:
            self.mapping[v], self.purity[v] = map_tracks(self.persons[v], labeled[v])

    def evaluate(self, videos: list[str], params: TransferParams) -> tuple[Score, dict[str, Score]]:
        per_video = {}
        for v in videos:
            objects = load_objects(self.objects_dir / f"{self.videos[v]}_tracks.csv", params)
            events = detect_events(v, self.persons[v], objects, params)
            per_video[v] = score_video(events, self.gold[v], self.mapping[v], self.tolerance)
        return combine(list(per_video.values())), per_video


def grid() -> list[TransferParams]:
    base = TransferParams()
    thresholds = {"box": (0.05, 0.1, 0.15, 0.2, 0.3), "center": (0.05, 0.08, 0.12, 0.16, 0.2, 0.3)}
    return [
        replace(base, distance_to=mode, contact_threshold=c, contest_margin=m, min_hold_frames=h, max_gap_frames=g)
        for mode, values in thresholds.items()
        for c, m, h, g in itertools.product(values, (0.0, 0.03), (3, 5, 8), (4, 8, 15))
    ]


def show(label: str, score: Score) -> None:
    mean_error = sum(abs(e) for e in score.errors) / len(score.errors) if score.errors else float("nan")
    print(f"{label:28} TP={score.tp:2} FP={score.fp:2} FN={score.fn:2}  precision={score.precision:.2f} "
          f"recall={score.recall:.2f} F1={score.f1:.2f}  mean |transfer-frame error|={mean_error:.1f} frames")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=Path("data/annotations"))
    parser.add_argument("--splits", type=Path, default=Path("data/splits/videos.csv"))
    parser.add_argument("--persons", type=Path, default=Path("results/persons"))
    parser.add_argument("--objects", type=Path, default=Path("results/tracking_finetuned"))
    parser.add_argument("--tolerance", type=int, default=15, help="Frames of slack around a labeled window.")
    args = parser.parse_args()
    try:
        ev = Evaluator(args.annotations, args.splits, args.persons, args.objects, args.tolerance)
    except (OSError, ValueError, KeyError) as exc:
        print(f"Cannot evaluate: {exc}", file=sys.stderr)
        return 1
    train = [v for v, s in ev.split.items() if s == "train"]
    val = [v for v, s in ev.split.items() if s == "val"]
    print("Track-to-person mapping purity (share of votes for the chosen person):")
    for v in ev.videos:
        print(f"  {v}: " + ", ".join(f"track {t}->{p} ({ev.purity[v][t]:.0%})" for t, p in sorted(ev.mapping[v].items())))
    default = TransferParams()
    print("\nCurrent library defaults:")
    for name, vids in (("train videos", train), ("validation videos", val)):
        show(name, ev.evaluate(vids, default)[0])
    ranked = sorted(grid(), key=lambda p: (-ev.evaluate(train, p)[0].f1, -ev.evaluate(train, p)[0].precision))
    best = ranked[0]
    print(f"\nBest on TRAIN videos only: {best}")
    train_score, _ = ev.evaluate(train, best)
    val_score, per_video = ev.evaluate(val, best)
    show("train videos", train_score)
    show("validation videos", val_score)
    for v, s in per_video.items():
        show(f"  {v}", s)
        print(f"    labeled events whose window contains a predicted event of that object: {s.timing_only}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
