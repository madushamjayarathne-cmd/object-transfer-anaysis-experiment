"""Detect object transfers in a video from saved person and object detections."""

import argparse
import sys
from dataclasses import replace
from pathlib import Path

from src.history.event_store import EventStore, TransferEvent
from src.interaction.hand_detection import Person, load_persons
from src.interaction.possession import Box, frame_holders, load_objects
from src.transfer.rules import TransferParams
from src.transfer.state_machine import TransferStateMachine


def detect_events(
    video: str, persons: dict[int, list[Person]], objects: dict[str, dict[int, Box]], params: TransferParams,
) -> list[TransferEvent]:
    last = max([*persons, *(f for frames in objects.values() for f in frames)], default=-1)
    events = []
    for object_class, boxes in sorted(objects.items()):
        machine = TransferStateMachine(video, object_class, params)
        for frame, holder in enumerate(frame_holders(persons, boxes, last + 1, params)):
            event = machine.update(frame, holder)
            if event is not None:
                events.append(event)
    return events


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--persons", type=Path, default=Path("results/persons"))
    parser.add_argument("--objects", type=Path, default=Path("results/tracking_finetuned"))
    parser.add_argument("--output", type=Path, default=Path("results/events.csv"))
    parser.add_argument("--contact-threshold", type=float, default=TransferParams.contact_threshold)
    parser.add_argument("--distance-to", choices=("box", "center"), default=TransferParams.distance_to)
    parser.add_argument("--contest-margin", type=float, default=TransferParams.contest_margin)
    parser.add_argument("--min-hold-frames", type=int, default=TransferParams.min_hold_frames)
    parser.add_argument("--max-gap-frames", type=int, default=TransferParams.max_gap_frames)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    try:
        params = replace(
            TransferParams(), contact_threshold=args.contact_threshold,
            distance_to=args.distance_to, contest_margin=args.contest_margin, min_hold_frames=args.min_hold_frames,
            max_gap_frames=args.max_gap_frames)
        store = EventStore()
        files = sorted(args.persons.glob("*_persons.csv"))
        if not files:
            raise ValueError(f"No *_persons.csv files in {args.persons}")
        for path in files:
            stem = path.name.removesuffix("_persons.csv")
            objects = load_objects(args.objects / f"{stem}_tracks.csv", params)
            for event in detect_events(stem, load_persons(path), objects, params):
                store.add(event)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        store.to_csv(args.output)
    except (OSError, ValueError) as exc:
        print(f"Analysis failed: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote {len(store.events)} events to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
