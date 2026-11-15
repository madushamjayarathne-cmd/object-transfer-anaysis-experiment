"""Stored transfer events and the holder history reconstructed from them."""

import csv
from dataclasses import astuple, dataclass, fields
from pathlib import Path


@dataclass(frozen=True)
class TransferEvent:
    video: str
    object_class: str
    giver: int
    receiver: int
    start_frame: int  # last frame the giver was seen holding the object
    transfer_frame: int  # first frame of the receiver's confirmed holding run
    end_frame: int  # frame at which the receiver's hold was confirmed

    def __post_init__(self) -> None:
        if self.giver == self.receiver:
            raise ValueError("Giver and receiver must differ.")
        if not 0 <= self.start_frame <= self.end_frame or not self.start_frame <= self.transfer_frame <= self.end_frame:
            raise ValueError("Event frames must be ordered within start..end.")


FIELDS = [f.name for f in fields(TransferEvent)]


class EventStore:
    def __init__(self, events: list[TransferEvent] | None = None) -> None:
        self.events: list[TransferEvent] = list(events or [])

    def add(self, event: TransferEvent) -> None:
        self.events.append(event)

    def for_video(self, video: str) -> list[TransferEvent]:
        return [e for e in self.events if e.video == video]

    def object_history(self, video: str, object_class: str) -> list[dict]:
        """Return holder intervals for one object: who held it, and from/to which frames.

        The first interval starts at frame 0 because the earlier holder is not observed
        before the first transfer; the last interval is open-ended (``to_frame`` is None).
        """
        events = sorted(
            (e for e in self.events if e.video == video and e.object_class == object_class),
            key=lambda e: e.transfer_frame,
        )
        if not events:
            return []
        history = [{"holder": events[0].giver, "from_frame": 0, "to_frame": events[0].transfer_frame}]
        for current, following in zip(events, events[1:] + [None]):
            history.append({
                "holder": current.receiver,
                "from_frame": current.transfer_frame,
                "to_frame": following.transfer_frame if following else None,
            })
        return history

    def to_csv(self, path: Path) -> None:
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(FIELDS)
            writer.writerows(astuple(e) for e in self.events)

    @classmethod
    def from_csv(cls, path: Path) -> "EventStore":
        with path.open(newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            if reader.fieldnames != FIELDS:
                raise ValueError(f"Unexpected headers in {path}")
            return cls([
                TransferEvent(
                    r["video"], r["object_class"], int(r["giver"]), int(r["receiver"]),
                    int(r["start_frame"]), int(r["transfer_frame"]), int(r["end_frame"]))
                for r in reader
            ])
