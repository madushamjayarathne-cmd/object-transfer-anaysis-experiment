"""Turn a stream of per-frame holders into transfer events.

States: no confirmed holder -> HELD(A). A transfer is reported when a different person B
holds the object for ``min_hold_frames`` frames (gaps up to ``max_gap_frames`` tolerated),
after which the state becomes HELD(B). Frames with no clear holder are neutral: they do not
change the confirmed holder, and they expire a pending candidate only after a long gap.
"""

from src.history.event_store import TransferEvent
from src.transfer.rules import TransferParams


class TransferStateMachine:
    def __init__(self, video: str, object_class: str, params: TransferParams) -> None:
        self.video, self.object_class, self.params = video, object_class, params
        self.holder: int | None = None
        self._last_seen: dict[int, int] = {}
        self._candidate: int | None = None
        self._first = self._last = self._count = 0

    def _reset_candidate(self) -> None:
        self._candidate = None
        self._count = 0

    def update(self, frame: int, holder: int | None) -> TransferEvent | None:
        """Consume one frame's holder (None if unclear). Returns an event when confirmed."""
        if holder is None:
            if self._candidate is not None and frame - self._last > self.params.max_gap_frames:
                self._reset_candidate()
            return None
        self._last_seen[holder] = frame
        if holder == self.holder:
            self._reset_candidate()
            return None
        if holder != self._candidate or frame - self._last > self.params.max_gap_frames:
            self._candidate, self._first, self._count = holder, frame, 0
        self._count += 1
        self._last = frame
        if self._count < self.params.min_hold_frames:
            return None
        previous, first = self.holder, self._first
        self.holder = holder
        self._reset_candidate()
        if previous is None:
            return None
        return TransferEvent(
            self.video, self.object_class, previous, holder,
            min(self._last_seen[previous], first), first, frame)
