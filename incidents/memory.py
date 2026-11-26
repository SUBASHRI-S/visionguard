"""Equipment does not vanish: remember what each tracked person was seen wearing.

Why: on a real clip, a worker in a clearly visible vest was called "no vest" on more than a third
of frames, because the detector finds the vest in some frames and misses it in others (it depends
on viewing angle). Treating every miss as "no vest" turns detector flicker into false incidents.

The rule: if a tracked person was seen **with** an item within the last ``seconds``, then a later
"no" for that item is downgraded to "unknown", because a detector miss is far more likely than
the person removing the item in that time. A person never seen with the item is unaffected, so a
real violator is still caught (after the normal debounce). The price: someone who really takes a
vest off is noticed ``seconds`` later than they otherwise would be.

Time is video time (seconds), not frames, so the behaviour does not depend on frame rate or stride.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping, Sequence

from safety.contracts import PPEStatus

ITEMS = ("helmet", "vest")


class MemoryConfigError(ValueError):
    """Raised when memory settings are invalid."""


@dataclass(frozen=True)
class MemoryConfig:
    seconds: float = 5.0  # PLACEHOLDER chosen by reasoning, not tuned; 0 would mean "no memory"

    def __post_init__(self) -> None:
        if isinstance(self.seconds, bool) or not isinstance(self.seconds, (int, float)) or not 0 < self.seconds <= 600:
            raise MemoryConfigError(f"seconds must be a number between 0 and 600, got {self.seconds!r}")


class PPEMemory:
    """One instance per stream (it keys on track ids, which are only unique within a stream)."""

    def __init__(self, config: MemoryConfig | None = None) -> None:
        self.config = config or MemoryConfig()
        self._last_yes: dict[tuple[int, str], float] = {}
        self.overrides = 0  # how many "no" answers were downgraded; reported so the effect is visible

    def apply(self, statuses: Sequence[PPEStatus], track_ids: Mapping[int, int], timestamp: float) -> list[PPEStatus]:
        horizon = self.config.seconds
        # forget old entries so memory cannot grow without bound over a long stream
        self._last_yes = {k: t for k, t in self._last_yes.items() if 0 <= timestamp - t <= horizon}
        out: list[PPEStatus] = []
        for status in statuses:
            track = track_ids.get(status.person_index)
            if track is None:                      # no stable identity, so nothing to remember against
                out.append(status)
                continue
            changes: dict[str, object] = {}
            reasons = list(status.reasons)
            for item in ITEMS:
                answer = getattr(status, item)
                key = (track, item)
                if answer == "yes":
                    self._last_yes[key] = timestamp
                elif answer == "no" and key in self._last_yes:
                    age = timestamp - self._last_yes[key]
                    changes[item] = "unknown"
                    reasons.append(f"{item} not detected now but seen {age:.1f}s ago")
                    self.overrides += 1
            out.append(replace(status, reasons=tuple(reasons), **changes) if changes else status)
        return out
