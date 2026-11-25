"""What rules receive (FrameContext) and what they produce (Event).

A rule looks at one frame and returns events. 'compliant' produces no event. 'violation' means
the evidence shows a breach. 'uncertain' means we could not tell; it is reported so that later
logic can hold an open incident steady instead of treating the gap as 'all clear'.
"""
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from safety.contracts import Detection, PPEStatus
from safety.zones import Zone

Outcome = Literal["violation", "uncertain"]
Severity = Literal["low", "medium", "high"]


@dataclass(frozen=True)
class FrameContext:
    camera_id: str
    frame_index: int
    timestamp: float
    frame_width: int
    frame_height: int
    statuses: tuple[PPEStatus, ...]
    forklifts: tuple[Detection, ...] = ()
    zones: tuple[Zone, ...] = ()
    track_ids: dict[int, int] = field(default_factory=dict)  # person_index -> track id (from tracking)


@dataclass(frozen=True)
class Event:
    rule_id: str
    camera_id: str
    frame_index: int
    timestamp: float
    outcome: Outcome
    severity: Severity
    person_index: int
    track_id: int | None
    details: dict[str, Any]


class Rule(Protocol):
    rule_id: str

    def evaluate(self, ctx: FrameContext) -> list[Event]: ...


def make_event(
    rule_id: str,
    ctx: FrameContext,
    status: PPEStatus,
    outcome: Outcome,
    severity: Severity,
    **details: Any,
) -> Event:
    box = status.person.box
    return Event(
        rule_id=rule_id,
        camera_id=ctx.camera_id,
        frame_index=ctx.frame_index,
        timestamp=ctx.timestamp,
        outcome=outcome,
        severity=severity,
        person_index=status.person_index,
        track_id=ctx.track_ids.get(status.person_index),
        details={"box": (box.x1, box.y1, box.x2, box.y2), **details},
    )
