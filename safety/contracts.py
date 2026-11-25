"""Plain data types shared by the safety modules.

Nothing here touches a model, a camera or a database, so everything built on these types can
be unit-tested with made-up boxes.
"""
from dataclasses import dataclass
from typing import Literal

Answer = Literal["yes", "no", "unknown"]

PERSON = "person"
HELMET = "helmet"
VEST = "vest"
FORKLIFT = "forklift"


@dataclass(frozen=True)
class Box:
    """Axis-aligned box in pixels: (x1, y1) top-left, (x2, y2) bottom-right."""

    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        if self.x2 < self.x1 or self.y2 < self.y1:
            raise ValueError(f"invalid box: {self}")

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return (self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2

    @property
    def foot_point(self) -> tuple[float, float]:
        """Bottom-middle of the box: where a standing person touches the ground."""
        return (self.x1 + self.x2) / 2, self.y2

    def intersection_area(self, other: "Box") -> float:
        width = min(self.x2, other.x2) - max(self.x1, other.x1)
        height = min(self.y2, other.y2) - max(self.y1, other.y1)
        return max(0.0, width) * max(0.0, height)

    def contains_point(self, x: float, y: float) -> bool:
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2


@dataclass(frozen=True)
class Detection:
    class_name: str
    confidence: float
    box: Box


@dataclass(frozen=True)
class PPEStatus:
    """What we conclude about one person.

    ``helmet`` and ``vest`` are 'yes', 'no' or 'unknown'. Only 'no' may ever raise a
    violation. ``reasons`` explains every 'unknown' so it can be shown and audited.
    ``is_operator`` marks a person who appears to be inside a forklift.
    """

    person_index: int
    person: Detection
    helmet: Answer
    vest: Answer
    is_operator: bool = False
    reasons: tuple[str, ...] = ()
