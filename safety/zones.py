"""Restricted zones: polygons in normalised (0-1) coordinates, tested with the foot point.

Normalised coordinates survive a change of video resolution. The test uses the person's foot
point (bottom-middle of the box) because feet show where someone stands. When the feet may be
cut off by the bottom of the frame, or the detection is weak, the answer is 'unknown'.
"""
from dataclasses import dataclass
from typing import Any

from safety.config import ZoneConfig
from safety.contracts import Answer, Detection


class ZoneError(ValueError):
    """Raised for an invalid zone definition."""


Point = tuple[float, float]


def _area(polygon: tuple[Point, ...]) -> float:
    total = 0.0
    for i, (x1, y1) in enumerate(polygon):
        x2, y2 = polygon[(i + 1) % len(polygon)]
        total += x1 * y2 - x2 * y1
    return abs(total) / 2


@dataclass(frozen=True)
class Zone:
    name: str
    polygon: tuple[Point, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ZoneError("zone name must be a non-empty string")
        if len(self.polygon) < 3:
            raise ZoneError("a zone needs at least 3 points")
        for point in self.polygon:
            if len(point) != 2 or any(
                isinstance(v, bool) or not isinstance(v, (int, float)) or not 0.0 <= v <= 1.0
                for v in point
            ):
                raise ZoneError(f"zone points must be (x, y) numbers between 0 and 1, got {point}")
        if _area(self.polygon) < 1e-6:
            raise ZoneError("zone polygon has no area (points are in a line or identical)")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Zone":
        if not isinstance(data, dict) or set(data) != {"name", "polygon"}:
            raise ZoneError("a zone is a mapping with exactly the keys 'name' and 'polygon'")
        points = tuple(tuple(p) for p in data["polygon"])  # type: ignore[arg-type]
        return cls(name=data["name"], polygon=points)  # type: ignore[arg-type]


def point_in_polygon(x: float, y: float, polygon: tuple[Point, ...]) -> bool:
    """Ray casting. A point exactly on an edge counts as inside."""
    eps = 1e-12
    inside = False
    for i, (x1, y1) in enumerate(polygon):
        x2, y2 = polygon[(i + 1) % len(polygon)]
        on_line = abs((x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)) <= eps
        if on_line and min(x1, x2) - eps <= x <= max(x1, x2) + eps and (
            min(y1, y2) - eps <= y <= max(y1, y2) + eps
        ):
            return True
        if (y1 > y) != (y2 > y):
            crossing_x = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if crossing_x > x:
                inside = not inside
    return inside


@dataclass(frozen=True)
class ZoneResult:
    answer: Answer  # 'yes' = inside, 'no' = outside, 'unknown' = cannot tell
    reasons: tuple[str, ...] = ()


def person_in_zone(
    person: Detection,
    zone: Zone,
    frame_width: int,
    frame_height: int,
    config: ZoneConfig | None = None,
) -> ZoneResult:
    config = config or ZoneConfig()
    reasons: list[str] = []
    if person.confidence < config.min_person_confidence:
        reasons.append("person confidence too low")
    if person.box.y2 >= frame_height - config.edge_margin_px:
        reasons.append("feet may be cut off by the bottom of the frame")
    if reasons:
        return ZoneResult("unknown", tuple(reasons))
    fx, fy = person.box.foot_point
    inside = point_in_polygon(fx / frame_width, fy / frame_height, zone.polygon)
    return ZoneResult("yes" if inside else "no")
