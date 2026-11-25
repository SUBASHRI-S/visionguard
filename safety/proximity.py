"""Person-to-forklift proximity, measured in the image.

THIS IS A SCREENING HEURISTIC, NOT PHYSICAL DISTANCE. Perspective, camera angle and height all
distort it. The distance is measured from the person's foot point to the forklift's ground edge
(the bottom edge of its box), then divided by the person's box height. Dividing by height gives
a rough scale-invariance: the same scene filmed at twice the resolution gives the same ratio.

Operators (people inside a forklift) are skipped. Unreliable people give 'unknown', never a warning.
"""
import math
from dataclasses import dataclass

from safety.config import ProximityConfig
from safety.contracts import Answer, Box, Detection, PPEStatus


@dataclass(frozen=True)
class ProximityResult:
    person_index: int
    forklift_index: int
    ratio: float | None  # distance / person height; None when not computed
    near: Answer  # 'yes' = closer than warn_ratio, 'no' = farther, 'unknown' = cannot tell
    reasons: tuple[str, ...] = ()


def distance_to_ground_edge(point: tuple[float, float], forklift: Box) -> float:
    """Distance from a point to the forklift's bottom edge (a horizontal segment)."""
    px, py = point
    nearest_x = min(max(px, forklift.x1), forklift.x2)
    return math.hypot(px - nearest_x, py - forklift.y2)


def check_proximity(
    statuses: list[PPEStatus],
    forklifts: list[Detection],
    frame_height: int,
    config: ProximityConfig | None = None,
) -> list[ProximityResult]:
    """One result per (non-operator person, forklift) pair."""
    config = config or ProximityConfig()
    results: list[ProximityResult] = []
    for status in statuses:
        if status.is_operator:
            continue
        person = status.person
        reasons: list[str] = []
        if person.confidence < config.min_person_confidence:
            reasons.append("person confidence too low")
        if person.box.height < config.min_person_height_px:
            reasons.append("person too small to measure against")
        if person.box.y2 >= frame_height - config.edge_margin_px:
            reasons.append("feet may be cut off by the bottom of the frame")
        for forklift_index, forklift in enumerate(forklifts):
            if reasons:
                results.append(
                    ProximityResult(status.person_index, forklift_index, None, "unknown", tuple(reasons))
                )
                continue
            ratio = distance_to_ground_edge(person.box.foot_point, forklift.box) / person.box.height
            near: Answer = "yes" if ratio <= config.warn_ratio else "no"
            results.append(ProximityResult(status.person_index, forklift_index, ratio, near))
    return results
