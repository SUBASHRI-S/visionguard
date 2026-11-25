"""Thresholds for PPE association, validated at load time.

Defaults are PLACEHOLDERS chosen from the structure of the problem, not tuned on data. Real
values must be chosen on validation detections and recorded with the experiment that chose them.
"""
from dataclasses import dataclass, fields
from typing import Any


class SafetyConfigError(ValueError):
    """Raised when a safety configuration is invalid. ``errors`` lists every problem."""

    def __init__(self, errors: list[str]) -> None:
        super().__init__("; ".join(errors))
        self.errors = errors


@dataclass(frozen=True)
class PPEConfig:
    head_region_fraction: float = 0.25   # top share of the person box that counts as the head
    torso_top_fraction: float = 0.20     # torso region runs from here ...
    torso_bottom_fraction: float = 0.65  # ... to here, as shares of the person box height
    vest_overlap_threshold: float = 0.5  # share of the vest box that must lie in the torso region
    helmet_min_width_ratio: float = 0.10  # helmet width / person width: sanity bounds
    helmet_max_width_ratio: float = 0.80
    min_person_confidence: float = 0.40  # below this the person is too uncertain to judge
    min_person_height_px: float = 80.0   # smaller people are 'unknown' (small objects are our weak spot)
    edge_margin_px: float = 5.0          # a box this close to an edge may be cut off
    max_occlusion: float = 0.5           # share of a person covered by another person
    operator_containment: float = 0.7    # share of a person inside a forklift box to call them operator
    helmet_min_confidence: float = 0.0   # matched helmet weaker than this -> 'unknown' (0 = off)
    vest_min_confidence: float = 0.0     # same for vests

    def __post_init__(self) -> None:
        errors: list[str] = []

        def unit(name: str, low: float = 0.0, high: float = 1.0) -> None:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                errors.append(f"{name} must be a number")
            elif not low <= value <= high:
                errors.append(f"{name} must be between {low} and {high}")

        for name in (
            "head_region_fraction", "torso_top_fraction", "torso_bottom_fraction",
            "vest_overlap_threshold", "helmet_min_width_ratio", "helmet_max_width_ratio",
            "min_person_confidence", "max_occlusion", "operator_containment",
            "helmet_min_confidence", "vest_min_confidence",
        ):
            unit(name)
        unit("min_person_height_px", 0.0, 10000.0)
        unit("edge_margin_px", 0.0, 1000.0)
        if not errors:
            if self.head_region_fraction <= 0:
                errors.append("head_region_fraction must be above 0")
            if self.torso_top_fraction >= self.torso_bottom_fraction:
                errors.append("torso_top_fraction must be below torso_bottom_fraction")
            if self.helmet_min_width_ratio >= self.helmet_max_width_ratio:
                errors.append("helmet_min_width_ratio must be below helmet_max_width_ratio")
        if errors:
            raise SafetyConfigError(errors)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "PPEConfig":
        """Build from a mapping, rejecting unknown keys so typos cannot pass silently."""
        data = data or {}
        if not isinstance(data, dict):
            raise SafetyConfigError(["ppe config must be a mapping"])
        known = {f.name for f in fields(cls)}
        extra = sorted(set(data) - known)
        if extra:
            raise SafetyConfigError([f"unknown ppe key(s) {extra}"])
        return cls(**data)


def _check_ranges(obj: object, ranges: dict[str, tuple[float, float]]) -> None:
    errors: list[str] = []
    for name, (low, high) in ranges.items():
        value = getattr(obj, name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            errors.append(f"{name} must be a number")
        elif not low <= value <= high:
            errors.append(f"{name} must be between {low} and {high}")
    if errors:
        raise SafetyConfigError(errors)


def _from_mapping(cls: type, data: dict[str, Any] | None, label: str) -> Any:
    data = data or {}
    if not isinstance(data, dict):
        raise SafetyConfigError([f"{label} config must be a mapping"])
    extra = sorted(set(data) - {f.name for f in fields(cls)})
    if extra:
        raise SafetyConfigError([f"unknown {label} key(s) {extra}"])
    return cls(**data)


@dataclass(frozen=True)
class ZoneConfig:
    """When is a person's foot point trustworthy enough to decide 'inside' or 'outside'?"""

    min_person_confidence: float = 0.40
    edge_margin_px: float = 5.0  # feet this close to the bottom edge may be cut off

    def __post_init__(self) -> None:
        _check_ranges(self, {"min_person_confidence": (0, 1), "edge_margin_px": (0, 1000)})

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ZoneConfig":
        return _from_mapping(cls, data, "zone")


@dataclass(frozen=True)
class ProximityConfig:
    """Image-space person-to-forklift screening. PLACEHOLDER values, to be tuned on real clips."""

    warn_ratio: float = 1.0  # warn when distance is below this many person-heights
    min_person_confidence: float = 0.40
    min_person_height_px: float = 80.0  # dividing by a tiny height makes the ratio noise
    edge_margin_px: float = 5.0

    def __post_init__(self) -> None:
        _check_ranges(
            self,
            {
                "warn_ratio": (0.01, 20),
                "min_person_confidence": (0, 1),
                "min_person_height_px": (0, 10000),
                "edge_margin_px": (0, 1000),
            },
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ProximityConfig":
        return _from_mapping(cls, data, "proximity")
