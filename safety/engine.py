"""Run the enabled rules over one frame of detections.

One rule crashing must not stop the others or the video pipeline, so each rule runs in
isolation: a failure is logged with its rule id and the remaining rules still run.
"""
import logging
from typing import Any

from safety.association import associate
from safety.config import PPEConfig, ProximityConfig, SafetyConfigError, ZoneConfig
from safety.contracts import FORKLIFT, Detection
from safety.events import Event, FrameContext, Rule
from safety.rules import NoHelmetRule, NoVestRule, ProximityRule, ZoneIntrusionRule
from safety.zones import Zone

logger = logging.getLogger(__name__)

ALL_RULES = ("no_helmet", "no_vest", "zone_intrusion", "forklift_proximity")
ENGINE_KEYS = {"enabled_rules", "ppe", "zone", "proximity"}


class SafetyEngine:
    def __init__(self, rules: list[Rule], ppe_config: PPEConfig | None = None) -> None:
        ids = [r.rule_id for r in rules]
        if len(set(ids)) != len(ids):
            raise SafetyConfigError([f"duplicate rule ids: {sorted(i for i in ids if ids.count(i) > 1)}"])
        self.rules = list(rules)
        self.ppe_config = ppe_config or PPEConfig()

    @classmethod
    def from_config(cls, data: dict[str, Any] | None = None) -> "SafetyEngine":
        """Build from a validated mapping. Unknown keys and unknown rule ids are rejected."""
        data = data or {}
        if not isinstance(data, dict):
            raise SafetyConfigError(["engine config must be a mapping"])
        errors: list[str] = []
        extra = sorted(set(data) - ENGINE_KEYS)
        if extra:
            errors.append(f"unknown engine key(s) {extra}")
        enabled = data.get("enabled_rules", list(ALL_RULES))
        if not isinstance(enabled, list) or not enabled:
            errors.append("enabled_rules must be a non-empty list")
            enabled = []
        unknown = sorted({r for r in enabled if r not in ALL_RULES}, key=str)
        if unknown:
            errors.append(f"unknown rule id(s) {unknown}; known: {list(ALL_RULES)}")
        if errors:
            raise SafetyConfigError(errors)

        ppe = PPEConfig.from_dict(data.get("ppe"))
        zone = ZoneConfig.from_dict(data.get("zone"))
        proximity = ProximityConfig.from_dict(data.get("proximity"))
        factories = {
            "no_helmet": lambda: NoHelmetRule(),
            "no_vest": lambda: NoVestRule(),
            "zone_intrusion": lambda: ZoneIntrusionRule(zone),
            "forklift_proximity": lambda: ProximityRule(proximity),
        }
        return cls([factories[rule_id]() for rule_id in enabled], ppe)

    def build_context(
        self,
        detections: list[Detection],
        *,
        camera_id: str,
        frame_index: int,
        timestamp: float,
        frame_width: int,
        frame_height: int,
        zones: tuple[Zone, ...] = (),
        track_ids: dict[int, int] | None = None,
    ) -> FrameContext:
        return FrameContext(
            camera_id=camera_id,
            frame_index=frame_index,
            timestamp=timestamp,
            frame_width=frame_width,
            frame_height=frame_height,
            statuses=tuple(associate(detections, frame_width, frame_height, self.ppe_config)),
            forklifts=tuple(d for d in detections if d.class_name == FORKLIFT),
            zones=tuple(zones),
            track_ids=dict(track_ids or {}),
        )

    def evaluate(self, ctx: FrameContext) -> list[Event]:
        events: list[Event] = []
        for rule in self.rules:
            try:
                events.extend(rule.evaluate(ctx))
            except Exception:  # noqa: BLE001 - isolate a failing rule, keep the pipeline alive
                logger.exception("rule %s failed on frame %s", rule.rule_id, ctx.frame_index)
        return events

    def process(self, detections: list[Detection], **frame: Any) -> list[Event]:
        """Convenience: detections in, events out. ``frame`` takes build_context's keyword arguments."""
        return self.evaluate(self.build_context(detections, **frame))
