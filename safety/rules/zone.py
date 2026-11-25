"""Restricted-zone intrusion, decided on the person's foot point."""
from safety.config import ZoneConfig
from safety.events import Event, FrameContext, make_event
from safety.zones import person_in_zone


class ZoneIntrusionRule:
    rule_id = "zone_intrusion"

    def __init__(self, config: ZoneConfig | None = None, ignore_operators: bool = True) -> None:
        """``ignore_operators``: a person seated in a forklift is assumed authorised to drive it."""
        self.config = config or ZoneConfig()
        self.ignore_operators = ignore_operators

    def evaluate(self, ctx: FrameContext) -> list[Event]:
        events: list[Event] = []
        for zone in ctx.zones:
            for status in ctx.statuses:
                if self.ignore_operators and status.is_operator:
                    continue
                result = person_in_zone(
                    status.person, zone, ctx.frame_width, ctx.frame_height, self.config
                )
                if result.answer == "no":
                    continue
                outcome = "violation" if result.answer == "yes" else "uncertain"
                events.append(
                    make_event(
                        self.rule_id, ctx, status, outcome, "high",
                        zone=zone.name, reasons=list(result.reasons),
                    )
                )
        return events
