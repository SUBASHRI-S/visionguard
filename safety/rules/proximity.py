"""Person too close to a forklift (image-space screening heuristic, not physical distance)."""
from safety.config import ProximityConfig
from safety.events import Event, FrameContext, make_event
from safety.proximity import check_proximity


class ProximityRule:
    rule_id = "forklift_proximity"

    def __init__(self, config: ProximityConfig | None = None) -> None:
        self.config = config or ProximityConfig()

    def evaluate(self, ctx: FrameContext) -> list[Event]:
        by_index = {s.person_index: s for s in ctx.statuses}
        events: list[Event] = []
        for result in check_proximity(
            list(ctx.statuses), list(ctx.forklifts), ctx.frame_height, self.config
        ):
            if result.near == "no":
                continue
            outcome = "violation" if result.near == "yes" else "uncertain"
            events.append(
                make_event(
                    self.rule_id, ctx, by_index[result.person_index], outcome, "high",
                    forklift_index=result.forklift_index,
                    ratio=result.ratio,
                    reasons=list(result.reasons),
                )
            )
        return events
