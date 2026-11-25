"""Missing-PPE rules. Only an explicit 'no' is a violation; 'unknown' is reported as uncertain."""
from safety.events import Event, FrameContext, Severity, make_event


class _MissingPPERule:
    rule_id: str
    item: str  # attribute of PPEStatus: 'helmet' or 'vest'
    severity: Severity

    def evaluate(self, ctx: FrameContext) -> list[Event]:
        events: list[Event] = []
        for status in ctx.statuses:
            answer = getattr(status, self.item)
            if answer == "yes":
                continue
            outcome = "violation" if answer == "no" else "uncertain"
            events.append(
                make_event(
                    self.rule_id, ctx, status, outcome, self.severity,
                    item=self.item, reasons=list(status.reasons),
                )
            )
        return events


class NoHelmetRule(_MissingPPERule):
    rule_id = "no_helmet"
    item = "helmet"
    severity: Severity = "high"


class NoVestRule(_MissingPPERule):
    rule_id = "no_vest"
    item = "vest"
    severity: Severity = "medium"
