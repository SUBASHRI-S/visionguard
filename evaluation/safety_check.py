"""How much does the detector's mistakes change the safety conclusions?

For every image we run the same PPE association twice: once on the ground-truth boxes (what a
perfect detector would give) and once on the model's detections. Persons are matched between the
two, and for each match we compare the helmet/vest answer.

IMPORTANT LIMIT: the 'truth' here is derived from the labels with the same association rules, so
this measures what the DETECTOR costs the safety logic. It does not prove the rules themselves
are right; that needs hand-labelled real clips (see docs/REAL_FOOTAGE_PROTOCOL.md).
"""
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from safety.contracts import Box, PPEStatus

ANSWERS = ("yes", "no", "unknown")
ITEMS = ("helmet", "vest")


@dataclass
class ImageComparison:
    matched: list[tuple[PPEStatus, PPEStatus]] = field(default_factory=list)  # (truth, predicted)
    truth_unmatched: list[PPEStatus] = field(default_factory=list)  # persons the detector missed
    pred_unmatched: list[PPEStatus] = field(default_factory=list)  # persons that are not there

    @property
    def disagreements(self) -> int:
        count = len(self.truth_unmatched) + len(self.pred_unmatched)
        for truth, pred in self.matched:
            count += sum(getattr(truth, i) != getattr(pred, i) for i in ITEMS)
        return count


def _iou(a: Box, b: Box) -> float:
    inter = a.intersection_area(b)
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def compare_image(
    truth: list[PPEStatus], pred: list[PPEStatus], iou_threshold: float = 0.5
) -> ImageComparison:
    """Greedy one-to-one matching of persons by box overlap, best overlap first."""
    pairs = sorted(
        (
            (_iou(t.person.box, p.person.box), ti, pi)
            for ti, t in enumerate(truth)
            for pi, p in enumerate(pred)
        ),
        reverse=True,
    )
    used_t: set[int] = set()
    used_p: set[int] = set()
    result = ImageComparison()
    for overlap, ti, pi in pairs:
        if overlap < iou_threshold:
            break
        if ti in used_t or pi in used_p:
            continue
        used_t.add(ti)
        used_p.add(pi)
        result.matched.append((truth[ti], pred[pi]))
    result.truth_unmatched = [t for i, t in enumerate(truth) if i not in used_t]
    result.pred_unmatched = [p for i, p in enumerate(pred) if i not in used_p]
    return result


def summarize(comparisons: list[ImageComparison]) -> dict[str, Any]:
    """Count outcomes. table[item][truth_answer] -> how the detector-based answer came out."""
    table: dict[str, dict[str, Counter]] = {i: {a: Counter() for a in ANSWERS} for i in ITEMS}
    matched = missed = extra = 0
    for comparison in comparisons:
        matched += len(comparison.matched)
        missed += len(comparison.truth_unmatched)
        extra += len(comparison.pred_unmatched)
        for truth, pred in comparison.matched:
            for item in ITEMS:
                table[item][getattr(truth, item)][getattr(pred, item)] += 1
        for truth in comparison.truth_unmatched:
            for item in ITEMS:
                table[item][getattr(truth, item)]["person_missed"] += 1
    return {
        "images": len(comparisons),
        "persons_matched": matched,
        "persons_missed": missed,
        "persons_extra": extra,
        "table": {i: {a: dict(c) for a, c in rows.items()} for i, rows in table.items()},
    }


def _pct(n: int, total: int) -> str:
    return f"{n} ({100 * n / total:.0f}%)" if total else "0"


def to_markdown(
    summary: dict[str, Any], meta: dict[str, Any], events: dict[str, dict[str, int]]
) -> str:
    lines = [
        f"# Safety check: {meta['run_name']} on the {meta['split']} split",
        "",
        f"- Dataset version `{meta['dataset_version']}`, detector confidence {meta['conf']}, "
        f"inference size {meta['imgsz']}, person match IoU {meta['iou']}",
        f"- {summary['images']} images. Persons: {summary['persons_matched']} matched, "
        f"{summary['persons_missed']} missed by the detector, {summary['persons_extra']} detected but not in the labels",
        "- Safety thresholds are the untuned placeholder defaults unless a config was given.",
        "- 'Truth' = the same association rules applied to the ground-truth labels. This measures "
        "what detector errors do to the safety conclusions, not whether the rules are right.",
        "",
    ]
    for item in ITEMS:
        lines += [f"## {item.capitalize()}: what the system concluded, by what the labels say", ""]
        lines += [
            "| Labels say | People | Said yes | Said no | Said unknown | Person not detected |",
            "|---|---|---|---|---|---|",
        ]
        for answer, label in (("no", "no (a real violation)"), ("yes", "yes (compliant)"),
                              ("unknown", "unknown (unclear in the labels)")):
            row = summary["table"][item][answer]
            total = sum(row.values())
            lines.append(
                f"| {label} | {total} | {_pct(row.get('yes', 0), total)} | {_pct(row.get('no', 0), total)} | "
                f"{_pct(row.get('unknown', 0), total)} | {_pct(row.get('person_missed', 0), total)} |"
            )
        lines.append("")
    lines += ["## Events the engine raised on the model's detections", "",
              "| Rule | Violations | Uncertain |", "|---|---|---|"]
    for rule_id, counts in sorted(events.items()):
        lines.append(f"| {rule_id} | {counts.get('violation', 0)} | {counts.get('uncertain', 0)} |")
    lines.append("")
    return "\n".join(lines)
