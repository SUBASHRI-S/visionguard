"""Error analysis: where and why does the detector fail?

Predictions are matched to ground-truth boxes image by image at a fixed confidence threshold
(the threshold the safety rules will use), not the near-zero threshold mAP uses. Each ground
truth box ends up as found, missed, or found under the wrong class, and each unmatched
prediction is a false alarm. Use the VALIDATION split for this, so the test split stays clean.
"""
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

Box = tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels

SIZE_BUCKETS = (("small", 0.0, 0.01), ("medium", 0.01, 0.05), ("large", 0.05, 1.01))


@dataclass(frozen=True)
class GroundTruth:
    class_id: int
    box: Box


@dataclass(frozen=True)
class Prediction:
    class_id: int
    confidence: float
    box: Box


@dataclass
class ImageMatch:
    found: list[tuple[GroundTruth, Prediction]] = field(default_factory=list)
    wrong_class: list[tuple[GroundTruth, Prediction]] = field(default_factory=list)
    missed: list[GroundTruth] = field(default_factory=list)
    false_alarms: list[Prediction] = field(default_factory=list)

    @property
    def error_count(self) -> int:
        return len(self.wrong_class) + len(self.missed) + len(self.false_alarms)


def parse_labels(text: str, width: int, height: int) -> list[GroundTruth]:
    """YOLO label text (class cx cy w h, normalised) to pixel boxes."""
    boxes = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        class_id = int(float(parts[0]))
        cx, cy, w, h = (float(v) for v in parts[1:5])
        boxes.append(
            GroundTruth(
                class_id,
                ((cx - w / 2) * width, (cy - h / 2) * height,
                 (cx + w / 2) * width, (cy + h / 2) * height),
            )
        )
    return boxes


def iou(a: Box, b: Box) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def size_bucket(box: Box, width: int, height: int) -> str:
    """Bucket by box area as a share of the image: small <1%, medium 1-5%, large >5%."""
    share = max(0.0, (box[2] - box[0]) * (box[3] - box[1])) / (width * height)
    for name, low, high in SIZE_BUCKETS:
        if low <= share < high:
            return name
    return "large"


def match_image(
    truths: Sequence[GroundTruth], predictions: Sequence[Prediction], iou_threshold: float = 0.5
) -> ImageMatch:
    """Greedy matching, highest confidence first. Same class and IoU >= threshold is a hit."""
    result = ImageMatch()
    free_truths = list(truths)
    leftovers: list[Prediction] = []
    for pred in sorted(predictions, key=lambda p: p.confidence, reverse=True):
        candidates = [
            (iou(t.box, pred.box), index)
            for index, t in enumerate(free_truths)
            if t.class_id == pred.class_id
        ]
        best = max(candidates, default=(0.0, -1))
        if best[0] >= iou_threshold:
            result.found.append((free_truths.pop(best[1]), pred))
        else:
            leftovers.append(pred)

    for pred in leftovers:
        candidates = [(iou(t.box, pred.box), index) for index, t in enumerate(free_truths)]
        best = max(candidates, default=(0.0, -1))
        if best[0] >= iou_threshold:
            result.wrong_class.append((free_truths.pop(best[1]), pred))
        else:
            result.false_alarms.append(pred)
    result.missed = free_truths
    return result


def analyze(
    images: Sequence[Path],
    label_dir: Path,
    predict: Callable[[Path], tuple[int, int, list[Prediction]]],
    class_names: list[str],
    iou_threshold: float = 0.5,
) -> tuple[dict[str, Any], dict[str, ImageMatch]]:
    """Run ``predict`` on each image and aggregate the match results.

    ``predict(path)`` returns (width, height, predictions). Returns the summary and the
    per-image matches (for drawing the worst images).
    """
    per_class = {n: Counter() for n in class_names}
    confusion: dict[str, Counter] = {n: Counter() for n in class_names}
    size_found: dict[str, dict[str, Counter]] = {
        n: {b[0]: Counter() for b in SIZE_BUCKETS} for n in class_names
    }
    matches: dict[str, ImageMatch] = {}

    for path in images:
        width, height, predictions = predict(path)
        label_file = label_dir / f"{path.stem}.txt"
        text = label_file.read_text() if label_file.exists() else ""
        match = match_image(parse_labels(text, width, height), predictions, iou_threshold)
        matches[path.name] = match

        for truth, _ in match.found:
            name = class_names[truth.class_id]
            per_class[name]["tp"] += 1
            bucket = size_found[name][size_bucket(truth.box, width, height)]
            bucket["gt"] += 1
            bucket["found"] += 1
        for truth in match.missed:
            name = class_names[truth.class_id]
            per_class[name]["fn"] += 1
            confusion[name]["background"] += 1
            size_found[name][size_bucket(truth.box, width, height)]["gt"] += 1
        for truth, pred in match.wrong_class:
            name = class_names[truth.class_id]
            per_class[name]["fn"] += 1
            per_class[class_names[pred.class_id]]["fp"] += 1
            confusion[name][class_names[pred.class_id]] += 1
            size_found[name][size_bucket(truth.box, width, height)]["gt"] += 1
        for pred in match.false_alarms:
            per_class[class_names[pred.class_id]]["fp"] += 1

    summary: dict[str, Any] = {"per_class": {}, "recall_by_size": {}, "confusions": {}}
    for name in class_names:
        tp, fn, fp = (per_class[name][k] for k in ("tp", "fn", "fp"))
        summary["per_class"][name] = {
            "tp": tp, "fn": fn, "fp": fp,
            "recall": tp / (tp + fn) if tp + fn else None,
            "precision": tp / (tp + fp) if tp + fp else None,
        }
        summary["recall_by_size"][name] = {
            bucket: {
                "objects": c["gt"],
                "recall": c["found"] / c["gt"] if c["gt"] else None,
            }
            for bucket, c in size_found[name].items()
        }
        summary["confusions"][name] = dict(confusion[name])
    summary["images"] = len(images)
    summary["images_with_errors"] = sum(1 for m in matches.values() if m.error_count)
    return summary, matches


def worst_images(matches: dict[str, ImageMatch], limit: int = 20) -> list[tuple[str, int]]:
    ranked = sorted(matches.items(), key=lambda kv: (-kv[1].error_count, kv[0]))
    return [(name, m.error_count) for name, m in ranked[:limit] if m.error_count]


def to_markdown(summary: dict[str, Any], meta: dict[str, Any], worst: list[tuple[str, int]]) -> str:
    def pct(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.3f}"

    lines = [
        f"# Error analysis: {meta['run_name']} on the {meta['split']} split",
        "",
        f"- Dataset version `{meta['dataset_version']}`, confidence threshold {meta['conf']}, "
        f"IoU threshold {meta['iou']}",
        f"- {summary['images']} images, {summary['images_with_errors']} with at least one error",
        "- Precision and recall here are at one fixed confidence threshold, so they differ from "
        "the mAP-based numbers (which sweep the threshold).",
        "",
        "## Per class",
        "",
        "| Class | Found | Missed or wrong class | False alarms | Recall | Precision |",
        "|---|---|---|---|---|---|",
    ]
    for name, row in summary["per_class"].items():
        lines.append(
            f"| {name} | {row['tp']} | {row['fn']} | {row['fp']} | "
            f"{pct(row['recall'])} | {pct(row['precision'])} |"
        )
    lines += ["", "## Recall by object size", "",
              "Size is the box area as a share of the image: small <1%, medium 1-5%, large >5%.", "",
              "| Class | Small (n) | Medium (n) | Large (n) |", "|---|---|---|---|"]
    for name, buckets in summary["recall_by_size"].items():
        cells = [
            f"{pct(buckets[b]['recall'])} ({buckets[b]['objects']})" for b in ("small", "medium", "large")
        ]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines += ["", "## What each class was mistaken for", ""]
    any_confusion = False
    for name, row in summary["confusions"].items():
        if row:
            any_confusion = True
            parts = ", ".join(f"{k}: {v}" for k, v in sorted(row.items(), key=lambda kv: -kv[1]))
            lines.append(f"- Real **{name}** objects not found as {name}: {parts}")
    if not any_confusion:
        lines.append("- None.")
    lines += ["", "(`background` means the object was missed entirely.)", "",
              "## Images with the most errors", ""]
    lines += [f"- `{name}`: {count} error(s)" for name, count in worst] or ["- None."]
    return "\n".join(lines) + "\n"


def draw_image(image, match: ImageMatch, class_names: list[str]):
    """Draw ground truth (green), correct hits (blue), misses (red) and false alarms (orange)."""
    import cv2

    canvas = image.copy()

    def rect(box: Box, color: tuple[int, int, int], label: str, thickness: int = 2) -> None:
        p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
        cv2.rectangle(canvas, p1, p2, color, thickness)
        cv2.putText(canvas, label, (p1[0], max(12, p1[1] - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    for truth in match.missed:
        rect(truth.box, (0, 0, 255), f"MISSED {class_names[truth.class_id]}", 3)
    for truth, pred in match.wrong_class:
        rect(truth.box, (0, 0, 255), f"is {class_names[truth.class_id]}", 3)
        rect(pred.box, (0, 165, 255), f"said {class_names[pred.class_id]} {pred.confidence:.2f}")
    for pred in match.false_alarms:
        rect(pred.box, (0, 165, 255), f"FALSE {class_names[pred.class_id]} {pred.confidence:.2f}")
    for truth, pred in match.found:
        rect(pred.box, (255, 128, 0), f"{class_names[pred.class_id]} {pred.confidence:.2f}", 1)
    return canvas
