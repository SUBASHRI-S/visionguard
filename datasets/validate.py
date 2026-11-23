"""Validate a dataset stored in YOLO detection format.

Expected layout:
    <root>/images/<split>/*.jpg
    <root>/labels/<split>/*.txt   (one line per object: class cx cy w h, all 0-1)
"""
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass
class ValidationReport:
    num_images: int = 0
    class_counts: Counter = field(default_factory=Counter)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def _validate_label_file(
    path: Path, split: str, num_classes: int, report: ValidationReport
) -> None:
    seen: set[str] = set()
    for line_no, raw in enumerate(path.read_text().splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        where = f"[{split}] {path.name}:{line_no}"
        parts = line.split()
        if len(parts) != 5:
            report.errors.append(f"{where} expected 5 values, got {len(parts)}")
            continue
        try:
            class_id = int(parts[0])
            cx, cy, w, h = (float(v) for v in parts[1:])
        except ValueError:
            report.errors.append(f"{where} non-numeric value")
            continue
        if not 0 <= class_id < num_classes:
            report.errors.append(f"{where} class id {class_id} out of range")
            continue
        if not all(0.0 <= v <= 1.0 for v in (cx, cy, w, h)):
            report.errors.append(f"{where} coordinate outside 0-1")
            continue
        if w <= 0 or h <= 0:
            report.errors.append(f"{where} box has zero width or height")
            continue
        if line in seen:
            report.warnings.append(f"{where} duplicate box")
        seen.add(line)
        report.class_counts[class_id] += 1


def _validate_split(
    root: Path, split: str, num_classes: int, report: ValidationReport
) -> None:
    image_dir = root / "images" / split
    label_dir = root / "labels" / split
    for folder in (image_dir, label_dir):
        if not folder.is_dir():
            report.errors.append(f"missing folder: {folder}")
            return

    images = {
        p.stem: p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
    }
    labels = {p.stem: p for p in label_dir.glob("*.txt")}
    report.num_images += len(images)

    for stem in sorted(labels.keys() - images.keys()):
        report.errors.append(f"[{split}] label without image: {stem}.txt")

    for stem in sorted(images):
        label_path = labels.get(stem)
        if label_path is None:
            report.warnings.append(
                f"[{split}] no label file (treated as background): {images[stem].name}"
            )
        else:
            _validate_label_file(label_path, split, num_classes, report)


def validate_dataset(
    root: Path, class_names: list[str], splits: tuple[str, ...] = ("train", "val")
) -> ValidationReport:
    """Check a YOLO-format dataset and return a report of errors and warnings."""
    report = ValidationReport()
    for split in splits:
        _validate_split(root, split, len(class_names), report)
    for class_id, name in enumerate(class_names):
        if report.class_counts[class_id] == 0:
            report.warnings.append(f"class '{name}' has no instances")
    return report