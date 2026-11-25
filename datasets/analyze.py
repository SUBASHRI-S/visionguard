"""Measure what is actually in a YOLO-format dataset (class balance, object sizes, lighting).

Size and brightness buckets are heuristics for error analysis (Milestone 4), not
physical measurements. Object size is the box area as a fraction of image area.
"""
from collections import Counter
from itertools import combinations
from pathlib import Path

from PIL import Image

from datasets.validate import IMAGE_EXTENSIONS


def _size_bucket(area: float, small: float, medium: float) -> str:
    if area < small:
        return "small"
    if area < medium:
        return "medium"
    return "large"


def _brightness(image: Image.Image) -> float:
    gray = image.convert("L").resize((64, 64), Image.Resampling.BILINEAR)
    data = gray.tobytes()
    return sum(data) / len(data)


def _parse_labels(path: Path) -> tuple[list[tuple[int, float]], int]:
    """Return ([(class_id, relative_area)], malformed_line_count)."""
    objects: list[tuple[int, float]] = []
    malformed = 0
    if not path.exists():
        return objects, malformed
    for raw in path.read_text().splitlines():
        parts = raw.split()
        if not parts:
            continue
        try:
            class_id = int(parts[0])
            width, height = float(parts[3]), float(parts[4])
        except (ValueError, IndexError):
            malformed += 1
            continue
        objects.append((class_id, width * height))
    return objects, malformed


def _analyze_split(root, split, class_names, small, medium, dark_below, bright_above):
    image_dir = root / "images" / split
    label_dir = root / "labels" / split
    images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)

    class_instances: Counter = Counter()
    class_images: Counter = Counter()
    class_sizes: dict[str, Counter] = {name: Counter() for name in class_names}
    brightness: Counter = Counter()
    resolutions: Counter = Counter()
    cooccurrence: Counter = Counter()
    unreadable: list[str] = []
    background = 0
    malformed_total = 0
    objects_per_image: list[int] = []

    for image_path in images:
        try:
            with Image.open(image_path) as image:
                resolutions[image.size] += 1
                level = _brightness(image)
        except (OSError, ValueError, ImportError):  # ImportError: ultralytics patches PIL.Image.open
            unreadable.append(image_path.name)
            continue
        if level < dark_below:
            brightness["dark"] += 1
        elif level > bright_above:
            brightness["bright"] += 1
        else:
            brightness["normal"] += 1

        objects, malformed = _parse_labels(label_dir / f"{image_path.stem}.txt")
        malformed_total += malformed
        objects = [(cid, area) for cid, area in objects if 0 <= cid < len(class_names)]
        objects_per_image.append(len(objects))
        if not objects:
            background += 1
        present = set()
        for class_id, area in objects:
            name = class_names[class_id]
            class_instances[name] += 1
            class_sizes[name][_size_bucket(area, small, medium)] += 1
            present.add(name)
        for name in present:
            class_images[name] += 1
        for a, b in combinations(sorted(present), 2):
            cooccurrence[f"{a}+{b}"] += 1

    readable = len(images) - len(unreadable)
    return {
        "images": len(images),
        "readable_images": readable,
        "unreadable_images": unreadable,
        "background_images": background,
        "malformed_label_lines": malformed_total,
        "distinct_resolutions": len(resolutions),
        "most_common_resolutions": [
            {"size": list(size), "count": count} for size, count in resolutions.most_common(5)
        ],
        "brightness": {k: brightness.get(k, 0) for k in ("dark", "normal", "bright")},
        "objects_per_image_mean": (
            round(sum(objects_per_image) / len(objects_per_image), 2) if objects_per_image else 0
        ),
        "objects_per_image_max": max(objects_per_image, default=0),
        "classes": {
            name: {
                "instances": class_instances.get(name, 0),
                "images": class_images.get(name, 0),
                "small": class_sizes[name].get("small", 0),
                "medium": class_sizes[name].get("medium", 0),
                "large": class_sizes[name].get("large", 0),
            }
            for name in class_names
        },
        "cooccurrence": dict(sorted(cooccurrence.items())),
    }


def analyze_dataset(
    root: Path,
    class_names: list[str],
    splits: tuple[str, ...] = ("train", "val"),
    *,
    small_area: float = 0.005,
    medium_area: float = 0.05,
    dark_below: float = 60,
    bright_above: float = 180,
) -> dict:
    """Return a JSON-serialisable report for each split."""
    return {
        "root": str(root),
        "class_names": class_names,
        "thresholds": {
            "small_area_fraction": small_area,
            "medium_area_fraction": medium_area,
            "dark_below_mean_gray": dark_below,
            "bright_above_mean_gray": bright_above,
        },
        "splits": {
            split: _analyze_split(
                root, split, class_names, small_area, medium_area, dark_below, bright_above
            )
            for split in splits
        },
    }


def report_to_markdown(report: dict, min_instances: int = 30) -> str:
    """Human-readable version of ``analyze_dataset`` output."""
    t = report["thresholds"]
    lines = [
        "# Dataset analysis",
        "",
        f"Source folder: `{report['root']}`",
        "",
        "Object size = box area as a fraction of image area "
        f"(small < {t['small_area_fraction']}, medium < {t['medium_area_fraction']}). "
        f"Brightness = mean gray level (dark < {t['dark_below_mean_gray']}, "
        f"bright > {t['bright_above_mean_gray']}). These are heuristics for error analysis.",
        "",
    ]
    for split, data in report["splits"].items():
        lines += [
            f"## Split: {split}",
            "",
            f"- images: {data['images']} (unreadable: {len(data['unreadable_images'])}, "
            f"without objects: {data['background_images']})",
            f"- distinct resolutions: {data['distinct_resolutions']}",
            f"- objects per image: mean {data['objects_per_image_mean']}, "
            f"max {data['objects_per_image_max']}",
            f"- brightness: {data['brightness']}",
            "",
            "| class | instances | images | small | medium | large | note |",
            "|---|---|---|---|---|---|---|",
        ]
        for name, c in data["classes"].items():
            note = ""
            if c["instances"] == 0:
                note = "none"
            elif c["instances"] < min_instances:
                note = f"few (<{min_instances}): metrics will be noisy"
            lines.append(
                f"| {name} | {c['instances']} | {c['images']} | {c['small']} | "
                f"{c['medium']} | {c['large']} | {note} |"
            )
        lines.append("")
    return "\n".join(lines)
