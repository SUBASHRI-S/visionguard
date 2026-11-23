"""Validate the raw Construction-PPE dataset (labels, class ids, coordinates).

Run from the project root:
    python -m scripts.run_validation
"""
import argparse
from pathlib import Path

from datasets.construction_ppe import SOURCE_CLASS_NAMES
from datasets.validate import validate_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("datasets/raw/construction-ppe"))
    args = parser.parse_args()

    if not args.root.is_dir():
        raise SystemExit(f"dataset folder not found: {args.root}")

    report = validate_dataset(args.root, SOURCE_CLASS_NAMES, splits=("train", "val"))
    print(f"images: {report.num_images}")
    print(f"errors: {len(report.errors)}  warnings: {len(report.warnings)}")
    print("class counts:")
    for class_id, name in enumerate(SOURCE_CLASS_NAMES):
        print(f"  {class_id:2d} {name:10s} {report.class_counts[class_id]}")
    for line in report.errors[:20]:
        print("ERROR  ", line)
    for line in report.warnings[:20]:
        print("WARNING", line)
    raise SystemExit(0 if report.ok else 1)


if __name__ == "__main__":
    main()
