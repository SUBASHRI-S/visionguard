"""Analyze the real Construction-PPE dataset and write a report.

Run from the project root:
    python -m scripts.analyze_dataset
"""
import argparse
import json
from pathlib import Path

from datasets.analyze import analyze_dataset, report_to_markdown
from datasets.construction_ppe import SOURCE_CLASS_NAMES


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("datasets/raw/construction-ppe"))
    parser.add_argument("--out-md", type=Path, default=Path("docs/dataset_analysis.md"))
    parser.add_argument("--out-json", type=Path, default=Path("experiments/dataset_analysis.json"))
    args = parser.parse_args()

    if not args.root.is_dir():
        raise SystemExit(f"dataset folder not found: {args.root}")

    report = analyze_dataset(args.root, SOURCE_CLASS_NAMES, splits=("train", "val"))
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(report_to_markdown(report))
    args.out_json.write_text(json.dumps(report, indent=2))
    print(f"wrote {args.out_md} and {args.out_json}")
    for split, data in report["splits"].items():
        print(f"[{split}] images={data['images']} unreadable={len(data['unreadable_images'])} "
              f"without_objects={data['background_images']} brightness={data['brightness']}")
        for name, c in data["classes"].items():
            print(f"    {name:10s} instances={c['instances']:5d} images={c['images']:4d} "
                  f"small={c['small']} medium={c['medium']} large={c['large']}")


if __name__ == "__main__":
    main()
