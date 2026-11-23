"""Validate the raw dataset, build a leakage-safe split, and write a trainable dataset.

Run from the project root:
    python -m scripts.make_split --profile core --source-sha256 <hash>
"""
import argparse
import json
import shutil
from pathlib import Path

from datasets.construction_ppe import PROFILES, SOURCE_CLASS_NAMES
from datasets.materialize import materialize
from datasets.split import make_split
from datasets.validate import IMAGE_EXTENSIONS, validate_dataset


def _images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("datasets/raw/construction-ppe"))
    parser.add_argument("--profile", choices=sorted(PROFILES), default="core")
    parser.add_argument("--out", type=Path, default=None,
                        help="output folder (default: datasets/processed/<profile>)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--hash-threshold", type=int, default=4)
    parser.add_argument("--keep-leaking", action="store_true",
                        help="flag but do not remove train images that duplicate test images")
    parser.add_argument("--source-sha256", default=None, help="SHA256 of the downloaded zip")
    parser.add_argument("--mode", choices=["copy", "link"], default="copy")
    parser.add_argument("--manifest-dir", type=Path, default=Path("datasets/versions"))
    args = parser.parse_args()

    out_dir = args.out or Path("datasets/processed") / args.profile

    report = validate_dataset(args.root, SOURCE_CLASS_NAMES, splits=("train", "val"))
    if not report.ok:
        for line in report.errors[:20]:
            print("ERROR", line)
        raise SystemExit(f"source dataset has {len(report.errors)} errors; fix them first")
    print(f"source validated: {report.num_images} images, {len(report.warnings)} warnings")

    split = make_split(
        _images(args.root / "images" / "train"),
        _images(args.root / "images" / "val"),
        val_fraction=args.val_fraction,
        seed=args.seed,
        hash_threshold=args.hash_threshold,
        drop_leaking=not args.keep_leaking,
    )
    print(f"train={len(split.train)} val={len(split.val)} test={len(split.test)}")
    print(f"train images that near-duplicate a test image: {len(split.leaking_train)}")
    print(f"train images removed for leakage: {len(split.excluded_train)}")
    print(f"unreadable images skipped: {len(split.unreadable)}")

    manifest = materialize(
        args.root, split, out_dir, args.profile,
        source_sha256=args.source_sha256, mode=args.mode,
    )
    args.manifest_dir.mkdir(parents=True, exist_ok=True)
    version_file = args.manifest_dir / f"{manifest['dataset_version']}.json"
    shutil.copy(out_dir / "manifest.json", version_file)

    print(f"dataset_version: {manifest['dataset_version']}")
    print(f"data config:     {out_dir / 'data.yaml'}")
    print(f"manifest copy:   {version_file}  (commit this file; it is small)")
    print(json.dumps({k: v["instances"] for k, v in manifest["counts"].items()}, indent=2))


if __name__ == "__main__":
    main()
