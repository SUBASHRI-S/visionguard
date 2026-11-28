"""Build the core_sh17 dataset: the prepared core dataset plus selected SH17 training photos.

Run from the project root, after `python -m scripts.rebuild_dataset --version d1dc2ab4d1a9`:
    python -m scripts.extend_with_sh17 --sh17 /kaggle/input/datasets/mugheesahmad/sh17-dataset-for-ppe-detection

Writes datasets/processed/core_sh17 and datasets/versions/<version>.json. Commit the small
manifest only. Never commit the images (licence: CC BY-NC-SA 4.0).
"""
import argparse
import json
from pathlib import Path

from datasets.sh17 import LICENCE_NOTE, extend_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sh17", type=Path, required=True, help="folder with images/ and labels/")
    parser.add_argument("--base", type=Path, default=Path("datasets/processed/core"))
    parser.add_argument("--out", type=Path, default=Path("datasets/processed/core_sh17"))
    parser.add_argument("--versions", type=Path, default=Path("datasets/versions"))
    parser.add_argument("--max-images", type=int, default=1500)
    parser.add_argument("--max-side", type=int, default=1280)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    try:
        result = extend_dataset(
            args.base,
            args.sh17,
            args.out,
            max_images=args.max_images,
            max_side=args.max_side,
            seed=args.seed,
        )
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"failed: {exc}") from exc

    manifest = result.manifest
    args.versions.mkdir(parents=True, exist_ok=True)
    (args.versions / f"{manifest['dataset_version']}.json").write_text(json.dumps(manifest, indent=2))
    selection = manifest["sh17_selection"]
    print(f"SH17 photos with a helmet or vest: {result.pool_size}")
    print(f"dropped as near-duplicates of val/test: {selection['dropped_near_duplicates']}")
    print(f"added to train: {selection['added_images']}  instances: {selection['added_instances']}")
    print(f"train images now: {manifest['counts']['train']['images']}")
    print(f"built core_sh17 version {manifest['dataset_version']} at {args.out}")
    print(LICENCE_NOTE)


if __name__ == "__main__":
    main()
