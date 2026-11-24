"""Rebuild a committed dataset version from the raw download.

Run from the project root:
    python -m scripts.rebuild_dataset --version d1dc2ab4d1a9
"""
import argparse
import json
from pathlib import Path

from datasets.rebuild import RebuildError, rebuild


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True, help="12-character dataset_version")
    parser.add_argument("--raw", type=Path, default=Path("datasets/raw/construction-ppe"))
    parser.add_argument("--versions", type=Path, default=Path("datasets/versions"))
    parser.add_argument("--out-root", type=Path, default=Path("datasets/processed"))
    parser.add_argument("--mode", choices=["copy", "link"], default="copy")
    args = parser.parse_args()

    manifest_path = args.versions / f"{args.version}.json"
    if not manifest_path.exists():
        raise SystemExit(f"no manifest at {manifest_path}")
    profile = json.loads(manifest_path.read_text())["profile"]
    out_dir = args.out_root / profile

    try:
        manifest = rebuild(args.version, args.raw, args.versions, out_dir, mode=args.mode)
    except (RebuildError, FileExistsError) as exc:
        raise SystemExit(f"rebuild failed: {exc}") from exc
    print(f"rebuilt {manifest['dataset_version']} ({profile}) at {out_dir}")


if __name__ == "__main__":
    main()
