"""Rebuild a dataset version on another machine (for example a Kaggle notebook).

The committed manifest records the split parameters and the exact file list. Rebuilding
replays the split from the raw download, checks the file lists match the manifest, writes the
dataset, and confirms the resulting dataset_version is identical. Any difference is an error
and the output folder is removed, so a wrong dataset can never be trained on by accident.
"""
import json
import shutil
from pathlib import Path

from datasets.materialize import materialize
from datasets.split import make_split
from datasets.validate import IMAGE_EXTENSIONS


class RebuildError(RuntimeError):
    """Raised when the rebuilt dataset does not match the committed manifest."""


def _images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)


def rebuild(
    version: str,
    raw_root: Path,
    versions_root: Path,
    out_dir: Path,
    *,
    mode: str = "copy",
) -> dict:
    """Rebuild ``version`` into ``out_dir`` and return its manifest."""
    stored = versions_root / f"{version}.json"
    if not stored.exists():
        raise RebuildError(f"no manifest for version {version} in {versions_root}")
    expected = json.loads(stored.read_text())

    split = make_split(
        _images(raw_root / "images" / "train"),
        _images(raw_root / "images" / "val"),
        val_fraction=expected["val_fraction"],
        seed=expected["seed"],
        hash_threshold=expected["hash_threshold"],
    )
    image_root = raw_root / "images"
    replayed = {
        "train": [p.relative_to(image_root).as_posix() for p in split.train],
        "val": [p.relative_to(image_root).as_posix() for p in split.val],
        "test": [p.relative_to(image_root).as_posix() for p in split.test],
    }
    for name, relatives in replayed.items():
        if relatives != expected["files"][name]:
            raise RebuildError(
                f"the {name} file list differs from the manifest "
                f"({len(relatives)} files now, {len(expected['files'][name])} expected). "
                "The raw download is probably a different file than the original."
            )

    manifest = materialize(
        raw_root,
        split,
        out_dir,
        expected["profile"],
        source_sha256=expected.get("source_zip_sha256"),
        mode=mode,
    )
    if manifest["dataset_version"] != version:
        shutil.rmtree(out_dir, ignore_errors=True)
        raise RebuildError(
            f"rebuilt version is {manifest['dataset_version']}, expected {version}. "
            "Output removed."
        )
    return manifest
