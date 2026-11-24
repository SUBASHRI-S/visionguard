"""Make sure the dataset on disk is exactly the version a training config asks for."""
import json
from pathlib import Path


class DatasetError(RuntimeError):
    """Raised when the prepared dataset does not match the requested version."""


def _count_files(folder: Path) -> int:
    return sum(1 for p in folder.iterdir() if p.is_file()) if folder.is_dir() else 0


def check_dataset(
    profile: str, version: str, processed_root: Path, versions_root: Path
) -> dict:
    """Return the manifest of ``version`` after verifying the prepared dataset matches it."""
    stored = versions_root / f"{version}.json"
    if not stored.exists():
        raise DatasetError(
            f"no manifest for dataset version {version} in {versions_root}. "
            "Check the version in your config, or commit the manifest file."
        )
    manifest = json.loads(stored.read_text())
    if manifest.get("profile") != profile:
        raise DatasetError(
            f"version {version} was built with profile '{manifest.get('profile')}', "
            f"but the config asks for '{profile}'"
        )

    folder = processed_root / profile
    rebuild_hint = f"python -m scripts.rebuild_dataset --version {version}"
    local_manifest = folder / "manifest.json"
    if not local_manifest.exists():
        raise DatasetError(f"{folder} is missing. Build it with: {rebuild_hint}")
    found = json.loads(local_manifest.read_text()).get("dataset_version")
    if found != version:
        raise DatasetError(
            f"{folder} holds dataset version {found}, but the config asks for {version}. "
            f"Rebuild with: {rebuild_hint}"
        )
    if not (folder / "data.yaml").exists():
        raise DatasetError(f"{folder / 'data.yaml'} is missing. Rebuild with: {rebuild_hint}")

    for split, counts in manifest["counts"].items():
        for kind in ("images", "labels"):
            actual = _count_files(folder / kind / split)
            if actual != counts["images"]:
                raise DatasetError(
                    f"{kind}/{split} has {actual} files, manifest expects {counts['images']}. "
                    f"Rebuild with: {rebuild_hint}"
                )
    return manifest
