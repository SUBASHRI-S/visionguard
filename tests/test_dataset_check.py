import json
from pathlib import Path

import pytest

from training.dataset_check import DatasetError, check_dataset

VERSION = "abc123abc123"


def build(tmp_path: Path, *, local_version=VERSION, n_files=2, profile="core"):
    versions = tmp_path / "versions"
    versions.mkdir()
    manifest = {"dataset_version": VERSION, "profile": "core",
                "counts": {"train": {"images": 2}, "val": {"images": 1}}}
    (versions / f"{VERSION}.json").write_text(json.dumps(manifest))
    processed = tmp_path / "processed"
    folder = processed / profile
    for kind in ("images", "labels"):
        for split, n in (("train", n_files), ("val", 1)):
            (folder / kind / split).mkdir(parents=True)
            for i in range(n):
                (folder / kind / split / f"{i}.x").write_text("x")
    (folder / "data.yaml").write_text("x")
    (folder / "manifest.json").write_text(json.dumps({"dataset_version": local_version}))
    return processed, versions


def test_matching_dataset_passes(tmp_path):
    processed, versions = build(tmp_path)
    assert check_dataset("core", VERSION, processed, versions)["dataset_version"] == VERSION


def test_unknown_version(tmp_path):
    processed, versions = build(tmp_path)
    with pytest.raises(DatasetError, match="no manifest"):
        check_dataset("core", "000000000000", processed, versions)


def test_wrong_profile(tmp_path):
    processed, versions = build(tmp_path)
    with pytest.raises(DatasetError, match="profile"):
        check_dataset("all", VERSION, processed, versions)


def test_local_dataset_is_a_different_version(tmp_path):
    processed, versions = build(tmp_path, local_version="ffffffffffff")
    with pytest.raises(DatasetError, match="rebuild"):
        check_dataset("core", VERSION, processed, versions)


def test_missing_folder(tmp_path):
    _, versions = build(tmp_path)
    with pytest.raises(DatasetError, match="missing"):
        check_dataset("core", VERSION, tmp_path / "empty", versions)


def test_file_count_mismatch(tmp_path):
    processed, versions = build(tmp_path, n_files=1)
    with pytest.raises(DatasetError, match="expects"):
        check_dataset("core", VERSION, processed, versions)
