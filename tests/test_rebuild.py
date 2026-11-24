import json
import shutil
from pathlib import Path

import pytest

from datasets.materialize import materialize
from datasets.rebuild import RebuildError, rebuild
from datasets.split import make_split

LABEL = "5 0.5 0.5 0.2 0.2\n"  # Person in the source class order


def build_raw(make_dataset, root: Path):
    items = [("train", f"t{i}", i, LABEL) for i in range(10)]
    items += [("val", f"v{i}", 100 + i, LABEL) for i in range(4)]
    make_dataset(root, items)


def images(folder: Path):
    return sorted(folder.iterdir())


@pytest.fixture
def original(tmp_path, make_dataset):
    raw = tmp_path / "raw"
    build_raw(make_dataset, raw)
    split = make_split(images(raw / "images" / "train"), images(raw / "images" / "val"))
    manifest = materialize(raw, split, tmp_path / "first", "core", source_sha256="abc")
    versions = tmp_path / "versions"
    versions.mkdir()
    (versions / f"{manifest['dataset_version']}.json").write_text(json.dumps(manifest))
    return raw, versions, manifest


def test_rebuild_reproduces_identical_version(tmp_path, original):
    raw, versions, manifest = original
    rebuilt = rebuild(manifest["dataset_version"], raw, versions, tmp_path / "second")
    assert rebuilt["dataset_version"] == manifest["dataset_version"]
    first = sorted(p.name for p in (tmp_path / "first" / "images" / "train").iterdir())
    second = sorted(p.name for p in (tmp_path / "second" / "images" / "train").iterdir())
    assert first == second


def test_unknown_version(tmp_path, original):
    raw, versions, _ = original
    with pytest.raises(RebuildError, match="no manifest"):
        rebuild("000000000000", raw, versions, tmp_path / "x")


def test_different_raw_data_is_detected(tmp_path, original):
    raw, versions, manifest = original
    extra = raw / "images" / "train" / "t0.jpg"
    shutil.copy(extra, raw / "images" / "train" / "zzz_extra.jpg")
    with pytest.raises(RebuildError, match="differs"):
        rebuild(manifest["dataset_version"], raw, versions, tmp_path / "x")


def test_changed_split_parameter_is_detected(tmp_path, original):
    raw, versions, manifest = original
    tampered = dict(manifest, val_fraction=0.5)
    (versions / f"{manifest['dataset_version']}.json").write_text(json.dumps(tampered))
    with pytest.raises(RebuildError):
        rebuild(manifest["dataset_version"], raw, versions, tmp_path / "x")
