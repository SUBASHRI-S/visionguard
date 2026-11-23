import json

import pytest

from datasets.materialize import materialize, remap_label_text
from datasets.split import make_split

# Source ids: helmet=0, vest=2, Person=5, forklift=10, gloves=1
LABEL = "5 0.5 0.5 0.2 0.4\n0 0.5 0.3 0.1 0.1\n1 0.4 0.6 0.05 0.05\n"


def _build(make_dataset, tmp_path):
    items = [("train", f"t{i}", i, LABEL) for i in range(10)]
    items += [("val", f"v{i}", 100 + i, LABEL) for i in range(3)]
    items.append(("train", "nolabel", 50, None))
    make_dataset(tmp_path / "src", items)
    root = tmp_path / "src"
    train = sorted((root / "images" / "train").glob("*.jpg"))
    test = sorted((root / "images" / "val").glob("*.jpg"))
    return root, train, test


def test_remap_changes_ids_and_drops_unmapped():
    out = remap_label_text(LABEL, {5: 0, 0: 1})
    assert out == "0 0.5 0.5 0.2 0.4\n1 0.5 0.3 0.1 0.1\n"


def test_remap_all_dropped_gives_empty_text():
    assert remap_label_text("1 0.5 0.5 0.1 0.1\n", {5: 0}) == ""


def test_remap_rejects_non_integer_class():
    with pytest.raises(ValueError):
        remap_label_text("abc 0.5 0.5 0.1 0.1\n", {5: 0})


def test_materialize_core_profile(make_dataset, tmp_path):
    root, train, test = _build(make_dataset, tmp_path)
    split = make_split(train, test, seed=1)
    out = tmp_path / "out"
    manifest = materialize(root, split, out, "core", source_sha256="abc")

    assert manifest["class_names"] == ["person", "helmet", "vest", "forklift"]
    assert manifest["source_zip_sha256"] == "abc"
    total = sum(c["images"] for c in manifest["counts"].values())
    assert total == len(split.train) + len(split.val) + len(split.test)

    any_label = next((out / "labels" / "test").glob("*.txt")).read_text()
    assert any_label == "0 0.5 0.5 0.2 0.4\n1 0.5 0.3 0.1 0.1\n"  # gloves dropped
    assert manifest["counts"]["test"]["instances"]["person"] == 3

    yaml_text = (out / "data.yaml").read_text()
    assert "train: images/train" in yaml_text
    assert "test: images/test" in yaml_text
    assert "  0: person" in yaml_text
    assert json.loads((out / "manifest.json").read_text())["dataset_version"]


def test_image_without_label_gets_empty_label_file(make_dataset, tmp_path):
    root, train, test = _build(make_dataset, tmp_path)
    split = make_split(train, test, seed=1)
    out = tmp_path / "out"
    manifest = materialize(root, split, out, "core")
    target = next(out.glob("labels/*/nolabel.txt"))
    assert target.read_text() == ""
    summary = manifest["counts"]
    assert sum(c["images_without_objects"] for c in summary.values()) == 1


def test_version_is_stable_and_changes_with_seed(make_dataset, tmp_path):
    root, train, test = _build(make_dataset, tmp_path)
    one = materialize(root, make_split(train, test, seed=1), tmp_path / "o1", "core")
    two = materialize(root, make_split(train, test, seed=1), tmp_path / "o2", "core")
    three = materialize(root, make_split(train, test, seed=2), tmp_path / "o3", "core")
    assert one["dataset_version"] == two["dataset_version"]
    assert one["dataset_version"] != three["dataset_version"]


def test_refuses_non_empty_output_folder(make_dataset, tmp_path):
    root, train, test = _build(make_dataset, tmp_path)
    split = make_split(train, test, seed=1)
    out = tmp_path / "out"
    out.mkdir()
    (out / "something.txt").write_text("x")
    with pytest.raises(FileExistsError):
        materialize(root, split, out, "core")


def test_unknown_profile_raises(make_dataset, tmp_path):
    root, train, test = _build(make_dataset, tmp_path)
    split = make_split(train, test, seed=1)
    with pytest.raises(ValueError):
        materialize(root, split, tmp_path / "out", "nope")


def test_link_mode_produces_same_files(make_dataset, tmp_path):
    root, train, test = _build(make_dataset, tmp_path)
    split = make_split(train, test, seed=1)
    out = tmp_path / "out"
    materialize(root, split, out, "all", mode="link")
    assert len(list(out.glob("images/*/*.jpg"))) == len(train) + len(test)
