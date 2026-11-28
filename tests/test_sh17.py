import json

import pytest
from PIL import Image

from datasets.materialize import materialize
from datasets.sh17 import (
    SH17_CLASS_NAMES,
    extend_dataset,
    find_candidates,
    has_target,
    sh17_id_map,
)
from datasets.split import make_split
from training.dataset_check import check_dataset

# Source (Construction-PPE) ids: helmet=0, vest=2, Person=5
BASE_LABEL = "5 0.5 0.5 0.2 0.4\n0 0.5 0.3 0.1 0.1\n"


def test_class_ids_match_the_checked_order():
    assert SH17_CLASS_NAMES[0] == "person"
    assert SH17_CLASS_NAMES[10] == "helmet"
    assert SH17_CLASS_NAMES[16] == "safety-vest"
    assert SH17_CLASS_NAMES[15] == "safety-suit"
    assert len(SH17_CLASS_NAMES) == 17


def test_id_map_goes_to_core_ids():
    assert sh17_id_map() == {0: 0, 10: 1, 16: 2}


def test_has_target_only_for_helmet_or_vest():
    assert has_target("0 0.5 0.5 0.1 0.1\n10 0.5 0.5 0.1 0.1\n")
    assert has_target("16 0.5 0.5 0.1 0.1\n")
    assert not has_target("0 0.5 0.5 0.1 0.1\n15 0.5 0.5 0.1 0.1\n")  # safety-suit is not a vest
    assert not has_target("")


def _base(make_dataset, tmp_path):
    items = [("train", f"t{i}", i, BASE_LABEL) for i in range(10)]
    items += [("val", f"v{i}", 100 + i, BASE_LABEL) for i in range(4)]
    make_dataset(tmp_path / "src", items)
    root = tmp_path / "src"
    split = make_split(
        sorted((root / "images" / "train").glob("*.jpg")),
        sorted((root / "images" / "val").glob("*.jpg")),
        seed=1,
    )
    out = tmp_path / "core"
    materialize(root, split, out, "core")
    return out


def _sh17(make_image, tmp_path, specs):
    """specs: (stem, seed, label_text, size)"""
    root = tmp_path / "sh17"
    (root / "labels").mkdir(parents=True)
    for stem, seed, label, size in specs:
        make_image(root / "images" / f"{stem}.jpeg", seed=seed, size=size)
        (root / "labels" / f"{stem}.txt").write_text(label)
    return root


SH17_LABEL = "0 0.5 0.5 0.3 0.6\n10 0.5 0.3 0.1 0.1\n12 0.5 0.3 0.1 0.1\n16 0.5 0.6 0.2 0.3\n"


def test_find_candidates_skips_photos_without_helmet_or_vest(make_image, tmp_path):
    root = _sh17(
        make_image,
        tmp_path,
        [("a", 1, SH17_LABEL, (64, 64)), ("b", 2, "0 0.5 0.5 0.3 0.6\n", (64, 64))],
    )
    assert [p.stem for p in find_candidates(root)] == ["a"]


def test_extend_adds_to_train_only_and_remaps(make_dataset, make_image, tmp_path):
    base = _base(make_dataset, tmp_path)
    root = _sh17(
        make_image,
        tmp_path,
        [(f"p{i}", 200 + i, SH17_LABEL, (64, 64)) for i in range(5)],
    )
    out = tmp_path / "core_sh17"
    result = extend_dataset(base, root, out, max_images=3, seed=7)

    manifest = result.manifest
    base_manifest = json.loads((base / "manifest.json").read_text())
    assert manifest["counts"]["train"]["images"] == base_manifest["counts"]["train"]["images"] + 3
    assert manifest["counts"]["val"] == base_manifest["counts"]["val"]
    assert manifest["counts"]["test"] == base_manifest["counts"]["test"]
    assert manifest["sh17_selection"]["added_instances"]["vest"] == 3
    assert manifest["sh17_selection"]["pool_size"] == 5

    added = sorted((out / "labels" / "train").glob("sh17_*.txt"))
    assert len(added) == 3
    # person -> 0, helmet -> 1, vest -> 2; head (12) is dropped
    assert added[0].read_text().splitlines() == [
        "0 0.5 0.5 0.3 0.6",
        "1 0.5 0.3 0.1 0.1",
        "2 0.5 0.6 0.2 0.3",
    ]
    assert not list((out / "labels" / "val").glob("sh17_*"))
    assert not list((out / "images" / "test").glob("sh17_*"))
    assert "core_sh17" == manifest["profile"]
    assert "CC BY-NC-SA" in manifest["sh17_licence"]


def test_selection_is_deterministic(make_dataset, make_image, tmp_path):
    base = _base(make_dataset, tmp_path)
    root = _sh17(
        make_image, tmp_path, [(f"p{i}", 300 + i, SH17_LABEL, (64, 64)) for i in range(8)]
    )
    one = extend_dataset(base, root, tmp_path / "o1", max_images=4, seed=3)
    two = extend_dataset(base, root, tmp_path / "o2", max_images=4, seed=3)
    other = extend_dataset(base, root, tmp_path / "o3", max_images=4, seed=4)
    assert one.manifest["dataset_version"] == two.manifest["dataset_version"]
    assert one.selected == two.selected
    assert other.selected != one.selected


def test_near_duplicate_of_test_image_is_dropped(make_dataset, make_image, tmp_path):
    base = _base(make_dataset, tmp_path)
    test_image = sorted((base / "images" / "test").glob("*.jpg"))[0]
    root = _sh17(make_image, tmp_path, [("fresh", 400, SH17_LABEL, (64, 64))])
    # same picture as a test image, saved under an SH17 name at another size
    with Image.open(test_image) as picture:
        picture.resize((128, 128)).save(root / "images" / "copy.jpeg")
    (root / "labels" / "copy.txt").write_text(SH17_LABEL)

    result = extend_dataset(base, root, tmp_path / "out", max_images=10, seed=1)
    assert result.dropped_duplicates == ["copy.jpeg"]
    assert result.selected == ["fresh.jpeg"]


def test_large_photos_are_shrunk(make_dataset, make_image, tmp_path):
    base = _base(make_dataset, tmp_path)
    root = _sh17(make_image, tmp_path, [("big", 500, SH17_LABEL, (400, 200))])
    out = tmp_path / "out"
    extend_dataset(base, root, out, max_images=5, max_side=100, seed=1)
    with Image.open(out / "images" / "train" / "sh17_big.jpg") as picture:
        assert picture.size == (100, 50)


def test_refuses_non_empty_output_and_non_core_base(make_dataset, make_image, tmp_path):
    base = _base(make_dataset, tmp_path)
    root = _sh17(make_image, tmp_path, [("a", 1, SH17_LABEL, (64, 64))])
    busy = tmp_path / "busy"
    busy.mkdir()
    (busy / "x").write_text("x")
    with pytest.raises(FileExistsError):
        extend_dataset(base, root, busy)

    manifest = json.loads((base / "manifest.json").read_text())
    manifest["profile"] = "all"
    (base / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        extend_dataset(base, root, tmp_path / "fresh")


def test_training_check_accepts_the_extended_dataset(make_dataset, make_image, tmp_path):
    base = _base(make_dataset, tmp_path)
    root = _sh17(
        make_image, tmp_path, [(f"p{i}", 600 + i, SH17_LABEL, (64, 64)) for i in range(3)]
    )
    processed = tmp_path / "processed"
    result = extend_dataset(base, root, processed / "core_sh17", max_images=3, seed=1)
    versions = tmp_path / "versions"
    versions.mkdir()
    version = result.manifest["dataset_version"]
    (versions / f"{version}.json").write_text(json.dumps(result.manifest))
    found = check_dataset("core_sh17", version, processed, versions)
    assert found["dataset_version"] == version
