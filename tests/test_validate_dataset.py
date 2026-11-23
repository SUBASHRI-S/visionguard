from pathlib import Path

from datasets.validate import validate_dataset

CLASSES = ["person", "helmet", "vest"]


def make_dataset(root: Path, files: dict[str, str | None]) -> None:
    """files maps 'split/name' -> label text (None = image without label file)."""
    for key, label_text in files.items():
        split, name = key.split("/")
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
        (root / "images" / split / f"{name}.jpg").write_bytes(b"")
        if label_text is not None:
            (root / "labels" / split / f"{name}.txt").write_text(label_text)


def good_files() -> dict[str, str | None]:
    label = "0 0.5 0.5 0.2 0.4\n1 0.5 0.35 0.1 0.1\n2 0.5 0.6 0.15 0.2\n"
    return {"train/a": label, "val/b": label}


def test_valid_dataset_passes(tmp_path):
    make_dataset(tmp_path, good_files())
    report = validate_dataset(tmp_path, CLASSES)
    assert report.ok
    assert report.num_images == 2
    assert report.class_counts[0] == 2


def test_class_id_out_of_range_is_error(tmp_path):
    files = good_files()
    files["train/a"] = "7 0.5 0.5 0.2 0.4\n"
    make_dataset(tmp_path, files)
    assert not validate_dataset(tmp_path, CLASSES).ok


def test_coordinate_above_one_is_error(tmp_path):
    files = good_files()
    files["train/a"] = "0 1.5 0.5 0.2 0.4\n"
    make_dataset(tmp_path, files)
    assert not validate_dataset(tmp_path, CLASSES).ok


def test_wrong_number_of_fields_is_error(tmp_path):
    files = good_files()
    files["train/a"] = "0 0.5 0.5 0.2\n"
    make_dataset(tmp_path, files)
    assert not validate_dataset(tmp_path, CLASSES).ok


def test_label_without_image_is_error(tmp_path):
    make_dataset(tmp_path, good_files())
    (tmp_path / "labels" / "train" / "orphan.txt").write_text("0 0.5 0.5 0.2 0.4\n")
    assert not validate_dataset(tmp_path, CLASSES).ok


def test_image_without_label_is_only_a_warning(tmp_path):
    files = good_files()
    files["train/c"] = None
    make_dataset(tmp_path, files)
    report = validate_dataset(tmp_path, CLASSES)
    assert report.ok
    assert any("no label file" in w for w in report.warnings)


def test_class_with_no_instances_warns(tmp_path):
    make_dataset(tmp_path, {"train/a": "0 0.5 0.5 0.2 0.4\n", "val/b": "0 0.5 0.5 0.2 0.4\n"})
    report = validate_dataset(tmp_path, CLASSES)
    assert any("helmet" in w for w in report.warnings)


def test_missing_split_folder_is_error(tmp_path):
    make_dataset(tmp_path, {"train/a": "0 0.5 0.5 0.2 0.4\n"})
    assert not validate_dataset(tmp_path, CLASSES).ok