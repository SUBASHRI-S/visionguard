import shutil

import pytest
from PIL import Image

from datasets.hashing import cluster_by_hash, dhash, hamming
from datasets.split import make_split


def _train_images(make_image, tmp_path, count=30):
    return [make_image(tmp_path / "train" / f"t{i}.jpg", seed=i) for i in range(count)]


def test_identical_images_have_zero_distance(make_image, tmp_path):
    a = make_image(tmp_path / "a.jpg", seed=1)
    b = tmp_path / "b.jpg"
    shutil.copy(a, b)
    assert hamming(dhash(a), dhash(b)) == 0


def test_different_images_are_far_apart(make_image, tmp_path):
    a = make_image(tmp_path / "a.jpg", seed=1)
    b = make_image(tmp_path / "b.jpg", seed=2)
    assert hamming(dhash(a), dhash(b)) > 10


def test_resized_copy_stays_close(make_image, tmp_path):
    a = make_image(tmp_path / "a.jpg", seed=3)
    resized = tmp_path / "a_big.jpg"
    with Image.open(a) as image:
        image.resize((128, 128), Image.Resampling.BILINEAR).save(resized)
    assert hamming(dhash(a), dhash(resized)) <= 8


def test_cluster_groups_duplicates(make_image, tmp_path):
    a = make_image(tmp_path / "a.jpg", seed=1)
    a_copy = tmp_path / "a_copy.jpg"
    shutil.copy(a, a_copy)
    b = make_image(tmp_path / "b.jpg", seed=2)
    clusters = cluster_by_hash({a: dhash(a), a_copy: dhash(a_copy), b: dhash(b)}, threshold=2)
    sizes = sorted(len(c) for c in clusters)
    assert sizes == [1, 2]


def test_splits_do_not_overlap_and_cover_all(make_image, tmp_path):
    train = _train_images(make_image, tmp_path)
    test = [make_image(tmp_path / "test" / f"x{i}.jpg", seed=1000 + i) for i in range(5)]
    result = make_split(train, test, val_fraction=0.2, seed=1)
    assert not set(result.train) & set(result.val)
    assert set(result.train) | set(result.val) == set(train)
    assert result.test == sorted(test)
    assert len(result.val) == pytest.approx(6, abs=1)


def test_split_is_deterministic_for_a_seed(make_image, tmp_path):
    train = _train_images(make_image, tmp_path)
    test = [make_image(tmp_path / "test" / "x.jpg", seed=999)]
    first = make_split(train, test, seed=7)
    second = make_split(train, test, seed=7)
    other = make_split(train, test, seed=8)
    assert first.val == second.val
    assert first.val != other.val


def test_duplicates_never_straddle_train_and_val(make_image, tmp_path):
    train = _train_images(make_image, tmp_path, count=20)
    copies = []
    for i in range(10):
        copy = tmp_path / "train" / f"copy{i}.jpg"
        shutil.copy(train[i], copy)
        copies.append(copy)
    test = [make_image(tmp_path / "test" / "x.jpg", seed=999)]
    for seed in range(5):
        result = make_split(train + copies, test, val_fraction=0.3, seed=seed)
        val = set(result.val)
        for original, copy in zip(train[:10], copies):
            assert (original in val) == (copy in val)


def test_train_image_duplicating_test_is_dropped(make_image, tmp_path):
    train = _train_images(make_image, tmp_path)
    test_image = make_image(tmp_path / "test" / "x.jpg", seed=999)
    leak = tmp_path / "train" / "leak.jpg"
    shutil.copy(test_image, leak)
    result = make_split([*train, leak], [test_image], seed=1)
    assert leak in result.leaking_train
    assert leak in result.excluded_train
    assert leak not in result.train and leak not in result.val


def test_keep_leaking_flags_but_keeps(make_image, tmp_path):
    train = _train_images(make_image, tmp_path, count=10)
    test_image = make_image(tmp_path / "test" / "x.jpg", seed=999)
    leak = tmp_path / "train" / "leak.jpg"
    shutil.copy(test_image, leak)
    result = make_split([*train, leak], [test_image], seed=1, drop_leaking=False)
    assert leak in result.leaking_train
    assert result.excluded_train == []
    assert leak in set(result.train) | set(result.val)


def test_unreadable_image_is_reported_not_crashing(make_image, tmp_path):
    train = _train_images(make_image, tmp_path, count=10)
    broken = tmp_path / "train" / "broken.jpg"
    broken.write_bytes(b"not an image")
    test = [make_image(tmp_path / "test" / "x.jpg", seed=999)]
    result = make_split([*train, broken], test, seed=1)
    assert result.unreadable == [broken]
    assert broken not in set(result.train) | set(result.val)


@pytest.mark.parametrize("fraction", [0, 1, -0.1, 1.5])
def test_invalid_val_fraction_raises(fraction):
    with pytest.raises(ValueError):
        make_split([], [], val_fraction=fraction)


def test_unreadable_after_ultralytics_import(make_image, tmp_path):
    """Importing ultralytics patches PIL.Image.open; a corrupt file must still be reported, not crash."""
    import ultralytics  # noqa: F401  (the import itself is the point)
    train = _train_images(make_image, tmp_path, count=10)
    broken = tmp_path / "train" / "broken.jpg"
    broken.write_bytes(b"not an image")
    test = [make_image(tmp_path / "test" / "x.jpg", seed=999)]
    assert make_split([*train, broken], test, seed=1).unreadable == [broken]
