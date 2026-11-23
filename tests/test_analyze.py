from datasets.analyze import analyze_dataset, report_to_markdown

CLASSES = ["person", "helmet"]


def test_counts_sizes_and_cooccurrence(make_dataset, tmp_path):
    items = [
        # one large person (area 0.2) + one small helmet (area 0.001)
        ("train", "a", 1, "0 0.5 0.5 0.4 0.5\n1 0.5 0.2 0.02 0.05\n"),
        ("train", "b", 2, "0 0.5 0.5 0.1 0.1\n"),  # medium person (area 0.01)
        ("train", "bg", 3, None),  # no label file
        ("val", "c", 4, "1 0.5 0.5 0.3 0.3\n"),
    ]
    make_dataset(tmp_path, items)
    report = analyze_dataset(tmp_path, CLASSES)
    train = report["splits"]["train"]

    assert train["images"] == 3
    assert train["background_images"] == 1
    assert train["classes"]["person"]["instances"] == 2
    assert train["classes"]["person"]["images"] == 2
    assert train["classes"]["person"]["large"] == 1
    assert train["classes"]["person"]["medium"] == 1
    assert train["classes"]["helmet"]["small"] == 1
    assert train["cooccurrence"] == {"helmet+person": 1}
    assert report["splits"]["val"]["classes"]["helmet"]["large"] == 1


def test_brightness_buckets(make_image, tmp_path):
    make_image(tmp_path / "images" / "train" / "black.jpg", color=(0, 0, 0))
    make_image(tmp_path / "images" / "train" / "white.jpg", color=(255, 255, 255))
    make_image(tmp_path / "images" / "train" / "gray.jpg", color=(120, 120, 120))
    (tmp_path / "labels" / "train").mkdir(parents=True)
    report = analyze_dataset(tmp_path, CLASSES, splits=("train",))
    assert report["splits"]["train"]["brightness"] == {"dark": 1, "normal": 1, "bright": 1}


def test_unreadable_image_is_counted_not_fatal(make_dataset, tmp_path):
    make_dataset(tmp_path, [("train", "ok", 1, "0 0.5 0.5 0.2 0.2\n")])
    (tmp_path / "images" / "train" / "broken.jpg").write_bytes(b"junk")
    report = analyze_dataset(tmp_path, CLASSES, splits=("train",))
    assert report["splits"]["train"]["unreadable_images"] == ["broken.jpg"]
    assert report["splits"]["train"]["readable_images"] == 1


def test_malformed_label_lines_are_counted(make_dataset, tmp_path):
    make_dataset(tmp_path, [("train", "a", 1, "0 0.5 0.5 0.2 0.2\ngarbage line\n")])
    report = analyze_dataset(tmp_path, CLASSES, splits=("train",))
    assert report["splits"]["train"]["malformed_label_lines"] == 1


def test_markdown_flags_rare_and_missing_classes(make_dataset, tmp_path):
    make_dataset(tmp_path, [("train", "a", 1, "0 0.5 0.5 0.2 0.2\n")])
    text = report_to_markdown(analyze_dataset(tmp_path, CLASSES, splits=("train",)), min_instances=5)
    assert "# Dataset analysis" in text
    assert "| person | 1 |" in text
    assert "few (<5)" in text
    assert "none" in text  # helmet has no instances
