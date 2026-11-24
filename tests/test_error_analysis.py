from pathlib import Path

import pytest

from evaluation.error_analysis import (
    GroundTruth, Prediction, analyze, iou, match_image, parse_labels, size_bucket,
    to_markdown, worst_images,
)

NAMES = ["person", "helmet"]


def gt(cls, box):
    return GroundTruth(cls, box)


def pr(cls, box, conf=0.9):
    return Prediction(cls, conf, box)


def test_parse_labels_converts_normalised_boxes():
    boxes = parse_labels("1 0.5 0.5 0.2 0.4\n\nbad line\n", 100, 200)
    assert len(boxes) == 1 and boxes[0].class_id == 1
    assert boxes[0].box == pytest.approx((40.0, 60.0, 60.0, 140.0))


def test_iou_cases():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(1 / 3)


def test_perfect_match():
    m = match_image([gt(0, (0, 0, 10, 10))], [pr(0, (0, 0, 10, 10))])
    assert len(m.found) == 1 and m.error_count == 0


def test_missed_and_false_alarm_are_separate():
    m = match_image([gt(0, (0, 0, 10, 10))], [pr(0, (50, 50, 60, 60))])
    assert len(m.missed) == 1 and len(m.false_alarms) == 1 and not m.found


def test_wrong_class_is_not_a_hit_and_not_a_false_alarm():
    m = match_image([gt(1, (0, 0, 10, 10))], [pr(0, (0, 0, 10, 10))])
    assert len(m.wrong_class) == 1
    assert not m.found and not m.missed and not m.false_alarms


def test_one_prediction_cannot_match_two_truths():
    truths = [gt(0, (0, 0, 10, 10)), gt(0, (0, 0, 10, 10))]
    m = match_image(truths, [pr(0, (0, 0, 10, 10))])
    assert len(m.found) == 1 and len(m.missed) == 1


def test_higher_confidence_prediction_wins_the_match():
    m = match_image(
        [gt(0, (0, 0, 10, 10))],
        [pr(0, (0, 0, 10, 10), 0.4), pr(0, (0, 0, 10, 10), 0.9)],
    )
    assert m.found[0][1].confidence == 0.9 and len(m.false_alarms) == 1


def test_low_iou_is_not_a_match():
    m = match_image([gt(0, (0, 0, 10, 10))], [pr(0, (8, 8, 18, 18))])
    assert not m.found


def test_size_buckets():
    assert size_bucket((0, 0, 5, 5), 100, 100) == "small"
    assert size_bucket((0, 0, 20, 20), 100, 100) == "medium"
    assert size_bucket((0, 0, 50, 50), 100, 100) == "large"


def test_analyze_aggregates_and_reports(tmp_path: Path):
    labels = tmp_path / "labels"
    labels.mkdir()
    # a.jpg: person found, helmet missed. b.jpg: helmet called person. c.jpg: false alarm.
    (labels / "a.txt").write_text("0 0.25 0.5 0.4 0.8\n1 0.75 0.5 0.4 0.8\n")
    (labels / "b.txt").write_text("1 0.5 0.5 0.5 0.5\n")
    predictions = {
        "a.jpg": [pr(0, (10, 10, 50, 90))],
        "b.jpg": [pr(0, (25, 25, 75, 75))],
        "c.jpg": [pr(1, (0, 0, 10, 10))],
    }

    def predict(path: Path):
        return 100, 100, predictions[path.name]

    images = [tmp_path / n for n in ("a.jpg", "b.jpg", "c.jpg")]
    summary, matches = analyze(images, labels, predict, NAMES)
    assert summary["per_class"]["person"] == {
        "tp": 1, "fn": 0, "fp": 1, "recall": 1.0, "precision": 0.5,
    }
    assert summary["per_class"]["helmet"]["fn"] == 2  # one missed, one called person
    assert summary["per_class"]["helmet"]["fp"] == 1
    assert summary["confusions"]["helmet"] == {"background": 1, "person": 1}
    assert summary["images_with_errors"] == 3
    ranked = worst_images(matches)
    assert ranked == [("a.jpg", 1), ("b.jpg", 1), ("c.jpg", 1)]
    text = to_markdown(
        summary, {"run_name": "r", "split": "val", "dataset_version": "v", "conf": 0.25, "iou": 0.5}, ranked
    )
    assert "Real **helmet** objects" in text and "person: 1" in text
