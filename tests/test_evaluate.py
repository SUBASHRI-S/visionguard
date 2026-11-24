from pathlib import Path

import pytest

from training.evaluate import (
    EvaluationError, evaluate, per_class_table, sha256_file, to_markdown,
)

NAMES = ["person", "helmet", "vest", "forklift"]


def test_per_class_table_maps_indices_to_names():
    table = per_class_table(
        NAMES, [0, 2], [(0.9, 0.8, 0.7, 0.6), (0.5, 0.4, 0.3, 0.2)],
        {"person": 10, "helmet": 0, "vest": 5, "forklift": 0},
    )
    assert table["person"]["map50"] == pytest.approx(0.7)
    assert table["vest"]["precision"] == pytest.approx(0.5)
    assert table["person"]["instances"] == 10


def test_class_missing_from_split_is_none_not_zero():
    table = per_class_table(NAMES, [0], [(1, 1, 1, 1)], {"person": 3})
    assert table["helmet"]["map50"] is None
    assert table["helmet"]["instances"] == 0


def test_sha256_is_stable(tmp_path):
    path = tmp_path / "f.bin"
    path.write_bytes(b"abc")
    assert sha256_file(path) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_markdown_marks_missing_values_and_includes_hash():
    result = {
        "run_name": "r", "split": "test", "dataset_version": "abc", "profile": "core",
        "images": 2, "weights_sha256": "deadbeef", "settings": {"imgsz": 640},
        "evaluated_at": "now", "instances": 5, "device": "cpu",
        "overall": {"precision": 0.9, "recall": 0.8, "map50": 0.7, "map50_95": 0.6},
        "per_class": {"person": per_class_table(NAMES, [], [], {})["person"]},
        "speed_ms": {"preprocess": 1.0, "inference": 2.0, "postprocess": 3.0},
    }
    text = to_markdown(result)
    assert "n/a" in text and "deadbeef" in text and "| **all** | 5 |" in text


def test_bad_split_and_missing_weights(tmp_path):
    kwargs = dict(run_name="r", profile="core", version="abc123abc123")
    with pytest.raises(EvaluationError, match="split"):
        evaluate(tmp_path / "w.pt", split="train", **kwargs)
    with pytest.raises(EvaluationError, match="not found"):
        evaluate(tmp_path / "w.pt", **kwargs)


def test_existing_test_result_is_not_overwritten(tmp_path):
    weights = tmp_path / "w.pt"
    weights.write_bytes(b"x")
    results = tmp_path / "results"
    results.mkdir()
    (results / "r_test.json").write_text("{}")
    with pytest.raises(EvaluationError, match="already exists"):
        evaluate(weights, run_name="r", profile="core", version="abc123abc123",
                 results_dir=results)
