import json
from pathlib import Path

import pytest

from training.tracking import (
    RunAlreadyLoggedError, best_row, clean_metric_name, fitness, log_run_to_mlflow, read_results,
)

CSV = (
    "epoch,train/box_loss,metrics/precision(B),metrics/recall(B),metrics/mAP50(B),metrics/mAP50-95(B)\n"
    "1,1.0,0.1,0.1,0.2,0.1\n"
    "2,0.9,0.5,0.4,0.6,0.4\n"
    "3,0.8,0.4,0.4,0.5,0.3\n"
)


def make_run(tmp_path: Path, name="r1") -> Path:
    run = tmp_path / name
    run.mkdir()
    (run / "results.csv").write_text(CSV)
    summary = {
        "config": {"name": name, "model": "yolo11n.pt", "dataset_profile": "core",
                   "dataset_version": "d1dc2ab4d1a9", "project": "runs",
                   "train": {"epochs": 3}, "augment": {},
                   "mlflow_tracking_uri": f"sqlite:///{tmp_path / 'mlflow.db'}",
                   "mlflow_experiment": "t"},
        "dataset": {"source_zip_sha256": "abc"},
        "git": {"commit": "deadbeef", "dirty": False},
        "environment": {"ultralytics": "x", "torch": "y", "gpu_name": None},
    }
    (run / "run_summary.json").write_text(json.dumps(summary))
    return run


def test_clean_metric_name():
    assert clean_metric_name("metrics/mAP50-95(B)") == "metrics/mAP50-95"
    assert clean_metric_name("  train/box_loss ") == "train/box_loss"


def test_read_results_skips_non_numbers(tmp_path):
    path = tmp_path / "r.csv"
    path.write_text("epoch,a,b\n1,2,nan\n2,x,3\n")
    assert read_results(path) == [{"epoch": 1.0, "a": 2.0}, {"epoch": 2.0, "b": 3.0}]


def test_best_row_uses_fitness_and_first_tie(tmp_path):
    path = tmp_path / "r.csv"
    path.write_text(CSV)
    rows = read_results(path)
    index, row = best_row(rows)
    assert index == 1 and row["epoch"] == 2
    assert fitness(rows[1]) == pytest.approx(0.1 * 0.6 + 0.9 * 0.4)
    assert best_row([{"metrics/mAP50": 0.5}, {"metrics/mAP50": 0.5}])[0] == 0
    with pytest.raises(ValueError):
        best_row([])


def test_log_run_and_duplicate_protection(tmp_path):
    import mlflow

    run = make_run(tmp_path)
    run_id = log_run_to_mlflow(run)
    data = mlflow.get_run(run_id).data
    assert data.tags["dataset_version"] == "d1dc2ab4d1a9"
    assert data.tags["git_commit"] == "deadbeef"
    assert data.metrics["best/epoch"] == 2.0
    assert data.metrics["best/metrics/mAP50-95"] == pytest.approx(0.4)
    assert data.params["train.epochs"] == "3"
    with pytest.raises(RunAlreadyLoggedError):
        log_run_to_mlflow(run)
    assert log_run_to_mlflow(run, force=True) != run_id


def test_log_requires_files(tmp_path):
    with pytest.raises(FileNotFoundError):
        log_run_to_mlflow(tmp_path)
