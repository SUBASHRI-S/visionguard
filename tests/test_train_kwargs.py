from pathlib import Path

from training.config import parse_config
from training.train import build_train_kwargs, git_info


def cfg(**extra):
    return parse_config({"name": "r", "model": "yolo11n.pt",
                         "dataset": {"profile": "core", "version": "d1dc2ab4d1a9"}, **extra})


def test_kwargs_contain_reproducibility_settings():
    kwargs = build_train_kwargs(cfg(), Path("d.yaml"), "cpu", Path("runs"))
    assert kwargs["seed"] == 42 and kwargs["deterministic"] is True
    assert kwargs["data"] == "d.yaml" and kwargs["name"] == "r" and kwargs["exist_ok"] is False
    assert kwargs["device"] == "cpu"
    assert "freeze" not in kwargs


def test_augment_and_freeze_are_passed_through():
    kwargs = build_train_kwargs(
        cfg(train={"freeze": 10}, augment={"fliplr": 0.0}), Path("d.yaml"), "0", Path("runs")
    )
    assert kwargs["freeze"] == 10 and kwargs["fliplr"] == 0.0


def test_git_info_outside_a_repo_does_not_crash(tmp_path):
    info = git_info(tmp_path)
    assert set(info) == {"commit", "dirty"}
