from pathlib import Path

import pytest

from training.config import ConfigError, load_config, parse_config

BASE = {
    "name": "run1",
    "model": "yolo11n.pt",
    "dataset": {"profile": "core", "version": "d1dc2ab4d1a9"},
}


def make(**overrides):
    data = {**BASE, **overrides}
    return data


def test_minimal_config_gets_defaults():
    cfg = parse_config(make())
    assert cfg.train["epochs"] == 100
    assert cfg.train["seed"] == 42
    assert cfg.mlflow_experiment == "visionguard"


def test_overrides_are_applied():
    cfg = parse_config(make(train={"epochs": 5, "imgsz": 320}, augment={"fliplr": 0.0}))
    assert cfg.train["epochs"] == 5 and cfg.train["imgsz"] == 320
    assert cfg.augment == {"fliplr": 0.0}


def test_all_problems_reported_at_once():
    with pytest.raises(ConfigError) as info:
        parse_config(
            {"name": "bad name!", "model": "x.onnx", "dataset": {"profile": "nope", "version": "zz"},
             "train": {"epochs": 0}}
        )
    assert len(info.value.errors) >= 5


@pytest.mark.parametrize(
    "train",
    [
        {"epochs": True},
        {"epochs": 1.5},
        {"imgsz": 330},
        {"lr0": 5},
        {"optimizer": "magic"},
        {"device": "gpu"},
        {"cos_lr": "yes"},
        {"freeze": -1},
    ],
)
def test_bad_train_values_rejected(train):
    with pytest.raises(ConfigError):
        parse_config(make(train=train))


def test_unknown_keys_rejected_so_typos_do_not_pass_silently():
    with pytest.raises(ConfigError):
        parse_config(make(train={"epochz": 5}))
    with pytest.raises(ConfigError):
        parse_config(make(extra=1))


def test_device_accepts_gpu_ids():
    assert parse_config(make(train={"device": 0})).train["device"] == "0"
    assert parse_config(make(train={"device": "0,1"})).train["device"] == "0,1"


def test_non_mapping_rejected():
    with pytest.raises(ConfigError):
        parse_config(["a"])


def test_load_missing_and_bad_yaml(tmp_path: Path):
    with pytest.raises(ConfigError):
        load_config(tmp_path / "nope.yaml")
    bad = tmp_path / "bad.yaml"
    bad.write_text("a: [unclosed")
    with pytest.raises(ConfigError):
        load_config(bad)


def test_shipped_configs_are_valid():
    for path in Path("experiments/configs").glob("*.yaml"):
        load_config(path)
