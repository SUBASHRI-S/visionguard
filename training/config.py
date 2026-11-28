"""Training configuration: YAML in, validated frozen object out.

Everything that changes a result lives in the YAML file, so a run can be reproduced from
that file plus the dataset version. Validation reports every problem at once, before any
compute is spent.
"""
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from datasets.construction_ppe import PROFILES

# core_sh17 = core plus SH17 training photos (datasets/sh17.py); built by scripts/extend_with_sh17.py
VALID_PROFILES = sorted({*PROFILES, "core_sh17"})

NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
VERSION_RE = re.compile(r"^[0-9a-f]{12}$")
DEVICE_RE = re.compile(r"^(auto|cpu|\d+(,\d+)*)$")

INT_KEYS: dict[str, tuple[int, int]] = {
    "epochs": (1, 1000),
    "imgsz": (32, 2048),
    "batch": (1, 512),
    "patience": (0, 1000),
    "workers": (0, 32),
    "seed": (0, 2**31 - 1),
    "close_mosaic": (0, 1000),
}
FLOAT_KEYS: dict[str, tuple[float, float]] = {"lr0": (1e-6, 1.0), "fraction": (0.01, 1.0)}
BOOL_KEYS = {"cos_lr", "plots"}
OPTIMIZERS = {"auto", "SGD", "Adam", "Adamax", "AdamW", "NAdam", "RAdam", "RMSProp"}
AUGMENT_KEYS: dict[str, tuple[float, float]] = {
    "hsv_h": (0.0, 1.0),
    "hsv_s": (0.0, 1.0),
    "hsv_v": (0.0, 1.0),
    "degrees": (0.0, 180.0),
    "translate": (0.0, 1.0),
    "scale": (0.0, 1.0),
    "fliplr": (0.0, 1.0),
    "flipud": (0.0, 1.0),
    "mosaic": (0.0, 1.0),
    "mixup": (0.0, 1.0),
}
DEFAULT_TRAIN: dict[str, Any] = {
    "epochs": 100,
    "imgsz": 640,
    "batch": 16,
    "patience": 30,
    "workers": 2,
    "seed": 42,
    "close_mosaic": 10,
    "lr0": 0.01,
    "fraction": 1.0,
    "cos_lr": False,
    "plots": True,
    "optimizer": "auto",
    "freeze": None,
    "device": "auto",
}
TOP_LEVEL_KEYS = {"name", "model", "dataset", "train", "augment", "mlflow", "output"}


class ConfigError(ValueError):
    """Raised for an invalid configuration. ``errors`` lists every problem found."""

    def __init__(self, errors: list[str]) -> None:
        super().__init__("; ".join(errors))
        self.errors = errors


@dataclass(frozen=True)
class TrainConfig:
    name: str
    model: str
    dataset_profile: str
    dataset_version: str
    train: dict[str, Any]
    augment: dict[str, float]
    mlflow_tracking_uri: str
    mlflow_experiment: str
    project: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _section(data: dict, key: str, allowed: set[str], errors: list[str]) -> dict:
    value = data.get(key, {})
    if value is None:
        value = {}
    if not isinstance(value, dict):
        errors.append(f"'{key}' must be a mapping")
        return {}
    extra = sorted(set(value) - allowed)
    if extra:
        errors.append(f"{key}: unknown key(s) {extra}")
    return value


def _validate_train(raw: dict, errors: list[str]) -> dict[str, Any]:
    train = dict(DEFAULT_TRAIN)
    for key, value in raw.items():
        if key not in DEFAULT_TRAIN:
            continue  # unknown keys already reported
        if key in INT_KEYS:
            low, high = INT_KEYS[key]
            if not _is_int(value) or not low <= value <= high:
                errors.append(f"train.{key} must be an integer between {low} and {high}")
                continue
            if key == "imgsz" and value % 32 != 0:
                errors.append("train.imgsz must be a multiple of 32")
                continue
        elif key in FLOAT_KEYS:
            low, high = FLOAT_KEYS[key]
            if not _is_number(value) or not low <= value <= high:
                errors.append(f"train.{key} must be a number between {low} and {high}")
                continue
            value = float(value)
        elif key in BOOL_KEYS:
            if not isinstance(value, bool):
                errors.append(f"train.{key} must be true or false")
                continue
        elif key == "optimizer":
            if value not in OPTIMIZERS:
                errors.append(f"train.optimizer must be one of {sorted(OPTIMIZERS)}")
                continue
        elif key == "freeze":
            if value is not None and not (_is_int(value) and 0 <= value <= 100):
                errors.append("train.freeze must be null or an integer between 0 and 100")
                continue
        elif key == "device":
            value = str(value) if _is_int(value) else value
            if not isinstance(value, str) or not DEVICE_RE.match(value):
                errors.append("train.device must be 'auto', 'cpu', or GPU ids like '0' or '0,1'")
                continue
        train[key] = value
    return train


def _validate_augment(raw: dict, errors: list[str]) -> dict[str, float]:
    augment: dict[str, float] = {}
    for key, value in raw.items():
        if key not in AUGMENT_KEYS:
            continue
        low, high = AUGMENT_KEYS[key]
        if not _is_number(value) or not low <= value <= high:
            errors.append(f"augment.{key} must be a number between {low} and {high}")
            continue
        augment[key] = float(value)
    return augment


def parse_config(data: Any) -> TrainConfig:
    """Validate a config mapping. Raises ConfigError listing every problem."""
    if not isinstance(data, dict):
        raise ConfigError(["config must be a mapping at the top level"])
    errors: list[str] = []

    extra = sorted(set(data) - TOP_LEVEL_KEYS)
    if extra:
        errors.append(f"unknown top-level key(s) {extra}")

    name = data.get("name")
    if not isinstance(name, str) or not NAME_RE.match(name):
        errors.append("name must be 1-64 characters: letters, digits, '_', '-', '.' (not first)")

    model = data.get("model")
    if not isinstance(model, str) or not model.endswith(".pt"):
        errors.append("model must be a pretrained weights file name ending in .pt")

    dataset = _section(data, "dataset", {"profile", "version"}, errors)
    profile = dataset.get("profile")
    if profile not in VALID_PROFILES:
        errors.append(f"dataset.profile must be one of {VALID_PROFILES}")
    version = dataset.get("version")
    if not isinstance(version, str) or not VERSION_RE.match(version):
        errors.append("dataset.version must be a 12-character lowercase hex dataset_version")

    train = _validate_train(_section(data, "train", set(DEFAULT_TRAIN), errors), errors)
    augment = _validate_augment(_section(data, "augment", set(AUGMENT_KEYS), errors), errors)

    mlflow_cfg = _section(data, "mlflow", {"tracking_uri", "experiment"}, errors)
    tracking_uri = mlflow_cfg.get("tracking_uri", "sqlite:///mlflow.db")
    experiment = mlflow_cfg.get("experiment", "visionguard")
    for label, value in (("mlflow.tracking_uri", tracking_uri), ("mlflow.experiment", experiment)):
        if not isinstance(value, str) or not value.strip():
            errors.append(f"{label} must be a non-empty string")

    output = _section(data, "output", {"project"}, errors)
    project = output.get("project", "runs")
    if not isinstance(project, str) or not project.strip():
        errors.append("output.project must be a non-empty string")

    if errors:
        raise ConfigError(errors)
    return TrainConfig(
        name=name,
        model=model,
        dataset_profile=profile,
        dataset_version=version,
        train=train,
        augment=augment,
        mlflow_tracking_uri=tracking_uri,
        mlflow_experiment=experiment,
        project=project,
    )


def load_config(path: Path) -> TrainConfig:
    """Read and validate a YAML config file."""
    try:
        data = yaml.safe_load(Path(path).read_text())
    except FileNotFoundError as exc:
        raise ConfigError([f"config file not found: {path}"]) from exc
    except yaml.YAMLError as exc:
        raise ConfigError([f"invalid YAML: {exc}"]) from exc
    return parse_config(data)
