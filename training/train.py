"""Train a detector from a YAML config and record everything needed to reproduce it.

Training writes plain files into the run folder (weights, results.csv, config.yaml,
run_summary.json). Logging to MLflow happens afterwards from those files, so a tracking
problem can never lose a finished run.
"""
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import yaml

from training.config import TrainConfig
from training.dataset_check import check_dataset


def git_info(repo: Path | None = None) -> dict[str, Any]:
    """Current commit and whether the working tree has uncommitted changes."""

    def run(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", *args], cwd=repo, capture_output=True, text=True, timeout=10, check=True
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return result.stdout.strip()

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain")
    return {"commit": commit, "dirty": None if status is None else bool(status)}


def environment_info() -> dict[str, Any]:
    """Library versions and hardware, recorded with every run."""
    import torch
    import ultralytics

    cuda = torch.cuda.is_available()
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "cuda_available": cuda,
        "gpu_name": torch.cuda.get_device_name(0) if cuda else None,
    }


def resolve_device(requested: str) -> str:
    """'auto' becomes GPU 0 when CUDA is available, otherwise 'cpu'."""
    if requested != "auto":
        return requested
    import torch

    return "0" if torch.cuda.is_available() else "cpu"


def build_train_kwargs(
    config: TrainConfig, data_yaml: Path, device: str, project: Path
) -> dict[str, Any]:
    """Arguments passed to Ultralytics ``model.train``."""
    kwargs: dict[str, Any] = {
        key: value for key, value in config.train.items() if key not in {"device", "freeze"}
    }
    if config.train.get("freeze") is not None:
        kwargs["freeze"] = config.train["freeze"]
    kwargs.update(config.augment)
    kwargs.update(
        data=str(data_yaml),
        device=device,
        project=str(project),
        name=config.name,
        exist_ok=False,
        pretrained=True,
        deterministic=True,
        val=True,
    )
    return kwargs


def train(
    config: TrainConfig,
    *,
    processed_root: Path = Path("datasets/processed"),
    versions_root: Path = Path("datasets/versions"),
    config_path: Path | None = None,
) -> Path:
    """Run one training job and return the run folder."""
    manifest = check_dataset(
        config.dataset_profile, config.dataset_version, processed_root, versions_root
    )
    project = Path(config.project).resolve()
    run_dir = project / config.name
    if run_dir.exists():
        raise FileExistsError(f"{run_dir} already exists; use a new run name in the config")

    from ultralytics import YOLO, settings

    settings.update({"mlflow": False})  # we log to MLflow ourselves, after training

    device = resolve_device(config.train["device"])
    data_yaml = (processed_root / config.dataset_profile / "data.yaml").resolve()
    kwargs = build_train_kwargs(config, data_yaml, device, project)

    started = time.time()
    model = YOLO(config.model)
    model.train(**kwargs)
    elapsed = time.time() - started

    run_dir = Path(model.trainer.save_dir)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config.to_dict(), sort_keys=False))
    summary = {
        "config": config.to_dict(),
        "config_file": str(config_path) if config_path else None,
        "dataset": {
            "version": manifest["dataset_version"],
            "profile": manifest["profile"],
            "source_zip_sha256": manifest.get("source_zip_sha256"),
            "class_names": manifest["class_names"],
            "counts": {k: v["images"] for k, v in manifest["counts"].items()},
        },
        "git": git_info(),
        "environment": environment_info(),
        "device_used": device,
        "train_seconds": round(elapsed, 1),
        "train_kwargs": {k: v for k, v in kwargs.items() if k != "project"},
    }
    (run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))
    return run_dir
