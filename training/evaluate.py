"""Evaluate trained weights on one dataset split and write an auditable result.

The test split is for the final, honest number. To keep it that way, a test evaluation for a
run is written once; running it again is refused unless ``force`` is set. Every result records
the weights hash, the dataset version and the settings, so the number can be traced.
"""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from training.dataset_check import check_dataset
from training.train import environment_info, git_info, resolve_device

SPLITS = ("val", "test")


class EvaluationError(RuntimeError):
    """Raised when an evaluation must not run or the model does not fit the dataset."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def per_class_table(
    class_names: list[str],
    ap_class_index: list[int],
    class_results: list[tuple[float, float, float, float]],
    instances: dict[str, int],
) -> dict[str, dict[str, Any]]:
    """Map Ultralytics per-class results to class names.

    A class with no examples in the split has no result; it is reported as ``None`` instead of
    a misleading zero.
    """
    by_index = dict(zip(ap_class_index, class_results))
    table: dict[str, dict[str, Any]] = {}
    for index, name in enumerate(class_names):
        result = by_index.get(index)
        row: dict[str, Any] = {"instances": instances.get(name, 0)}
        if result is None:
            row.update(precision=None, recall=None, map50=None, map50_95=None)
        else:
            precision, recall, map50, map50_95 = (float(v) for v in result)
            row.update(precision=precision, recall=recall, map50=map50, map50_95=map50_95)
        table[name] = row
    return table


def to_markdown(result: dict[str, Any]) -> str:
    def cell(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.3f}"

    lines = [
        f"# Evaluation: {result['run_name']} on the {result['split']} split",
        "",
        f"- Dataset version: `{result['dataset_version']}` (profile `{result['profile']}`)",
        f"- Images: {result['images']}",
        f"- Weights SHA256: `{result['weights_sha256']}`",
        f"- Settings: {json.dumps(result['settings'])}",
        f"- Evaluated at: {result['evaluated_at']}",
        "",
        "| Class | Instances | Precision | Recall | mAP50 | mAP50-95 |",
        "|---|---|---|---|---|---|",
    ]
    overall = result["overall"]
    lines.append(
        f"| **all** | {result['instances']} | {cell(overall['precision'])} | "
        f"{cell(overall['recall'])} | {cell(overall['map50'])} | {cell(overall['map50_95'])} |"
    )
    for name, row in result["per_class"].items():
        lines.append(
            f"| {name} | {row['instances']} | {cell(row['precision'])} | {cell(row['recall'])} | "
            f"{cell(row['map50'])} | {cell(row['map50_95'])} |"
        )
    speed = result["speed_ms"]
    lines += [
        "",
        f"Speed per image on `{result['device']}` (not a deployment benchmark): "
        f"preprocess {speed['preprocess']:.1f} ms, inference {speed['inference']:.1f} ms, "
        f"postprocess {speed['postprocess']:.1f} ms.",
        "",
        "Small classes have few instances, so their numbers are noisy.",
    ]
    return "\n".join(lines) + "\n"


def evaluate(
    weights: Path,
    *,
    run_name: str,
    profile: str,
    version: str,
    split: str = "test",
    processed_root: Path = Path("datasets/processed"),
    versions_root: Path = Path("datasets/versions"),
    results_dir: Path = Path("evaluation/results"),
    imgsz: int = 640,
    batch: int = 16,
    conf: float = 0.001,
    iou: float = 0.6,
    device: str = "auto",
    force: bool = False,
) -> dict[str, Any]:
    """Evaluate ``weights`` on ``split`` and write JSON and Markdown results."""
    if split not in SPLITS:
        raise EvaluationError(f"split must be one of {SPLITS}")
    weights = Path(weights)
    if not weights.is_file():
        raise EvaluationError(f"weights not found: {weights}")

    json_path = results_dir / f"{run_name}_{split}.json"
    if split == "test" and json_path.exists() and not force:
        raise EvaluationError(
            f"{json_path} already exists. The test split is evaluated once per run; "
            "use --force only if you are sure, and say so when you report the number."
        )

    manifest = check_dataset(profile, version, processed_root, versions_root)

    from ultralytics import YOLO

    model = YOLO(str(weights))
    model_names = [model.names[i] for i in sorted(model.names)]
    if model_names != manifest["class_names"]:
        raise EvaluationError(
            f"model classes {model_names} do not match dataset classes {manifest['class_names']}"
        )

    used_device = resolve_device(device)
    data_yaml = (processed_root / profile / "data.yaml").resolve()
    metrics = model.val(
        data=str(data_yaml),
        split=split,
        imgsz=imgsz,
        batch=batch,
        conf=conf,
        iou=iou,
        device=used_device,
        plots=True,
        project=str(results_dir.resolve() / "plots"),
        name=f"{run_name}_{split}",
        exist_ok=force,
        verbose=False,
    )
    box = metrics.box
    counts = manifest["counts"][split]
    ap_index = [int(i) for i in box.ap_class_index]
    result = {
        "run_name": run_name,
        "split": split,
        "profile": profile,
        "dataset_version": version,
        "class_names": manifest["class_names"],
        "images": counts["images"],
        "instances": sum(counts["instances"].values()),
        "weights": str(weights),
        "weights_sha256": sha256_file(weights),
        "overall": {
            "precision": float(box.mp),
            "recall": float(box.mr),
            "map50": float(box.map50),
            "map50_95": float(box.map),
        },
        "per_class": per_class_table(
            manifest["class_names"],
            ap_index,
            [box.class_result(i) for i in range(len(ap_index))],
            counts["instances"],
        ),
        "speed_ms": {k: float(v) for k, v in metrics.speed.items() if k != "loss"},
        "settings": {"imgsz": imgsz, "batch": batch, "conf": conf, "iou": iou},
        "device": used_device,
        "git": git_info(),
        "environment": environment_info(),
        "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(result, indent=2))
    (results_dir / f"{run_name}_{split}.md").write_text(to_markdown(result))
    return result
