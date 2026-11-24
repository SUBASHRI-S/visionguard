"""Read finished training runs and log them to MLflow.

Training writes plain files (results.csv, run_summary.json, weights). Logging to MLflow is a
separate step that only reads those files, so a run trained on a cloud GPU can be imported on
your own machine, and a failed MLflow call never loses a finished training run.
"""
import csv
import json
import math
import re
from pathlib import Path

_INVALID_CHARS = re.compile(r"[^A-Za-z0-9_\-. /]")
MAIN_METRICS = ("metrics/precision", "metrics/recall", "metrics/mAP50", "metrics/mAP50-95")
OPTIONAL_ARTIFACTS = {
    "config.yaml": "run",
    "args.yaml": "run",
    "results.csv": "run",
    "weights/best.pt": "weights",
    "results.png": "plots",
    "confusion_matrix.png": "plots",
    "confusion_matrix_normalized.png": "plots",
    "BoxPR_curve.png": "plots",
    "BoxF1_curve.png": "plots",
}


class RunAlreadyLoggedError(RuntimeError):
    """Raised when a run with the same name is already in the MLflow experiment."""


def clean_metric_name(name: str) -> str:
    """Make an Ultralytics column name valid for MLflow: drop '(B)', replace odd characters."""
    return _INVALID_CHARS.sub("_", name.strip().replace("(B)", ""))


def read_results(path: Path) -> list[dict[str, float]]:
    """Read results.csv into one dict per epoch, with cleaned metric names."""
    rows: list[dict[str, float]] = []
    with open(path, newline="") as handle:
        for raw in csv.DictReader(handle):
            row: dict[str, float] = {}
            for key, value in raw.items():
                if key is None:
                    continue
                try:
                    number = float(value)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(number):
                    row[clean_metric_name(key)] = number
            rows.append(row)
    return rows


def fitness(row: dict[str, float]) -> float:
    """Ultralytics' model-selection score: 10% mAP@50 + 90% mAP@50-95."""
    return 0.1 * row.get("metrics/mAP50", 0.0) + 0.9 * row.get("metrics/mAP50-95", 0.0)


def best_row(rows: list[dict[str, float]]) -> tuple[int, dict[str, float]]:
    """Return (index, row) of the epoch with the highest fitness (earliest wins ties)."""
    if not rows:
        raise ValueError("no epochs to choose from")
    index = max(range(len(rows)), key=lambda i: (fitness(rows[i]), -i))
    return index, rows[index]


def _flatten_params(config: dict) -> dict[str, str]:
    params = {
        "name": config["name"],
        "model": config["model"],
        "dataset_profile": config["dataset_profile"],
        "dataset_version": config["dataset_version"],
        "project": config["project"],
    }
    for key, value in config["train"].items():
        params[f"train.{key}"] = value
    for key, value in config["augment"].items():
        params[f"augment.{key}"] = value
    return {k: str(v) for k, v in params.items()}


def log_run_to_mlflow(
    run_dir: Path,
    tracking_uri: str | None = None,
    experiment: str | None = None,
    *,
    force: bool = False,
) -> str:
    """Log a finished run folder to MLflow and return the MLflow run id."""
    import mlflow  # imported here so the rest of the module works without it

    run_dir = Path(run_dir)
    summary_path = run_dir / "run_summary.json"
    results_path = run_dir / "results.csv"
    if not summary_path.exists() or not results_path.exists():
        raise FileNotFoundError(f"{run_dir} needs run_summary.json and results.csv")

    summary = json.loads(summary_path.read_text())
    config = summary["config"]
    name = config["name"]
    uri = tracking_uri or config["mlflow_tracking_uri"]
    experiment_name = experiment or config["mlflow_experiment"]
    rows = read_results(results_path)

    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment_name)
    existing = mlflow.search_runs(
        experiment_names=[experiment_name],
        filter_string=f"tags.vg_run_name = '{name}'",
        output_format="list",
    )
    if existing and not force:
        raise RunAlreadyLoggedError(
            f"run '{name}' is already in experiment '{experiment_name}'. "
            "Use a new name, or pass force to log a second copy."
        )

    git = summary.get("git", {})
    env = summary.get("environment", {})
    with mlflow.start_run(run_name=name) as run:
        mlflow.log_params(_flatten_params(config))
        mlflow.set_tags(
            {
                "vg_run_name": name,
                "dataset_version": config["dataset_version"],
                "dataset_source_sha256": str(summary["dataset"].get("source_zip_sha256")),
                "git_commit": str(git.get("commit")),
                "git_dirty": str(git.get("dirty")),
                "ultralytics_version": str(env.get("ultralytics")),
                "torch_version": str(env.get("torch")),
                "device_name": str(env.get("gpu_name") or "cpu"),
                "metrics_split": "validation (carved from official train); test split untouched",
            }
        )
        for index, row in enumerate(rows):
            step = int(row.get("epoch", index + 1))
            mlflow.log_metrics({k: v for k, v in row.items() if k != "epoch"}, step=step)
        if rows:
            best_index, best = best_row(rows)
            summary_metrics = {f"best/{k}": v for k, v in best.items() if k.startswith("metrics/")}
            summary_metrics["best/epoch"] = float(best.get("epoch", best_index + 1))
            summary_metrics["best/fitness"] = fitness(best)
            mlflow.log_metrics(summary_metrics)

        mlflow.log_artifact(str(summary_path), artifact_path="run")
        for relative, folder in OPTIONAL_ARTIFACTS.items():
            candidate = run_dir / relative
            if candidate.is_file():
                mlflow.log_artifact(str(candidate), artifact_path=folder)
        return run.info.run_id
