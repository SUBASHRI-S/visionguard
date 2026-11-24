"""Train from a config file, then log the finished run to MLflow.

Run from the project root:
    python -m scripts.train --config experiments/configs/smoke.yaml
    python -m scripts.train --config experiments/configs/n_core_baseline.yaml --no-mlflow
Import a run that was trained elsewhere (for example on Kaggle):
    python -m scripts.train --log-only runs/n_core_baseline
"""
import argparse
from pathlib import Path

from training.config import ConfigError, load_config
from training.tracking import RunAlreadyLoggedError, log_run_to_mlflow


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--log-only", type=Path, help="log an existing run folder to MLflow")
    parser.add_argument("--no-mlflow", action="store_true", help="train but skip MLflow logging")
    args = parser.parse_args()

    if args.log_only:
        run_dir = args.log_only
    elif args.config:
        try:
            config = load_config(args.config)
        except ConfigError as exc:
            for line in exc.errors:
                print("CONFIG ERROR:", line)
            raise SystemExit(2) from exc
        from training.dataset_check import DatasetError
        from training.train import train

        try:
            run_dir = train(config, config_path=args.config)
        except (DatasetError, FileExistsError) as exc:
            raise SystemExit(f"cannot start: {exc}") from exc
        print(f"training finished: {run_dir}")
    else:
        parser.error("give --config or --log-only")

    if args.no_mlflow:
        return
    try:
        run_id = log_run_to_mlflow(run_dir)
    except RunAlreadyLoggedError as exc:
        raise SystemExit(str(exc)) from exc
    print(f"logged to MLflow, run id {run_id}")


if __name__ == "__main__":
    main()
