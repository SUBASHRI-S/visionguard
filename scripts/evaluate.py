"""Evaluate a trained run on the validation or test split.

Run from the project root:
    python -m scripts.evaluate --run runs/n_core_baseline --split val
    python -m scripts.evaluate --run runs/n_core_baseline --split test
"""
import argparse
import json
from pathlib import Path

from training.dataset_check import DatasetError
from training.evaluate import EvaluationError, evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="run folder with run_summary.json")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--weights", type=Path, default=None, help="default: <run>/weights/best.pt")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--force", action="store_true", help="overwrite an existing test result")
    args = parser.parse_args()

    summary_path = args.run / "run_summary.json"
    if not summary_path.exists():
        raise SystemExit(f"{summary_path} not found")
    config = json.loads(summary_path.read_text())["config"]

    try:
        result = evaluate(
            args.weights or args.run / "weights" / "best.pt",
            run_name=config["name"],
            profile=config["dataset_profile"],
            version=config["dataset_version"],
            split=args.split,
            imgsz=config["train"]["imgsz"],
            batch=args.batch,
            device=args.device,
            force=args.force,
        )
    except (EvaluationError, DatasetError) as exc:
        raise SystemExit(f"cannot evaluate: {exc}") from exc

    overall = result["overall"]
    print(f"{config['name']} on {args.split} ({result['images']} images, {result['instances']} objects)")
    print(f"precision {overall['precision']:.3f}  recall {overall['recall']:.3f}  "
          f"mAP50 {overall['map50']:.3f}  mAP50-95 {overall['map50_95']:.3f}")
    for name, row in result["per_class"].items():
        if row["map50_95"] is None:
            print(f"  {name:10s} no examples in this split")
        else:
            print(f"  {name:10s} P {row['precision']:.3f}  R {row['recall']:.3f}  "
                  f"mAP50 {row['map50']:.3f}  mAP50-95 {row['map50_95']:.3f}  ({row['instances']} obj)")
    print(f"written to evaluation/results/{config['name']}_{args.split}.json and .md")


if __name__ == "__main__":
    main()
