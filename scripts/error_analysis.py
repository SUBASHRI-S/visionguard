"""Find out where the detector fails, using the validation split.

Run from the project root:
    python -m scripts.error_analysis --run runs/n_core_baseline
Use --split val (the default). Do not tune the model on the test split.
"""
import argparse
import json
from pathlib import Path

import cv2

from evaluation.error_analysis import (
    Prediction, analyze, draw_image, to_markdown, worst_images,
)
from inference.detector import Detector
from training.dataset_check import DatasetError, check_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--iou", type=float, default=0.5, help="match threshold")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--draw", type=int, default=20, help="how many worst images to draw")
    parser.add_argument("--out", type=Path, default=Path("evaluation/error_analysis"))
    args = parser.parse_args()

    if args.split == "test":
        print("WARNING: analysing the test split. Do not use what you learn here to tune the model.")

    config = json.loads((args.run / "run_summary.json").read_text())["config"]
    try:
        manifest = check_dataset(
            config["dataset_profile"], config["dataset_version"],
            Path("datasets/processed"), Path("datasets/versions"),
        )
    except DatasetError as exc:
        raise SystemExit(f"cannot analyse: {exc}") from exc

    root = Path("datasets/processed") / config["dataset_profile"]
    images = sorted((root / "images" / args.split).iterdir())
    detector = Detector(str(args.run / "weights" / "best.pt"), device=args.device)

    def predict(path: Path):
        image = cv2.imread(str(path))
        height, width = image.shape[:2]
        found = detector.predict(image, conf=args.conf)
        return width, height, [
            Prediction(d.class_id, d.confidence, (d.x1, d.y1, d.x2, d.y2)) for d in found
        ]

    summary, matches = analyze(
        images, root / "labels" / args.split, predict, manifest["class_names"], args.iou
    )
    worst = worst_images(matches, limit=max(args.draw, 20))
    meta = {"run_name": config["name"], "split": args.split,
            "dataset_version": config["dataset_version"], "conf": args.conf, "iou": args.iou}

    out_dir = args.out / f"{config['name']}_{args.split}"
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps({"meta": meta, **summary}, indent=2))
    (out_dir / "report.md").write_text(to_markdown(summary, meta, worst))
    by_name = {p.name: p for p in images}
    for name, _ in worst[: args.draw]:
        picture = draw_image(cv2.imread(str(by_name[name])), matches[name], manifest["class_names"])
        cv2.imwrite(str(out_dir / "images" / name), picture)

    print((out_dir / "report.md").read_text())
    print(f"report: {out_dir / 'report.md'}")
    print(f"worst images drawn in: {out_dir / 'images'}")


if __name__ == "__main__":
    main()
