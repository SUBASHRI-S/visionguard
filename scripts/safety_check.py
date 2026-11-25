"""Run the trained detector and the safety engine on real images and compare with the labels.

Run from the project root:
    python -m scripts.safety_check --run runs/n_core_imgsz960
Uses the validation split by default. Safety thresholds are untuned placeholders unless you
pass --safety-config with a YAML file (keys: enabled_rules, ppe, zone, proximity).
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import yaml

from evaluation.error_analysis import parse_labels
from evaluation.safety_check import ImageComparison, compare_image, summarize, to_markdown
from safety.contracts import Box, Detection, PPEStatus
from safety.engine import SafetyEngine
from training.dataset_check import DatasetError, check_dataset
from ultralytics import YOLO

COLOURS = {"all_ok": (0, 160, 0), "violation": (0, 0, 220), "unknown": (0, 200, 230)}


def draw_statuses(image, statuses: list[PPEStatus]):
    out = image.copy()
    for s in statuses:
        b = s.person.box
        if "no" in (s.helmet, s.vest):
            colour = COLOURS["violation"]
        elif "unknown" in (s.helmet, s.vest):
            colour = COLOURS["unknown"]
        else:
            colour = COLOURS["all_ok"]
        cv2.rectangle(out, (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2)), colour, 2)
        label = f"H:{s.helmet} V:{s.vest}" + (" op" if s.is_operator else "")
        cv2.putText(out, label, (int(b.x1), max(14, int(b.y1) - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--conf", type=float, default=0.35, help="detector confidence threshold")
    parser.add_argument("--iou", type=float, default=0.5, help="person match threshold")
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--safety-config", type=Path, default=None)
    parser.add_argument("--draw", type=int, default=20)
    parser.add_argument("--out", type=Path, default=Path("evaluation/safety_check"))
    args = parser.parse_args()
    if args.split == "test":
        print("WARNING: do not tune safety thresholds on the test split.")

    config = json.loads((args.run / "run_summary.json").read_text())["config"]
    try:
        manifest = check_dataset(config["dataset_profile"], config["dataset_version"],
                                 Path("datasets/processed"), Path("datasets/versions"))
    except DatasetError as exc:
        raise SystemExit(f"cannot run: {exc}") from exc
    names = manifest["class_names"]
    engine_cfg = yaml.safe_load(args.safety_config.read_text()) if args.safety_config else None
    engine = SafetyEngine.from_config(engine_cfg)

    root = Path("datasets/processed") / config["dataset_profile"]
    images = sorted((root / "images" / args.split).iterdir())
    imgsz = args.imgsz or config["train"]["imgsz"]
    model = YOLO(str(args.run / "weights" / "best.pt"))
    print(f"inference size: {imgsz}, {len(images)} images")

    comparisons: dict[str, ImageComparison] = {}
    pred_statuses: dict[str, list[PPEStatus]] = {}
    events: dict[str, Counter] = defaultdict(Counter)
    for path in images:
        image = cv2.imread(str(path))
        height, width = image.shape[:2]
        label_file = root / "labels" / args.split / f"{path.stem}.txt"
        truth_dets = [
            Detection(names[g.class_id], 1.0, Box(*g.box))
            for g in parse_labels(label_file.read_text() if label_file.exists() else "", width, height)
        ]
        result = model.predict(image, conf=args.conf, imgsz=imgsz, device=args.device, verbose=False)[0]
        pred_dets = [
            Detection(names[int(c)], float(s), Box(*[float(v) for v in b]))
            for b, c, s in zip(result.boxes.xyxy.tolist(), result.boxes.cls.tolist(),
                               result.boxes.conf.tolist())
        ]
        frame = dict(camera_id="eval", frame_index=0, timestamp=0.0,
                     frame_width=width, frame_height=height)
        truth_ctx = engine.build_context(truth_dets, **frame)
        pred_ctx = engine.build_context(pred_dets, **frame)
        for event in engine.evaluate(pred_ctx):
            events[event.rule_id][event.outcome] += 1
        comparisons[path.name] = compare_image(list(truth_ctx.statuses), list(pred_ctx.statuses), args.iou)
        pred_statuses[path.name] = list(pred_ctx.statuses)

    summary = summarize(list(comparisons.values()))
    meta = {"run_name": config["name"], "split": args.split, "imgsz": imgsz,
            "dataset_version": config["dataset_version"], "conf": args.conf, "iou": args.iou}
    out_dir = args.out / f"{config['name']}_{args.split}"
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    events_plain = {rule: dict(counts) for rule, counts in events.items()}
    (out_dir / "summary.json").write_text(json.dumps({"meta": meta, **summary, "events": events_plain}, indent=2))
    report = to_markdown(summary, meta, events_plain)
    (out_dir / "report.md").write_text(report)

    worst = sorted(comparisons.items(), key=lambda kv: kv[1].disagreements, reverse=True)
    for name, comparison in worst[: args.draw]:
        if comparison.disagreements == 0:
            break
        picture = draw_statuses(cv2.imread(str(root / "images" / args.split / name)), pred_statuses[name])
        cv2.imwrite(str(out_dir / "images" / name), picture)

    print(report)
    print(f"report: {out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
