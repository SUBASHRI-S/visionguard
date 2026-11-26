"""Run the full VisionGuard chain on one video file.

    python -m scripts.run_video data/clips/small/67460-522170651_medium.mp4 --run runs/n_core_imgsz960

Everything is written to outputs/video_runs/<clip name>/ (see pipeline/runner.py). The default
safety settings are the ones chosen on the validation split (experiments/safety/chosen.yaml).
Nothing here is tuned on video: read the report's caveats before quoting a result.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from incidents.temporal import IncidentConfigError, IncidentTracker, TemporalConfig
from pipeline.detector import DetectorError, YoloDetector
from pipeline.runner import VideoError, process_video
from pipeline.video_pipeline import VideoPipeline
from safety.config import SafetyConfigError
from safety.engine import SafetyEngine
from safety.zones import Zone, ZoneError
from trackers import PersonTracker, TrackerConfig, TrackerConfigError


def load_yaml(path: Path | None) -> dict | None:
    if path is None:
        return None
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", type=Path)
    ap.add_argument("--run", type=Path, default=Path("runs/n_core_imgsz960"), help="training run folder with weights/best.pt")
    ap.add_argument("--weights", type=Path, default=None, help="use this weights file instead of <run>/weights/best.pt")
    ap.add_argument("--imgsz", type=int, default=None, help="default: the size the run was trained at")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--safety-config", type=Path, default=Path("experiments/safety/chosen.yaml"))
    ap.add_argument("--tracker-config", type=Path, default=None)
    ap.add_argument("--temporal-config", type=Path, default=None)
    ap.add_argument("--zones", type=Path, default=None, help="yaml with a 'zones:' list of {name, polygon}")
    ap.add_argument("--camera-id", default=None, help="default: the video's file name")
    ap.add_argument("--stride", type=int, default=1, help="process every Nth frame (default 1 = all)")
    ap.add_argument("--max-seconds", type=float, default=None)
    ap.add_argument("--no-video", action="store_true", help="skip writing annotated.mp4")
    ap.add_argument("--out", type=Path, default=Path("outputs/video_runs"))
    args = ap.parse_args(argv)

    try:
        weights = args.weights or args.run / "weights" / "best.pt"
        imgsz = args.imgsz
        if imgsz is None:
            summary_file = args.run / "run_summary.json"
            if not summary_file.is_file():
                raise SystemExit(f"cannot find {summary_file}; pass --imgsz explicitly")
            imgsz = json.loads(summary_file.read_text())["config"]["train"]["imgsz"]
        safety_cfg = load_yaml(args.safety_config)
        zone_cfg = load_yaml(args.zones) or {}
        zones = [Zone.from_dict(z) for z in zone_cfg.get("zones", [])]
        tracker_cfg = TrackerConfig.from_dict(load_yaml(args.tracker_config))
        temporal_cfg = TemporalConfig.from_dict(load_yaml(args.temporal_config))
        engine = SafetyEngine.from_config(safety_cfg)
        print(f"model {weights}  inference size {imgsz}  conf {args.conf}  device {args.device}")
        detector = YoloDetector(weights, imgsz=imgsz, conf=args.conf, device=args.device)
    except (SafetyConfigError, IncidentConfigError, TrackerConfigError, ZoneError, DetectorError, ValueError) as exc:
        raise SystemExit(f"cannot start: {exc}") from exc

    camera_id = args.camera_id or args.video.name
    pipeline = VideoPipeline(detector, engine, PersonTracker(tracker_cfg), IncidentTracker(temporal_cfg),
                             camera_id=camera_id, zones=zones)
    out_dir = args.out / args.video.stem
    run_info = {
        "weights": str(weights), "imgsz": imgsz, "conf": args.conf, "device": args.device,
        "safety_config": safety_cfg or "defaults", "tracker_config": tracker_cfg.__dict__,
        "temporal_config": temporal_cfg.__dict__, "zones": [z.name for z in zones],
    }
    try:
        summary = process_video(
            args.video, pipeline, out_dir, frame_stride=args.stride, max_seconds=args.max_seconds,
            write_video=not args.no_video, run_info=run_info,
            progress=lambda done, total: print(f"  {done} frames processed"),
        )
    except (VideoError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from exc
    print(f"done in {summary['wall_seconds']}s -> {out_dir}")
    print(f"incidents: {len(summary['incidents'])}; tracks: {len(summary['tracks'])}")
    print(f"read {out_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
