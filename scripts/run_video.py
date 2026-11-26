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

import cv2

from incidents.memory import MemoryConfig, MemoryConfigError, PPEMemory
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


def probe_fps(video: Path) -> float:
    cap = cv2.VideoCapture(str(video))
    try:
        if not cap.isOpened():
            raise ValueError(f"cannot open video: {video}")
        return float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    finally:
        cap.release()


def load_zone_file(path: Path | None) -> tuple[list[Zone], Zone | None]:
    """Read ``zones:`` (restricted areas) and ``work_area:`` (where PPE is required) from one file.

    Polygons use fractions of the frame (0 to 1) so they survive a change of video size.
    """
    data = load_yaml(path) or {}
    if not isinstance(data, dict) or set(data) - {"zones", "work_area"}:
        raise ZoneError("the zone file may only contain 'zones' and 'work_area'")
    zones = [Zone.from_dict(z) for z in data.get("zones", [])]
    work_area = Zone.from_dict(data["work_area"]) if data.get("work_area") else None
    return zones, work_area


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
    ap.add_argument("--temporal-config", type=Path, default=None,
                    help="yaml with timing in processed FRAMES; if given it replaces the --*-seconds options")
    ap.add_argument("--open-after-seconds", type=float, default=3.0,
                    help="a violation must persist this long before an incident opens (default 3)")
    ap.add_argument("--close-after-seconds", type=float, default=2.0,
                    help="this long without the violation closes the incident (default 2)")
    ap.add_argument("--memory-seconds", type=float, default=5.0,
                    help="an item seen on a person this recently is never reported missing; 0 turns it off (default 5)")
    ap.add_argument("--zones", type=Path, default=None,
                    help="yaml with 'zones:' (restricted areas) and/or 'work_area:' (PPE is checked only inside it)")
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
        zones, work_area = load_zone_file(args.zones)
        tracker_cfg = TrackerConfig.from_dict(load_yaml(args.tracker_config))
        if args.temporal_config is not None:
            temporal_cfg = TemporalConfig.from_dict(load_yaml(args.temporal_config))
        else:
            temporal_cfg = TemporalConfig.from_seconds(
                fps=probe_fps(args.video), frame_stride=args.stride,
                open_after_seconds=args.open_after_seconds, close_after_seconds=args.close_after_seconds)
        memory = PPEMemory(MemoryConfig(args.memory_seconds)) if args.memory_seconds > 0 else None
        engine = SafetyEngine.from_config(safety_cfg)
        print(f"model {weights}  inference size {imgsz}  conf {args.conf}  device {args.device}")
        detector = YoloDetector(weights, imgsz=imgsz, conf=args.conf, device=args.device)
    except (SafetyConfigError, IncidentConfigError, TrackerConfigError, ZoneError, DetectorError,
            MemoryConfigError, ValueError) as exc:
        raise SystemExit(f"cannot start: {exc}") from exc

    camera_id = args.camera_id or args.video.name
    pipeline = VideoPipeline(detector, engine, PersonTracker(tracker_cfg), IncidentTracker(temporal_cfg),
                             camera_id=camera_id, zones=zones, work_area=work_area, memory=memory)
    out_dir = args.out / args.video.stem
    run_info = {
        "weights": str(weights), "imgsz": imgsz, "conf": args.conf, "device": args.device,
        "safety_config": safety_cfg or "defaults", "tracker_config": tracker_cfg.__dict__,
        "temporal_config": temporal_cfg.__dict__,
        "memory_seconds": args.memory_seconds if memory else "off", "zones": [z.name for z in zones], "work_area": work_area.name if work_area else "whole frame",
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
