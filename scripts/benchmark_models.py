"""Measure how long each model takes per frame on this machine (CPU by default).

Example (from the project root):
    python -m scripts.benchmark_models --video data/clips/heldout_v2/154950-809834548_medium.mp4 ^
        --runs runs/n_core_imgsz960 runs/s_core_imgsz960

What is timed: ``YoloDetector.detect`` on a frame (resize, network, post-processing), the same call
the video pipeline makes. Reading the video is not timed. The first few frames are warm-up and are
dropped. The result describes THIS machine and THIS clip; it is not a general benchmark.
"""
import argparse
import json
import platform
import statistics
import time
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from pipeline.detector import FrameDetector, YoloDetector


def sample_frames(video: Path, count: int) -> list[np.ndarray]:
    """Return ``count`` frames spread evenly through the video."""
    if count < 1:
        raise ValueError("count must be at least 1")
    cap = cv2.VideoCapture(str(video))
    try:
        if not cap.isOpened():
            raise ValueError(f"cannot open video: {video}")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total < 1:
            raise ValueError(f"video has no frames: {video}")
        indices = sorted({int(i * (total - 1) / max(count - 1, 1)) for i in range(count)})
        frames: list[np.ndarray] = []
        for index in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = cap.read()
            if ok:
                frames.append(frame)
        if not frames:
            raise ValueError(f"could not read any frame from: {video}")
        return frames
    finally:
        cap.release()


def time_detector(detector: FrameDetector, frames: Sequence[np.ndarray], warmup: int) -> list[float]:
    """Milliseconds per call, after ``warmup`` untimed calls on the first frame."""
    if not frames:
        raise ValueError("no frames to time")
    for _ in range(max(warmup, 0)):
        detector.detect(frames[0])
    times: list[float] = []
    for frame in frames:
        start = time.perf_counter()
        detector.detect(frame)
        times.append((time.perf_counter() - start) * 1000.0)
    return times


def summarize(times_ms: Sequence[float]) -> dict[str, float]:
    if not times_ms:
        raise ValueError("no timings")
    ordered = sorted(times_ms)
    p95_index = min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))
    median = statistics.median(ordered)
    return {
        "frames": float(len(ordered)),
        "median_ms": round(median, 1),
        "p95_ms": round(ordered[p95_index], 1),
        "min_ms": round(ordered[0], 1),
        "max_ms": round(ordered[-1], 1),
        "fps_at_median": round(1000.0 / median, 2) if median > 0 else 0.0,
    }


def machine_info(device: str) -> dict[str, str]:
    info = {"platform": platform.platform(), "processor": platform.processor(), "python": platform.python_version(), "device": device}
    try:
        import os

        info["cpu_count"] = str(os.cpu_count())
        import torch

        info["torch"] = torch.__version__
        info["torch_threads"] = str(torch.get_num_threads())
    except ImportError:
        pass
    return info


def to_markdown(results: dict[str, dict[str, float]], meta: dict[str, object]) -> str:
    lines = [
        f"# CPU timing: {meta['video']}",
        "",
        f"- Input size {meta['imgsz']}, {meta['frames']} frames sampled, {meta['warmup']} warm-up calls dropped",
        f"- Machine: {meta['machine']}",
        "- One clip, one machine, one run: treat differences under about 10 percent as noise.",
        "",
        "| Model | Median ms/frame | 95th percentile ms | Frames/s at median |",
        "|---|---|---|---|",
    ]
    for name, row in results.items():
        lines.append(f"| {name} | {row['median_ms']} | {row['p95_ms']} | {row['fps_at_median']} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--runs", type=Path, nargs="+", required=True, help="run folders containing weights/best.pt")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path, default=Path("evaluation/benchmarks"))
    args = parser.parse_args()

    frames = sample_frames(args.video, args.frames)
    results: dict[str, dict[str, float]] = {}
    for run in args.runs:
        weights = run / "weights" / "best.pt"
        detector = YoloDetector(weights, imgsz=args.imgsz, conf=0.35, device=args.device)
        results[run.name] = summarize(time_detector(detector, frames, args.warmup))
        print(f"{run.name}: median {results[run.name]['median_ms']} ms/frame")

    meta = {
        "video": args.video.name,
        "imgsz": args.imgsz,
        "frames": len(frames),
        "warmup": args.warmup,
        "machine": machine_info(args.device),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    stem = f"{args.device}_imgsz{args.imgsz}_{args.video.stem}"
    (args.out / f"{stem}.json").write_text(json.dumps({"meta": meta, "results": results}, indent=2), encoding="utf-8")
    report = to_markdown(results, meta)
    (args.out / f"{stem}.md").write_text(report, encoding="utf-8")
    print()
    print(report)


if __name__ == "__main__":
    main()
