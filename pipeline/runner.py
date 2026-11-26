"""Reads a video file, runs the pipeline over it and writes everything a reviewer needs.

Output folder (one per clip):
  annotated.mp4        the video with boxes, track ids and PPE answers drawn on
  incidents.jsonl      one line per incident opened/closed
  evidence/*.jpg       the annotated frame at each incident opening and closing
  per_frame.csv        one row per processed frame (people, events, open incidents, timings)
  summary.json         the run in numbers
  report.md            the run in words

Nothing here changes the source video.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from time import perf_counter
from typing import Callable

import cv2

from incidents.temporal import Transition
from pipeline.render import draw_result
from pipeline.report import TrackLog, count_events, to_markdown
from pipeline.video_pipeline import FrameResult, VideoPipeline


class VideoError(RuntimeError):
    """Raised when the video cannot be read."""


def transition_to_dict(t: Transition) -> dict:
    return {
        "kind": t.kind, "camera_id": t.camera_id, "rule_id": t.rule_id, "track_id": t.track_id,
        "episode": t.episode, "frame_index": t.frame_index, "timestamp": round(t.timestamp, 3),
        "started_frame": t.started_frame, "started_timestamp": round(t.started_timestamp, 3),
        "severity": t.severity, "violating_frames": t.violating_frames,
        "uncertain_frames": t.uncertain_frames, "reason": t.reason, "details": t.event.details,
    }


def process_video(
    video: Path,
    pipeline: VideoPipeline,
    out_dir: Path,
    *,
    frame_stride: int = 1,
    max_seconds: float | None = None,
    write_video: bool = True,
    run_info: dict | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> dict:
    if frame_stride < 1:
        raise ValueError("frame_stride must be at least 1")
    if max_seconds is not None and max_seconds <= 0:
        raise ValueError("max_seconds must be positive")
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise VideoError(f"cannot open video: {video}")
    writer = None
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        if not 0 < fps <= 240:
            raise VideoError(f"unusable frame rate ({fps}) in {video}")
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        limit = int(max_seconds * fps) if max_seconds else None

        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "evidence").mkdir(exist_ok=True)
        if write_video:
            writer = cv2.VideoWriter(str(out_dir / "annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                     fps / frame_stride, (width, height))
            if not writer.isOpened():
                raise VideoError("could not create annotated.mp4")

        track_log = TrackLog()
        all_events, incident_rows = [], []
        opened: dict[tuple, dict] = {}
        sums = {"detect": 0.0, "track": 0.0, "rules": 0.0}
        processed = 0
        last_index, last_ts = 0, 0.0
        started = perf_counter()

        with (out_dir / "incidents.jsonl").open("w", encoding="utf-8") as jl, \
                (out_dir / "per_frame.csv").open("w", newline="", encoding="utf-8") as pf:
            csv_writer = csv.writer(pf)
            csv_writer.writerow(["frame", "time_s", "persons", "tracked", "violation_events", "uncertain_events",
                                 "open_incidents", "detect_ms", "track_ms", "rules_ms"])

            last_annotated = None

            def record(t: Transition, annotated) -> None:
                row = transition_to_dict(t)
                jl.write(json.dumps(row, default=str) + "\n")
                key = (t.rule_id, t.track_id, t.episode)
                if t.kind == "opened":
                    opened[key] = {"rule_id": t.rule_id, "track_id": t.track_id, "episode": t.episode,
                                   "started_frame": t.started_frame, "started_timestamp": t.started_timestamp,
                                   "severity": t.severity}
                else:
                    rec = opened.pop(key, {"rule_id": t.rule_id, "track_id": t.track_id, "episode": t.episode,
                                           "started_frame": t.started_frame,
                                           "started_timestamp": t.started_timestamp, "severity": t.severity})
                    rec.update(ended_timestamp=t.timestamp, ended_frame=t.frame_index, reason=t.reason,
                               violating_frames=t.violating_frames, uncertain_frames=t.uncertain_frames)
                    incident_rows.append(rec)
                if annotated is not None:
                    name = f"{t.rule_id}_track{t.track_id}_ep{t.episode}_{t.kind}_f{t.frame_index}.jpg"
                    cv2.imwrite(str(out_dir / "evidence" / name), annotated)

            index = -1
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                index += 1
                if limit is not None and index >= limit:
                    break
                if index % frame_stride:
                    continue
                ts = index / fps
                result = pipeline.process_frame(frame, index, ts)
                processed += 1
                last_index, last_ts = index, ts
                all_events.extend(result.events)
                track_log.add(result)
                for k in sums:
                    sums[k] += result.timings_ms[k]
                annotated = draw_result(frame, result, camera_id=pipeline.camera_id)
                if writer:
                    writer.write(annotated)
                last_annotated = annotated
                for t in result.transitions:
                    record(t, annotated)
                csv_writer.writerow([
                    index, f"{ts:.3f}", len(result.statuses), len(result.track_ids),
                    sum(e.outcome == "violation" for e in result.events),
                    sum(e.outcome == "uncertain" for e in result.events), len(result.open_incidents),
                    f"{result.timings_ms['detect']:.1f}", f"{result.timings_ms['track']:.1f}",
                    f"{result.timings_ms['rules']:.1f}",
                ])
                if progress and processed % 25 == 0:
                    progress(processed, total)

            for t in pipeline.finish(last_index, last_ts):  # close anything still open
                record(t, last_annotated)
        elapsed = perf_counter() - started
    finally:
        cap.release()
        if writer:
            writer.release()

    n = max(processed, 1)
    summary = {
        "video": video.name, "source_frames": total, "fps": fps, "frame_size": [width, height],
        "frame_stride": frame_stride, "frames_processed": processed,
        "wall_seconds": round(elapsed, 2),
        "timing_ms": {k: v / n for k, v in sums.items()},
        "incidents": sorted(incident_rows, key=lambda r: (r["started_frame"], r["rule_id"], r["track_id"])),
        "ignored_untracked_events": pipeline.incidents.ignored_untracked_events,
        "tracks": track_log.rows(), "untracked_person_frames": track_log.untracked_person_frames,
        "events_per_rule": count_events(all_events),
        **(run_info or {}),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    (out_dir / "report.md").write_text(to_markdown(summary), encoding="utf-8")
    return summary
