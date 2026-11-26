"""What a run produces besides pictures: per-person timelines and a readable report.

``TrackLog`` answers "for each tracked person, what did we conclude and for how long?", which is
the first thing to read when an incident looks wrong. The report states what the run was and was
not (one clip, placeholder settings, this machine's speed) so nobody reads more into it than it holds.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

from pipeline.video_pipeline import FrameResult


@dataclass
class _Track:
    first_frame: int
    first_time: float
    last_frame: int = 0
    last_time: float = 0.0
    frames: int = 0
    operator_frames: int = 0
    helmet: Counter = field(default_factory=Counter)
    vest: Counter = field(default_factory=Counter)


class TrackLog:
    def __init__(self) -> None:
        self._tracks: dict[int, _Track] = {}
        self.untracked_person_frames = 0  # person detections that had no confirmed id yet

    def add(self, result: FrameResult) -> None:
        for s in result.statuses:
            track = result.track_ids.get(s.person_index)
            if track is None:
                self.untracked_person_frames += 1
                continue
            t = self._tracks.setdefault(track, _Track(result.frame_index, result.timestamp))
            t.last_frame, t.last_time = result.frame_index, result.timestamp
            t.frames += 1
            t.operator_frames += int(s.is_operator)
            t.helmet[s.helmet] += 1
            t.vest[s.vest] += 1

    def rows(self) -> list[dict[str, Any]]:
        rows = []
        for track_id, t in sorted(self._tracks.items()):
            rows.append({
                "track_id": track_id, "first_frame": t.first_frame, "last_frame": t.last_frame,
                "first_time": round(t.first_time, 2), "last_time": round(t.last_time, 2),
                "frames": t.frames, "operator_frames": t.operator_frames,
                "helmet": {k: t.helmet.get(k, 0) for k in ("yes", "no", "unknown")},
                "vest": {k: t.vest.get(k, 0) for k in ("yes", "no", "unknown")},
            })
        return rows


def to_markdown(summary: dict[str, Any]) -> str:
    s = {**{k: "n/a" for k in ("weights", "imgsz", "conf", "safety_config", "tracker_config", "temporal_config", "device")}, **summary}
    lines = [
        f"# Video run: {s['video']}",
        "",
        "## What this is",
        f"- One clip, processed with the model `{s['weights']}` at inference size {s['imgsz']}, "
        f"detector confidence {s['conf']}.",
        f"- PPE is checked for: {s.get('work_area', 'whole frame')}.",
        f"- Safety settings: {s['safety_config']}. Tracker: {s['tracker_config']}. Incident timing: {s['temporal_config']}.",
        "- Safety and incident settings are placeholders or were chosen on image validation data, "
        "not tuned on video. A single clip is an illustration, not an accuracy measurement.",
        "",
        "## Run",
        f"- Source frames: {s['source_frames']} at {s['fps']:.2f} fps; processed every {s['frame_stride']}th "
        f"frame = {s['frames_processed']} frames.",
        f"- Mean time per processed frame on this machine ({s['device']}): detect {s['timing_ms']['detect']:.0f} ms, "
        f"track {s['timing_ms']['track']:.1f} ms, rules and incidents {s['timing_ms']['rules']:.1f} ms. "
        "This is a rough single-run figure, not a benchmark.",
        "",
        "## Incidents",
    ]
    incidents = s["incidents"]
    if not incidents:
        lines.append("- none opened")
    for i in incidents:
        lines.append(
            f"- {i['rule_id']} on track {i['track_id']} (episode {i['episode']}): "
            f"{i['started_timestamp']:.2f}s -> {i.get('last_violation_timestamp', i['ended_timestamp']):.2f}s "
            f"(last violation; closed at {i['ended_timestamp']:.2f}s by {i['reason']}), "
            f"{i['violating_frames']} violating and {i['uncertain_frames']} uncertain frames"
        )
    lines += [
        f"- events ignored because the person had no track id yet: {s['ignored_untracked_events']}",
        "",
        "## Per-person timeline",
        "",
        "| track | first-last (s) | frames | helmet yes/no/unknown | vest yes/no/unknown | operator frames |",
        "|---|---|---|---|---|---|",
    ]
    for r in s["tracks"]:
        h, v = r["helmet"], r["vest"]
        lines.append(
            f"| {r['track_id']} | {r['first_time']:.2f}-{r['last_time']:.2f} | {r['frames']} | "
            f"{h['yes']}/{h['no']}/{h['unknown']} | {v['yes']}/{v['no']}/{v['unknown']} | {r['operator_frames']} |"
        )
    lines += ["", f"- person frames outside the work area (not checked): {s.get('outside_person_frames', 0)}",
              f"- person detections without a track id: {s['untracked_person_frames']}",
              "", "## Events per rule (frames)", ""]
    for rule, counts in sorted(s["events_per_rule"].items()):
        lines.append(f"- {rule}: {counts.get('violation', 0)} violation, {counts.get('uncertain', 0)} uncertain")
    if not s["events_per_rule"]:
        lines.append("- none")
    return "\n".join(lines) + "\n"


def count_events(results_events: list) -> dict[str, dict[str, int]]:
    counts: dict[str, Counter] = defaultdict(Counter)
    for e in results_events:
        counts[e.rule_id][e.outcome] += 1
    return {k: dict(v) for k, v in counts.items()}
