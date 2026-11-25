"""Adapter around Ultralytics' ByteTrack.

Why an adapter: the rest of VisionGuard speaks in our own types (``Detection``) and in
"person index -> track id" dictionaries. ByteTrack speaks in numpy arrays. This one file is the
only place that knows about the library, so a version change or a different tracker (BoT-SORT,
a re-identification model) touches one file, and everything else stays tested and untouched.

Behaviour worth knowing:
* Only person detections are tracked (helmets/vests/forklifts are not given IDs).
* A person gets ``None`` instead of an ID when ByteTrack has not confirmed them yet. A new
  track on a later frame needs a second matching frame before it is reported (the very first
  frame of a stream is the exception). Incident logic ignores events without an ID, so this
  delays a brand-new person by a frame instead of risking an ID that flips.
* IDs are per tracker instance and are never reused inside one stream. Make one tracker per
  camera/stream.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np

from safety.contracts import Detection


class TrackerConfigError(ValueError):
    """Raised when tracker settings are invalid."""


@dataclass(frozen=True)
class TrackerConfig:
    """Mirrors ultralytics/cfg/trackers/bytetrack.yaml (defaults copied from 8.4.216).

    track_high_thresh: detections at or above this are matched first.
    track_low_thresh:  detections between low and high get a second, rescue pass.
    new_track_thresh:  a detection must score at least this to start a new track.
    track_buffer:      processed frames a lost track is kept alive (longer = survives occlusion,
                       but raises the risk of swapping IDs).
    match_thresh:      matching threshold on (1 - IoU) cost; higher accepts looser matches.
    fuse_score:        blend detector confidence into the matching cost.
    """

    track_high_thresh: float = 0.25
    track_low_thresh: float = 0.10
    new_track_thresh: float = 0.25
    track_buffer: int = 30
    match_thresh: float = 0.80
    fuse_score: bool = True

    def __post_init__(self) -> None:
        for name in ("track_high_thresh", "track_low_thresh", "new_track_thresh", "match_thresh"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.0 <= value <= 1.0:
                raise TrackerConfigError(f"{name} must be a number between 0 and 1, got {value!r}")
        if self.track_low_thresh > self.track_high_thresh:
            raise TrackerConfigError("track_low_thresh must not be above track_high_thresh")
        if self.new_track_thresh < self.track_high_thresh:
            # ByteTrack only ever considers high-confidence detections for new tracks; a lower
            # value would silently behave like track_high_thresh, so reject the confusion.
            raise TrackerConfigError("new_track_thresh must not be below track_high_thresh")
        if isinstance(self.track_buffer, bool) or not isinstance(self.track_buffer, int) or self.track_buffer < 1:
            raise TrackerConfigError(f"track_buffer must be a whole number >= 1, got {self.track_buffer!r}")
        if not isinstance(self.fuse_score, bool):
            raise TrackerConfigError(f"fuse_score must be true or false, got {self.fuse_score!r}")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "TrackerConfig":
        data = dict(data or {})
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise TrackerConfigError(f"unknown tracker setting(s): {', '.join(unknown)}")
        return cls(**data)


class _Boxes:
    """The small numpy object ByteTrack reads: ``conf``, ``cls``, ``xywh`` + boolean indexing."""

    def __init__(self, conf: np.ndarray, cls: np.ndarray, xywh: np.ndarray) -> None:
        self.conf, self.cls, self.xywh = conf, cls, xywh

    def __len__(self) -> int:
        return len(self.conf)

    def __getitem__(self, index: Any) -> "_Boxes":
        return _Boxes(self.conf[index], self.cls[index], self.xywh[index])


class PersonTracker:
    """One instance per video stream. Call ``update`` once per processed frame."""

    def __init__(self, config: TrackerConfig | None = None) -> None:
        # Imported here so that importing this module (and our config validation) does not
        # require the optional tracking dependencies; a missing `lap` shows up with a clear message.
        try:
            from ultralytics.trackers.byte_tracker import BYTETracker
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise RuntimeError(
                "ByteTrack needs ultralytics and lap: python -m pip install -r requirements-tracking.txt"
            ) from exc
        self.config = config or TrackerConfig()
        self._cls = BYTETracker
        self._tracker = self._make()
        self.frames_seen = 0

    def _make(self) -> Any:
        return self._cls(SimpleNamespace(**{f.name: getattr(self.config, f.name) for f in fields(self.config)}))

    def reset(self) -> None:
        """Forget all tracks (use between unrelated videos)."""
        self._tracker = self._make()
        self.frames_seen = 0

    def update(self, persons: Sequence[Detection]) -> dict[int, int]:
        """Feed this frame's person detections; return ``{person_index: track_id}``.

        ``person_index`` is the position in ``persons``, the same order the safety engine uses
        for its per-person statuses. Persons ByteTrack has not confirmed are left out. Call this
        every processed frame, including frames with no people, because empty frames are how
        the tracker ages out lost tracks.
        """
        self.frames_seen += 1
        count = len(persons)
        if count:
            xyxy = np.array([[p.box.x1, p.box.y1, p.box.x2, p.box.y2] for p in persons], dtype=np.float32)
            xywh = np.stack(
                [(xyxy[:, 0] + xyxy[:, 2]) / 2, (xyxy[:, 1] + xyxy[:, 3]) / 2,
                 xyxy[:, 2] - xyxy[:, 0], xyxy[:, 3] - xyxy[:, 1]], axis=1,
            )
            conf = np.array([p.confidence for p in persons], dtype=np.float32)
        else:
            xywh, conf = np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)
        boxes = _Boxes(conf, np.zeros(count, dtype=np.float32), xywh)

        rows = self._tracker.update(boxes)
        result: dict[int, int] = {}
        for row in np.asarray(rows).reshape(-1, 8) if len(rows) else []:
            track_id, detection_index = int(row[4]), int(row[7])
            if 0 <= detection_index < count:
                result[detection_index] = track_id
        return result
