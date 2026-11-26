"""The per-frame core. No video files, no drawing, no disk: frame in, result out.

Keeping this free of I/O is what lets the whole chain be tested with a scripted detector, and lets
the same class later serve a live stream or the API.

Order of work for each processed frame:
  detect -> track persons -> associate PPE and run rules -> update incidents.

Details that are easy to get wrong:
* The tracker receives the person detections in the order they appear, and the safety engine
  numbers persons in that same order, so ``track_ids[person_index]`` lines up. A test locks this.
* The tracker is called on every processed frame, including frames with nobody in them, because
  that is how it ages out lost tracks.
* ``timestamp`` is video time in seconds (frame_index / fps), not the wall clock, so a replay gives
  the same answer however fast the computer is.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from time import perf_counter
from typing import Sequence

import numpy as np

from incidents.memory import PPEMemory
from incidents.temporal import IncidentTracker, Transition
from pipeline.detector import FrameDetector
from safety.contracts import PERSON, Detection, PPEStatus
from safety.engine import SafetyEngine
from safety.events import Event
from safety.zones import Zone, point_in_polygon
from trackers import PersonTracker


@dataclass(frozen=True)
class FrameResult:
    frame_index: int
    timestamp: float
    detections: tuple[Detection, ...]
    statuses: tuple[PPEStatus, ...]
    track_ids: dict[int, int]  # person_index -> track id (confirmed tracks only); indexes into `statuses`
    events: tuple[Event, ...]
    transitions: tuple[Transition, ...]
    open_incidents: tuple[tuple[str, str, int], ...]  # (camera_id, rule_id, track_id) open after this frame
    outside_persons: tuple[tuple[Detection, int | None], ...] = ()  # people outside the work area: (box, track id)
    timings_ms: dict[str, float] = field(default_factory=dict)


class VideoPipeline:
    def __init__(
        self,
        detector: FrameDetector,
        engine: SafetyEngine,
        tracker: PersonTracker,
        incidents: IncidentTracker,
        *,
        camera_id: str,
        zones: Sequence[Zone] = (),
        work_area: Zone | None = None,
        memory: PPEMemory | None = None,
    ) -> None:
        if not camera_id:
            raise ValueError("camera_id must not be empty")
        self.detector, self.engine, self.tracker, self.incidents = detector, engine, tracker, incidents
        self.camera_id = camera_id
        self.zones = tuple(zones)
        # PPE is only required inside the work area. People outside it (passers-by, a road behind
        # the site) are still tracked and drawn, but the safety rules never see them.
        self.work_area = work_area
        self.memory = memory  # optional: downgrade 'no' to 'unknown' for items seen a moment ago

    def process_frame(self, frame: np.ndarray, frame_index: int, timestamp: float) -> FrameResult:
        if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3 or frame.size == 0:
            raise ValueError("frame must be a non-empty height x width x 3 array")
        height, width = frame.shape[:2]

        t0 = perf_counter()
        detections = self.detector.detect(frame)
        t1 = perf_counter()
        persons = [d for d in detections if d.class_name == PERSON]
        all_track_ids = self.tracker.update(persons)
        t2 = perf_counter()
        outside: list[tuple[Detection, int | None]] = []
        if self.work_area is None:
            engine_detections, track_ids = detections, all_track_ids
        else:
            kept = [i for i, p in enumerate(persons) if self._in_work_area(p, width, height)]
            outside = [(p, all_track_ids.get(i)) for i, p in enumerate(persons) if i not in set(kept)]
            # the engine numbers people in the order they appear, so renumber the track ids to match
            track_ids = {new: all_track_ids[old] for new, old in enumerate(kept) if old in all_track_ids}
            engine_detections = [d for d in detections if d.class_name != PERSON] + [persons[i] for i in kept]
        ctx = self.engine.build_context(
            engine_detections, camera_id=self.camera_id, frame_index=frame_index, timestamp=timestamp,
            frame_width=width, frame_height=height, zones=self.zones, track_ids=track_ids,
        )
        if self.memory is not None:
            ctx = replace(ctx, statuses=tuple(self.memory.apply(ctx.statuses, ctx.track_ids, timestamp)))
        events = self.engine.evaluate(ctx)
        transitions = self.incidents.update(
            camera_id=self.camera_id, frame_index=frame_index, timestamp=timestamp,
            events=events, present_track_ids=set(track_ids.values()),
        )
        t3 = perf_counter()
        return FrameResult(
            frame_index=frame_index, timestamp=timestamp,
            detections=tuple(detections), statuses=ctx.statuses, track_ids=dict(track_ids),
            events=tuple(events), transitions=tuple(transitions),
            open_incidents=tuple(sorted(self.incidents.open_keys)), outside_persons=tuple(outside),
            timings_ms={"detect": (t1 - t0) * 1e3, "track": (t2 - t1) * 1e3, "rules": (t3 - t2) * 1e3},
        )

    def _in_work_area(self, person: Detection, width: int, height: int) -> bool:
        x, y = person.box.foot_point
        return point_in_polygon(x / width, y / height, self.work_area.polygon)  # type: ignore[union-attr]

    def finish(self, frame_index: int, timestamp: float) -> list[Transition]:
        """End of stream: close everything still open."""
        return self.incidents.finish(frame_index, timestamp)
