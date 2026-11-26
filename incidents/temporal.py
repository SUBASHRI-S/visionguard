"""Temporal logic: when does a stream of per-frame events become ONE incident?

A single frame is noisy: a helmet can vanish from detection for a frame, a person can flicker in
and out. So an incident only OPENS after N consecutive violating frames on one tracked person, and
only CLOSES after M consecutive clear frames, or when the track is lost for too long.

For every (camera, rule, track) the state is:

    idle --N violating frames--> open --M clear frames--> closed   (reason: cleared)
                                  open --track gone for > T seconds--> closed (reason: track_lost)
                                  open --stream ends--> closed (reason: stream_end)

Frame kinds for a key:   violating | uncertain | clear (person visible, no event) | absent (person gone)
* 'uncertain' frames HOLD the state: they never count towards opening and never towards closing.
  While idle, a long run of them (> uncertain_expire_frames) cancels a half-built streak.
* 'absent' frames also hold, until the track has been gone longer than track_lost_seconds.

Counts are in PROCESSED frames. If you skip frames before analysis, scale the settings to match.
Events without a track id cannot be followed over time and are ignored (and counted).
"""
import math
from dataclasses import dataclass, fields
from typing import Any, Literal

from safety.events import Event, Severity


class IncidentConfigError(ValueError):
    """Raised when temporal settings are invalid. ``errors`` lists every problem."""

    def __init__(self, errors: list[str]) -> None:
        super().__init__("; ".join(errors))
        self.errors = errors


@dataclass(frozen=True)
class TemporalConfig:
    """PLACEHOLDER values: tune on real clips (the 'tune' subset) and record the choice."""

    open_after_frames: int = 8
    close_after_frames: int = 30
    track_lost_seconds: float = 3.0
    uncertain_expire_frames: int = 60

    def __post_init__(self) -> None:
        errors: list[str] = []
        for name, low, high, integer in (
            ("open_after_frames", 1, 1000, True),
            ("close_after_frames", 1, 100000, True),
            ("uncertain_expire_frames", 1, 100000, True),
            ("track_lost_seconds", 0.0, 3600.0, False),
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                errors.append(f"{name} must be a number")
            elif integer and not isinstance(value, int):
                errors.append(f"{name} must be a whole number")
            elif not low <= value <= high:
                errors.append(f"{name} must be between {low} and {high}")
        if errors:
            raise IncidentConfigError(errors)

    @classmethod
    def from_seconds(
        cls,
        *,
        fps: float,
        frame_stride: int = 1,
        open_after_seconds: float = 3.0,
        close_after_seconds: float = 2.0,
        uncertain_expire_seconds: float = 2.5,
        track_lost_seconds: float = 3.0,
    ) -> "TemporalConfig":
        """Express the timing in seconds of video, so it means the same at 25 fps, 50 fps or with
        frames skipped. The tracker works in *processed* frames, so seconds are converted using
        the rate at which frames are actually processed (fps / frame_stride)."""
        if not 0 < fps <= 240:
            raise IncidentConfigError([f"fps must be between 0 and 240, got {fps}"])
        if isinstance(frame_stride, bool) or not isinstance(frame_stride, int) or frame_stride < 1:
            raise IncidentConfigError(["frame_stride must be a whole number >= 1"])
        rate = fps / frame_stride

        def frames(seconds: float, name: str) -> int:
            if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or seconds <= 0:
                raise IncidentConfigError([f"{name} must be a positive number of seconds"])
            return max(1, math.ceil(seconds * rate - 1e-9))

        return cls(
            open_after_frames=frames(open_after_seconds, "open_after_seconds"),
            close_after_frames=frames(close_after_seconds, "close_after_seconds"),
            uncertain_expire_frames=frames(uncertain_expire_seconds, "uncertain_expire_seconds"),
            track_lost_seconds=track_lost_seconds,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "TemporalConfig":
        data = data or {}
        if not isinstance(data, dict):
            raise IncidentConfigError(["temporal config must be a mapping"])
        extra = sorted(set(data) - {f.name for f in fields(cls)})
        if extra:
            raise IncidentConfigError([f"unknown temporal key(s) {extra}"])
        return cls(**data)


@dataclass(frozen=True)
class Transition:
    """An incident opened or closed. ``episode`` numbers repeated incidents for the same key."""

    kind: Literal["opened", "closed"]
    camera_id: str
    rule_id: str
    track_id: int
    episode: int
    frame_index: int
    timestamp: float
    started_frame: int  # first frame of the violating streak (not the frame that crossed N)
    started_timestamp: float
    severity: Severity
    event: Event  # opened: the event that opened it; closed: the last violating event
    violating_frames: int
    uncertain_frames: int
    reason: str | None = None  # closed only: cleared | track_lost | stream_end


@dataclass
class _State:
    status: Literal["idle", "open"] = "idle"
    violation_streak: int = 0
    streak_start_frame: int = 0
    streak_start_ts: float = 0.0
    uncertain_streak: int = 0
    clear_streak: int = 0
    last_seen_ts: float = 0.0
    last_violation: Event | None = None
    violating_frames: int = 0
    uncertain_frames: int = 0
    episode: int = 0


Key = tuple[str, str, int]


class IncidentTracker:
    def __init__(self, config: TemporalConfig | None = None) -> None:
        self.config = config or TemporalConfig()
        self._states: dict[Key, _State] = {}
        self._episodes: dict[Key, int] = {}
        self.ignored_untracked_events = 0

    # ---- public API ----
    def update(
        self,
        *,
        camera_id: str,
        frame_index: int,
        timestamp: float,
        events: list[Event],
        present_track_ids: set[int],
    ) -> list[Transition]:
        """Feed one processed frame. ``present_track_ids`` = every tracked person visible in it,
        including people with no events (those are the 'clear' frames)."""
        chosen: dict[Key, Event] = {}
        for event in events:
            if event.camera_id != camera_id:
                raise ValueError(f"event for camera {event.camera_id!r} given to frame of {camera_id!r}")
            if event.track_id is None:
                self.ignored_untracked_events += 1
                continue
            key = (camera_id, event.rule_id, event.track_id)
            previous = chosen.get(key)
            if previous is None or (previous.outcome == "uncertain" and event.outcome == "violation"):
                chosen[key] = event

        out: list[Transition] = []
        for key in [k for k in self._states if k[0] == camera_id and k not in chosen]:
            state = self._states[key]
            if key[2] in present_track_ids:
                state.last_seen_ts = timestamp
                out += self._on_clear(key, state, frame_index, timestamp)
            else:
                out += self._on_absent(key, state, frame_index, timestamp)
        for key, event in chosen.items():
            state = self._states.setdefault(key, _State())
            state.last_seen_ts = timestamp
            if event.outcome == "violation":
                out += self._on_violation(key, state, event, frame_index, timestamp)
            else:
                self._on_uncertain(key, state)
        return out

    def finish(self, frame_index: int, timestamp: float) -> list[Transition]:
        """End of stream: close every open incident and drop half-built streaks."""
        out = [
            self._close(key, state, frame_index, timestamp, "stream_end")
            for key, state in list(self._states.items())
            if state.status == "open"
        ]
        self._states.clear()
        return out

    def is_open(self, camera_id: str, rule_id: str, track_id: int) -> bool:
        state = self._states.get((camera_id, rule_id, track_id))
        return state is not None and state.status == "open"

    @property
    def open_keys(self) -> list[Key]:
        return [k for k, s in self._states.items() if s.status == "open"]

    # ---- frame handlers ----
    def _on_violation(self, key: Key, s: _State, event: Event, frame: int, ts: float) -> list[Transition]:
        s.clear_streak = 0
        s.uncertain_streak = 0
        s.last_violation = event
        if s.status == "open":
            s.violating_frames += 1
            return []
        if s.violation_streak == 0:
            s.streak_start_frame, s.streak_start_ts = frame, ts
        s.violation_streak += 1
        if s.violation_streak < self.config.open_after_frames:
            return []
        s.status = "open"
        s.violating_frames = s.violation_streak
        s.uncertain_frames = 0
        self._episodes[key] = self._episodes.get(key, 0) + 1
        s.episode = self._episodes[key]
        return [self._transition("opened", key, s, event, frame, ts, None)]

    def _on_uncertain(self, key: Key, s: _State) -> None:
        s.uncertain_streak += 1
        if s.status == "open":
            s.uncertain_frames += 1
        elif s.uncertain_streak > self.config.uncertain_expire_frames:
            del self._states[key]  # a long run of 'cannot tell' cancels a half-built streak

    def _on_clear(self, key: Key, s: _State, frame: int, ts: float) -> list[Transition]:
        s.uncertain_streak = 0
        if s.status == "idle":
            del self._states[key]
            return []
        s.clear_streak += 1
        if s.clear_streak >= self.config.close_after_frames:
            return [self._close(key, s, frame, ts, "cleared")]
        return []

    def _on_absent(self, key: Key, s: _State, frame: int, ts: float) -> list[Transition]:
        if ts - s.last_seen_ts <= self.config.track_lost_seconds:
            return []
        if s.status == "open":
            return [self._close(key, s, frame, ts, "track_lost")]
        del self._states[key]
        return []

    # ---- helpers ----
    def _close(self, key: Key, s: _State, frame: int, ts: float, reason: str) -> Transition:
        assert s.last_violation is not None
        transition = self._transition("closed", key, s, s.last_violation, frame, ts, reason)
        self._states.pop(key, None)
        return transition

    @staticmethod
    def _transition(
        kind: Literal["opened", "closed"], key: Key, s: _State, event: Event,
        frame: int, ts: float, reason: str | None,
    ) -> Transition:
        return Transition(
            kind=kind, camera_id=key[0], rule_id=key[1], track_id=key[2], episode=s.episode,
            frame_index=frame, timestamp=ts, started_frame=s.streak_start_frame,
            started_timestamp=s.streak_start_ts, severity=event.severity, event=event,
            violating_frames=s.violating_frames, uncertain_frames=s.uncertain_frames, reason=reason,
        )
