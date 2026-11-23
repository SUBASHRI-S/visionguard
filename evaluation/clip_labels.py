"""Schema and validation for hand-labelled real-footage clips (replay evaluation).

Each clip lists the time intervals where a safety incident is truly happening.
The manifest also records where the clip came from and whether we may use it,
because the training data contains generated images and real footage is our
honest check (see docs/REAL_FOOTAGE_PROTOCOL.md).
"""
import json
from dataclasses import dataclass, field
from pathlib import Path

ALLOWED_RULES = {"no_helmet", "no_vest", "zone_intrusion", "proximity"}
SOURCE_TYPES = {"own_recording", "third_party"}
SUBSETS = {"tune", "final"}


class ClipLabelError(ValueError):
    """Raised when a clip manifest is invalid. ``errors`` lists every problem found."""

    def __init__(self, errors: list[str]) -> None:
        super().__init__("; ".join(errors))
        self.errors = errors


@dataclass(frozen=True)
class IncidentLabel:
    rule_id: str
    start_s: float
    end_s: float
    ambiguous: bool = False
    note: str = ""


@dataclass(frozen=True)
class ClipLabel:
    clip_id: str
    path: str
    camera_id: str
    fps: float
    subset: str
    source_type: str
    licence: str
    incidents: list[IncidentLabel] = field(default_factory=list)
    duration_s: float | None = None


def validate_manifest(data: dict) -> list[str]:
    """Return a list of human-readable problems (empty list means valid)."""
    errors: list[str] = []
    clips = data.get("clips")
    if not isinstance(clips, list) or not clips:
        return ["manifest needs a non-empty 'clips' list"]

    seen_ids: set[str] = set()
    for index, clip in enumerate(clips):
        label = clip.get("clip_id", f"#{index}") if isinstance(clip, dict) else f"#{index}"
        where = f"clip {label}"
        if not isinstance(clip, dict):
            errors.append(f"{where}: must be an object")
            continue

        for key in ("clip_id", "path", "camera_id", "source_type", "licence", "subset"):
            if not isinstance(clip.get(key), str) or not clip[key].strip():
                errors.append(f"{where}: missing or empty '{key}'")
        if clip.get("clip_id") in seen_ids:
            errors.append(f"{where}: duplicate clip_id")
        seen_ids.add(clip.get("clip_id"))

        if clip.get("subset") not in SUBSETS:
            errors.append(f"{where}: subset must be one of {sorted(SUBSETS)}")
        fps = clip.get("fps")
        if not isinstance(fps, (int, float)) or fps <= 0:
            errors.append(f"{where}: fps must be a positive number")

        source_type = clip.get("source_type")
        if source_type not in SOURCE_TYPES:
            errors.append(f"{where}: source_type must be one of {sorted(SOURCE_TYPES)}")
        elif source_type == "own_recording" and clip.get("consent_obtained") is not True:
            errors.append(f"{where}: own recordings need consent_obtained = true")
        elif source_type == "third_party" and not str(clip.get("url", "")).strip():
            errors.append(f"{where}: third-party clips need the source 'url'")

        duration = clip.get("duration_s")
        if duration is not None and (not isinstance(duration, (int, float)) or duration <= 0):
            errors.append(f"{where}: duration_s must be a positive number")

        incidents = clip.get("incidents")
        if not isinstance(incidents, list):
            errors.append(f"{where}: 'incidents' must be a list (use [] for a compliant clip)")
            continue
        spans: dict[str, list[tuple[float, float]]] = {}
        for i, incident in enumerate(incidents):
            iw = f"{where} incident #{i}"
            if not isinstance(incident, dict):
                errors.append(f"{iw}: must be an object")
                continue
            rule = incident.get("rule_id")
            start, end = incident.get("start_s"), incident.get("end_s")
            if rule not in ALLOWED_RULES:
                errors.append(f"{iw}: rule_id must be one of {sorted(ALLOWED_RULES)}")
            if not all(isinstance(v, (int, float)) for v in (start, end)):
                errors.append(f"{iw}: start_s and end_s must be numbers")
                continue
            if start < 0 or end <= start:
                errors.append(f"{iw}: need 0 <= start_s < end_s")
                continue
            if isinstance(duration, (int, float)) and end > duration:
                errors.append(f"{iw}: end_s {end} is beyond duration_s {duration}")
            for other_start, other_end in spans.get(rule, []):
                if start < other_end and other_start < end:
                    errors.append(f"{iw}: overlaps another '{rule}' interval in the same clip")
            spans.setdefault(rule, []).append((start, end))
    return errors


def parse_manifest(data: dict) -> list[ClipLabel]:
    """Validate and convert a manifest dict. Raises ClipLabelError listing all problems."""
    errors = validate_manifest(data)
    if errors:
        raise ClipLabelError(errors)
    clips: list[ClipLabel] = []
    for clip in data["clips"]:
        incidents = [
            IncidentLabel(
                rule_id=i["rule_id"],
                start_s=float(i["start_s"]),
                end_s=float(i["end_s"]),
                ambiguous=bool(i.get("ambiguous", False)),
                note=str(i.get("note", "")),
            )
            for i in clip["incidents"]
        ]
        clips.append(
            ClipLabel(
                clip_id=clip["clip_id"],
                path=clip["path"],
                camera_id=clip["camera_id"],
                fps=float(clip["fps"]),
                subset=clip["subset"],
                source_type=clip["source_type"],
                licence=clip["licence"],
                incidents=incidents,
                duration_s=float(clip["duration_s"]) if clip.get("duration_s") else None,
            )
        )
    return clips


def load_manifest(path: Path) -> list[ClipLabel]:
    """Read a JSON manifest file and return validated clip labels."""
    try:
        data = json.loads(Path(path).read_text())
    except json.JSONDecodeError as exc:
        raise ClipLabelError([f"invalid JSON: {exc}"]) from exc
    return parse_manifest(data)
