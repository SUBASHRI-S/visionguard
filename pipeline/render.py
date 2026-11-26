"""Drawing on frames. Never changes the input frame; always returns a copy."""
from __future__ import annotations

import cv2
import numpy as np

from pipeline.video_pipeline import FrameResult

GREY = (150, 150, 150)
GREEN, YELLOW, RED, BLUE, MAGENTA = (0, 160, 0), (0, 200, 230), (0, 0, 220), (255, 120, 0), (200, 0, 200)


def draw_result(frame: np.ndarray, result: FrameResult, *, camera_id: str) -> np.ndarray:
    out = frame.copy()
    for d in result.detections:
        if d.class_name in ("helmet", "vest"):
            b = d.box
            colour = BLUE if d.class_name == "helmet" else MAGENTA
            cv2.rectangle(out, (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2)), colour, 1)
        elif d.class_name == "forklift":
            b = d.box
            cv2.rectangle(out, (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2)), (0, 140, 255), 2)
    for person, track in result.outside_persons:
        b = person.box
        cv2.rectangle(out, (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2)), GREY, 1)
        cv2.putText(out, f"#{track if track is not None else '?'} outside area", (int(b.x1), max(14, int(b.y1) - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, GREY, 1)
    open_tracks = {track for (cam, _rule, track) in result.open_incidents if cam == camera_id}
    for s in result.statuses:
        b = s.person.box
        track = result.track_ids.get(s.person_index)
        if track is not None and track in open_tracks:
            colour, thickness = RED, 4  # an incident is open for this person
        elif "no" in (s.helmet, s.vest):
            colour, thickness = RED, 1  # violating this frame, no incident yet
        elif "unknown" in (s.helmet, s.vest):
            colour, thickness = YELLOW, 2
        else:
            colour, thickness = GREEN, 2
        cv2.rectangle(out, (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2)), colour, thickness)
        label = f"#{track if track is not None else '?'} H:{s.helmet} V:{s.vest}" + (" op" if s.is_operator else "")
        cv2.putText(out, label, (int(b.x1), max(14, int(b.y1) - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2)
    banner = f"t={result.timestamp:6.2f}s  frame {result.frame_index}  open incidents: {len(open_tracks)}"
    cv2.rectangle(out, (0, 0), (out.shape[1], 20), (0, 0, 0), -1)
    cv2.putText(out, banner, (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return out
