"""Turns a video frame into our own ``Detection`` objects.

``FrameDetector`` is the only thing the pipeline needs from a model: "give me detections for this
frame". Anything with a ``detect`` method works, which is how the tests run the whole pipeline with
a scripted fake instead of a neural network.

``YoloDetector`` is the real one. It must run at the image size the model was trained at (960 for
the adopted model); running a 960 model at 640 measurably loses small helmets and vests
(docs/EXPERIMENTS.md), so the size is explicit and printed, never silently defaulted.
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping, Protocol, Sequence

import numpy as np

from safety.contracts import FORKLIFT, HELMET, PERSON, VEST, Box, Detection

REQUIRED_CLASSES = (PERSON, HELMET, VEST)
KNOWN_CLASSES = (PERSON, HELMET, VEST, FORKLIFT)


class DetectorError(RuntimeError):
    """Raised when the model cannot be used for this project."""


class FrameDetector(Protocol):
    def detect(self, frame_bgr: np.ndarray) -> list[Detection]: ...


def results_to_detections(
    names: Mapping[int, str],
    xyxy: Sequence[Sequence[float]],
    conf: Sequence[float],
    cls: Sequence[float],
) -> list[Detection]:
    """Pure conversion of raw model output. Boxes that are empty or inverted are dropped, and so
    are classes this project does not use; neither should be able to crash a rule later."""
    detections: list[Detection] = []
    for box, score, class_id in zip(xyxy, conf, cls):
        name = names.get(int(class_id))
        if name not in KNOWN_CLASSES:
            continue
        x1, y1, x2, y2 = (float(v) for v in box)
        if not (x2 > x1 and y2 > y1):
            continue
        detections.append(Detection(name, float(score), Box(x1, y1, x2, y2)))
    return detections


class YoloDetector:
    def __init__(self, weights: Path, *, imgsz: int, conf: float = 0.35, device: str = "cpu") -> None:
        if imgsz < 32:
            raise ValueError(f"imgsz must be at least 32, got {imgsz}")
        if not 0.0 < conf < 1.0:
            raise ValueError(f"conf must be between 0 and 1, got {conf}")
        weights = Path(weights)
        if not weights.is_file():
            raise DetectorError(f"weights not found: {weights}")
        from ultralytics import YOLO  # imported here: heavy, and not needed to test the rest

        self.weights, self.imgsz, self.conf, self.device = weights, imgsz, conf, device
        self._model = YOLO(str(weights))
        self.names: dict[int, str] = dict(self._model.names)
        missing = [c for c in REQUIRED_CLASSES if c not in self.names.values()]
        if missing:
            raise DetectorError(f"model has no class(es) {missing}; it knows {sorted(self.names.values())}")

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        result = self._model.predict(frame_bgr, conf=self.conf, imgsz=self.imgsz, device=self.device, verbose=False)[0]
        boxes = result.boxes
        return results_to_detections(
            self.names, boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy(), boxes.cls.cpu().numpy()
        )
