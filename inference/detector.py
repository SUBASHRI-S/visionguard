"""Object detector wrapper around Ultralytics YOLO."""
from dataclasses import dataclass

import numpy as np
from ultralytics import YOLO


@dataclass(frozen=True)
class Detection:
    """One detected object. Box is in pixel coordinates (top-left, bottom-right)."""

    class_id: int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float


class Detector:
    """Loads a YOLO model once and runs detection on BGR images (OpenCV format)."""

    def __init__(self, weights: str = "yolo11n.pt", device: str = "cpu") -> None:
        self.weights = weights
        self.device = device
        self._model = YOLO(weights)

    def predict(self, image: np.ndarray, conf: float = 0.25) -> list[Detection]:
        if not isinstance(image, np.ndarray) or image.ndim != 3:
            raise ValueError("image must be a numpy array of shape (H, W, 3)")
        if not 0.0 <= conf <= 1.0:
            raise ValueError("conf must be between 0 and 1")

        result = self._model.predict(
            image, conf=conf, device=self.device, verbose=False
        )[0]

        boxes = result.boxes
        detections: list[Detection] = []
        for cls_id, score, xyxy in zip(
            boxes.cls.tolist(), boxes.conf.tolist(), boxes.xyxy.tolist()
        ):
            detections.append(
                Detection(
                    class_id=int(cls_id),
                    class_name=result.names[int(cls_id)],
                    confidence=float(score),
                    x1=xyxy[0],
                    y1=xyxy[1],
                    x2=xyxy[2],
                    y2=xyxy[3],
                )
            )
        return detections