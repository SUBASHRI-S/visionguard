from pathlib import Path

import cv2
import numpy as np
import pytest

from inference.detector import Detector

BUS_IMAGE = Path(__file__).parent.parent / "bus.jpg"


@pytest.fixture(scope="module")
def detector() -> Detector:
    return Detector()  # loaded once for all tests in this file


def test_blank_image_returns_list(detector):
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    assert detector.predict(blank) == []


def test_rejects_non_image_input(detector):
    with pytest.raises(ValueError):
        detector.predict("not an image")  # type: ignore[arg-type]


def test_rejects_invalid_confidence(detector):
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        detector.predict(blank, conf=1.5)


@pytest.mark.skipif(not BUS_IMAGE.exists(), reason="bus.jpg sample not present")
def test_detects_bus_and_people(detector):
    image = cv2.imread(str(BUS_IMAGE))
    names = {d.class_name for d in detector.predict(image)}
    assert "bus" in names
    assert "person" in names


@pytest.mark.skipif(not BUS_IMAGE.exists(), reason="bus.jpg sample not present")
def test_higher_threshold_gives_fewer_or_equal_detections(detector):
    image = cv2.imread(str(BUS_IMAGE))
    low = detector.predict(image, conf=0.25)
    high = detector.predict(image, conf=0.8)
    assert len(high) <= len(low)