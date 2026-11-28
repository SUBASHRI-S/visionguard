import cv2
import numpy as np
import pytest

from scripts.benchmark_models import sample_frames, summarize, time_detector, to_markdown


class CountingDetector:
    def __init__(self) -> None:
        self.calls = 0

    def detect(self, frame_bgr):
        self.calls += 1
        return []


def make_video(path, frames=20):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 48))
    for i in range(frames):
        writer.write(np.full((48, 64, 3), i * 10 % 255, np.uint8))
    writer.release()


def test_summarize_values():
    row = summarize([10.0, 20.0, 30.0, 40.0, 100.0])
    assert row["median_ms"] == 30.0
    assert row["min_ms"] == 10.0 and row["max_ms"] == 100.0
    assert row["p95_ms"] == 100.0
    assert row["fps_at_median"] == pytest.approx(33.33, abs=0.01)


def test_summarize_rejects_empty():
    with pytest.raises(ValueError):
        summarize([])


def test_warmup_calls_are_not_timed():
    detector = CountingDetector()
    frames = [np.zeros((4, 4, 3), np.uint8)] * 3
    times = time_detector(detector, frames, warmup=5)
    assert len(times) == 3
    assert detector.calls == 8


def test_time_detector_needs_frames():
    with pytest.raises(ValueError):
        time_detector(CountingDetector(), [], warmup=1)


def test_sample_frames_even_spread(tmp_path):
    video = tmp_path / "v.mp4"
    make_video(video, 20)
    frames = sample_frames(video, 5)
    assert 1 <= len(frames) <= 5
    assert frames[0].shape == (48, 64, 3)


def test_sample_frames_errors(tmp_path):
    with pytest.raises(ValueError):
        sample_frames(tmp_path / "missing.mp4", 3)
    video = tmp_path / "v.mp4"
    make_video(video, 5)
    with pytest.raises(ValueError):
        sample_frames(video, 0)


def test_markdown_lists_each_model():
    text = to_markdown(
        {"n": summarize([10.0, 12.0]), "s": summarize([30.0, 33.0])},
        {"video": "c.mp4", "imgsz": 960, "frames": 2, "warmup": 5, "machine": {"device": "cpu"}},
    )
    assert "| n |" in text and "| s |" in text and "c.mp4" in text
