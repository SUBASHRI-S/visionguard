import cv2
import numpy as np
import pytest

from scripts.shrink_clip import main, shrink


def make_video(path, frames=60, fps=30, size=(320, 240)):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for i in range(frames):
        w.write(np.full((size[1], size[0], 3), i * 4 % 255, dtype=np.uint8))
    w.release()
    return path


def test_trims_and_resizes(tmp_path):
    src = make_video(tmp_path / "a.mp4")
    info = shrink(src, tmp_path / "o.mp4", start=0.5, seconds=1.0, width=160)
    assert info["size"] == (160, 120) and info["frames"] == 30


def test_never_upscales_and_keeps_even_size(tmp_path):
    src = make_video(tmp_path / "a.mp4", size=(322, 242))
    info = shrink(src, tmp_path / "o.mp4", start=0, seconds=1, width=1000)
    assert info["size"][0] % 2 == 0 and info["size"][1] % 2 == 0 and info["size"][0] <= 322


def test_original_untouched(tmp_path):
    src = make_video(tmp_path / "a.mp4")
    before = src.read_bytes()
    shrink(src, tmp_path / "o.mp4", start=0, seconds=1, width=160)
    assert src.read_bytes() == before


def test_short_video_returns_what_exists(tmp_path):
    src = make_video(tmp_path / "a.mp4", frames=15)
    assert shrink(src, tmp_path / "o.mp4", start=0, seconds=10, width=160)["frames"] == 15


def test_start_past_end_errors_and_leaves_no_file(tmp_path):
    src = make_video(tmp_path / "a.mp4", frames=15)
    out = tmp_path / "o.mp4"
    with pytest.raises(ValueError):
        shrink(src, out, start=50, seconds=1, width=160)
    assert not out.exists()


def test_missing_file_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        shrink(tmp_path / "nope.mp4", tmp_path / "o.mp4", start=0, seconds=1, width=160)


@pytest.mark.parametrize("kw", [dict(seconds=0), dict(seconds=-1), dict(start=-1), dict(width=10)])
def test_bad_arguments(tmp_path, kw):
    src = make_video(tmp_path / "a.mp4")
    args = dict(start=0, seconds=1, width=160) | kw
    with pytest.raises(ValueError):
        shrink(src, tmp_path / "o.mp4", **args)


def test_cli_default_output_location_and_exit_codes(tmp_path, capsys):
    src = make_video(tmp_path / "clip.mp4")
    assert main([str(src), "--seconds", "1", "--width", "160"]) == 0
    assert (tmp_path / "small" / "clip.mp4").exists()
    assert main([str(tmp_path / "missing.mp4")]) == 1
    assert "error" in capsys.readouterr().err
