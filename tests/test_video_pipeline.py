import csv
import json

import cv2
import numpy as np
import pytest

from incidents.temporal import IncidentTracker, TemporalConfig
from pipeline.detector import DetectorError, YoloDetector, results_to_detections
from pipeline.render import draw_result
from pipeline.runner import VideoError, process_video
from pipeline.video_pipeline import VideoPipeline
from safety.contracts import Box, Detection
from safety.engine import SafetyEngine
from trackers import PersonTracker

W, H = 1280, 720


def person(x=200, y=150, w=120, h=420, conf=0.9):
    return Detection("person", conf, Box(x, y, x + w, y + h))


def helmet_on(p: Detection):
    b = p.box
    return Detection("helmet", 0.9, Box(b.x1 + 30, b.y1 + 8, b.x1 + 80, b.y1 + 55))


def vest_on(p: Detection):
    b = p.box
    return Detection("vest", 0.9, Box(b.x1 + 5, b.y1 + 95, b.x2 - 5, b.y1 + 270))


class Scripted:
    """A fake detector: call number n returns script[n] (empty after the script ends)."""

    def __init__(self, script):
        self.script, self.calls = script, 0

    def detect(self, frame):
        i, self.calls = self.calls, self.calls + 1
        return list(self.script[i]) if i < len(self.script) else []


def make_pipeline(script, *, open_after=3, close_after=4, lost=0.5, rules=("no_helmet",)):
    return VideoPipeline(
        Scripted(script),
        SafetyEngine.from_config({"enabled_rules": list(rules)}),
        PersonTracker(),
        IncidentTracker(TemporalConfig(open_after_frames=open_after, close_after_frames=close_after,
                                       track_lost_seconds=lost)),
        camera_id="cam",
    )


def blank():
    return np.zeros((H, W, 3), dtype=np.uint8)


def run_frames(pipeline, count, fps=10):
    results = [pipeline.process_frame(blank(), i, i / fps) for i in range(count)]
    return results


# ---------------------------------------------------------------- core

def test_unhelmeted_person_opens_one_incident_and_stream_end_closes_it():
    p = person()
    pipeline = make_pipeline([[p, vest_on(p)]] * 10)
    results = run_frames(pipeline, 10)
    opened = [t for r in results for t in r.transitions if t.kind == "opened"]
    assert len(opened) == 1 and opened[0].rule_id == "no_helmet" and opened[0].frame_index == 2
    closed = pipeline.finish(9, 0.9)
    assert len(closed) == 1 and closed[0].reason == "stream_end"
    assert closed[0].track_id == opened[0].track_id


def test_compliant_person_raises_nothing():
    p = person()
    pipeline = make_pipeline([[p, helmet_on(p), vest_on(p)]] * 10, rules=("no_helmet", "no_vest"))
    results = run_frames(pipeline, 10)
    assert all(not r.events and not r.transitions for r in results)
    assert pipeline.finish(9, 0.9) == []


def test_track_id_lines_up_with_the_right_person_even_when_order_swaps():
    """Person A has no helmet, person B has one. Detections are listed in swapped order mid-way;
    the incident must stay on A's track id."""
    a, b = person(x=100), person(x=700)
    first = [a, b, helmet_on(b)]
    swapped = [b, helmet_on(b), a]
    pipeline = make_pipeline([first] * 3 + [swapped] * 6)
    results = run_frames(pipeline, 9)
    a_track = results[0].track_ids[0]          # in the first frames person index 0 is A
    opened = [t for r in results for t in r.transitions if t.kind == "opened"]
    assert [t.track_id for t in opened] == [a_track]
    last = results[-1]
    helmetless = [s for s in last.statuses if s.helmet == "no"]
    assert len(helmetless) == 1 and last.track_ids[helmetless[0].person_index] == a_track


def test_ids_stay_with_the_same_person_when_someone_new_is_listed_first():
    a, c = person(x=100), person(x=800)
    script = [[a]] * 4 + [[c, a]] * 8           # a newcomer c is listed BEFORE a from frame 4 on
    results = run_frames(make_pipeline(script), 12)
    ids_of_a = set()
    for r in results:
        for s in r.statuses:
            if s.person.box.x1 == a.box.x1:
                ids_of_a.add(r.track_ids.get(s.person_index))
    assert len(ids_of_a) == 1 and None not in ids_of_a
    ids_of_c = {r.track_ids[s.person_index] for r in results for s in r.statuses
                if s.person.box.x1 == c.box.x1 and s.person_index in r.track_ids}
    assert len(ids_of_c) == 1 and ids_of_c.isdisjoint(ids_of_a)


def test_incident_closes_when_person_leaves():
    p = person()
    script = [[p, vest_on(p)]] * 6 + [[]] * 12
    pipeline = make_pipeline(script, lost=0.4)
    results = run_frames(pipeline, 18)
    closed = [t for r in results for t in r.transitions if t.kind == "closed"]
    assert len(closed) == 1 and closed[0].reason == "track_lost"


def test_empty_frames_do_not_crash():
    pipeline = make_pipeline([[]] * 5)
    assert all(not r.statuses and not r.transitions for r in run_frames(pipeline, 5))


def test_result_has_timings_and_open_incident_keys():
    p = person()
    pipeline = make_pipeline([[p]] * 5)
    last = run_frames(pipeline, 5)[-1]
    assert set(last.timings_ms) == {"detect", "track", "rules"} and all(v >= 0 for v in last.timings_ms.values())
    assert last.open_incidents == (("cam", "no_helmet", last.track_ids[0]),)


@pytest.mark.parametrize("bad", [np.zeros((10, 10), np.uint8), np.zeros((0, 10, 3), np.uint8), "frame", None])
def test_bad_frame_rejected(bad):
    with pytest.raises(ValueError):
        make_pipeline([[]]).process_frame(bad, 0, 0.0)


def test_empty_camera_id_rejected():
    with pytest.raises(ValueError):
        VideoPipeline(Scripted([]), SafetyEngine.from_config(None), PersonTracker(), IncidentTracker(), camera_id="")


# ---------------------------------------------------------------- detector wrapper

def test_results_to_detections_filters_and_converts():
    names = {0: "person", 1: "helmet", 2: "vest", 3: "forklift", 4: "banana"}
    xyxy = [[0, 0, 10, 20], [5, 5, 5, 9], [9, 9, 3, 12], [1, 1, 8, 8], [0, 0, 5, 5]]
    out = results_to_detections(names, xyxy, [0.9, 0.8, 0.7, 0.6, 0.5], [0, 1, 2, 3, 4])
    assert [d.class_name for d in out] == ["person", "forklift"]   # empty/inverted boxes and unknown class dropped
    assert isinstance(out[0].confidence, float) and out[0].box.x2 == 10.0


def test_results_to_detections_empty():
    assert results_to_detections({0: "person"}, [], [], []) == []


def test_yolo_detector_rejects_bad_settings(tmp_path):
    weights = tmp_path / "w.pt"
    weights.write_bytes(b"x")
    with pytest.raises(ValueError):
        YoloDetector(weights, imgsz=8)
    with pytest.raises(ValueError):
        YoloDetector(weights, imgsz=960, conf=1.5)
    with pytest.raises(DetectorError, match="not found"):
        YoloDetector(tmp_path / "missing.pt", imgsz=960)


# ---------------------------------------------------------------- drawing

def test_draw_returns_a_copy_and_marks_the_frame():
    p = person()
    pipeline = make_pipeline([[p]] * 5)
    result = run_frames(pipeline, 5)[-1]
    frame = blank()
    out = draw_result(frame, result, camera_id="cam")
    assert out.shape == frame.shape and out is not frame
    assert frame.sum() == 0 and out.sum() > 0


# ---------------------------------------------------------------- runner on a real (synthetic) file

def make_video(path, frames=30, fps=10, size=(320, 240)):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for i in range(frames):
        w.write(np.full((size[1], size[0], 3), 40 + i, dtype=np.uint8))
    w.release()
    return path


def small_person():
    return Detection("person", 0.9, Box(80, 30, 200, 230))


def test_runner_writes_everything_and_counts_one_incident(tmp_path):
    video = make_video(tmp_path / "clip.mp4")
    p = small_person()
    pipeline = make_pipeline([[p]] * 30)
    out = tmp_path / "out"
    summary = process_video(video, pipeline, out, run_info={"weights": "fake.pt", "imgsz": 960, "conf": 0.35})
    assert summary["frames_processed"] == 30 and len(summary["incidents"]) == 1
    assert summary["incidents"][0]["reason"] == "stream_end"
    for name in ("annotated.mp4", "incidents.jsonl", "per_frame.csv", "summary.json", "report.md"):
        assert (out / name).is_file(), name
    lines = (out / "incidents.jsonl").read_text().splitlines()
    assert [json.loads(l)["kind"] for l in lines] == ["opened", "closed"]
    assert len(list((out / "evidence").glob("*.jpg"))) == 2
    rows = list(csv.reader((out / "per_frame.csv").open()))
    assert len(rows) == 31 and rows[0][0] == "frame"
    assert json.loads((out / "summary.json").read_text())["frames_processed"] == 30
    assert "## Incidents" in (out / "report.md").read_text()


def test_runner_compliant_clip_has_no_incidents_and_says_so(tmp_path):
    video = make_video(tmp_path / "clip.mp4")
    p = small_person()
    pipeline = make_pipeline([[p, helmet_on(p)]] * 30)
    summary = process_video(video, pipeline, tmp_path / "out")
    assert summary["incidents"] == []
    assert "none opened" in (tmp_path / "out" / "report.md").read_text()


def test_runner_times_are_video_seconds_not_frame_numbers(tmp_path):
    video = make_video(tmp_path / "clip.mp4", frames=30, fps=10)
    p = small_person()
    script = [[]] * 10 + [[p]] * 20           # the person appears at frame 10 (1.0 s)
    summary = process_video(video, make_pipeline(script), tmp_path / "out")
    (incident,) = summary["incidents"]
    assert incident["started_frame"] == 11     # confirmed one frame after first sighting
    assert incident["started_timestamp"] == pytest.approx(1.1)


def test_runner_stride_and_max_seconds(tmp_path):
    video = make_video(tmp_path / "clip.mp4", frames=30, fps=10)
    summary = process_video(video, make_pipeline([[]] * 30), tmp_path / "o1", frame_stride=3)
    assert summary["frames_processed"] == 10
    summary = process_video(video, make_pipeline([[]] * 30), tmp_path / "o2", max_seconds=1.0)
    assert summary["frames_processed"] == 10


def test_runner_track_timeline_is_reported(tmp_path):
    video = make_video(tmp_path / "clip.mp4")
    p = small_person()
    summary = process_video(video, make_pipeline([[p, vest_on(p)]] * 30), tmp_path / "out")
    (row,) = summary["tracks"]
    assert row["frames"] == 30 and row["helmet"] == {"yes": 0, "no": 30, "unknown": 0}


def test_runner_errors(tmp_path):
    with pytest.raises(VideoError):
        process_video(tmp_path / "missing.mp4", make_pipeline([]), tmp_path / "o")
    video = make_video(tmp_path / "clip.mp4")
    with pytest.raises(ValueError):
        process_video(video, make_pipeline([]), tmp_path / "o", frame_stride=0)
    with pytest.raises(ValueError):
        process_video(video, make_pipeline([]), tmp_path / "o", max_seconds=0)


def test_runner_does_not_modify_the_source(tmp_path):
    video = make_video(tmp_path / "clip.mp4")
    before = video.read_bytes()
    process_video(video, make_pipeline([[]] * 30), tmp_path / "out")
    assert video.read_bytes() == before


# ---------------------------------------------------------------- command line

def test_cli_reports_bad_start_cleanly(tmp_path):
    from scripts.run_video import main
    video = make_video(tmp_path / "clip.mp4")
    safety = tmp_path / "s.yaml"
    safety.write_text("ppe: {}\n")
    common = [str(video), "--weights", str(tmp_path / "nope.pt"), "--imgsz", "960", "--safety-config", str(safety)]
    with pytest.raises(SystemExit, match="weights not found"):
        main(common)
    bad = tmp_path / "temporal.yaml"
    bad.write_text("open_after_frame: 3\n")
    with pytest.raises(SystemExit, match="cannot start"):
        main(common + ["--temporal-config", str(bad)])


# ---------------------------------------------------------------- work area

from safety.zones import Zone, ZoneError

# the lower half of the frame: where the workers stand; the far background is outside
LOWER_HALF = Zone("foreground", ((0.0, 0.5), (1.0, 0.5), (1.0, 1.0), (0.0, 1.0)))


def make_area_pipeline(script, area=LOWER_HALF):
    return VideoPipeline(
        Scripted(script), SafetyEngine.from_config({"enabled_rules": ["no_helmet"]}), PersonTracker(),
        IncidentTracker(TemporalConfig(open_after_frames=3, close_after_frames=4, track_lost_seconds=0.5)),
        camera_id="cam", work_area=area,
    )


def far_person(x=900):          # feet at y=270 of 720 (37%): above the line, outside the area
    return person(x=x, y=150, w=60, h=120)


def near_person(x=100):         # feet at y=570 of 720 (79%): inside the area
    return person(x=x, y=150, w=120, h=420)


def test_person_outside_work_area_is_not_checked_but_is_reported():
    pipeline = make_area_pipeline([[far_person()]] * 8)
    results = run_frames(pipeline, 8)
    assert all(not r.statuses and not r.events and not r.transitions for r in results)
    last = results[-1]
    assert len(last.outside_persons) == 1 and last.outside_persons[0][1] is not None


def test_person_inside_work_area_is_checked():
    results = run_frames(make_area_pipeline([[near_person()]] * 8), 8)
    assert any(t.kind == "opened" for r in results for t in r.transitions)


def test_track_ids_follow_the_right_person_when_one_is_outside():
    """The outside person is listed first. Track ids must be renumbered for the people the engine sees."""
    out, inn = far_person(), near_person()
    pipeline = make_area_pipeline([[out, inn]] * 8)
    results = run_frames(pipeline, 8)
    outside_id = results[-1].outside_persons[0][1]
    opened = [t for r in results for t in r.transitions if t.kind == "opened"]
    assert len(opened) == 1 and opened[0].track_id != outside_id
    assert results[-1].track_ids == {0: opened[0].track_id}


def test_no_work_area_means_everyone_is_checked():
    results = run_frames(make_area_pipeline([[far_person()]] * 8, area=None), 8)
    assert results[-1].statuses and not results[-1].outside_persons


def test_runner_counts_outside_frames_and_reports_them(tmp_path):
    video = make_video(tmp_path / "clip.mp4", frames=10)
    pipeline = make_area_pipeline([[Detection("person", 0.9, Box(200, 20, 240, 90))]] * 10)
    summary = process_video(video, pipeline, tmp_path / "out", run_info={"work_area": "foreground"})
    assert summary["outside_person_frames"] == 10 and summary["incidents"] == []
    assert "outside the work area" in (tmp_path / "out" / "report.md").read_text()


def test_report_shows_last_violation_time_separately_from_close_time(tmp_path):
    video = make_video(tmp_path / "clip.mp4", frames=30, fps=10)
    p = small_person()
    script = [[p]] * 6 + [[]] * 24                      # leaves after 0.6 s
    summary = process_video(video, make_pipeline(script, lost=0.5), tmp_path / "out")
    (incident,) = summary["incidents"]
    assert incident["last_violation_timestamp"] < incident["ended_timestamp"]
    assert incident["last_violation_timestamp"] == pytest.approx(0.5)
    assert "last violation" in (tmp_path / "out" / "report.md").read_text()


def test_zone_file_loader(tmp_path):
    from scripts.run_video import load_zone_file
    good = tmp_path / "z.yaml"
    good.write_text("work_area:\n  name: yard\n  polygon: [[0,0.5],[1,0.5],[1,1],[0,1]]\n"
                    "zones:\n  - name: pit\n    polygon: [[0.1,0.1],[0.3,0.1],[0.3,0.3]]\n")
    zones, area = load_zone_file(good)
    assert [z.name for z in zones] == ["pit"] and area.name == "yard"
    assert load_zone_file(None) == ([], None)
    bad = tmp_path / "bad.yaml"
    bad.write_text("work_areas: {}\n")
    with pytest.raises(ZoneError):
        load_zone_file(bad)
    bad.write_text("work_area:\n  name: x\n  polygon: [[0,0],[1,1]]\n")
    with pytest.raises(ZoneError):
        load_zone_file(bad)
