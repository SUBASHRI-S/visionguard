from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np
import pytest

from incidents.store import (ALLOWED_STATUS_CHANGES, STATUSES, IncidentNotFound, IncidentStore,
                             InvalidStatusChange)
from incidents.temporal import IncidentTracker, TemporalConfig, Transition
from pipeline.runner import process_video
from pipeline.video_pipeline import VideoPipeline
from safety.contracts import Box, Detection
from safety.engine import SafetyEngine
from safety.events import Event
from trackers import PersonTracker


@pytest.fixture
def store(tmp_path):
    s = IncidentStore(f"sqlite:///{tmp_path / 'db' / 'vg.db'}")
    s.create_schema()
    yield s
    s.dispose()


def event(rule="no_helmet", camera="cam", track=1, ts=1.0, details=None):
    return Event(rule_id=rule, camera_id=camera, frame_index=int(ts * 10), timestamp=ts, outcome="violation",
                 severity="high", person_index=0, track_id=track, details=details or {"box": (1.0, 2.0, 3.0, 4.0)})


def transition(kind="opened", rule="no_helmet", camera="cam", track=1, episode=1, started=1.0, ts=4.0,
               reason=None, violating=30, uncertain=2):
    return Transition(kind=kind, camera_id=camera, rule_id=rule, track_id=track, episode=episode,
                      frame_index=int(ts * 10), timestamp=ts, started_frame=int(started * 10),
                      started_timestamp=started, severity="high", event=event(rule, camera, track, ts),
                      violating_frames=violating, uncertain_frames=uncertain, reason=reason)


# ---------------------------------------------------------------- recording what the temporal logic decided

def test_opened_creates_an_active_open_incident(store):
    iid = store.record_transition(transition(), run_id="r1", evidence_path="ev/a.jpg")
    rec = store.get(iid)
    assert (rec.rule_id, rec.camera_id, rec.track_id, rec.episode) == ("no_helmet", "cam", 1, 1)
    assert rec.status == "open" and rec.active is True and rec.severity == "high"
    assert rec.started_s == 1.0 and rec.opened_s == 4.0 and rec.ended_s is None
    assert [e["kind"] for e in rec.evidence] == ["opened"] and rec.evidence[0]["path"] == "ev/a.jpg"
    assert [h["kind"] for h in rec.history] == ["opened"]


def test_closed_ends_the_incident_but_leaves_review_status_alone(store):
    iid = store.record_transition(transition(), run_id="r1")
    store.record_transition(transition("closed", ts=9.0, reason="cleared", violating=80), run_id="r1",
                            evidence_path="ev/b.jpg")
    rec = store.get(iid)
    assert rec.active is False and rec.status == "open"
    assert rec.ended_s == 9.0 and rec.end_reason == "cleared" and rec.last_violation_s == 9.0
    assert rec.violating_frames == 80
    assert [e["kind"] for e in rec.evidence] == ["closed"]          # only the close carried a picture here
    assert [h["kind"] for h in rec.history] == ["opened", "ended"]


def test_recording_the_same_transition_twice_changes_nothing(store):
    a = store.record_transition(transition(), run_id="r1", evidence_path="ev/a.jpg")
    b = store.record_transition(transition(), run_id="r1", evidence_path="ev/a.jpg")
    c1 = store.record_transition(transition("closed", ts=9.0, reason="cleared"), run_id="r1")
    c2 = store.record_transition(transition("closed", ts=9.0, reason="cleared"), run_id="r1")
    assert a == b == c1 == c2
    rec = store.get(a)
    assert len(rec.evidence) == 1 and [h["kind"] for h in rec.history] == ["opened", "ended"]
    assert store.list_incidents()[1] == 1


def test_a_close_without_an_open_still_produces_a_row(store):
    iid = store.record_transition(transition("closed", ts=9.0, reason="stream_end"), run_id="r1")
    rec = store.get(iid)
    assert rec.active is False and rec.end_reason == "stream_end"


def test_runs_tracks_rules_and_episodes_are_separate_incidents(store):
    ids = {
        store.record_transition(transition(), run_id="r1"),
        store.record_transition(transition(), run_id="r2"),
        store.record_transition(transition(track=2), run_id="r1"),
        store.record_transition(transition(rule="no_vest"), run_id="r1"),
        store.record_transition(transition(episode=2), run_id="r1"),
        store.record_transition(transition(camera="cam2"), run_id="r1"),
    }
    assert len(ids) == 6


def test_event_details_survive_as_json(store):
    iid = store.record_transition(transition(), run_id="r1")
    assert store.get(iid).details == {"box": [1.0, 2.0, 3.0, 4.0]}


def test_numpy_numbers_in_details_are_stored(store):
    t = transition()
    t = Transition(**{**t.__dict__, "event": event(details={"score": np.float32(0.5), "n": np.int64(3)})})
    assert store.get(store.record_transition(t, run_id="r1")).details == {"score": 0.5, "n": 3}


def test_empty_run_id_rejected(store):
    with pytest.raises(ValueError):
        store.record_transition(transition(), run_id="")


# ---------------------------------------------------------------- human review

def test_review_workflow_is_recorded_with_who_and_when(store):
    iid = store.record_transition(transition(), run_id="r1")
    store.set_status(iid, "acknowledged", actor="asha", note="looking")
    rec = store.set_status(iid, "resolved", actor="asha", note="gave him a helmet")
    assert rec.status == "resolved"
    changes = [h for h in rec.history if h["kind"] == "status_change"]
    assert [(c["from_status"], c["to_status"], c["actor"]) for c in changes] == [
        ("open", "acknowledged", "asha"), ("acknowledged", "resolved", "asha")]
    assert changes[1]["note"] == "gave him a helmet"


def test_review_status_can_change_while_the_violation_is_still_active(store):
    iid = store.record_transition(transition(), run_id="r1")
    rec = store.set_status(iid, "acknowledged", actor="asha")
    assert rec.active is True and rec.status == "acknowledged"


@pytest.mark.parametrize("start", STATUSES)
@pytest.mark.parametrize("target", STATUSES)
def test_every_status_change_follows_the_allowed_table(store, start, target):
    iid = store.record_transition(transition(), run_id="r1")
    path = {"open": [], "acknowledged": ["acknowledged"], "resolved": ["resolved"], "false_positive": ["false_positive"]}
    for step in path[start]:
        store.set_status(iid, step, actor="x")
    if target in ALLOWED_STATUS_CHANGES[start]:
        assert store.set_status(iid, target, actor="x").status == target
    else:
        with pytest.raises(InvalidStatusChange):
            store.set_status(iid, target, actor="x")
        assert store.get(iid).status == start


def test_bad_review_input(store):
    iid = store.record_transition(transition(), run_id="r1")
    with pytest.raises(InvalidStatusChange):
        store.set_status(iid, "banana", actor="x")
    with pytest.raises(ValueError):
        store.set_status(iid, "acknowledged", actor="  ")
    with pytest.raises(ValueError):
        store.set_status(iid, "acknowledged", actor="x", note="n" * 501)
    with pytest.raises(IncidentNotFound):
        store.set_status(9999, "acknowledged", actor="x")
    assert store.get(iid).status == "open"


def test_unknown_incident(store):
    with pytest.raises(IncidentNotFound):
        store.get(1)


# ---------------------------------------------------------------- listing and stats

def populated(store):
    store.record_transition(transition(camera="cam1", rule="no_helmet", track=1), run_id="rA")
    store.record_transition(transition(camera="cam1", rule="no_vest", track=2), run_id="rA")
    store.record_transition(transition("closed", camera="cam1", rule="no_vest", track=2, ts=8, reason="cleared"),
                            run_id="rA")
    store.record_transition(transition(camera="cam2", rule="no_helmet", track=3), run_id="rB")
    store.set_status(1, "false_positive", actor="x")


def test_list_is_newest_first_and_counts_total(store):
    populated(store)
    items, total = store.list_incidents()
    assert total == 3 and [i.id for i in items] == [3, 2, 1]


@pytest.mark.parametrize("kw, expected_ids", [
    (dict(camera_id="cam1"), {1, 2}), (dict(rule_id="no_helmet"), {1, 3}), (dict(status="false_positive"), {1}),
    (dict(active=True), {1, 3}), (dict(active=False), {2}), (dict(run_id="rB"), {3}),
    (dict(camera_id="cam1", rule_id="no_helmet"), {1}), (dict(camera_id="nope"), set()),
])
def test_list_filters(store, kw, expected_ids):
    populated(store)
    items, total = store.list_incidents(**kw)
    assert {i.id for i in items} == expected_ids and total == len(expected_ids)


def test_list_pagination_keeps_the_real_total(store):
    populated(store)
    page, total = store.list_incidents(limit=2, offset=1)
    assert total == 3 and [i.id for i in page] == [2, 1]


def test_list_time_window(store):
    populated(store)
    now = datetime.now(timezone.utc)
    assert store.list_incidents(since=now - timedelta(minutes=1))[1] == 3
    assert store.list_incidents(since=now + timedelta(minutes=1))[1] == 0
    assert store.list_incidents(until=now - timedelta(minutes=1))[1] == 0


@pytest.mark.parametrize("kw", [dict(limit=0), dict(limit=501), dict(offset=-1), dict(status="banana")])
def test_list_rejects_bad_arguments(store, kw):
    with pytest.raises(ValueError):
        store.list_incidents(**kw)


def test_list_items_are_summaries_and_get_has_detail(store):
    populated(store)
    assert store.list_incidents()[0][0].history == () and store.get(1).history


def test_stats(store):
    populated(store)
    s = store.stats()
    assert s["total"] == 3 and s["active"] == 2
    assert s["by_rule"] == {"no_helmet": 2, "no_vest": 1}
    assert s["by_status"] == {"open": 2, "false_positive": 1}
    assert s["by_camera"] == {"cam1": 2, "cam2": 1}
    assert sum(s["by_day"].values()) == 3
    with pytest.raises(ValueError):
        store.stats(days=0)


def test_empty_database_stats(store):
    assert store.stats()["total"] == 0 and store.list_incidents() == ([], 0)


# ---------------------------------------------------------------- durability and the runner

def test_data_survives_reopening_the_file(tmp_path):
    url = f"sqlite:///{tmp_path / 'x.db'}"
    first = IncidentStore(url)
    first.create_schema()
    iid = first.record_transition(transition(), run_id="r1")
    first.dispose()
    second = IncidentStore(url)
    second.create_schema()                       # must not wipe anything
    assert second.get(iid).rule_id == "no_helmet"
    second.dispose()


def test_in_memory_database_works():
    s = IncidentStore("sqlite://")
    s.create_schema()
    assert s.record_transition(transition(), run_id="r") == 1


def test_deleting_an_incident_removes_its_history(store):
    iid = store.record_transition(transition(), run_id="r1", evidence_path="e.jpg")
    store.set_status(iid, "acknowledged", actor="x")
    from incidents.store import EvidenceRow, IncidentEventRow, IncidentRow
    from sqlalchemy import func, select
    with store._session() as session, session.begin():
        session.delete(session.get(IncidentRow, iid))
    with store._session() as session:
        assert session.scalar(select(func.count()).select_from(IncidentEventRow)) == 0
        assert session.scalar(select(func.count()).select_from(EvidenceRow)) == 0


class Scripted:
    def __init__(self, script):
        self.script, self.calls = script, 0

    def detect(self, frame):
        i, self.calls = self.calls, self.calls + 1
        return list(self.script[i]) if i < len(self.script) else []


def test_runner_saves_incidents_with_evidence(tmp_path, store):
    video = tmp_path / "clip.mp4"
    w = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10, (320, 240))
    for i in range(30):
        w.write(np.full((240, 320, 3), 40 + i, dtype=np.uint8))
    w.release()
    person = Detection("person", 0.9, Box(80, 30, 200, 230))
    pipeline = VideoPipeline(Scripted([[person]] * 30), SafetyEngine.from_config({"enabled_rules": ["no_helmet"]}),
                             PersonTracker(), IncidentTracker(TemporalConfig(open_after_frames=3, close_after_frames=4)),
                             camera_id="camA")
    summary = process_video(video, pipeline, tmp_path / "out", sink=store, run_id="run-1")
    items, total = store.list_incidents(run_id="run-1")
    assert total == len(summary["incidents"]) == 1 and summary["run_id"] == "run-1"
    rec = store.get(items[0].id)
    assert rec.camera_id == "camA" and rec.active is False and rec.end_reason == "stream_end"
    assert {e["kind"] for e in rec.evidence} == {"opened", "closed"}
    assert all(Path(e["path"]).is_file() for e in rec.evidence)


def test_runner_without_a_sink_still_works(tmp_path):
    video = tmp_path / "clip.mp4"
    w = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10, (320, 240))
    for _ in range(5):
        w.write(np.zeros((240, 320, 3), dtype=np.uint8))
    w.release()
    pipeline = VideoPipeline(Scripted([]), SafetyEngine.from_config(None), PersonTracker(), IncidentTracker(),
                             camera_id="c")
    assert process_video(video, pipeline, tmp_path / "o")["run_id"].startswith("clip-")


def test_database_refuses_orphan_rows(store):
    """The database itself, not just our Python, must reject evidence pointing at a missing incident."""
    from incidents.store import EvidenceRow
    from sqlalchemy.exc import IntegrityError
    with pytest.raises(IntegrityError):
        with store._session() as session, session.begin():
            session.add(EvidenceRow(incident_id=424242, kind="opened", path="x.jpg"))
