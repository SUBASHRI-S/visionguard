import os

import pytest
import yaml

from incidents.temporal import IncidentTracker, TemporalConfig
from safety.contracts import Box, Detection
from safety.engine import SafetyEngine
from trackers import PersonTracker, TrackerConfig, TrackerConfigError


def person(x, y=100, w=60, h=160, conf=0.9) -> Detection:
    return Detection("person", conf, Box(x, y, x + w, y + h))


# ---------------------------------------------------------------- config

def test_defaults_match_ultralytics_bytetrack_yaml():
    import ultralytics
    path = os.path.join(os.path.dirname(ultralytics.__file__), "cfg", "trackers", "bytetrack.yaml")
    with open(path, encoding="utf-8") as fh:
        library = yaml.safe_load(fh)
    cfg = TrackerConfig()
    for key in ("track_high_thresh", "track_low_thresh", "new_track_thresh", "track_buffer", "match_thresh", "fuse_score"):
        assert getattr(cfg, key) == library[key], key


def test_from_dict_overrides_and_none():
    assert TrackerConfig.from_dict({"track_buffer": 60}).track_buffer == 60
    assert TrackerConfig.from_dict(None) == TrackerConfig()


@pytest.mark.parametrize("bad", [
    {"track_high_thresh": 1.5},
    {"track_low_thresh": -0.1},
    {"match_thresh": "0.8"},
    {"track_high_thresh": True},
    {"track_low_thresh": 0.5, "track_high_thresh": 0.3, "new_track_thresh": 0.5},
    {"new_track_thresh": 0.1},
    {"track_buffer": 0},
    {"track_buffer": 2.5},
    {"fuse_score": "yes"},
])
def test_bad_config_rejected(bad):
    with pytest.raises(TrackerConfigError):
        TrackerConfig.from_dict(bad)


def test_unknown_key_rejected():
    with pytest.raises(TrackerConfigError, match="unknown"):
        TrackerConfig.from_dict({"track_buffr": 10})


# ---------------------------------------------------------------- tracking behaviour

def test_first_frame_people_get_distinct_ids():
    ids = PersonTracker().update([person(50), person(300)])
    assert set(ids) == {0, 1}
    assert ids[0] != ids[1]


def test_empty_frame_returns_empty():
    t = PersonTracker()
    assert t.update([]) == {}
    assert t.update([person(10)]) == {}        # first sighting after the start: not yet confirmed
    assert len(t.update([person(12)])) == 1    # confirmed on the second sighting


def test_id_is_stable_while_person_moves():
    t = PersonTracker()
    seen = {t.update([person(100 + 5 * f)])[0] for f in range(15)}
    assert len(seen) == 1


def test_two_people_keep_their_own_ids():
    t = PersonTracker()
    first = t.update([person(100), person(400)])
    for f in range(1, 12):
        out = t.update([person(100 + 4 * f), person(400 - 4 * f)])
    assert out[0] == first[0] and out[1] == first[1]


def test_index_follows_input_order_not_id_order():
    t = PersonTracker()
    a = t.update([person(100), person(400)])
    b = t.update([person(402), person(102)])  # same people, listed in swapped order
    assert b[0] == a[1] and b[1] == a[0]


def test_new_person_reported_one_frame_late_and_without_none():
    t = PersonTracker()
    t.update([person(100)])
    out = t.update([person(102), person(400)])      # newcomer, first sighting
    assert out == {0: next(iter(out.values()))} and 1 not in out
    out = t.update([person(104), person(401)])      # second sighting: now confirmed
    assert set(out) == {0, 1}


def test_short_gap_keeps_same_id():
    t = PersonTracker()
    before = t.update([person(100)])[0]
    for _ in range(3):
        assert t.update([]) == {}
    assert t.update([person(104)])[0] == before


def test_gap_longer_than_buffer_gives_new_id():
    t = PersonTracker(TrackerConfig(track_buffer=5))
    before = t.update([person(100)])[0]
    for _ in range(12):
        t.update([])
    t.update([person(100)])
    after = t.update([person(101)])[0]
    assert after != before


def test_ids_are_never_reused_in_one_stream():
    t = PersonTracker(TrackerConfig(track_buffer=3))
    seen = []
    for start in (100, 500, 900):                   # three different people, one after another
        for _ in range(4):
            out = t.update([person(start)])
        seen.append(out[0])
        for _ in range(10):
            t.update([])
    assert len(set(seen)) == 3


def test_low_confidence_detection_cannot_start_a_track():
    t = PersonTracker()
    assert t.update([person(100, conf=0.15)]) == {}
    assert t.update([person(101, conf=0.15)]) == {}


def test_low_confidence_detection_keeps_existing_track():
    t = PersonTracker()
    t.update([person(100)])
    original = t.update([person(102)])[0]
    out = t.update([person(104, conf=0.15)])        # ByteTrack's second pass rescues it
    assert out.get(0) == original


def test_reset_forgets_everything():
    t = PersonTracker()
    for f in range(5):
        t.update([person(100 + f)])
    t.reset()
    assert t.frames_seen == 0
    # a different person far away: only a truly fresh tracker confirms them on the first frame with id 1
    assert t.update([person(900)]) == {0: 1}


def test_two_trackers_do_not_share_state():
    a, b = PersonTracker(), PersonTracker()
    a.update([person(100)]); a.update([person(100)])
    assert b.update([person(300)]) == {0: 1}


# ---------------------------------------------------------------- with the rest of the system

def test_tracker_feeds_safety_engine_and_incident_tracker_end_to_end():
    """A person without a helmet, tracked across frames, becomes one incident for that track id."""
    engine = SafetyEngine.from_config({"enabled_rules": ["no_helmet"]})
    incidents = IncidentTracker(TemporalConfig(open_after_frames=3, close_after_frames=4, track_lost_seconds=0.5))
    tracker = PersonTracker()
    opened = closed = None
    for f in range(16):
        detections = [person(200 + 3 * f, y=150, w=120, h=420)] if f < 8 else []
        persons = [d for d in detections if d.class_name == "person"]
        ids = tracker.update(persons)
        events = engine.process(detections, camera_id="cam", frame_index=f, timestamp=f / 10,
                                frame_width=1280, frame_height=720, track_ids=ids)
        present = set(ids.values())
        for tr in incidents.update(camera_id="cam", frame_index=f, timestamp=f / 10,
                                   events=events, present_track_ids=present):
            if tr.kind == "opened":
                opened = tr
            else:
                closed = tr
    assert opened is not None and closed is not None
    assert opened.track_id == closed.track_id
    assert closed.reason == "track_lost"
    assert opened.rule_id == "no_helmet"
