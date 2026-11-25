import logging

import pytest

from safety.config import SafetyConfigError
from safety.contracts import Box, Detection
from safety.engine import ALL_RULES, SafetyEngine
from safety.zones import Zone

W, H = 1000, 800


def det(name, x1, y1, x2, y2, conf=0.9):
    return Detection(name, conf, Box(x1, y1, x2, y2))


def frame(**extra):
    base = dict(camera_id="cam1", frame_index=7, timestamp=1.5, frame_width=W, frame_height=H)
    base.update(extra)
    return base


PERSON = det("person", 400, 200, 500, 500)
HELMET = det("helmet", 430, 205, 470, 240)
VEST = det("vest", 410, 270, 490, 380)
ZONE = Zone("press", ((0.3, 0.3), (0.7, 0.3), (0.7, 0.8), (0.3, 0.8)))  # x 300..700, y 240..640


def events_for(detections, rules=None, **extra):
    engine = SafetyEngine.from_config({"enabled_rules": rules} if rules else None)
    return engine.process(detections, **frame(**extra))


def by_rule(events):
    return {e.rule_id: e for e in events}


# ---------- PPE rules ----------
def test_fully_equipped_worker_gives_no_events():
    assert events_for([PERSON, HELMET, VEST]) == []


def test_missing_helmet_is_a_violation_with_context():
    (event,) = events_for([PERSON, VEST], rules=["no_helmet"])
    assert event.outcome == "violation" and event.severity == "high"
    assert (event.camera_id, event.frame_index, event.timestamp) == ("cam1", 7, 1.5)
    assert event.details["box"] == (400, 200, 500, 500)
    assert event.track_id is None


def test_missing_vest_is_a_violation():
    (event,) = events_for([PERSON, HELMET], rules=["no_vest"])
    assert event.rule_id == "no_vest" and event.outcome == "violation"


def test_tiny_person_is_uncertain_never_a_violation():
    tiny = det("person", 100, 100, 130, 150)
    events = events_for([tiny], rules=["no_helmet", "no_vest"])
    assert {e.outcome for e in events} == {"uncertain"}
    assert "person too small to judge" in events[0].details["reasons"]


def test_track_id_is_attached_when_known():
    (event,) = events_for([PERSON, VEST], rules=["no_helmet"], track_ids={0: 42})
    assert event.track_id == 42


# ---------- zone rule ----------
def test_person_in_zone_is_a_violation():
    (event,) = events_for([PERSON, HELMET, VEST], rules=["zone_intrusion"], zones=(ZONE,))
    assert event.outcome == "violation" and event.details["zone"] == "press"


def test_person_outside_zone_gives_nothing():
    person = det("person", 800, 200, 900, 500)
    assert events_for([person], rules=["zone_intrusion"], zones=(ZONE,)) == []


def test_zone_rule_uncertain_when_feet_cut_off():
    person = det("person", 400, 400, 500, 800)
    (event,) = events_for([person], rules=["zone_intrusion"], zones=(ZONE,))
    assert event.outcome == "uncertain"


def test_no_zones_means_no_zone_events():
    assert events_for([PERSON], rules=["zone_intrusion"]) == []


def test_forklift_operator_is_not_an_intruder():
    forklift = det("forklift", 350, 150, 650, 600)
    events = events_for([PERSON, forklift], rules=["zone_intrusion"], zones=(ZONE,))
    assert events == []


# ---------- proximity rule ----------
def test_close_to_forklift_is_a_violation():
    forklift = det("forklift", 100, 100, 300, 300)
    person = det("person", 320, 150, 400, 350)
    (event,) = events_for([person, forklift], rules=["forklift_proximity"])
    assert event.outcome == "violation"
    assert event.details["ratio"] < 1.0 and event.details["forklift_index"] == 0


def test_far_from_forklift_gives_nothing():
    forklift = det("forklift", 100, 100, 300, 300)
    person = det("person", 800, 150, 900, 450)
    assert events_for([person, forklift], rules=["forklift_proximity"]) == []


def test_proximity_uncertain_for_small_person():
    forklift = det("forklift", 100, 100, 300, 300)
    person = det("person", 310, 250, 330, 300)
    (event,) = events_for([person, forklift], rules=["forklift_proximity"])
    assert event.outcome == "uncertain"


# ---------- engine ----------
def test_all_rules_run_by_default_and_events_follow_rule_order():
    forklift = det("forklift", 100, 100, 300, 300)
    person = det("person", 320, 250, 420, 550)  # no PPE, beside the forklift, feet inside the zone
    events = events_for([person, forklift], zones=(ZONE,))
    assert [e.rule_id for e in events] == list(ALL_RULES)
    assert {e.outcome for e in events} == {"violation"}


def test_enabled_rules_subset_respected():
    engine = SafetyEngine.from_config({"enabled_rules": ["no_vest"]})
    assert [r.rule_id for r in engine.rules] == ["no_vest"]


def test_config_overrides_reach_the_rules():
    engine = SafetyEngine.from_config({"ppe": {"min_person_height_px": 400}})
    events = engine.process([PERSON, VEST], **frame())  # 300 px tall now counts as too small
    assert [e.outcome for e in events] == ["uncertain"]


@pytest.mark.parametrize(
    "bad",
    [
        {"enabled_rules": ["no_helmett"]},
        {"enabled_rules": []},
        {"enabled_rules": "no_helmet"},
        {"surprise": 1},
        {"ppe": {"head_region_fraction": 5}},
        {"proximity": {"warn_ratioo": 1}},
    ],
)
def test_bad_engine_config_rejected(bad):
    with pytest.raises(SafetyConfigError):
        SafetyEngine.from_config(bad)


def test_duplicate_rule_ids_rejected():
    from safety.rules import NoHelmetRule

    with pytest.raises(SafetyConfigError):
        SafetyEngine([NoHelmetRule(), NoHelmetRule()])


def test_a_crashing_rule_does_not_stop_the_others(caplog):
    class Boom:
        rule_id = "boom"

        def evaluate(self, ctx):
            raise RuntimeError("kaboom")

    from safety.rules import NoHelmetRule

    engine = SafetyEngine([Boom(), NoHelmetRule()])
    with caplog.at_level(logging.ERROR):
        events = engine.process([PERSON, VEST], **frame())
    assert [e.rule_id for e in events] == ["no_helmet"]
    assert "rule boom failed" in caplog.text


def test_empty_frame_gives_no_events():
    assert events_for([]) == []
