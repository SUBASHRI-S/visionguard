import pytest

from incidents.temporal import IncidentConfigError, IncidentTracker, TemporalConfig
from safety.events import Event

FPS = 10.0


def make_event(rule="no_helmet", track=1, outcome="violation", frame=0, camera="cam1", severity="high"):
    return Event(rule, camera, frame, frame / FPS, outcome, severity, 0, track, {})


def run(sequence, config=None, track=1, rule="no_helmet", tracker=None, start=0, camera="cam1"):
    """V violating, U uncertain, C clear (person visible), A absent. One character per frame."""
    tracker = tracker or IncidentTracker(config or TemporalConfig(
        open_after_frames=3, close_after_frames=4, track_lost_seconds=1.0, uncertain_expire_frames=5))
    out = []
    for i, ch in enumerate(sequence, start=start):
        events = []
        if ch == "V":
            events = [make_event(rule, track, "violation", i, camera)]
        elif ch == "U":
            events = [make_event(rule, track, "uncertain", i, camera)]
        present = set() if ch == "A" else {track}
        out += tracker.update(camera_id=camera, frame_index=i, timestamp=i / FPS,
                              events=events, present_track_ids=present)
    return out, tracker


def kinds(transitions):
    return [(t.kind, t.frame_index, t.reason) for t in transitions]


# ---------- opening ----------
def test_opens_only_after_n_consecutive_violating_frames():
    out, _ = run("VV")
    assert out == []
    out, _ = run("VVV")
    assert kinds(out) == [("opened", 2, None)]


def test_opened_transition_reports_the_start_of_the_streak():
    out, _ = run("CCVVV")
    (opened,) = out
    assert opened.frame_index == 4 and opened.started_frame == 2
    assert opened.started_timestamp == pytest.approx(0.2)


def test_flicker_never_opens_an_incident():
    out, _ = run("VCVCVCVCVCVC")
    assert out == []


def test_a_clear_frame_resets_the_streak():
    out, _ = run("VVCVV")
    assert out == []
    out, _ = run("VVCVVV")
    assert kinds(out) == [("opened", 5, None)]


def test_uncertain_frames_hold_the_streak_but_do_not_count():
    out, _ = run("VVUUV")  # only 3 violating frames, interrupted by uncertainty
    assert kinds(out) == [("opened", 4, None)]
    out, _ = run("VUUUU")
    assert out == []


def test_long_uncertainty_cancels_a_half_built_streak():
    # uncertain_expire_frames=5: six uncertain frames in a row wipe the streak
    out, _ = run("VVUUUUUUVV")
    assert out == []


# ---------- while open ----------
def test_more_violations_while_open_do_not_open_again():
    out, _ = run("VVVVVVVV")
    assert [t.kind for t in out] == ["opened"]


def test_closes_after_m_clear_frames():
    out, _ = run("VVVCCCC")
    assert kinds(out) == [("opened", 2, None), ("closed", 6, "cleared")]


def test_a_short_gap_does_not_close_it():
    out, _ = run("VVVCCCVCCC")  # 3 clear, violation resets, 3 clear: never reaches 4
    assert [t.kind for t in out] == ["opened"]


def test_a_detector_dropout_inside_a_long_violation_stays_one_incident():
    out, _ = run("VVVVVVCCVVVVVV")
    assert [t.kind for t in out] == ["opened"]


def test_uncertain_frames_never_close_an_incident():
    out, tracker = run("VVV" + "U" * 50)
    assert [t.kind for t in out] == ["opened"]
    assert tracker.is_open("cam1", "no_helmet", 1)


def test_uncertain_does_not_advance_or_reset_clear_count():
    out, _ = run("VVVCCUCC")  # 4 clear frames in total, uncertain in between holds
    assert kinds(out) == [("opened", 2, None), ("closed", 7, "cleared")]


def test_closed_transition_carries_counts_and_last_violation():
    out, _ = run("VVVUVCCCC")
    closed = out[-1]
    assert closed.kind == "closed"
    assert closed.violating_frames == 4 and closed.uncertain_frames == 1
    assert closed.event.frame_index == 4
    assert closed.started_frame == 0


# ---------- track lost ----------
def test_open_incident_closes_when_track_is_gone_too_long():
    out, _ = run("VVV" + "A" * 15)  # 1.0 s at 10 fps = more than 10 absent frames
    assert out[-1].kind == "closed" and out[-1].reason == "track_lost"


def test_short_absence_keeps_incident_open_and_it_can_continue():
    out, tracker = run("VVVAAAVV")
    assert [t.kind for t in out] == ["opened"]
    assert tracker.is_open("cam1", "no_helmet", 1)


def test_absent_halfway_through_a_streak_drops_it():
    out, _ = run("VV" + "A" * 15 + "V")
    assert out == []


# ---------- independence ----------
def test_tracks_and_rules_are_independent():
    tracker = IncidentTracker(TemporalConfig(open_after_frames=2, close_after_frames=2,
                                             track_lost_seconds=1.0, uncertain_expire_frames=5))
    out = []
    for i in range(3):
        events = [make_event("no_helmet", 1, frame=i), make_event("no_vest", 1, frame=i),
                  make_event("no_helmet", 2, frame=i)]
        out += tracker.update(camera_id="cam1", frame_index=i, timestamp=i / FPS,
                              events=events, present_track_ids={1, 2})
    assert sorted((t.rule_id, t.track_id) for t in out) == [
        ("no_helmet", 1), ("no_helmet", 2), ("no_vest", 1)]


def test_cameras_are_independent():
    tracker = IncidentTracker(TemporalConfig(open_after_frames=2, close_after_frames=2,
                                             track_lost_seconds=1.0, uncertain_expire_frames=5))
    a, _ = run("VV", tracker=tracker, camera="camA")
    b, _ = run("VC", tracker=tracker, camera="camB")
    assert [t.camera_id for t in a] == ["camA"] and b == []


def test_violation_beats_uncertain_for_the_same_key_in_one_frame():
    tracker = IncidentTracker(TemporalConfig(open_after_frames=1))
    events = [make_event(outcome="uncertain"), make_event(outcome="violation")]
    out = tracker.update(camera_id="cam1", frame_index=0, timestamp=0.0, events=events,
                         present_track_ids={1})
    assert [t.kind for t in out] == ["opened"]


# ---------- repeats, end of stream, bad input ----------
def test_a_new_incident_after_closing_gets_a_new_episode_number():
    out, _ = run("VVVCCCCVVVCCCC")
    assert [(t.kind, t.episode) for t in out] == [
        ("opened", 1), ("closed", 1), ("opened", 2), ("closed", 2)]


def test_finish_closes_open_incidents_and_drops_streaks():
    out, tracker = run("VVV")
    closing = tracker.finish(frame_index=3, timestamp=0.3)
    assert [(t.kind, t.reason) for t in closing] == [("closed", "stream_end")]
    assert tracker.open_keys == []
    out, tracker = run("VV")
    assert tracker.finish(2, 0.2) == []


def test_events_without_a_track_id_are_ignored_and_counted():
    tracker = IncidentTracker(TemporalConfig(open_after_frames=1))
    out = tracker.update(camera_id="cam1", frame_index=0, timestamp=0.0,
                         events=[make_event(track=None)], present_track_ids=set())
    assert out == [] and tracker.ignored_untracked_events == 1


def test_event_for_the_wrong_camera_is_an_error():
    tracker = IncidentTracker()
    with pytest.raises(ValueError):
        tracker.update(camera_id="cam1", frame_index=0, timestamp=0.0,
                       events=[make_event(camera="camX")], present_track_ids={1})


def test_realistic_noisy_stream_gives_exactly_one_incident():
    # a real violation (V) with detector dropouts (C), uncertainty (U) and a single early blip
    out, _ = run("VCCCCVVVUVVCVVVVUUVVVVCCCCC")
    assert [t.kind for t in out] == ["opened", "closed"]


# ---------- config ----------
def test_defaults_are_valid_and_overridable():
    assert TemporalConfig().open_after_frames == 8
    assert TemporalConfig.from_dict({"close_after_frames": 5}).close_after_frames == 5


@pytest.mark.parametrize("bad", [
    {"open_after_frames": 0}, {"open_after_frames": 2.5}, {"close_after_frames": -1},
    {"track_lost_seconds": -0.1}, {"uncertain_expire_frames": "a"}, {"surprise": 1},
])
def test_bad_config_rejected(bad):
    with pytest.raises(IncidentConfigError):
        TemporalConfig.from_dict(bad)
