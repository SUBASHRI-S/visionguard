import pytest

from safety.association import associate
from safety.config import PPEConfig, SafetyConfigError
from safety.contracts import Box, Detection

W, H = 1000, 800


def det(name, x1, y1, x2, y2, conf=0.9):
    return Detection(name, conf, Box(x1, y1, x2, y2))


# A clear, standing person: 100 wide, 300 tall, away from every edge.
PERSON = det("person", 400, 200, 500, 500)
HELMET_ON_HEAD = det("helmet", 430, 205, 470, 240)   # inside the top 25% (200..275)
VEST_ON_TORSO = det("vest", 410, 270, 490, 380)      # torso region is y 260..395


def run(*detections, config=None):
    return associate(list(detections), W, H, config)


def test_helmet_and_vest_present():
    (status,) = run(PERSON, HELMET_ON_HEAD, VEST_ON_TORSO)
    assert (status.helmet, status.vest) == ("yes", "yes")
    assert status.reasons == ()


def test_clear_person_without_ppe_is_no():
    (status,) = run(PERSON)
    assert (status.helmet, status.vest) == ("no", "no")


def test_helmet_far_from_head_does_not_match():
    helmet_on_legs = det("helmet", 430, 450, 470, 485)
    (status,) = run(PERSON, helmet_on_legs)
    assert status.helmet == "no"


def test_helmet_in_hand_next_to_person_does_not_match():
    helmet_beside = det("helmet", 520, 210, 560, 245)  # centre outside the person box
    (status,) = run(PERSON, helmet_beside)
    assert status.helmet == "no"


def test_implausibly_big_helmet_rejected():
    huge = det("helmet", 400, 200, 500, 270)  # as wide as the person
    (status,) = run(PERSON, huge)
    assert status.helmet == "no"


def test_vest_must_overlap_torso():
    vest_elsewhere = det("vest", 410, 450, 490, 500)
    (status,) = run(PERSON, vest_elsewhere)
    assert status.vest == "no"


def test_one_helmet_goes_to_one_person_only():
    left = det("person", 400, 200, 500, 500)
    right = det("person", 470, 200, 570, 500)
    shared = det("helmet", 478, 205, 492, 235, conf=0.8)  # centre x=485 is in both heads
    statuses = run(left, right, shared)
    assert sorted(s.helmet for s in statuses) == ["no", "yes"]


def test_better_scored_helmet_wins_the_person():
    weak = det("helmet", 430, 205, 470, 240, conf=0.4)
    strong = det("helmet", 432, 206, 468, 240, conf=0.95)
    (status,) = run(PERSON, weak, strong)
    assert status.helmet == "yes"


def test_small_person_is_unknown_not_no():
    tiny = det("person", 100, 100, 130, 150)  # 50 px tall
    (status,) = run(tiny)
    assert (status.helmet, status.vest) == ("unknown", "unknown")
    assert "person too small to judge" in status.reasons


def test_positive_evidence_beats_small_size():
    person = det("person", 100, 100, 160, 170)  # 70 px tall, below the minimum
    helmet = det("helmet", 118, 102, 142, 118)
    (status,) = run(person, helmet)
    assert status.helmet == "yes"
    assert status.vest == "unknown"


def test_low_confidence_person_is_unknown():
    (status,) = run(det("person", 400, 200, 500, 500, conf=0.2))
    assert status.helmet == "unknown"
    assert "person confidence too low" in status.reasons


def test_head_cut_by_top_edge_is_unknown_for_helmet_only():
    person = det("person", 400, 0, 500, 300)
    (status,) = run(person)
    assert status.helmet == "unknown"
    assert status.vest == "no"


def test_torso_cut_by_side_edge_is_unknown_for_vest_only():
    person = det("person", 0, 200, 100, 500)
    (status,) = run(person)
    assert status.vest == "unknown"
    assert status.helmet == "no"


def test_mostly_hidden_person_is_unknown():
    front = det("person", 400, 200, 600, 700)
    hidden = det("person", 420, 220, 560, 640)  # almost entirely behind the first
    statuses = run(front, hidden)
    assert statuses[1].helmet == "unknown"
    assert "person mostly hidden by another person" in statuses[1].reasons


def test_person_inside_forklift_is_operator():
    forklift = det("forklift", 300, 150, 700, 600)
    (status,) = run(PERSON, forklift)
    assert status.is_operator is True


def test_person_beside_forklift_is_not_operator():
    forklift = det("forklift", 600, 150, 900, 600)
    (status,) = run(PERSON, forklift)
    assert status.is_operator is False


def test_no_persons_gives_empty_list():
    assert run(HELMET_ON_HEAD, VEST_ON_TORSO) == []


def test_statuses_follow_person_order():
    a = det("person", 50, 200, 150, 500)
    b = det("person", 400, 200, 500, 500)
    statuses = run(a, b, HELMET_ON_HEAD)
    assert [s.person_index for s in statuses] == [0, 1]
    assert statuses[1].helmet == "yes"


def test_config_defaults_valid_and_overridable():
    assert PPEConfig().head_region_fraction == 0.25
    assert PPEConfig.from_dict({"min_person_height_px": 120}).min_person_height_px == 120


def test_config_rejects_unknown_key():
    with pytest.raises(SafetyConfigError):
        PPEConfig.from_dict({"head_regoin_fraction": 0.3})


@pytest.mark.parametrize(
    "bad",
    [
        {"head_region_fraction": 1.5},
        {"torso_top_fraction": 0.7, "torso_bottom_fraction": 0.6},
        {"helmet_min_width_ratio": 0.9},
        {"min_person_height_px": -1},
        {"vest_overlap_threshold": "high"},
    ],
)
def test_config_rejects_bad_values(bad):
    with pytest.raises(SafetyConfigError):
        PPEConfig.from_dict(bad)


def test_box_rejects_inverted_coordinates():
    with pytest.raises(ValueError):
        Box(10, 10, 5, 20)
