import math

import pytest

from safety.association import associate
from safety.config import ProximityConfig, SafetyConfigError, ZoneConfig
from safety.contracts import Box, Detection
from safety.proximity import check_proximity, distance_to_ground_edge
from safety.zones import Zone, ZoneError, person_in_zone, point_in_polygon

W, H = 1000, 800


def det(name, x1, y1, x2, y2, conf=0.9):
    return Detection(name, conf, Box(x1, y1, x2, y2))


SQUARE = ((0.2, 0.2), (0.6, 0.2), (0.6, 0.6), (0.2, 0.6))
L_SHAPE = ((0.0, 0.0), (0.6, 0.0), (0.6, 0.3), (0.3, 0.3), (0.3, 0.6), (0.0, 0.6))


# ---------- point in polygon ----------
@pytest.mark.parametrize(
    "x,y,expected",
    [(0.4, 0.4, True), (0.1, 0.4, False), (0.7, 0.4, False), (0.4, 0.7, False), (0.4, 0.1, False)],
)
def test_point_in_square(x, y, expected):
    assert point_in_polygon(x, y, SQUARE) is expected


def test_points_on_edge_and_corner_count_as_inside():
    assert point_in_polygon(0.2, 0.4, SQUARE) is True
    assert point_in_polygon(0.2, 0.2, SQUARE) is True
    assert point_in_polygon(0.4, 0.6, SQUARE) is True


def test_concave_polygon_notch_is_outside():
    assert point_in_polygon(0.15, 0.45, L_SHAPE) is True   # in the vertical bar
    assert point_in_polygon(0.45, 0.15, L_SHAPE) is True   # in the horizontal bar
    assert point_in_polygon(0.45, 0.45, L_SHAPE) is False  # in the notch


# ---------- zone definition ----------
def test_zone_from_dict_ok():
    zone = Zone.from_dict({"name": "press", "polygon": [[0.1, 0.1], [0.5, 0.1], [0.5, 0.5]]})
    assert zone.name == "press" and len(zone.polygon) == 3


@pytest.mark.parametrize(
    "bad",
    [
        {"name": "", "polygon": SQUARE},
        {"name": "z", "polygon": ((0.1, 0.1), (0.5, 0.5))},
        {"name": "z", "polygon": ((0.1, 0.1), (0.5, 0.5), (0.9, 0.9))},  # a line, no area
        {"name": "z", "polygon": ((0.1, 0.1), (1.5, 0.1), (0.5, 0.5))},  # outside 0-1
        {"name": "z", "polygon": ((0.1, 0.1), (0.5, "a"), (0.5, 0.5))},
    ],
)
def test_bad_zone_rejected(bad):
    with pytest.raises(ZoneError):
        Zone(bad["name"], tuple(tuple(p) for p in bad["polygon"]))


def test_zone_from_dict_rejects_extra_keys():
    with pytest.raises(ZoneError):
        Zone.from_dict({"name": "z", "polygon": SQUARE, "colour": "red"})


# ---------- person in zone ----------
ZONE = Zone("z", SQUARE)  # in a 1000x800 frame: x 200..600, y 160..480


def test_person_with_feet_inside_zone():
    person = det("person", 300, 200, 400, 400)  # feet (350, 400)
    assert person_in_zone(person, ZONE, W, H).answer == "yes"


def test_person_with_feet_outside_zone():
    person = det("person", 700, 200, 800, 400)
    assert person_in_zone(person, ZONE, W, H).answer == "no"


def test_head_in_zone_but_feet_outside_is_outside():
    person = det("person", 300, 100, 400, 500)  # head at y=100..., feet y=500 are below the zone
    assert person_in_zone(person, ZONE, W, H).answer == "no"


def test_feet_cut_off_is_unknown():
    person = det("person", 300, 400, 400, 800)
    result = person_in_zone(person, ZONE, W, H)
    assert result.answer == "unknown"
    assert "feet may be cut off by the bottom of the frame" in result.reasons


def test_low_confidence_person_is_unknown():
    person = det("person", 300, 200, 400, 400, conf=0.2)
    assert person_in_zone(person, ZONE, W, H).answer == "unknown"


def test_zone_decision_survives_resolution_change():
    small = det("person", 300, 200, 400, 400)
    assert person_in_zone(small, ZONE, W, H).answer == "yes"
    big = det("person", 600, 400, 800, 800)  # same scene scaled x2; feet y=800 in a 1600-high frame
    assert person_in_zone(big, ZONE, W * 2, H * 2).answer == "yes"


# ---------- proximity ----------
def statuses_for(*detections):
    return associate(list(detections), W, H)


def test_distance_to_ground_edge_cases():
    forklift = Box(400, 100, 600, 300)
    assert distance_to_ground_edge((500, 300), forklift) == 0          # on the edge
    assert distance_to_ground_edge((500, 340), forklift) == 40         # straight below
    assert distance_to_ground_edge((300, 300), forklift) == 100        # level, to the left
    assert distance_to_ground_edge((300, 340), forklift) == pytest.approx(math.hypot(100, 40))


def test_person_close_to_forklift_is_near():
    forklift = det("forklift", 400, 100, 700, 300)
    person = det("person", 720, 150, 800, 350)  # foot (760, 350), height 200
    (result,) = check_proximity(statuses_for(person, forklift), [forklift], H)
    assert result.near == "yes"
    assert result.ratio == pytest.approx(math.hypot(60, 50) / 200)


def test_person_far_from_forklift_is_not_near():
    forklift = det("forklift", 100, 100, 300, 300)
    person = det("person", 800, 150, 900, 450)  # height 300, foot (850, 450)
    (result,) = check_proximity(statuses_for(person, forklift), [forklift], H)
    assert result.near == "no"
    assert result.ratio > 1.0


def test_operator_is_skipped():
    forklift = det("forklift", 300, 150, 700, 600)
    operator = det("person", 400, 200, 500, 500)
    assert check_proximity(statuses_for(operator, forklift), [forklift], H) == []


def test_small_person_gives_unknown_not_a_warning():
    forklift = det("forklift", 400, 100, 700, 300)
    tiny = det("person", 710, 250, 730, 300)  # 50 px tall, right beside the forklift
    (result,) = check_proximity(statuses_for(tiny, forklift), [forklift], H)
    assert result.near == "unknown" and result.ratio is None
    assert "person too small to measure against" in result.reasons


def test_feet_cut_off_gives_unknown():
    forklift = det("forklift", 400, 100, 700, 300)
    person = det("person", 720, 500, 800, 800)
    (result,) = check_proximity(statuses_for(person, forklift), [forklift], H)
    assert result.near == "unknown"


def test_one_result_per_person_forklift_pair():
    f1 = det("forklift", 100, 100, 300, 300)
    f2 = det("forklift", 700, 100, 900, 300)
    person = det("person", 400, 200, 500, 500)
    results = check_proximity(statuses_for(person, f1, f2), [f1, f2], H)
    assert [(r.person_index, r.forklift_index) for r in results] == [(0, 0), (0, 1)]


def test_ratio_is_scale_invariant():
    """The same scene at twice the resolution gives the same ratio (the point of dividing by height)."""
    forklift = det("forklift", 400, 100, 700, 300)
    person = det("person", 720, 150, 800, 350)
    a = check_proximity(statuses_for(person, forklift), [forklift], H)[0]
    big_f = det("forklift", 800, 200, 1400, 600)
    big_p = det("person", 1440, 300, 1600, 700)
    big_statuses = associate([big_p, big_f], W * 2, H * 2)
    b = check_proximity(big_statuses, [big_f], H * 2)[0]
    assert a.ratio == pytest.approx(b.ratio)


def test_no_forklifts_gives_no_results():
    person = det("person", 400, 200, 500, 500)
    assert check_proximity(statuses_for(person), [], H) == []


def test_warn_ratio_is_configurable():
    forklift = det("forklift", 100, 100, 300, 300)
    person = det("person", 800, 150, 900, 450)
    strict = ProximityConfig(warn_ratio=5.0)
    (result,) = check_proximity(statuses_for(person, forklift), [forklift], H, strict)
    assert result.near == "yes"


# ---------- config ----------
def test_new_configs_reject_bad_input():
    with pytest.raises(SafetyConfigError):
        ProximityConfig.from_dict({"warn_ratioo": 1})
    with pytest.raises(SafetyConfigError):
        ProximityConfig.from_dict({"warn_ratio": 0})
    with pytest.raises(SafetyConfigError):
        ZoneConfig.from_dict({"min_person_confidence": 2})
    assert ZoneConfig.from_dict(None).edge_margin_px == 5.0
