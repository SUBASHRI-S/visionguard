from evaluation.safety_check import compare_image, summarize, to_markdown
from safety.contracts import Box, Detection, PPEStatus


def status(index, x1, y1, x2, y2, helmet="yes", vest="yes"):
    person = Detection("person", 0.9, Box(x1, y1, x2, y2))
    return PPEStatus(person_index=index, person=person, helmet=helmet, vest=vest)


def test_matching_is_by_overlap_and_one_to_one():
    truth = [status(0, 0, 0, 100, 300), status(1, 200, 0, 300, 300)]
    pred = [status(0, 205, 0, 305, 300), status(1, 2, 0, 102, 300)]  # listed in the other order
    result = compare_image(truth, pred)
    assert len(result.matched) == 2
    assert result.matched[0][0].person.box.x1 in (0, 200)
    for t, p in result.matched:
        assert abs(t.person.box.x1 - p.person.box.x1) < 10
    assert result.truth_unmatched == [] and result.pred_unmatched == []


def test_missed_and_extra_persons_counted():
    truth = [status(0, 0, 0, 100, 300), status(1, 500, 0, 600, 300)]
    pred = [status(0, 2, 0, 102, 300), status(1, 800, 0, 900, 300)]
    result = compare_image(truth, pred)
    assert len(result.matched) == 1
    assert len(result.truth_unmatched) == 1 and len(result.pred_unmatched) == 1


def test_low_overlap_is_not_a_match():
    result = compare_image([status(0, 0, 0, 100, 300)], [status(0, 90, 0, 190, 300)])
    assert result.matched == []


def test_disagreements_count_each_item():
    truth = [status(0, 0, 0, 100, 300, helmet="no", vest="yes")]
    pred = [status(0, 0, 0, 100, 300, helmet="yes", vest="unknown")]
    assert compare_image(truth, pred).disagreements == 2


def test_summary_table_tracks_missed_violations():
    truth = [status(0, 0, 0, 100, 300, helmet="no"), status(1, 500, 0, 600, 300, helmet="no")]
    pred = [status(0, 0, 0, 100, 300, helmet="yes")]  # a real violation hidden; another person missed
    summary = summarize([compare_image(truth, pred)])
    row = summary["table"]["helmet"]["no"]
    assert row == {"yes": 1, "person_missed": 1}
    assert summary["persons_matched"] == 1 and summary["persons_missed"] == 1


def test_summary_counts_caught_violations_and_false_violations():
    truth = [status(0, 0, 0, 100, 300, helmet="no"), status(1, 500, 0, 600, 300, helmet="yes")]
    pred = [status(0, 0, 0, 100, 300, helmet="no"), status(1, 500, 0, 600, 300, helmet="no")]
    table = summarize([compare_image(truth, pred)])["table"]["helmet"]
    assert table["no"] == {"no": 1}
    assert table["yes"] == {"no": 1}


def test_markdown_has_the_key_sections():
    meta = {"run_name": "r", "split": "val", "dataset_version": "abc", "conf": 0.35,
            "imgsz": 960, "iou": 0.5}
    summary = summarize([compare_image([status(0, 0, 0, 100, 300)], [status(0, 0, 0, 100, 300)])])
    text = to_markdown(summary, meta, {"no_helmet": {"violation": 2}})
    assert "## Helmet" in text and "## Vest" in text
    assert "| no_helmet | 2 | 0 |" in text
    assert "not whether the rules are right" in text


def test_empty_input_does_not_crash():
    summary = summarize([])
    assert summary["images"] == 0
    assert "Helmet" in to_markdown(summary, {"run_name": "r", "split": "val", "dataset_version": "x",
                                              "conf": 0.3, "imgsz": 640, "iou": 0.5}, {})


def test_missing_cause_separates_detector_misses_from_rule_rejections():
    from evaluation.safety_check import false_violation_causes, missing_cause

    person = status(0, 400, 200, 500, 500, helmet="no")
    far_helmet = Detection("helmet", 0.9, Box(430, 300, 470, 335))  # on the person, not at the head
    elsewhere = Detection("helmet", 0.9, Box(800, 100, 840, 140))
    assert missing_cause(person, [far_helmet], "helmet") == "rejected_by_rules"
    assert missing_cause(person, [elsewhere], "helmet") == "not_detected"
    assert missing_cause(person, [], "helmet") == "not_detected"

    truth = [status(0, 400, 200, 500, 500, helmet="yes")]
    comparison = compare_image(truth, [person])
    causes = false_violation_causes({"a.jpg": comparison}, {"a.jpg": [far_helmet]})
    assert causes["helmet"] == {"rejected_by_rules": 1}
    assert causes["vest"] == {}


def test_report_includes_causes_section_when_given():
    meta = {"run_name": "r", "split": "val", "dataset_version": "x", "conf": 0.3, "imgsz": 640, "iou": 0.5}
    text = to_markdown(summarize([]), meta, {}, {"helmet": {"not_detected": 3}, "vest": {}})
    assert "Why compliant people were called violators" in text
    assert "| helmet | 3 | 0 |" in text


def _with_helmet(status_obj, confidence):
    from dataclasses import replace

    helmet = Detection("helmet", confidence, Box(430, 205, 470, 240))
    return replace(status_obj, helmet="yes", helmet_detection=helmet)


def test_confidence_profile_splits_wrong_from_right_acceptances():
    from evaluation.safety_check import confidence_profile

    truth = [status(0, 0, 0, 100, 300, helmet="no"), status(1, 500, 0, 600, 300, helmet="yes")]
    pred = [_with_helmet(status(0, 0, 0, 100, 300), 0.40), _with_helmet(status(1, 500, 0, 600, 300), 0.90)]
    profile = confidence_profile({"a.jpg": compare_image(truth, pred)})
    assert profile["helmet"] == {"wrongly_accepted": [0.4], "correctly_accepted": [0.9]}
    assert profile["vest"] == {"wrongly_accepted": [], "correctly_accepted": []}


def test_profile_section_in_report():
    meta = {"run_name": "r", "split": "val", "dataset_version": "x", "conf": 0.3, "imgsz": 640, "iou": 0.5}
    profile = {"helmet": {"wrongly_accepted": [0.4, 0.5], "correctly_accepted": [0.9]},
               "vest": {"wrongly_accepted": [], "correctly_accepted": []}}
    text = to_markdown(summarize([]), meta, {}, None, profile)
    assert "Confidence of the matched item" in text
    assert "| helmet | 0.40 / 0.50 / 0.50 (2) | 0.90 / 0.90 / 0.90 (1) |" in text
