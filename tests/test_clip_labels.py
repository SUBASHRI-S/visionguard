import copy
import json

import pytest

from evaluation.clip_labels import ClipLabelError, load_manifest, parse_manifest, validate_manifest


def good_manifest() -> dict:
    return {
        "clips": [
            {
                "clip_id": "site_a_01",
                "path": "datasets/raw/real_clips/site_a_01.mp4",
                "camera_id": "cam1",
                "fps": 25,
                "duration_s": 40,
                "subset": "final",
                "source_type": "own_recording",
                "consent_obtained": True,
                "licence": "own work",
                "incidents": [
                    {"rule_id": "no_helmet", "start_s": 3.0, "end_s": 9.5},
                    {"rule_id": "zone_intrusion", "start_s": 20, "end_s": 26, "ambiguous": True},
                ],
            },
            {
                "clip_id": "site_b_02",
                "path": "datasets/raw/real_clips/site_b_02.mp4",
                "camera_id": "cam2",
                "fps": 30,
                "subset": "tune",
                "source_type": "third_party",
                "url": "https://example.org/clip",
                "licence": "CC BY 4.0",
                "incidents": [],
            },
        ]
    }


def test_valid_manifest_parses():
    clips = parse_manifest(good_manifest())
    assert [c.clip_id for c in clips] == ["site_a_01", "site_b_02"]
    assert clips[0].incidents[1].ambiguous is True
    assert clips[1].incidents == []
    assert clips[0].duration_s == 40.0


def mutated(change) -> list[str]:
    data = copy.deepcopy(good_manifest())
    change(data)
    return validate_manifest(data)


def test_empty_manifest_is_invalid():
    assert validate_manifest({}) != []
    assert validate_manifest({"clips": []}) != []


def test_own_recording_requires_consent():
    errors = mutated(lambda d: d["clips"][0].update(consent_obtained=False))
    assert any("consent" in e for e in errors)


def test_third_party_requires_url():
    errors = mutated(lambda d: d["clips"][1].pop("url"))
    assert any("url" in e for e in errors)


def test_unknown_rule_is_rejected():
    errors = mutated(lambda d: d["clips"][0]["incidents"][0].update(rule_id="nonsense"))
    assert any("rule_id" in e for e in errors)


def test_interval_must_be_ordered():
    errors = mutated(lambda d: d["clips"][0]["incidents"][0].update(start_s=10, end_s=5))
    assert any("start_s < end_s" in e for e in errors)


def test_interval_beyond_duration_is_rejected():
    errors = mutated(lambda d: d["clips"][0]["incidents"][0].update(end_s=99))
    assert any("beyond duration" in e for e in errors)


def test_overlapping_same_rule_is_rejected():
    def change(d):
        d["clips"][0]["incidents"].append({"rule_id": "no_helmet", "start_s": 8, "end_s": 12})

    assert any("overlaps" in e for e in mutated(change))


def test_overlap_between_different_rules_is_allowed():
    def change(d):
        d["clips"][0]["incidents"].append({"rule_id": "no_vest", "start_s": 4, "end_s": 6})

    assert mutated(change) == []


def test_duplicate_clip_id_is_rejected():
    errors = mutated(lambda d: d["clips"][1].update(clip_id="site_a_01"))
    assert any("duplicate" in e for e in errors)


def test_bad_subset_and_fps_are_rejected():
    errors = mutated(lambda d: d["clips"][0].update(subset="maybe", fps=0))
    assert any("subset" in e for e in errors)
    assert any("fps" in e for e in errors)


def test_all_problems_are_reported_together():
    data = good_manifest()
    data["clips"][0]["consent_obtained"] = False
    data["clips"][1].pop("url")
    with pytest.raises(ClipLabelError) as info:
        parse_manifest(data)
    assert len(info.value.errors) == 2


def test_load_manifest_from_file_and_bad_json(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(good_manifest()))
    assert len(load_manifest(path)) == 2
    path.write_text("{not json")
    with pytest.raises(ClipLabelError):
        load_manifest(path)
