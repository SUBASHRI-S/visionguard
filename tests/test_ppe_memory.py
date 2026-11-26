import pytest

from incidents.memory import MemoryConfig, MemoryConfigError, PPEMemory
from incidents.temporal import IncidentConfigError, TemporalConfig
from safety.contracts import Box, Detection, PPEStatus


def status(index=0, helmet="yes", vest="yes", reasons=()):
    return PPEStatus(person_index=index, person=Detection("person", 0.9, Box(0, 0, 100, 300)),
                     helmet=helmet, vest=vest, reasons=tuple(reasons))


# ---------------------------------------------------------------- memory

def test_recently_seen_item_is_not_reported_missing():
    mem = PPEMemory(MemoryConfig(5.0))
    mem.apply([status(vest="yes")], {0: 7}, timestamp=1.0)
    (out,) = mem.apply([status(vest="no")], {0: 7}, timestamp=2.5)
    assert out.vest == "unknown" and any("vest not detected now but seen 1.5s ago" in r for r in out.reasons)
    assert mem.overrides == 1


def test_never_seen_item_stays_no():
    mem = PPEMemory()
    (out,) = mem.apply([status(vest="no")], {0: 7}, timestamp=1.0)
    assert out.vest == "no" and mem.overrides == 0


def test_memory_expires():
    mem = PPEMemory(MemoryConfig(5.0))
    mem.apply([status(vest="yes")], {0: 7}, timestamp=1.0)
    (out,) = mem.apply([status(vest="no")], {0: 7}, timestamp=6.5)    # 5.5 s later
    assert out.vest == "no"


def test_boundary_is_inclusive_of_the_horizon():
    mem = PPEMemory(MemoryConfig(5.0))
    mem.apply([status(vest="yes")], {0: 7}, timestamp=1.0)
    assert mem.apply([status(vest="no")], {0: 7}, timestamp=6.0)[0].vest == "unknown"


def test_items_are_remembered_separately():
    mem = PPEMemory()
    mem.apply([status(helmet="yes", vest="no")], {0: 7}, timestamp=1.0)
    (out,) = mem.apply([status(helmet="no", vest="no")], {0: 7}, timestamp=2.0)
    assert out.helmet == "unknown" and out.vest == "no"               # only the helmet was ever seen


def test_people_are_remembered_separately():
    mem = PPEMemory()
    mem.apply([status(0, vest="yes"), status(1, vest="no")], {0: 1, 1: 2}, timestamp=1.0)
    out = mem.apply([status(0, vest="no"), status(1, vest="no")], {0: 1, 1: 2}, timestamp=2.0)
    assert [o.vest for o in out] == ["unknown", "no"]


def test_person_without_track_id_is_untouched():
    mem = PPEMemory()
    mem.apply([status(vest="yes")], {0: 7}, timestamp=1.0)
    (out,) = mem.apply([status(vest="no")], {}, timestamp=2.0)
    assert out.vest == "no"


def test_yes_and_unknown_pass_through_and_input_is_not_modified():
    mem = PPEMemory()
    original = status(helmet="unknown", vest="no")
    mem.apply([status(vest="yes")], {0: 7}, timestamp=1.0)
    (out,) = mem.apply([original], {0: 7}, timestamp=2.0)
    assert out.helmet == "unknown" and original.vest == "no"


def test_memory_refreshes_while_the_item_keeps_being_seen():
    mem = PPEMemory(MemoryConfig(2.0))
    for t in (0.0, 1.5, 3.0, 4.5):
        mem.apply([status(vest="yes")], {0: 7}, timestamp=t)
    assert mem.apply([status(vest="no")], {0: 7}, timestamp=6.0)[0].vest == "unknown"


def test_old_entries_are_dropped_so_memory_stays_small():
    mem = PPEMemory(MemoryConfig(1.0))
    for track in range(100):
        mem.apply([status(vest="yes")], {0: track}, timestamp=track * 10.0)
    assert len(mem._last_yes) <= 2


def test_time_going_backwards_forgets_instead_of_guessing():
    mem = PPEMemory()
    mem.apply([status(vest="yes")], {0: 7}, timestamp=10.0)
    assert mem.apply([status(vest="no")], {0: 7}, timestamp=2.0)[0].vest == "no"


@pytest.mark.parametrize("bad", [0, -1, 601, "5", True, None])
def test_bad_memory_config(bad):
    with pytest.raises(MemoryConfigError):
        MemoryConfig(bad)


# ---------------------------------------------------------------- timing in seconds

def test_from_seconds_converts_with_fps():
    cfg = TemporalConfig.from_seconds(fps=25, open_after_seconds=3, close_after_seconds=2, uncertain_expire_seconds=2.5)
    assert (cfg.open_after_frames, cfg.close_after_frames, cfg.uncertain_expire_frames) == (75, 50, 63)
    assert cfg.track_lost_seconds == 3.0


def test_from_seconds_accounts_for_frame_stride():
    cfg = TemporalConfig.from_seconds(fps=50, frame_stride=2, open_after_seconds=3)
    assert cfg.open_after_frames == 75            # 25 processed frames per second


def test_from_seconds_never_gives_zero_frames():
    assert TemporalConfig.from_seconds(fps=1, open_after_seconds=0.1).open_after_frames == 1


@pytest.mark.parametrize("kw", [dict(fps=0), dict(fps=500), dict(fps=25, frame_stride=0),
                                dict(fps=25, open_after_seconds=0), dict(fps=25, close_after_seconds=-1),
                                dict(fps=25, open_after_seconds=True)])
def test_from_seconds_rejects_bad_input(kw):
    with pytest.raises(IncidentConfigError):
        TemporalConfig.from_seconds(**kw)
