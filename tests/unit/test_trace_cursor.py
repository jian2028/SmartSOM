"""Independent recorded readers seek exact committed ticks without running physics."""

from copy import deepcopy

import pytest
from test_schedule_replay import recorded_hand

from smartsom.trace.production import Playback


def test_independent_readers_and_detached_rows_preserve_the_same_episode(tmp_path):
    root = tmp_path / "run"
    recorded_hand(root)
    a, b = Playback(root), Playback(root)
    expected = [deepcopy(a.row(t)) for t in range(9)]
    for tick in (0, 8, 3, 7, 1, 0, 4):
        assert a.row(tick) == b.row(tick) == expected[tick]
        row = a.row(tick)
        row["state"]["agvs"]["agv"]["cell"][0] = 999
        assert a.row(tick) == b.row(tick) == expected[tick]


@pytest.mark.parametrize("tick", [-1, 9, True, False, 0.0, 1.0, "0", None])
def test_invalid_recorded_tick_does_not_mutate_reader_or_file(tmp_path, tick):
    root = tmp_path / "run"
    recorded_hand(root)
    playback = Playback(root)
    before = (
        deepcopy(playback.manifest),
        tuple(playback.offsets),
        (root / "trace.jsonl").read_bytes(),
    )
    with pytest.raises(ValueError, match="outside"):
        playback.row(tick)
    assert (
        playback.manifest,
        tuple(playback.offsets),
        (root / "trace.jsonl").read_bytes(),
    ) == before
