"""Archived physics is read, never reexecuted by the current simulator."""

import shutil
from pathlib import Path

from smartsom.trace.production import Playback


def recorded_hand(root):
    shutil.copytree(
        Path(__file__).resolve().parents[1] / "fixtures/historical-hand", root
    )


def test_historical_frames_remain_byte_identical_after_reading(tmp_path):
    root = tmp_path / "run"
    recorded_hand(root)
    before = {p.name: p.read_bytes() for p in root.iterdir()}
    playback = Playback(root)
    assert playback.last_tick == 8
    assert playback.row(3)["state"]["machines"]["machine"]["job"] is not None
    for tick in (8, 0, 5, 1, 8):
        assert playback.row(tick)["state"]["tick"] == tick
    assert before == {p.name: p.read_bytes() for p in root.iterdir()}
