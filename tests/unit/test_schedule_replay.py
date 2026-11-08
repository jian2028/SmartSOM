"""Current recordings are read without replaying simulator physics."""

from smartsom.trace.production import Playback
from tests.helpers.recording import recording


def recorded_hand(root):
    return recording(root)


def test_current_frames_remain_byte_identical_after_reading(tmp_path):
    root = tmp_path / "run"
    recorded_hand(root)
    before = {p.name: p.read_bytes() for p in root.iterdir()}
    playback = Playback(root)
    assert playback.last_tick > 8
    assert any(
        any(m["job"] is not None for m in row["state"]["machines"].values())
        for row in (playback.row(t) for t in range(playback.last_tick + 1))
    )
    for tick in (8, 0, 5, 1, playback.last_tick):
        assert playback.row(tick)["state"]["tick"] == tick
    assert before == {p.name: p.read_bytes() for p in root.iterdir()}
