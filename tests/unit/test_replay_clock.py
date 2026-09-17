"""Deterministic presentation-clock contracts without Qt or wall-clock sleeps."""

import pytest

from smartsom.studio.replay_clock import ReplayClock


def test_pause_resume_speed_and_end():
    clock = ReplayClock(12)
    clock.resume(now=0)
    assert clock.sample(now=0.04) == pytest.approx(0.4)
    clock.pause(now=0.04)
    assert clock.sample(now=100) == pytest.approx(0.4)
    clock.resume(now=100)
    clock.speed(0, now=100)
    assert clock.sample(now=100.2) == pytest.approx(0.9)
    clock.speed(6, now=100.2)
    assert clock.sample(now=101) == 12
    assert not clock.playing
    clock.resume(now=102)
    assert not clock.playing


@pytest.mark.parametrize(
    "position,back,forward", [(10.4, 10, 11), (10, 9, 11), (0, 0, 1), (12, 11, 12)]
)
def test_integer_and_fractional_steps(position, back, forward):
    clock = ReplayClock(12, position=position, playing=True)
    clock.step(-1)
    assert clock.position == back and not clock.playing
    clock.position = position
    clock.step(1)
    assert clock.position == forward and not clock.playing


def test_seek_and_single_initial_frame():
    clock = ReplayClock(0)
    clock.resume(now=0)
    assert not clock.playing
    with pytest.raises(ValueError):
        clock.seek(1)
    with pytest.raises(ValueError):
        clock.seek(0.5)
