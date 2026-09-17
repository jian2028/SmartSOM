"""Presentation time, independent of recorded integer simulation time."""

import math
from dataclasses import dataclass
from time import monotonic

TICK_SECONDS = (0.4, 0.2, 0.1, 0.05, 0.025, 0.01, 0.001)


@dataclass
class ReplayClock:
    last_tick: int
    position: float = 0.0
    playing: bool = False
    seconds_per_tick: float = 0.1
    anchor: float = 0.0

    def sample(self, now=None):
        now = monotonic() if now is None else now
        if self.playing:
            self.position = min(
                self.last_tick,
                self.position + max(0, now - self.anchor) / self.seconds_per_tick,
            )
            if self.position == self.last_tick:
                self.playing = False
        self.anchor = now
        return self.position

    def pause(self, now=None):
        self.sample(now)
        self.playing = False

    def resume(self, now=None):
        self.anchor = monotonic() if now is None else now
        self.playing = self.position < self.last_tick

    def speed(self, index, now=None):
        self.sample(now)
        self.seconds_per_tick = TICK_SECONDS[index]

    def seek(self, tick):
        if type(tick) is not int or not 0 <= tick <= self.last_tick:
            raise ValueError("tick is outside recorded trajectory")
        self.position = float(tick)
        self.playing = False

    def step(self, direction):
        # Operate on the frozen visible position, never on an unseen timer sample.
        tick = (
            math.ceil(self.position) - 1
            if direction < 0
            else math.floor(self.position) + 1
        )
        self.seek(max(0, min(self.last_tick, tick)))
