"""Useful-throughput feedback for a frozen, homogeneous experiment queue.

No framework, learner or hardware access lives here. The driver supplies verified
commits and the complete live membership. Windows include validation/checkpoint
time; startup and membership changes are discarded rather than timed as training.
"""

import math
import statistics


class OnlineConcurrency:
    def __init__(self, ceiling, *, hint=1, window_seconds=30.0, history=()):
        if type(ceiling) is not int or ceiling < 1:
            raise ValueError("online concurrency ceiling must be positive")
        self.ceiling = ceiling
        self.limit = 1
        self.hint = min(ceiling, max(1, hint))
        self.window_seconds = window_seconds
        self.history = list(history)
        self.reason = "warming up one real experiment"
        self._members = ()
        self._seen = {}
        self._baseline = None
        self._started = None
        self._windows = []
        self._accepted = None
        self._saturated = False
        self._latest_window = None
        self._latest_gain = None

    def reset_window(self):
        self._baseline, self._started = None, None
        self._windows = []

    def suspend(self):
        self.reset_window()
        self._members, self._seen = (), {}

    def observe(self, members, identity, *, ticks, updates, now):
        """Return a decision only after two stable windows of complete updates.

        Each member must commit two updates per window. Rates use one common
        driver wall clock, not a sum of workers' CPU times or per-update rates.
        Cached hints shorten confirmation below a previously observed limit;
        they never bypass live RSS/admission checks or directly set concurrency.
        """
        members = tuple(sorted(members))
        if identity not in members or not members:
            self.reset_window()
            return None
        if (
            type(ticks) is not int
            or ticks < 0
            or type(updates) is not int
            or updates < 1
            or not math.isfinite(now)
        ):
            return None
        previous = self._seen.get(identity)
        if previous and (updates <= previous[1] or ticks < previous[0]):
            return None
        self._seen[identity] = (ticks, updates)
        if members != self._members:
            self._members = members
            self._seen = {identity: (ticks, updates)}
            self.reset_window()
        if len(members) != self.limit:
            # Stopping, finishing and resource-limited tail waves are not probes.
            return None
        if not all(member in self._seen for member in members):
            return None
        if self._baseline is None:
            self._baseline = {member: self._seen[member] for member in members}
            self._started = now
            return None
        elapsed = now - self._started
        minimum = (
            self.window_seconds / 2
            if self.limit <= self.hint and self.hint > 1
            else self.window_seconds
        )
        if elapsed < minimum or any(
            self._seen[member][1] - self._baseline[member][1] < 2 for member in members
        ):
            return None
        ticks = sum(
            self._seen[member][0] - self._baseline[member][0] for member in members
        )
        self._baseline = {member: self._seen[member] for member in members}
        self._started = now
        if ticks <= 0:
            return None
        self._windows.append(
            {
                "physical_ticks": ticks,
                "wall_seconds": elapsed,
                "throughput": ticks / elapsed,
            }
        )
        self._windows = self._windows[-2:]
        self._latest_window = {**self._windows[-1], "concurrency": self.limit}
        if len(self._windows) < 2:
            return None
        rates = [window["throughput"] for window in self._windows]
        rate = statistics.median(rates)
        if max(rates) > min(rates) * 1.15:
            self.reason = "throughput windows unstable; holding concurrency"
            return None
        count = self.limit
        gain = None if self._accepted is None else rate / self._accepted[1] - 1
        self._latest_gain = gain
        accepted = self._accepted is None or count == self._accepted[0] or gain >= 0.05
        row = {
            "concurrency": count,
            "throughput": rate,
            "window_rates": rates,
            "windows": list(self._windows),
            "members": list(members),
            "window_min_seconds": minimum,
            "accepted": accepted,
            "gain": gain,
        }
        self.history.append(row)
        if accepted:
            self._accepted = (count, rate)
            if not self._saturated and count < self.ceiling:
                self.limit = count + 1
                self.reason = "stable useful throughput; testing one more experiment"
            else:
                self.reason = "holding measured concurrency"
        else:
            self.limit = self._accepted[0]
            self._saturated = True
            self.reason = (
                "less than 5% throughput gain; returning to measured concurrency"
            )
        self.reset_window()
        return {**row, "next_limit": self.limit, "reason": self.reason}

    def summary(self):
        return {
            "limit": self.limit,
            "ceiling": self.ceiling,
            "cached_hint": self.hint,
            "best_concurrency": self._accepted[0] if self._accepted else None,
            "best_throughput": self._accepted[1] if self._accepted else None,
            "latest_throughput": self._latest_window["throughput"]
            if self._latest_window
            else None,
            "measured_concurrency": self._latest_window["concurrency"]
            if self._latest_window
            else None,
            "gain": self._latest_gain,
            "window_count": len(self._windows),
            "observing": self._baseline is not None
            and len(self._members) == self.limit,
            "reason": self.reason,
            "measurements": self.history,
        }
