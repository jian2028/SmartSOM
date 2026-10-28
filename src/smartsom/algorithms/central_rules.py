"""Runnable centralized baselines returning semantic task identities."""

import random


class CentralRule:
    def __init__(self, method="rule", seed=0):
        if method not in {"rule", "random"}:
            raise ValueError("unknown centralized baseline")
        self.method, self.rng = method, random.Random(seed)

    def choose(self, observation, candidates):
        if self.method == "random":
            return self.rng.choice(candidates).identity
        productive = [c for c in candidates if c.kind != "advance"]
        if not productive:
            return candidates[-1].identity
        jobs = {j["id"]: j for j in observation["jobs"]}
        return min(
            productive,
            key=lambda c: (
                c.kind != "process",
                jobs[c.job]["due_at"],
                c.mode not in (None, "normal"),
                c.identity,
            ),
        ).identity
