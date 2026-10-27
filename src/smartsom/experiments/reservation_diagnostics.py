"""Optional bounded counters; no trajectory copies, JSON encoding or hashes."""

from collections import Counter


class DecisionCounters:
    def __init__(self):
        self.decisions = Counter()
        self.special_actions = Counter()
        self.physical_ticks = 0

    def decision(self, request, action):
        self.decisions[request.group + "." + request.stage] += 1
        if action in (("STOP",), ("EMPTY_WAIT",)):
            self.special_actions[action[0]] += 1

    def transition(self, elapsed, reward):
        self.physical_ticks += elapsed

    def summary(self):
        return {
            "decisions": dict(self.decisions),
            "special_actions": dict(self.special_actions),
            "physical_ticks": self.physical_ticks,
        }
