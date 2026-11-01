"""Fixed production baselines and the shared, occupancy-aware grid Mover."""

import random


class BaseRule:
    def __init__(self, method="rule", seed=0):
        if method not in ("rule", "random"):
            raise ValueError("unknown Base baseline")
        self.method = method
        self.rng = random.Random(seed)

    def get_state(self):
        return self.rng.getstate()

    def set_state(self, state):
        from smartsom.experiments.reservation_session import tuples

        self.rng.setstate(tuples(state))

    def choose(self, request, critic=None):
        candidates = request.candidates
        if self.method == "random":
            return self.rng.choice(candidates).action
        if request.group == "dispatcher" and candidates[-1].action == ("EMPTY_WAIT",):
            candidates = candidates[:-1]
        # EDD for process/service; normal mode; nearest eligible resource with
        # ready supply preferred; accept all feasible requests without STOP.
        if request.group == "machine":
            normal = [c for c in candidates if c.action[1] == "normal"]
            selected = min(
                normal or candidates,
                key=lambda c: (c.features[0], c.features[12], c.identity),
            )
        elif request.group == "dispatcher":
            if request.stage == "pickup":
                selected = min(
                    candidates,
                    key=lambda c: (
                        c.features[2] == 0,
                        c.features[0],
                        c.features[3],
                        c.identity,
                    ),
                )
            else:
                selected = min(
                    candidates,
                    key=lambda c: (c.features[0] + c.features[2], c.identity),
                )
        elif request.stage == "accept":
            selected = min(
                (c for c in candidates if c.action != ("STOP",)),
                key=lambda c: (c.features[0], c.identity),
            )
        else:
            due = 4 if request.group == "pre" else 0
            selected = min(candidates, key=lambda c: (c.features[due], c.identity))
        return selected.action


def create_policy(*, seed=0, control="base"):
    if control == "centralized":
        from smartsom.algorithms.central_rules import CentralPolicy

        return CentralPolicy("rule", seed)
    return BaseRule("rule", seed)


create_policy.CONTRACT = "smartsom.policy/v1"
