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
        values = request.inputs["candidates"] if self.method == "rule" else None
        if self.method == "random":
            return self.rng.choice(candidates).action
        if request.group == "dispatcher" and candidates[-1].action == ("EMPTY_WAIT",):
            candidates = candidates[:-1]
        # EDD for process/service; critical-ratio speed; nearest eligible resource with
        # ready supply preferred; accept all feasible requests without STOP.
        if request.group == "machine":
            jobs = request.inputs["machine_jobs"]
            job = min(
                jobs,
                key=lambda j: (jobs[j]["due_ticks"], jobs[j]["normal_ticks"], j),
            )
            due, work = jobs[job]["due_ticks"], jobs[job]["remaining_normal_ticks"]
            # Integer comparisons preserve the exact CR=1 and CR=1.2 boundaries.
            mode = "fast" if due < work else "normal" if 5 * due < 6 * work else "slow"
            available = [c for c in candidates if c.action[0] == job]
            selected = min(
                available,
                key=lambda c: (
                    c.action[1] != mode,
                    c.action[1] != "normal",
                    values[c.identity]["nominal_duration"],
                    c.identity,
                ),
            )
        elif request.group == "dispatcher":
            if request.stage == "pickup":
                selected = min(
                    candidates,
                    key=lambda c: (
                        values[c.identity]["ready_count"] == 0,
                        values[c.identity]["distance"],
                        values[c.identity]["ready_min_due"],
                        c.identity,
                    ),
                )
            else:
                selected = min(
                    candidates,
                    key=lambda c: (
                        values[c.identity]["distance"]
                        + values[c.identity]["ready_count"],
                        c.identity,
                    ),
                )
        elif request.stage == "accept":
            selected = min(
                (c for c in candidates if c.action != ("STOP",)),
                key=lambda c: (values[c.identity]["distance"], c.identity),
            )
        else:
            selected = min(
                candidates, key=lambda c: (values[c.identity]["due_slack"], c.identity)
            )
        return selected.action


def create_policy(*, seed=0, control="base"):
    if control == "centralized":
        from smartsom.algorithms.central_rules import CentralPolicy

        return CentralPolicy("rule", seed)
    return BaseRule("rule", seed)


create_policy.CONTRACT = "smartsom.policy/v1"
