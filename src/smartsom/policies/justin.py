"""Soft dispatch preferences only; shared protocol owns admission and feasibility."""

from smartsom.policies.rule import BaseRule


def observation(facts, request):
    from smartsom.observations.rules import candidate_facts

    return {**facts, "candidates": candidate_facts(facts, request)}


def create_observation():
    return observation


create_observation.OUTPUT_SCHEMA = "smartsom.justin-public-facts/v1"


class JustinPolicy:
    def __init__(self, seed):
        self.default = BaseRule(seed=seed)

    def choose(self, request, critic=None):
        if request.group == "machine":
            # Original normal_first: prefer normal mode, then shortest processing.
            return min(
                request.candidates,
                key=lambda c: (
                    c.action[1] != "normal",
                    request.inputs["candidates"][c.identity]["nominal_duration"],
                    c.identity,
                ),
            ).action
        if request.group != "dispatcher":
            return self.default.choose(request, critic)
        facts = request.inputs
        vehicles = {r["id"]: r for r in facts["vehicles"]}
        buffers = {r["id"]: r for r in facts["buffers"]}
        jobs = {r["id"]: r for r in facts["jobs"]}
        candidates = [c for c in request.candidates if c.action != ("EMPTY_WAIT",)]
        if not candidates:
            return request.candidates[0].action

        def score(candidate):
            owner = candidate.action[0]
            buffer = buffers[owner]
            incoming = [
                r
                for r in vehicles.values()
                if r["id"] != request.owner
                and r["target"]
                and r["target"]["owner"] == owner
            ]
            cells = facts["topology"]["ports"].get(owner, ())
            occupied = any(
                r["id"] != request.owner and r["cell"] in cells
                for r in vehicles.values()
            )
            vehicle = vehicles[request.owner]
            ongoing = bool(vehicle["target"] and vehicle["target"]["owner"] == owner)
            if request.stage == "pickup":
                ready = buffer["ready"]
                empty_inbound = [r for r in incoming if not r["job"]]
                # Current accepted pickup credits replace old intention-only supply.
                tier = 0 if ready else (1 if buffer["quota"] > 0 else 2)
                excess = (
                    max(0, len(empty_inbound) + 1 - len(ready))
                    if ready
                    else len(empty_inbound)
                )
                step = max((jobs[j]["step"] for j in ready), default=-1)
                return (
                    tier,
                    excess,
                    len(empty_inbound),
                    -step if ready else 0,
                    request.inputs["candidates"][candidate.identity]["distance"],
                    not ongoing,
                    candidate.identity,
                )
            capacities = tuple(buffer["capacity"].values())
            inventory = sum(len(rows) for rows in buffer["inventory"].values())
            full = all(c is not None for c in capacities) and inventory >= sum(
                capacities
            )
            return (
                full,
                not ongoing,
                len(incoming),
                occupied,
                request.inputs["candidates"][candidate.identity]["distance"],
                candidate.identity,
            )

        return min(candidates, key=score).action

    def get_state(self):
        return self.default.get_state()

    def set_state(self, state):
        self.default.set_state(state)


def create_policy(*, seed=0, control="base"):
    if control != "base":
        raise ValueError("Justin adaptation requires the reservation protocol")
    return JustinPolicy(seed)


create_policy.CONTRACT = "smartsom.policy/v1"
