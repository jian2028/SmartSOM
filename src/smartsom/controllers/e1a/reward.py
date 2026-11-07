"""Shipment objective adapted to the current simulator's physical clock."""

from smartsom.rewards.default import Reward as StatelessReward


def starved(view):
    machines = view["machines"]
    buffers = {b["id"]: b for b in view["buffers"]}
    excluded = {m["id"] for m in machines} | {
        b["id"]
        for b in buffers.values()
        if b["role"] in ("system_input", "system_output")
    }
    waiting = [
        (j["remaining_operations"][0]["type"], j["location"])
        for j in view["jobs"]
        if j["location"] not in excluded
        and j["location"] != "queue"
        and j["quality"] != "FAIL"
        and j["remaining_operations"]
    ]
    return sum(
        m["status"] == "IDLE"
        and not m["down"]
        and not any(buffers[m["pre"]]["inventory"].values())
        and any(op in m["capabilities"] and at != m["pre"] for op, at in waiting)
        for m in machines
    )


class Reward(StatelessReward):
    public_snapshots = True

    def __init__(
        self,
        *,
        shipment_weight,
        passing_weight,
        tardiness_weight,
        starvation_weight,
        reference_jobs,
        reference_ticks,
    ):
        self.weights = (
            shipment_weight,
            passing_weight,
            tardiness_weight,
            starvation_weight,
        )
        self.jobs, self.ticks = reference_jobs, reference_ticks

    def __call__(self, transition):
        c = transition.outcomes
        ship, passing, late, starvation = self.weights
        reward = (
            ship * c["shipments"] / self.jobs
            + passing * c["passing_change"]
            - late * c["overdue_time"] / (self.jobs * self.ticks)
        )
        reward -= (
            starvation
            * starved(transition.before)
            * (transition.end - transition.start)
            / (max(1, len(transition.before["machines"])) * self.ticks)
        )
        return {"team": reward}


def create_reward(
    *,
    orders,
    quality_penalty,
    shipment_weight=0.5,
    passing_weight=0.1,
    tardiness_weight=2.0,
    starvation_weight=1.0,
    reference_jobs=128,
    reference_ticks=4096,
):
    import math

    if (
        reference_jobs <= 0
        or reference_ticks <= 0
        or not all(
            math.isfinite(v) and v >= 0
            for v in (
                shipment_weight,
                passing_weight,
                tardiness_weight,
                starvation_weight,
            )
        )
    ):
        raise ValueError(
            "reward weights must be finite/nonnegative and references positive"
        )
    return Reward(
        shipment_weight=shipment_weight,
        passing_weight=passing_weight,
        tardiness_weight=tardiness_weight,
        starvation_weight=starvation_weight,
        reference_jobs=reference_jobs,
        reference_ticks=reference_ticks,
    )


create_reward.CONTRACT = "smartsom.reward/v1"
