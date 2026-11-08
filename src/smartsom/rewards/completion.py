"""Elapsed-time, overdue-demand and failed-shipment costs with fixed scaling."""

from math import isfinite


class Reward:
    public_snapshots = False
    requires_events = False
    requires_actions = False

    def __init__(
        self, *, orders, quality_penalty, makespan_weight, tardiness_weight, scale
    ):
        if orders <= 0 or not isfinite(scale) or scale <= 0:
            raise ValueError("completion reward requires positive orders and scale")
        if any(
            not isfinite(x) or x < 0
            for x in (quality_penalty, makespan_weight, tardiness_weight)
        ):
            raise ValueError("completion costs must be finite and nonnegative")
        self.orders = orders
        self.quality_penalty = quality_penalty
        self.makespan_weight = makespan_weight
        self.tardiness_weight = tardiness_weight
        self.scale = scale

    def __call__(self, transition):
        return {
            "team": -(
                self.makespan_weight * (transition.end - transition.start)
                + self.tardiness_weight
                * transition.outcomes["overdue_time"]
                / self.orders
                + self.quality_penalty * transition.outcomes["failed"] / self.orders
            )
            / self.scale
        }

    def begin_episode(self):
        pass

    def get_state(self):
        return {}

    def set_state(self, state):
        if state:
            raise ValueError("stateless completion reward received state")


def create_reward(
    *, orders, quality_penalty, makespan_weight=1.0, tardiness_weight=1.0, scale=1000.0
):
    return Reward(
        orders=orders,
        quality_penalty=quality_penalty,
        makespan_weight=makespan_weight,
        tardiness_weight=tardiness_weight,
        scale=scale,
    )


create_reward.CONTRACT = "smartsom.reward/v1"
