"""Elapsed-time, overdue-demand and failed-shipment costs with fixed scaling."""

from math import isfinite


class Reward:
    public_snapshots = False
    requires_events = False
    requires_actions = False

    def __init__(
        self,
        *,
        orders,
        quality_penalty,
        makespan_weight,
        tardiness_weight,
        scale,
        truncation_fixed_penalty=0.0,
        truncation_unfinished_weight=0.0,
    ):
        if orders <= 0 or not isfinite(scale) or scale <= 0:
            raise ValueError("completion reward requires positive orders and scale")
        if any(
            not isfinite(x) or x < 0
            for x in (
                quality_penalty,
                makespan_weight,
                tardiness_weight,
                truncation_fixed_penalty,
                truncation_unfinished_weight,
            )
        ):
            raise ValueError("completion costs must be finite and nonnegative")
        self.orders = orders
        self.quality_penalty = quality_penalty
        self.makespan_weight = makespan_weight
        self.tardiness_weight = tardiness_weight
        self.scale = scale
        self.truncation_fixed_penalty = truncation_fixed_penalty
        self.truncation_unfinished_weight = truncation_unfinished_weight
        self.deadline_penalty_enabled = bool(
            truncation_fixed_penalty or truncation_unfinished_weight
        )
        self.begin_episode()

    def __call__(self, transition):
        penalty = 0.0
        if self.deadline_penalty_enabled:
            self.shipments += transition.outcomes["shipments"]
            unfinished = self.orders - self.shipments
            if (
                transition.end_reason == "drain_deadline"
                and not self.deadline_charged
                and unfinished > 0
            ):
                penalty = (
                    self.truncation_fixed_penalty
                    + self.truncation_unfinished_weight * unfinished / self.orders
                )
                self.deadline_charged = True
        return {
            "team": -(
                self.makespan_weight * (transition.end - transition.start)
                + self.tardiness_weight
                * transition.outcomes["overdue_time"]
                / self.orders
                + self.quality_penalty * transition.outcomes["failed"] / self.orders
            )
            / self.scale
            - penalty
        }

    def begin_episode(self):
        self.shipments = 0
        self.deadline_charged = False

    def get_state(self):
        return (
            {"shipments": self.shipments, "deadline_charged": self.deadline_charged}
            if self.deadline_penalty_enabled
            else {}
        )

    def set_state(self, state):
        if not self.deadline_penalty_enabled:
            if state:
                raise ValueError("stateless completion reward received state")
            return
        self.shipments = state["shipments"]
        self.deadline_charged = state["deadline_charged"]


def create_reward(
    *,
    orders,
    quality_penalty,
    makespan_weight=1.0,
    tardiness_weight=1.0,
    scale=1000.0,
    truncation_fixed_penalty=0.0,
    truncation_unfinished_weight=0.0,
):
    return Reward(
        orders=orders,
        quality_penalty=quality_penalty,
        makespan_weight=makespan_weight,
        tardiness_weight=tardiness_weight,
        scale=scale,
        truncation_fixed_penalty=truncation_fixed_penalty,
        truncation_unfinished_weight=truncation_unfinished_weight,
    )


create_reward.CONTRACT = "smartsom.reward/v1"
