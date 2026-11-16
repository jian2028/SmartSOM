"""RLlib task objective; separate from historical native reward semantics.

Inputs are trainer/evaluation facts, never actor observations. A replicated
multi-agent team reward must not be summed to obtain the factory cost.
"""

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True, slots=True)
class TaskCostSnapshot:
    elapsed: int
    overdue: int
    failed: int
    orders: int

    def reward_since(self, previous):
        if self.orders != previous.orders:
            raise ValueError("task order inventory changed within an episode")
        return interval_reward(
            elapsed=self.elapsed - previous.elapsed,
            overdue=self.overdue - previous.overdue,
            failed=self.failed - previous.failed,
            orders=self.orders,
        )


def cost_snapshot(sim):
    """Observed task totals, including overdue orders not yet released.

    The fixed task ends at its last delivery even if a dynamic simulator keeps
    advancing. Due dates are trainer-only inputs; do not expose them to actors
    before the scenario's public reveal boundary.
    """
    orders = len(sim.demands)
    if not orders:
        raise ValueError("task cost requires a positive number of orders")
    elapsed = (
        max(sim.shipment_times.values()) if len(sim.completed) == orders else sim.tick
    )
    overdue = sum(
        max(0, sim.shipment_times.get(key, elapsed) - demand.due_at)
        for key, demand in sim.demands.items()
    )
    return TaskCostSnapshot(
        elapsed, overdue, sim.privileged_output_quality()["bad_shipped"], orders
    )


def interval_reward(*, elapsed, overdue, failed, orders):
    """Cost of one observed physical interval, including zero-time decisions."""
    if orders <= 0:
        raise ValueError("task cost requires a positive number of orders")
    if any(not isfinite(x) or x < 0 for x in (elapsed, overdue, failed)):
        raise ValueError("task cost increments must be finite and nonnegative")
    return -(elapsed / 5000 + overdue / (300 * orders) + 1000 * failed / (300 * orders))


def complete_objective(*, makespan, mean_tardiness, final_passing_rate):
    """J for a complete case; callers must not replace missing goals with zero."""
    if any(
        value is None or not isfinite(value)
        for value in (makespan, mean_tardiness, final_passing_rate)
    ):
        raise ValueError("complete task objective requires three finite metrics")
    if makespan < 0 or mean_tardiness < 0 or not 0 <= final_passing_rate <= 1:
        raise ValueError("task metrics are outside their physical ranges")
    return (
        makespan / 5000 + mean_tardiness / 300 + (1000 / 300) * (1 - final_passing_rate)
    )


def task_metrics(sim):
    """Read committed simulator totals without changing its legacy reward ledger."""
    orders = len(sim.demands)
    completed = len(sim.completed)
    complete = completed == orders
    quality = sim.privileged_output_quality()
    observed = cost_snapshot(sim)
    makespan = max(sim.shipment_times.values()) if complete else None
    tardiness = (
        sum(
            max(0, sim.shipment_times[key] - sim.demands[key].due_at)
            for key in sim.completed
        )
        / orders
        if complete
        else None
    )
    passing = quality["good_shipped"] / orders if complete else None
    return {
        "objective_schema": "smartsom.normalized-task/v1",
        "makespan": makespan,
        "mean_tardiness": tardiness,
        "final_passing_rate": passing,
        "joint_cost": complete_objective(
            makespan=makespan, mean_tardiness=tardiness, final_passing_rate=passing
        )
        if complete
        else None,
        "observed_cost": -interval_reward(
            elapsed=observed.elapsed,
            overdue=observed.overdue,
            failed=observed.failed,
            orders=orders,
        ),
        "unfinished_fraction": (orders - completed) / orders,
        "truncated": sim.done and not complete,
    }
