"""Hand-calculated objectives and real physical-interval accounting."""

from types import SimpleNamespace

import pytest

from smartsom.rewards.normalized_task import (
    complete_objective,
    cost_snapshot,
    interval_reward,
    task_metrics,
)


def test_complete_objective_and_interval_telescoping():
    # Two orders delivered at 5000 and 10000, due at 4800 and 9600;
    # one failed delivery: J = 2 + 1 + 5/3 = 14/3.
    expected = 14 / 3
    assert complete_objective(
        makespan=10000, mean_tardiness=300, final_passing_rate=0.5
    ) == pytest.approx(expected)
    rewards = [
        interval_reward(elapsed=5000, overdue=200, failed=0, orders=2),
        interval_reward(elapsed=0, overdue=0, failed=0, orders=2),
        interval_reward(elapsed=5000, overdue=400, failed=1, orders=2),
    ]
    assert sum(rewards) == pytest.approx(-expected)


def test_incomplete_case_reports_observed_cost_without_complete_goals():
    sim = SimpleNamespace(
        demands={"a": SimpleNamespace(due_at=1), "b": SimpleNamespace(due_at=2)},
        completed={"a"},
        shipment_times={"a": 3},
        tick=5,
        status="truncated",
        done=True,
        privileged_output_quality=lambda: {"good_shipped": 0, "bad_shipped": 1},
        privileged_task_totals=lambda: {"overdue_time": 5},
    )
    report = task_metrics(sim)
    assert report["joint_cost"] is None
    assert report["makespan"] is None
    assert report["mean_tardiness"] is None
    assert report["final_passing_rate"] is None
    assert report["unfinished_fraction"] == 0.5
    assert report["truncated"]
    assert report["observed_cost"] == pytest.approx(5 / 5000 + 5 / 600 + 1000 / 600)


@pytest.mark.parametrize("passing", [None, -0.1, 1.1, float("nan")])
def test_missing_or_invalid_complete_goal_is_not_optimized(passing):
    with pytest.raises(ValueError):
        complete_objective(makespan=10, mean_tardiness=0, final_passing_rate=passing)


@pytest.mark.parametrize(
    "mode,release,due", [("finite", 0, 5), ("finite", 10, 5), ("dynamic", 0, 5)]
)
def test_real_factory_intervals_sum_to_complete_objective(mode, release, due):
    from dataclasses import replace

    from smartsom.engine.production import ProductionSimulator
    from smartsom.traffic.rule_mover import SharedMover
    from tests.unit.test_base_reservations import decide, options, sim_with

    scenario = sim_with(steps=("A",), n=1).scenario
    scenario = replace(
        scenario,
        mode=mode,
        tick_limit=120,
        demands=tuple(
            replace(d, release_at=release, due_at=due) for d in scenario.demands
        ),
    )
    sim = ProductionSimulator(scenario, contract="base")
    sim.protocol.configure(options())
    mover = SharedMover(sim.protocol.routes)
    total = 0.0
    while not sim.done:
        decide(sim)
        before = cost_snapshot(sim)
        result = sim.protocol.advance(
            mover.moves(sim.protocol.traffic_view(), sim.tick)
        )
        assert not result["rejections"]
        total += cost_snapshot(sim).reward_since(before)
    assert len(sim.completed) == len(sim.demands)
    old_reward = sim.total_reward
    report = task_metrics(sim)
    assert total == pytest.approx(-report["joint_cost"])
    assert report["observed_cost"] == pytest.approx(report["joint_cost"])
    assert sim.total_reward == old_reward
