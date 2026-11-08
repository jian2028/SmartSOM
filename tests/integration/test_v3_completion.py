"""V3 costs and completion boundaries, with independent hand-calculated expectations."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from smartsom.algorithms.base_policy import build_policy
from smartsom.config.base_diagnostic import load_inputs
from smartsom.domain.training import RewardTransition
from smartsom.experiments.reservation_periodic_evaluation import candidate_key
from smartsom.experiments.reservation_runtime import episode
from smartsom.representations.v3.rules import choose_hpof
from smartsom.rewards.completion import create_reward

ROOT = Path(__file__).resolve().parents[2]


def test_fixed_denominator_costs_telescope():
    reward = create_reward(orders=100, quality_penalty=1000)
    # 10 ticks, 200 order-ticks overdue, one failed delivery: -(10+2+10)/1000.
    first = RewardTransition(
        0, 10, (), None, None, (), {"overdue_time": 200, "failed": 1}, False, False
    )
    second = RewardTransition(
        10, 20, (), None, None, (), {"overdue_time": 100, "failed": 0}, True, False
    )
    assert reward(first)["team"] == pytest.approx(-0.022)
    assert reward(second)["team"] == pytest.approx(-0.011)
    assert reward(first)["team"] + reward(second)["team"] == pytest.approx(-0.033)
    assert reward.get_state() == {}


def test_hpof_lexicographic_resident_backlog():
    inventory = {"a": [1, 2, 3], "b": [4, 5, 6], "pa": [7, 8, 9, 10], "pb": [11]}
    facts = SimpleNamespace(
        component_cache={},
        scales=(200, 50, 20),
        roles={"a": "machine_post", "b": "machine_post"},
        capacity={"a": {"slot": 4}, "b": {"slot": 4}},
        station={"a": "ma", "b": "mb"},
        pre={"ma": "pa", "mb": "pb"},
        machine_state={"ma": {"job": "ja"}, "mb": {"job": "jb"}},
        inventory=lambda owner: inventory[owner],
        ready=lambda owner: inventory[owner],
    )
    request = SimpleNamespace(group="dispatcher", stage="pickup")
    candidates = [
        SimpleNamespace(identity="z", action=("a",)),
        SimpleNamespace(identity="a", action=("b",)),
    ]
    assert choose_hpof(facts, request, candidates) == "z"
    inventory["b"].append(12)
    facts.component_cache.clear()
    assert choose_hpof(facts, request, candidates) == "a"
    inventory["a"].clear()
    inventory["b"].clear()
    facts.component_cache.clear()
    assert choose_hpof(facts, request, candidates) is None


@pytest.mark.parametrize("condition", ["normal", "high"])
def test_completion_extends_same_world_to_output(condition):
    _, factory, workload, options, _ = load_inputs(
        ROOT / "experiments/v3/small/justin-101/experiment.yaml"
    )
    workload = workload.model_copy(update={"orders": 4, "initial_orders": 2})
    reports = []
    for drain, mode in [(100, "all_delivered"), (10000, "drain_deadline")]:
        policy = build_policy(options, 101, training=False)
        report, rewards, _ = episode(
            factory,
            workload,
            options,
            policy,
            19,
            condition,
            drain,
            episode_end=mode,
            metrics_level="off",
        )
        assert report["status"] == "completed"
        assert report["completed"] == 4
        assert report["makespan"] == report["tick"]
        assert sum(rewards) == pytest.approx(
            -(report["tick"] + report["mean_tardiness"] + 1000 * report["failed"] / 4)
            / 1000
        )
        assert report["total_tardiness"] == report["observed_overdue"]
        reports.append(report)
    assert any("extended_to" in row for row in reports[0]["horizon_checks"])
    for key in (
        "tick",
        "mean_tardiness",
        "failed",
        "learning_return",
        "physics",
        "rule_actions",
    ):
        assert reports[0][key] == reports[1][key]


def test_best_selection_uses_declared_learning_cost():
    old = [{"joint_cost": 10, "learning_cost": 50}]
    new = [{"joint_cost": 20, "learning_cost": 40}]
    assert candidate_key(old, 50, "argmax_complete_cost") < candidate_key(
        new, 100, "argmax_complete_cost"
    )
    assert candidate_key(
        new, 100, "argmax_complete_cost", "learning_cost"
    ) < candidate_key(old, 50, "argmax_complete_cost", "learning_cost")
    incomplete = [{"joint_cost": None, "learning_cost": None}]
    assert candidate_key(
        incomplete, 1, "argmax_complete_cost", "learning_cost"
    ) > candidate_key(new, 100, "argmax_complete_cost", "learning_cost")


@pytest.mark.parametrize("ownership", ["shared", "independent"])
def test_complete_learning_update_preserves_terminal_tail(ownership):
    import time

    _, factory, workload, options, _ = load_inputs(
        ROOT / f"experiments/v3/small/base-{ownership}-101/experiment.yaml"
    )
    workload = workload.model_copy(update={"orders": 2, "initial_orders": 2})
    policy = build_policy(options, 101, training=True)
    report, rewards, final = episode(
        factory,
        workload,
        options,
        policy,
        19,
        "normal",
        100,
        episode_end="all_delivered",
        metrics_level="off",
        wall_deadline=time.monotonic() + 20,
    )
    assert report["status"] == "completed"
    assert rewards.end_reason == "all_delivered"
    policy.update(rewards, terminal=True, final_critic=final)
    assert policy.learner.truncated_tails == 0
    assert any(
        row["next"] is None
        for replay in policy.learner.replay.values()
        for row in replay
    )
    assert policy.updates > 0


@pytest.mark.parametrize("budget", [50, 100, 150])
def test_completion_respects_budget_across_initial_horizon(budget):
    _, factory, workload, options, _ = load_inputs(
        ROOT / "experiments/v3/small/justin-101/experiment.yaml"
    )
    workload = workload.model_copy(update={"orders": 4, "initial_orders": 4})
    report, rewards, _ = episode(
        factory,
        workload,
        options,
        build_policy(options, 101, training=False),
        19,
        "normal",
        100,
        episode_end="all_delivered",
        max_ticks=budget,
        metrics_level="off",
    )
    assert report["tick"] == budget
    assert rewards.end_reason == "training_budget_cut"
    assert report["makespan"] is None
    assert report["learning_cost"] is None


def test_completion_has_no_intermediate_truncated_transition(monkeypatch):
    from smartsom.algorithms.training import TrainingComponents

    original = TrainingComponents.transition
    states = []

    def observe(self, protocol, *args, **kwargs):
        states.append(protocol.core.status)
        return original(self, protocol, *args, **kwargs)

    monkeypatch.setattr(TrainingComponents, "transition", observe)
    test_completion_extends_same_world_to_output("high")
    assert "truncated" not in states
