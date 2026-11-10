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
        ROOT / "experiments/v3/small/rule-normal/experiment.yaml"
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
        assert (
            report["completion_tail_ticks"] == report["tick"] - report["last_arrival"]
        )
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


def test_delivery_selection_orders_completion_delivery_cost_then_episode():
    def row(delivered, cost):
        return {
            "completed": delivered,
            "total_orders": 100,
            "learning_cost": cost if delivered == 100 else None,
            "learning_return": -cost,
        }

    def key(rows, ep=50):
        return candidate_key(rows, ep, "argmax_delivery_cost", "learning_cost")

    assert key([row(100, 100), row(1, 100)]) < key([row(99, 1), row(99, 1)])
    assert key([row(100, 100), row(90, 100)]) < key([row(100, 1), row(80, 1)])
    assert key([row(100, 3), row(90, 15)], 100) < key([row(100, 8), row(90, 20)], 50)
    assert key([row(100, 3), row(100, 5)]) < key([row(100, 4), row(100, 5)])
    assert key([row(100, 3)], 50) < key([row(100, 3)], 100)
    with pytest.raises(ValueError, match="learning_cost"):
        candidate_key([row(100, 3)], 50, "argmax_delivery_cost")


@pytest.mark.parametrize("mode", ["slow", "normal", "fast"])
def test_rule_processing_mode_prefers_requested_mode_then_shortest(mode):
    from smartsom.policies.justin import create_policy

    candidates = tuple(
        SimpleNamespace(identity=f"{job}-{speed}", action=(job, speed))
        for job in ("long", "short")
        for speed in ("slow", "normal", "fast")
    )
    request = SimpleNamespace(
        group="machine",
        candidates=candidates,
        inputs={
            "candidates": {
                c.identity: {"nominal_duration": 100 if c.action[0] == "long" else 50}
                for c in candidates
            }
        },
    )
    policy = create_policy(processing_mode=mode)
    assert policy.choose(request) == ("short", mode)
    if mode == "normal":
        assert create_policy().choose(request) == policy.choose(request)


@pytest.mark.parametrize("ownership", ["shared", "independent"])
def test_complete_learning_update_preserves_terminal_tail(ownership):
    import time

    _, factory, workload, options, _ = load_inputs(
        ROOT / f"experiments/v3/small/base-{ownership}/experiment.yaml"
    )
    # Exercise optimization with a tiny workload; campaign warmup is tested
    # separately and must not prevent this terminal-return regression probe.
    options.learning_policy.learner.params.update(
        warmup_transitions=0, update_every=1, updates_per_transition=1.0
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
        ROOT / "experiments/v3/small/rule-normal/experiment.yaml"
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
    assert report["completion_tail_ticks"] is None
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


def test_completion_tail_summary_excludes_censored_cases():
    from smartsom.telemetry.evaluation_summary import completion_tail_summary

    result = completion_tail_summary(
        [
            {"status": "completed", "completion_tail_ticks": 101},
            {"status": "completed", "completion_tail_ticks": 201},
            {"status": "truncated", "completion_tail_ticks": None, "tick": 99999},
            {"status": "completed"},
        ]
    )
    assert result["max_ticks"] == 201
    assert result["median_ticks"] == 151
    assert result["suggested_drain_ticks"] == 302
    assert result["noncompleted_cases"] == 1
    assert result["missing_completed_cases"] == 1
    assert not result["complete_coverage"]
    assert completion_tail_summary([])["suggested_drain_ticks"] is None


@pytest.mark.parametrize(
    "reason, penalty",
    [
        ("drain_deadline", 16),
        ("training_budget_cut", 0),
        ("wall_deadline", 0),
        ("all_delivered", 0),
    ],
)
def test_deadline_penalty_uses_fixed_demand_count_and_cause(reason, penalty):
    reward = create_reward(
        orders=10,
        quality_penalty=1000,
        truncation_fixed_penalty=10,
        truncation_unfinished_weight=10,
    )
    # Four shipments include one failed shipment: six demands remain, not seven.
    first = RewardTransition(
        0,
        10,
        (),
        None,
        None,
        (),
        {"overdue_time": 20, "failed": 1, "shipments": 4},
        False,
        False,
    )
    assert reward(first)["team"] == pytest.approx(-0.112)
    state = reward.get_state()
    restored = create_reward(
        orders=10,
        quality_penalty=1000,
        truncation_fixed_penalty=10,
        truncation_unfinished_weight=10,
    )
    restored.set_state(state)
    last = RewardTransition(
        10,
        20,
        (),
        None,
        None,
        (),
        {"overdue_time": 30, "failed": 0, "shipments": 0},
        False,
        True,
        reason,
    )
    assert restored(last)["team"] == pytest.approx(-0.013 - penalty)
    assert restored(last)["team"] == pytest.approx(-0.013)
    restored.begin_episode()
    assert restored.get_state() == {"shipments": 0, "deadline_charged": False}


@pytest.mark.parametrize("ownership", ["shared", "independent"])
@pytest.mark.parametrize("budget_cut", [False, True])
def test_deadline_learning_retains_penalty_but_budget_omits_unresolved_tail(
    ownership, budget_cut
):
    _, factory, workload, options, _ = load_inputs(
        ROOT / f"experiments/v3/small/base-{ownership}/experiment.yaml"
    )
    options.learning_policy.learner.params.update(
        warmup_transitions=0, update_every=1, updates_per_transition=1.0
    )
    workload = workload.model_copy(update={"orders": 4, "initial_orders": 4})
    policy = build_policy(options, 101, training=True)
    report, rewards, final = episode(
        factory,
        workload,
        options,
        policy,
        19,
        "normal",
        100 if budget_cut else 50,
        max_ticks=50 if budget_cut else None,
        metrics_level="off",
    )
    assert report["tick"] == 50
    assert report["status"] == "truncated"
    assert rewards.end_reason == (
        "training_budget_cut" if budget_cut else "drain_deadline"
    )
    physical_cost = (
        50 + report["observed_overdue"] / 4 + 1000 * report["failed"] / 4
    ) / 1000
    expected_penalty = 0 if budget_cut else 10 + 10 * (4 - report["completed"]) / 4
    assert sum(rewards) == pytest.approx(-physical_cost - expected_penalty)
    records = list(policy.learner.records)
    assert records
    policy.update(rewards, terminal=False, final_critic=final)
    tails = [
        r for rows in policy.learner.replay.values() for r in rows if r["next"] is None
    ]
    if budget_cut:
        assert policy.learner.truncated_tails > 0
        assert not tails
    else:
        assert policy.learner.truncated_tails == 0
        assert tails
        for tail in tails:
            assert tail["reward"] == pytest.approx(sum(rewards[tail["request"].tick :]))
        assert policy.updates > 0


@pytest.mark.parametrize("budget_cut", [False, True])
@pytest.mark.parametrize("batched", [False, True])
def test_central_ppo_deadline_bootstrap_switch(monkeypatch, budget_cut, batched):
    import smartsom.learning.returns as returns

    _, factory, workload, options, _ = load_inputs(
        ROOT / "experiments/v3/small/central/experiment.yaml"
    )
    workload = workload.model_copy(update={"orders": 4, "initial_orders": 4})
    policy = build_policy(options, 101, training=True)
    report, rewards, final = episode(
        factory,
        workload,
        options,
        policy,
        19,
        "normal",
        100 if budget_cut else 50,
        max_ticks=50 if budget_cut else None,
        metrics_level="off",
    )
    assert report["status"] == "truncated"
    original = returns.prepare_returns
    terminals = []

    def observe(*args, **kwargs):
        terminals.append(kwargs["terminal"])
        return original(*args, **kwargs)

    monkeypatch.setattr(returns, "prepare_returns", observe)
    if batched:
        policy.learner.update_rollouts([(policy.records, rewards, False, final)])
    else:
        policy.update(rewards, terminal=False, final_critic=final)
    assert terminals and all(t == (not budget_cut) for t in terminals)


def test_terminal_replay_discards_boundary_bootstrap_request():
    from smartsom.domain.training import RewardTimeline
    from smartsom.learning.dqn.replay import transitions

    rewards = RewardTimeline()
    rewards.extend([-1.0, -20.0])
    rewards.final_requests["machine", "m"] = SimpleNamespace(
        group="machine", owner="m", stage="process", tick=2
    )
    records = [
        dict(
            request=SimpleNamespace(
                group="machine", owner="m", stage="process", tick=0
            ),
            index=0,
        )
    ]
    rows, omitted = transitions(
        records, rewards, terminal={"machine": True}, scale={"machine": 1.0}
    )
    assert omitted == 0
    assert rows["machine"][0]["next"] is None
    assert rows["machine"][0]["reward"] == -21


@pytest.mark.parametrize("processes", [0, 2])
def test_v3_fragment_deadline_cost_and_terminal_replay_survive_restore(
    tmp_path, processes
):
    import yaml

    from smartsom.config.reservation import compile_reservation
    from smartsom.experiments import reservation
    from smartsom.experiments.reservation_session import ReservationSession
    from tests.integration.test_training_sampling_controls import assert_state_equal

    source = ROOT / "experiments/v3/small/base-shared/experiment.yaml"
    data = yaml.safe_load(source.read_text())
    scenario = yaml.safe_load((source.parent / data["scenario"]).read_text())
    scenario["jobs"]["orders"] = {"total": 4, "initial": 2}
    (tmp_path / "scenario.yaml").write_text(yaml.safe_dump(scenario))
    algorithm = yaml.safe_load((source.parent / data["algorithm"]).read_text())
    algorithm["production"]["policies"]["dqn"]["learner"]["params"].update(
        warmup_transitions=0, update_every=1, updates_per_transition=1.0
    )
    (tmp_path / "algorithm.yaml").write_text(yaml.safe_dump(algorithm))
    data.update(
        scenario="scenario.yaml",
        algorithm="algorithm.yaml",
        train_episodes=2,
        drain_ticks=50,
        final_evaluation=False,
        evaluation={"mode": "limited", "checkpoints": []},
        checkpoint_retention="all",
        training_conditions=["normal"],
        tracking={"mode": "disabled"},
        validation={"milestones": [], "selection": "last"},
        test={"seeds": 1, "conditions": ["normal"]},
        sampling={
            "num_envs": processes or 1,
            "processes": processes,
            "fragment_ticks": 8,
        },
    )
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(data))
    plan = compile_reservation(path, task="train")
    root = reservation.allocate(plan, root=tmp_path / "run")
    args, _ = reservation.arguments(root)
    session = ReservationSession(args)
    try:
        while not session.step()["done"]:
            pass
        assert len(session.rows) == 2
        for report in session.rows:
            assert report["termination_reason"] == "drain_deadline"
            physical_cost = (
                report["tick"]
                + report["observed_overdue"] / 4
                + 1000 * report["failed"] / 4
            ) / 1000
            assert report["learning_return"] == pytest.approx(
                -physical_cost - 10 - 10 * report["unfinished"] / 4
            )
        learner = session.policy.learner
        assert learner.truncated_tails == 0
        tails = [
            r for replay in learner.replay.values() for r in replay if r["next"] is None
        ]
        assert tails and all(r["reward"] < -10 for r in tails)
        assert session.policy.updates > 0
        expected = session.policy.get_state()
    finally:
        session.close()
    restored = ReservationSession(args)
    try:
        assert_state_equal(expected, restored.policy.get_state())
        assert restored.step()["done"]
    finally:
        restored.close()


@pytest.mark.parametrize("final_drain", [250, 10000])
def test_extended_window_charges_only_actual_deadline(final_drain):
    _, factory, workload, options, _ = load_inputs(
        ROOT / "experiments/v3/small/rule-normal/experiment.yaml"
    )
    workload = workload.model_copy(update={"orders": 4, "initial_orders": 4})

    def extend(sim, history):
        return final_drain if len(history) == 1 else None

    results = []
    for drain, callback in [(50, extend), (final_drain, None)]:
        report, rewards, _ = episode(
            factory,
            workload,
            options,
            build_policy(options, 101, training=False),
            19,
            "normal",
            drain,
            extend=callback,
            metrics_level="off",
        )
        results.append((report, sum(rewards)))
    assert results[0][0]["status"] == results[1][0]["status"]
    assert results[0][0]["tick"] == results[1][0]["tick"]
    assert results[0][1] == pytest.approx(results[1][1])
    assert results[0][0]["completed"] > 0
    assert (results[0][0]["status"] == "completed") == (final_drain == 10000)
