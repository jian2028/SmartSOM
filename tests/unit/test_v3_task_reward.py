"""Explicit task-cost units, oracle isolation, telescoping and real learning tails."""

import copy
import json
from dataclasses import replace
from pathlib import Path

import pytest
from test_travel_time_matrix import driver, scenario
from test_v3_blind_output import ready_output

from smartsom import api
from smartsom.config.codec import canonical_json, primitive
from smartsom.config.experiment import prepare
from smartsom.config.experiment_v3 import (
    RewardSpecV3,
    ShipmentTaskReward,
    TrainingOptionsV3,
)
from smartsom.engine.production import ProductionSimulator
from smartsom.experiments.composable import TrainingSession, allocate, episode_metrics
from smartsom.experiments.task_reward import shipment_reward

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("defective", [False, True])
def test_shipment_and_oracle_passing_reward_without_public_leak(defective):
    sim, _, _ = ready_output(defective=defective)
    public_before = sim.protocol.public_view()
    result = driver(sim).tick()
    spec = ShipmentTaskReward()
    private = sim.privileged_reward_components()
    value = shipment_reward(spec, private)
    assert value == pytest.approx(0.5 / 128 + (0.0 if defective else 0.3))
    assert sum(sim.metrics.values()) >= 0
    assert (
        not {"passing_change", "overdue_time", "good_shipped", "bad_shipped"}
        & result.keys()
    )
    assert "task" not in public_before
    assert episode_metrics(sim, task_reward=spec)["return"] == pytest.approx(value)
    assert result["reward"] != value  # Raw legacy ledger is separate, never stacked.


def test_due_crossing_external_fifo_and_future_exclusion():
    case = scenario(jobs=8)
    from smartsom.domain.factory_design import PoolStorage

    factory = replace(
        case.factory,
        buffers=tuple(
            replace(b, storage=PoolStorage(capacity=1))
            if b.role == "system_input"
            else b
            for b in case.factory.buffers
        ),
    )
    case = replace(case, factory=factory)
    demands = tuple(
        replace(d, due_at=0, release_at=0 if i < 6 else 2)
        for i, d in enumerate(case.demands)
    )
    sim = ProductionSimulator(
        replace(case, demands=demands, tick_limit=3, mode="dynamic"), contract="v3"
    )
    assert (
        sim.queue
    )  # Initial physical input capacity excludes part of released demand.
    spec = ShipmentTaskReward()
    values = []
    overdue = []
    for _ in range(3):
        driver(sim).tick()
        c = sim.privileged_reward_components()
        values.append(shipment_reward(spec, c))
        overdue.append(c["overdue_time"])
    assert overdue == [6, 6, 8]
    assert sim._overdue_time == 20
    assert sum(values) == pytest.approx(-0.2 * 20 / (128 * 4096))
    assert sim.total_reward < sum(values)  # No old unfinished-cap penalty added.


def test_replacement_keeps_due_clock_and_no_inspection_or_scrap_cost():
    sim, vehicle, job = ready_output(defective=True, mode="dynamic")
    demand = sim.jobs[job]["demand"]
    sim.demands[demand] = replace(sim.demands[demand], due_at=0)
    sim.jobs[job]["quality"] = "FAIL"
    sim._transfer(vehicle, ("drop", job, next(iter(sim.scrap)), "sink"))
    driver(sim).tick()
    c = sim.privileged_reward_components()
    assert c == {"shipments": 0, "passing_change": 0.0, "overdue_time": 1}
    assert shipment_reward(ShipmentTaskReward(), c) == pytest.approx(
        -0.2 / (128 * 4096)
    )


def test_batch_passing_difference_telescopes_independent_of_order():
    spec = ShipmentTaskReward()
    whole = shipment_reward(spec, dict(shipments=2, passing_change=0.5, overdue_time=3))
    for first_good in (True, False):
        p = float(first_good)
        first = shipment_reward(
            spec, dict(shipments=1, passing_change=p, overdue_time=1)
        )
        second = shipment_reward(
            spec, dict(shipments=1, passing_change=0.5 - p, overdue_time=2)
        )
        assert first + second == pytest.approx(whole)


def test_config_roundtrip_and_rejects_stacked_or_discounted_task():
    spec = RewardSpecV3(
        task=ShipmentTaskReward(reference_jobs=256, reference_ticks=8192)
    )
    assert RewardSpecV3.model_validate_json(canonical_json(primitive(spec))) == spec
    with pytest.raises(ValueError, match="gamma=1"):
        TrainingOptionsV3(groups=("machine",), reward=spec)
    with pytest.raises(ValueError, match="cannot stack"):
        RewardSpecV3.model_validate(
            {"task": {}, "team": {"name": "builtin.reward_scale", "version": "1"}}
        )
    for params in (
        {"reference_jobs": 0},
        {"passing_weight": float("nan")},
        {"tardiness_weight": -1.0},
    ):
        with pytest.raises(ValueError):
            ShipmentTaskReward(**params)
    assert TrainingOptionsV3(groups=("machine",)).gamma == 0.99
    assert TrainingOptionsV3(groups=("machine",)).reward.task is None


def task_prepared(
    name, tmp_path, *, horizon=8, total=8, validation=False, smoothing=None
):
    config = api.load_config(ROOT / "configs/test/runs" / (name + ".yaml"))
    config.training.gamma = 1.0
    config.training.reward = RewardSpecV3(
        task=ShipmentTaskReward(), passing_rate_smoothing=smoothing
    )
    config.training.total_ticks = total
    config.training.ticks_per_update = 4
    config.scenario_overrides["tick_limit"] = horizon
    config.validation.enabled = validation
    if validation:
        config.validation.best_mode = "custom"
        config.validation.metric = "return"
        config.validation.direction = "max"
        config.validation.failure_policy = "ineligible"
        config.validation.every_updates = 1
        config.validation.replications = 1
    config.output.root = str(tmp_path)
    config.runtime.num_envs = 1
    config.runtime.sampling_processes = 0
    prepared = prepare(config)
    parameters = json.loads(prepared.parameters_json)
    parameters.update(batch_size=2)
    if config.training.algorithm == "dqn":
        parameters.update(warmup_ticks=0, target_update_ticks=4)
    else:
        parameters.update(n_epochs=1)
    return replace(prepared, parameters_json=canonical_json(parameters))


@pytest.mark.parametrize("name", ["train_all_ppo", "train_all_dqn"])
def test_real_task_tail_rewards_mid_batch_bootstrap_and_exact_resume(name, tmp_path):
    pytest.importorskip("ray")
    prepared = task_prepared(name, tmp_path)
    root, record, prepared = allocate(prepared, "training")
    session = TrainingSession(prepared, root, record)
    try:
        # Tiny engineering fixture: all original due dates are zero, including FIFO.
        sim = session.sims[0]
        sim.demands = {d: replace(row, due_at=0) for d, row in sim.demands.items()}
        session.step_update()
        saved = copy.deepcopy(session.state_dict())
        assert not session.sims[0].done  # Training cuts do not end the task.
        if name.endswith("dqn"):
            assert session.collector.pending
            assert all(
                not row["terminated"]
                for replay in session.replays.values()
                for row in replay.rows
            )
        session.step_update()
        sim = session.sims[0]
        assert sim.done and sim.status == "truncated"
        score = shipment_reward(
            prepared.config.training.reward.task, sim.privileged_task_totals()
        )
        assert sum(a["reward"] for a in session.actions) == pytest.approx(score)
        assert len(session.episode_results) == 1
        row = session.episode_results[0]
        assert row["return"] == pytest.approx(score)
        assert row["task_window_complete"]
        assert row["task_reward_contract"] == primitive(
            prepared.config.training.reward.task
        )
        assert episode_metrics(sim, task_reward=prepared.config.training.reward.task)[
            "task_window_complete"
        ]
        assert not session.collector.active and not session.collector.pending
        assert all(
            c["censored_truncations"] == 0 for c in session.collector.counts.values()
        )
        if name.endswith("dqn"):
            terminal = [
                row
                for replay in session.replays.values()
                for row in replay.rows
                if row["terminated"]
            ]
            assert terminal and any(row["reward"] < 0 for row in terminal)
        expected_actions = copy.deepcopy(session.actions)
        expected_replay = {
            g: copy.deepcopy(r.state_dict()) for g, r in session.replays.items()
        }
        expected_diagnostics = copy.deepcopy(session.learner_diagnostics())
        session.close()
        session = TrainingSession(prepared, root, record)
        session.restore(saved)
        session.step_update()
        assert session.actions == expected_actions
        assert session.learner_diagnostics() == expected_diagnostics
        assert {
            g: r.state_dict() for g, r in session.replays.items()
        } == expected_replay
        assert sum(a["reward"] for a in session.actions) == pytest.approx(score)
        # A reward contract mismatch rejects before mutable training state loads.
        wrong = copy.deepcopy(saved)
        wrong["task_reward_contract"]["passing_weight"] = 0.9
        old_tick = session.ticks
        with pytest.raises(ValueError, match="task reward contract"):
            session.restore(wrong)
        assert session.ticks == old_tick
    finally:
        session.close()


def test_task_validation_accepts_incomplete_shipping_window_and_maximizes_return(
    tmp_path,
):
    pytest.importorskip("ray")
    prepared = task_prepared("train_all_ppo", tmp_path, validation=True)
    root, record, prepared = allocate(prepared, "training")
    session = TrainingSession(prepared, root, record)
    try:
        session.step_update()
        selection = json.loads((root / "logs/selection-000001.json").read_text())
        assert selection["selected"] and selection["reason"] == "first_eligible"
        candidate = selection["candidate"]
        assert candidate["completed"] == candidate["episodes"] == 1
        assert candidate["metrics"]["makespan"] is None
        assert candidate["metrics"]["return"] == 0.0
        assert session.best_score == candidate
    finally:
        session.close()


def smoothing():
    from smartsom.config.extensions import PassingRateSmoothing

    return PassingRateSmoothing(pseudocount=9)


def ledger():
    from smartsom.experiments.passing_reward import PassingRewardLedger

    return PassingRewardLedger(ShipmentTaskReward(), smoothing())


def interval(book, tick, s, g, *, terminal=False, overdue=0):
    return book.interval(
        0,
        0,
        tick=tick,
        totals=dict(
            shipments=s, passing_change=g / s if s else 0.0, overdue_time=overdue
        ),
        good=g,
        terminal=terminal,
    )


@pytest.mark.parametrize(
    "counts",
    [
        [(0, 0)],
        [(1, 0)],
        [(1, 1)],
        [(1, 1), (2, 1)],
        [(2, 1)],
        [(1, 1), (3, 2), (3, 2)],
    ],
)
def test_passing_smoothing_complete_episode_closes_original_objective(counts):
    book = ledger()
    rows = [
        interval(book, i, s, g, terminal=i == len(counts), overdue=i * 3)
        for i, (s, g) in enumerate(counts, 1)
    ]
    s, g = counts[-1]
    expected = shipment_reward(
        ShipmentTaskReward(),
        dict(
            shipments=s,
            passing_change=g / s if s else 0.0,
            overdue_time=len(counts) * 3,
        ),
    )
    assert sum(row["training_reward"] for row in rows) == pytest.approx(
        expected, abs=1e-12
    )
    assert sum(
        row["quality_smoothed_reward"] + row["quality_terminal_compensation"]
        for row in rows
    ) == pytest.approx(0.3 * g / s if s else 0.0)
    assert book.summary(0, 0)["terminal_compensations"] == 1
    assert book.summary(0, 0)["quality_closure_error"] == pytest.approx(0, abs=1e-12)
    with pytest.raises(ValueError, match="already committed"):
        interval(book, len(counts) + 1, s, g, terminal=True)


def test_first_good_delivery_and_terminal_compensation_are_private_and_separate():
    sim, _, _ = ready_output(mode="dynamic")
    sim.scenario = replace(sim.scenario, tick_limit=3)
    book = ledger()
    rows = []
    while not sim.done:
        outcome = driver(sim).tick()
        quality = sim.privileged_output_quality()
        rows.append(
            book.interval(
                0,
                0,
                tick=sim.tick,
                totals=sim.privileged_task_totals(),
                good=quality["good_shipped"],
                terminal=sim.done,
            )
        )
        assert (
            not {
                "passing_true",
                "passing_smooth",
                "quality_smoothed_reward",
                "quality_terminal_compensation",
            }
            & outcome.keys()
        )
        assert (
            not {"passing_true", "passing_smooth", "good_shipped"}
            & sim.protocol.public_view().keys()
        )
    assert rows[0]["quality_smoothed_reward"] == pytest.approx(0.03)
    assert rows[0]["quality_terminal_compensation"] == 0
    assert rows[-1]["quality_smoothed_reward"] == 0
    assert rows[-1]["quality_terminal_compensation"] == pytest.approx(0.27)
    assert sum(row["training_reward"] for row in rows) == pytest.approx(
        episode_metrics(sim, task_reward=ShipmentTaskReward())["return"]
    )


def test_immediate_normal_terminal_pays_point_zero_three_plus_point_two_seven():
    sim, _, _ = ready_output()
    driver(sim).tick()
    assert sim.done and sim.status == "completed"
    row = ledger().interval(
        0, 0, tick=sim.tick, totals=sim.privileged_task_totals(), good=1, terminal=True
    )
    assert row["quality_smoothed_reward"] == pytest.approx(0.03)
    assert row["quality_terminal_compensation"] == pytest.approx(0.27)


def test_4096_window_terminal_differs_from_256_update_and_budget_cuts():
    book = ledger()
    for tick in range(1, 4097):
        row = interval(book, tick, 1, 1, terminal=tick == 4096)
        if tick in (256, 2048, 4095):
            assert row["quality_terminal_compensation"] == 0
            assert book.summary(0, 0)["terminal_compensations"] == 0
            saved = book.state_dict()
            restored = ledger()
            restored.load_state_dict(saved)
            assert restored.state_dict() == saved
    assert row["quality_terminal_compensation"] == pytest.approx(0.27)
    assert book.summary(0, 0)["training_return"] == pytest.approx(0.5 / 128 + 0.3)


def test_smoothing_identity_rejects_missing_task_and_discounts():
    with pytest.raises(ValueError, match="explicit shipment"):
        TrainingOptionsV3(
            groups=("dispatcher",),
            gamma=1,
            reward=RewardSpecV3(passing_rate_smoothing=smoothing()),
        )
    with pytest.raises(ValueError, match="gamma=1"):
        TrainingOptionsV3(
            groups=("dispatcher",),
            gamma=0.99,
            reward=RewardSpecV3(
                task=ShipmentTaskReward(), passing_rate_smoothing=smoothing()
            ),
        )


@pytest.mark.parametrize("algorithm", ["ppo", "dqn"])
def test_terminal_compensation_reaches_owner_without_last_tick_decision(algorithm):
    """Collector/replay contract using a synthetic value input, not a live run."""
    from types import SimpleNamespace

    from smartsom.learning.production_collection import PhysicalCollector, Replay

    encoded = dict(context=[0.0], candidates=[[0.0]], prefix=[], mask=[True])
    policy = SimpleNamespace(value_input=lambda *args: (encoded, 1000.0))
    coordinator = SimpleNamespace(
        records=[],
        sim=SimpleNamespace(protocol=SimpleNamespace(public_view=lambda: {})),
    )
    key = (0, 0, "vehicle", "dispatcher")
    collector = PhysicalCollector(["dispatcher"], 1)
    collector.active[key] = "dispatcher"
    if algorithm == "dqn":
        collector.pending[key] = dict(
            input=encoded, action=0, reward=0.03, dt=2, choice_decision=True
        )
    else:
        collector.trajectories[key] = [
            dict(
                inputs=[],
                actions=[],
                value_input=encoded,
                log_probability=0.0,
                value=0.0,
                actor_mask=False,
                value_mask=True,
                reward=0.03,
                dt=2,
                terminated=False,
                bootstrap=1000.0,
            )
        ]
    collector.collect(
        0,
        0,
        coordinator,
        {},
        dict(dispatcher=policy),
        dict(dispatcher=0.27),
        algorithm=algorithm,
        terminated=True,
    )
    rows = collector.drain(algorithm, lam=1)["dispatcher"]
    assert not collector.pending and not collector.active
    if algorithm == "ppo":
        assert rows[-1]["bootstrap"] == 0 and rows[-1]["terminated"]
        assert rows[0]["return"] == pytest.approx(0.30)
    else:
        assert len(rows) == 1 and rows[0]["dt"] == 3 and rows[0]["terminated"]
        replay = Replay(2, 23)
        replay.add(rows[0])
        assert replay.rows[0]["reward"] == pytest.approx(0.30)


def test_partial_training_potential_does_not_replace_true_evaluation_objective():
    sim, _, _ = ready_output(mode="dynamic")
    driver(sim).tick()
    book = ledger()
    trained = book.interval(
        0, 0, tick=sim.tick, totals=sim.privileged_task_totals(), good=1, terminal=False
    )
    assert trained["quality_smoothed_reward"] == pytest.approx(0.03)
    row = episode_metrics(sim, task_reward=ShipmentTaskReward())
    assert row["return"] == pytest.approx(0.5 / 128 + 0.30)
    assert trained["training_reward"] == pytest.approx(0.5 / 128 + 0.03)
    assert "training_reward_contract" not in row


@pytest.mark.parametrize("name", ["train_all_ppo", "train_all_dqn"])
def test_real_smoothed_terminal_and_partial_budget_restore_without_double_payment(
    name, tmp_path
):
    prepared = task_prepared(name, tmp_path, horizon=8, total=12, smoothing=smoothing())
    root, record, prepared = allocate(prepared, "training")
    session = TrainingSession(prepared, root, record)
    saved = []
    try:
        for _ in range(3):
            session.step_update()
            saved.append(copy.deepcopy(session.state_dict()))
        expected_actions = copy.deepcopy(session.actions)
        expected_ledger = copy.deepcopy(session.training_reward.state_dict())
        assert session.training_reward.summary(0, 0)["terminal_compensations"] == 1
        assert session.training_reward.summary(0, 1)["terminal_compensations"] == 0
        assert session.training_reward.summary(0, 1)["terminal"] is False
        assert all(
            row["quality_terminal_compensation"] == 0
            for row in session.actions
            if row["episode"] == 1
        )
        for boundary, state in enumerate(saved):
            assert state["training_reward_state"]["episodes"][0]["terminal"] == (
                boundary > 0
            )
        session.close()
        for state in saved:
            session = TrainingSession(prepared, root, record)
            session.restore(state)
            while session.ticks < 12:
                session.step_update()
            assert session.actions == expected_actions
            assert session.training_reward.state_dict() == expected_ledger
            wrong = copy.deepcopy(state)
            wrong["training_reward_contract"]["passing_rate_smoothing"][
                "pseudocount"
            ] = 10
            before = session.ticks
            with pytest.raises(ValueError, match="training reward contract"):
                session.restore(wrong)
            assert session.ticks == before
            session.close()
    finally:
        session.close()


def assert_native_reward_state_equal(actual, expected):
    """Compare the complete continuation, including optimizer tensors and RNGs."""
    import numpy as np
    import torch

    if isinstance(expected, torch.Tensor):
        assert torch.equal(actual, expected)
    elif isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_native_reward_state_equal(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert type(actual) is type(expected) and len(actual) == len(expected)
        for a, b in zip(actual, expected, strict=True):
            assert_native_reward_state_equal(a, b)
    elif hasattr(expected, "snapshot"):
        assert actual.snapshot() == expected.snapshot()
    else:
        assert actual == expected


@pytest.mark.parametrize("name", ["train_all_ppo", "train_all_dqn"])
def test_live_nonzero_compensation_exact_native_continuation(name, tmp_path):
    """Three real ticks, a good shipment and native learning; no mocked rewards."""
    from test_v3_dispatch_liveness import loaded

    sim = ProductionSimulator(
        replace(scenario("auto", jobs=2), mode="finite", tick_limit=3),
        contract="v3",
    )
    vehicle, job = next(iter(sim.agvs)), next(iter(sim.jobs))
    # A hand-controlled initial physical state, as in the blind-output tests.
    # The second job remains real work, giving owners sparse decision spans.
    sim.jobs[job].update(step=1, quality="UNKNOWN", defective=False)
    output = next(o for o, role in sim.roles.items() if role == "system_output")
    loaded(sim, vehicle, job, output)
    prepared = task_prepared(name, tmp_path, horizon=3, total=3, smoothing=smoothing())
    recipe = json.loads(prepared.training_inputs_json)
    recipe["frozen_world"] = primitive(sim.scenario)
    prepared = replace(
        prepared,
        scenario_json=canonical_json(sim.scenario),
        training_inputs_json=canonical_json(recipe),
    )
    root, record, prepared = allocate(prepared, "training")
    session = TrainingSession(prepared, root, record)
    session.sims = [sim]
    saved = []
    try:
        for tick in range(1, 4):
            session.advance()
            saved.append(copy.deepcopy(session.state_dict()))
            summary = session.training_reward.summary(0, 0)
            assert summary["good"] == summary["shipments"] == 1
            assert summary["terminal_compensations"] == int(tick == 3)
            assert session.actions[-1]["quality_terminal_compensation"] == (
                pytest.approx(0.27) if tick == 3 else 0
            )
        assert session.actions[0]["quality_smoothed_reward"] == pytest.approx(0.03)
        assert sim.done and sim.status == "truncated"
        assert not session.collector.pending and not session.collector.active
        if name.endswith("dqn"):
            terminal_rows = [
                row for row in session.replays["dispatcher"].rows if row["terminated"]
            ]
            assert terminal_rows and all(row["reward"] >= 0.27 for row in terminal_rows)
            # The formerly loaded owner has no final-tick dispatcher decision;
            # its actual pending transition spans both physical ticks.
            assert vehicle not in dict(session.actions[-1]["actions"]["dispatchers"])
            assert any(row["dt"] == 2 for row in terminal_rows)
        else:
            assert all(
                rows[-1]["terminated"]
                and rows[-1]["bootstrap"] == 0
                and rows[-1]["reward"] == pytest.approx(0.27)
                for rows in session.collector.trajectories.values()
            )
            session.optimize_ppo()
        assert session.optimizations["dispatcher"] > 0
        expected = copy.deepcopy(session.state_dict())
        summary = session.training_reward.summary(0, 0)
        assert summary["quality_closure_error"] == pytest.approx(0)
        assert sum(row["reward"] for row in session.actions) == pytest.approx(
            session.episode_results[0]["return"]
        )
        session.close()
        for state in saved:
            session = TrainingSession(prepared, root, record)
            session.restore(copy.deepcopy(state))
            assert_native_reward_state_equal(session.state_dict(), state)
            while session.ticks < 3:
                session.advance()
            if name.endswith("ppo"):
                session.optimize_ppo()
            assert_native_reward_state_equal(session.state_dict(), expected)
            session.close()
        # Restoring the already optimized terminal checkpoint is a no-op at
        # this total budget: no ledger mutation, replay insertion or payment.
        session = TrainingSession(prepared, root, record)
        session.restore(copy.deepcopy(expected))
        session.step_update()
        assert_native_reward_state_equal(session.state_dict(), expected)
    finally:
        session.close()
