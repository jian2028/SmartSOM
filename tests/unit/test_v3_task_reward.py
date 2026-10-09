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


def task_prepared(name, tmp_path, *, horizon=8, total=8, validation=False):
    config = api.load_config(ROOT / "configs/test/runs" / (name + ".yaml"))
    config.training.gamma = 1.0
    config.training.reward = RewardSpecV3(task=ShipmentTaskReward())
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
