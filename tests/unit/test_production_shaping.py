"""Public progress, pressure, exact terminal closure and native DQN recovery."""

import copy

import pytest

from smartsom.config.codec import primitive
from smartsom.config.experiment_v3 import (
    RewardSpecV3,
    ShipmentTaskReward,
    TrainingOptionsV3,
)
from smartsom.config.extensions import PassingRateSmoothing, ProductionPotentialShaping
from smartsom.experiments.passing_reward import PassingRewardLedger
from smartsom.experiments.production_shaping import (
    ProductionShapingLedger,
    blocking_clocks,
    blocking_potential,
    make_training_reward,
    progress_potential,
)

TASK = ShipmentTaskReward(reference_jobs=1, reference_ticks=100)
ROLES = {"pre": "machine_pre", "post": "machine_post", "input": "system_input"}
CAPACITIES = {"pre": {"slot": 1}, "post": {"slot": 1}}


def view(tick=0, *, location="input", step=0, elapsed=0, status="IDLE"):
    job = "order/attempt/1"
    return {
        "tick": tick,
        "released": ["order"],
        "shipped": [],
        "jobs": {
            job: {
                "demand": "order",
                "attempt": 1,
                "step": step,
                "original_steps": [{"nominal_ticks": 10, "reference_ticks": 10}],
                "location": location,
                "quality": "UNKNOWN",
                "since": 0,
            }
        },
        "machines": {
            "machine": {
                "job": job if location == "machine" else None,
                "status": status,
                "elapsed": elapsed,
            }
        },
        "storage": {
            "pre": {"slot": [job] if location == "pre" else []},
            "post": {"slot": [job] if location == "post" else []},
        },
        "agvs": {"vehicle": {"job": job if location == "vehicle" else None}},
    }


def spec(**kwargs):
    return ProductionPotentialShaping(
        progress_weight=0.1, blocking_weight=0.1, **kwargs
    )


def test_progress_credits_work_and_required_drop_not_motion_or_oracle():
    config = spec(handoff_reference_ticks=10.0)
    values = [
        progress_potential(v, TASK, config, ROLES)
        for v in (
            view(),
            view(location="vehicle"),
            view(location="pre"),
            view(location="machine", status="PROCESSING", elapsed=5),
            view(location="machine", status="BLOCKED", step=1, elapsed=10),
            view(location="post", step=1),
            view(location="vehicle", step=1),
        )
    ]
    assert values == pytest.approx([0, 0, 1 / 3, 0.5, 2 / 3, 2 / 3, 2 / 3])
    shipped = view(step=1)
    shipped["shipped"] = ["order"]
    assert progress_potential(shipped, TASK, config, ROLES) == 1
    a = view(location="machine", status="PROCESSING", elapsed=5)
    b = copy.deepcopy(a)
    b["machines"]["machine"].update(remaining=900, nominal=800, mode="fast")
    b["jobs"]["order/attempt/1"]["defective"] = True
    b["announced"] = [{"demand_id": "future", "release_at": 9}]
    assert progress_potential(a, TASK, config, ROLES) == progress_potential(
        b, TASK, config, ROLES
    )


def test_replacement_old_scrap_and_external_fifo_cannot_farm_progress():
    v = view(location="post", step=1)
    row = v["jobs"]["order/attempt/1"]
    row["attempt"] = 2  # Core reports the current attempt on historical rows.
    assert progress_potential(v, TASK, spec(), ROLES) == 0
    row["attempt"] = 1
    row["quality"] = "FAIL"
    assert progress_potential(v, TASK, spec(), ROLES) == 0


def test_pressure_age_and_physical_unblocking_credit():
    config = spec(blocking_age_ticks=4, cargo_grace_ticks=4)
    v = view(0, location="machine", status="BLOCKED", step=1)
    clocks = blocking_clocks(v)
    assert blocking_potential(v, config, ROLES, CAPACITIES, clocks) == 0
    for tick in range(1, 5):
        v["tick"] = tick
        clocks = blocking_clocks(v, clocks)
    assert blocking_potential(v, config, ROLES, CAPACITIES, clocks) == pytest.approx(
        -1 / 3
    )
    freed = view(4, location="vehicle", step=1)
    assert blocking_clocks(freed, clocks) == {}
    assert blocking_potential(freed, config, ROLES, CAPACITIES, {}) == 0
    # Retargeting cannot erase physical cargo age.
    late = view(8, location="vehicle")
    late["agvs"]["vehicle"]["target"] = {"owner": "new-target"}
    assert blocking_potential(late, config, ROLES, CAPACITIES, {}) == pytest.approx(
        -1 / 3
    )
    queued = view(4, location="pre")
    assert blocking_potential(queued, config, ROLES, CAPACITIES, {}) == pytest.approx(
        -1 / 6
    )


@pytest.mark.parametrize("weights", [(0.1, 0), (0, 0.1), (0.1, 0.1), (0, 0)])
@pytest.mark.parametrize("smooth", [None, PassingRateSmoothing(pseudocount=9.0)])
def test_episode_telescoping_partial_cut_and_exact_restore(weights, smooth):
    config = ProductionPotentialShaping(
        progress_weight=weights[0], blocking_weight=weights[1]
    )
    book = ProductionShapingLedger(TASK, smooth, config)
    states = [
        view(),
        view(1, location="pre"),
        view(2, location="machine", status="PROCESSING", elapsed=1),
        view(3, location="post", step=1),
    ]
    total = 0
    for tick in range(1, 4):
        row = book.interval(
            0,
            0,
            tick=tick,
            totals={"shipments": 0, "passing_change": 0.0, "overdue_time": tick},
            good=0,
            terminal=tick == 3,
            before=states[tick - 1],
            after=states[tick],
            roles=ROLES,
            capacities=CAPACITIES,
        )
        total += row["training_reward"]
        assert book.summary(0, 0)["production_closure_error"] == pytest.approx(0)
        restored = ProductionShapingLedger(TASK, smooth, config)
        restored.load_state_dict(book.state_dict())
        assert restored.state_dict() == book.state_dict()
        book = restored
        if tick < 3:
            assert row["production_terminal_compensation"] == 0
    assert total == pytest.approx(-0.2 * 3 / 100)
    assert book.summary(0, 0)["pending_production_compensation"] == 0
    broken = book.state_dict()
    broken["episodes"][0]["progress_return"] += 1
    with pytest.raises(ValueError, match="closure"):
        book.load_state_dict(broken)
    with pytest.raises(ValueError, match="physical interval"):
        book.interval(
            0,
            0,
            tick=4,
            totals={},
            good=0,
            terminal=True,
            before=states[3],
            after=view(4),
            roles=ROLES,
            capacities=CAPACITIES,
        )


def test_nonzero_initial_potential_is_centered_and_base_factory_is_unchanged():
    initial = view(location="pre")
    after = copy.deepcopy(initial)
    after["tick"] = 1
    book = ProductionShapingLedger(TASK, None, spec())
    row = book.interval(
        0,
        0,
        tick=1,
        totals={"shipments": 0, "passing_change": 0.0, "overdue_time": 0},
        good=0,
        terminal=True,
        before=initial,
        after=after,
        roles=ROLES,
        capacities=CAPACITIES,
    )
    assert row["training_reward"] == pytest.approx(0)
    smoothing = PassingRateSmoothing(pseudocount=9.0)
    assert isinstance(make_training_reward(TASK, smoothing), PassingRewardLedger)
    assert make_training_reward(TASK, None) is None


def test_shaping_requires_gamma1_task_and_rejects_nonfinite_parameters():
    with pytest.raises(ValueError, match="explicit shipment"):
        TrainingOptionsV3(
            groups=("dispatcher",),
            gamma=1,
            reward=RewardSpecV3(production_potential_shaping=spec()),
        )
    with pytest.raises(ValueError, match="gamma=1"):
        TrainingOptionsV3(
            groups=("dispatcher",),
            gamma=0.99,
            reward=RewardSpecV3(task=TASK, production_potential_shaping=spec()),
        )
    for kwargs in (
        {"progress_weight": float("nan")},
        {"blocking_weight": -1.0},
        {"blocking_age_ticks": 0},
        {"cargo_grace_ticks": -1},
    ):
        with pytest.raises(ValueError):
            ProductionPotentialShaping(**kwargs)


@pytest.mark.parametrize("weights", [(0.1, 0), (0, 0.1), (0.1, 0.1)])
def test_real_dqn_restore_terminal_flush_and_true_objective_selection(
    tmp_path, weights
):
    pytest.importorskip("ray")
    from test_v3_task_reward import assert_native_reward_state_equal, task_prepared

    from smartsom.experiments.composable import TrainingSession, allocate

    smoothing = PassingRateSmoothing(pseudocount=9.0)
    prepared = task_prepared(
        "train_all_dqn", tmp_path, horizon=12, total=16, smoothing=smoothing
    )
    configured = prepared.config
    configured.training.reward = configured.training.reward.model_copy(
        update={
            "production_potential_shaping": ProductionPotentialShaping(
                progress_weight=weights[0],
                blocking_weight=weights[1],
                cargo_grace_ticks=0,
                blocking_age_ticks=2,
            ),
        }
    )
    from dataclasses import replace

    from smartsom.config.codec import canonical_json

    prepared = replace(prepared, config_json=canonical_json(primitive(configured)))
    root, record, prepared = allocate(prepared, "training")
    session = TrainingSession(prepared, root, record)
    try:
        session.step_update()
        saved = copy.deepcopy(session.state_dict())
        while session.ticks < 16:
            session.step_update()
        expected = copy.deepcopy(session.state_dict())
        row = session.training_reward.summary(0, 0)
        assert row["terminal"] and row["production_closure_error"] == pytest.approx(0)
        assert row["training_return"] == pytest.approx(row["objective_return"])
        assert session.episode_results[0]["return"] == pytest.approx(
            row["objective_return"]
        )
        assert not session.training_reward.summary(0, 1)["terminal"]
        assert any(
            r["terminated"] for replay in session.replays.values() for r in replay.rows
        )
        session.close()
        session = TrainingSession(prepared, root, record)
        session.restore(saved)
        while session.ticks < 16:
            session.step_update()
        assert_native_reward_state_equal(session.state_dict(), expected)
        wrong = copy.deepcopy(saved)
        wrong["training_reward_contract"]["production_potential_shaping"][
            "progress_weight"
        ] = 0.5
        with pytest.raises(ValueError, match="training reward contract"):
            session.restore(wrong)
    finally:
        session.close()


def test_zero_weight_shaping_preserves_actual_base_dqn_trajectory(tmp_path):
    """Adding the conversion with zero weights must not change learning or RNGs."""
    from dataclasses import replace

    from test_v3_task_reward import assert_native_reward_state_equal, task_prepared

    from smartsom.config.codec import canonical_json
    from smartsom.experiments.composable import TrainingSession, allocate

    snapshots = []
    for enabled in (False, True):
        prepared = task_prepared(
            "train_all_dqn",
            tmp_path / str(enabled),
            horizon=12,
            total=16,
            smoothing=PassingRateSmoothing(pseudocount=9.0),
        )
        if enabled:
            config = prepared.config
            config.training.reward = config.training.reward.model_copy(
                update={"production_potential_shaping": ProductionPotentialShaping()}
            )
            prepared = replace(prepared, config_json=canonical_json(primitive(config)))
        root, record, prepared = allocate(prepared, "training")
        session = TrainingSession(prepared, root, record)
        try:
            while session.ticks < 16:
                session.step_update()
            snapshots.append(copy.deepcopy(session.state_dict()))
        finally:
            session.close()
    base, zero = snapshots
    zero["reward"]["spec"]["reward"].pop("production_potential_shaping")
    for key in (
        "sims",
        "episodes",
        "ticks",
        "updates",
        "policies",
        "learners",
        "collector",
        "replays",
        "target_clock",
        "optimizations",
        "reward",
        "random",
        "torch_random",
        "numpy_random",
        "python_random",
    ):
        assert_native_reward_state_equal(zero[key], base[key])
    assert [a["reward"] for a in zero["actions"]] == [
        a["reward"] for a in base["actions"]
    ]
