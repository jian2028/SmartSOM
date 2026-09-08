"""Shared projection acceptance, runnable without Gym, NumPy, Ray or Torch."""

import os
import subprocess
import sys
from dataclasses import FrozenInstanceError, replace
from itertools import product
from pathlib import Path

import pytest
from test_fjsp_engine import flexible_case
from test_holding import holding_factory
from test_static_engine import op, problem

from smartsom.algorithms import SPTPolicy
from smartsom.dispatch import WaitNextEvent
from smartsom.domain import (
    ArrivalPlan,
    Job,
    JobArrival,
    MachineBuffers,
    MachineOutage,
    MachineOutagePlan,
    ProcessingTime,
    ProcessingTimePlan,
)
from smartsom.domain.quality import (
    QualityDraw,
    QualityDrawPlan,
    QualityMode,
    QualitySpeedSpec,
)
from smartsom.engine import SimulationResult, Simulator, replay, replay_schedule
from smartsom.learning import LearningProjection, ProjectionSpec, validate_capacity
from smartsom.learning.episode import EpisodeInput
from smartsom.learning.projection import visible_future_event
from smartsom.modules.quality import prepare_quality

SPEC = ProjectionSpec(4, 3, 6)


def module_case(
    ja, mb, upt, agv, buffers, quality, holding, trigger="dispatch_available"
):
    f, w = problem(
        Job("A", (op("A1", "M1", 2), op("A2", "M2", 3, "A1"))),
        Job("B", (op("B1", "M2", 2),)),
    )
    f = holding_factory(f, capacity=1, count=2)
    f = replace(
        f,
        buffers=tuple(MachineBuffers(m.machine_id, 1, 1) for m in f.machines),
        quality_speed=QualitySpeedSpec(
            (QualityMode("M0", "1.2", ".1"), QualityMode("M1", ".8", ".5"))
        ),
    )
    times = (
        ProcessingTimePlan(
            tuple(
                ProcessingTime(
                    o.operation_id,
                    m.processing_mode_id,
                    m.nominal_ticks,
                    m.nominal_ticks + 1,
                )
                for o in w.operations
                for m in o.modes
            )
        )
        if upt
        else None
    )
    q = (
        prepare_quality(
            f,
            w,
            QualityDrawPlan(
                tuple(QualityDraw(o.operation_id, 2**51) for o in w.operations)
            ),
            processing_times=times,
        )
        if quality
        else None
    )
    return EpisodeInput(
        f,
        w,
        arrivals=ArrivalPlan((JobArrival("A", 0, 0), JobArrival("B", 2, 1)))
        if ja
        else None,
        decision_trigger=trigger,
        processing_times=times,
        machine_events=MachineOutagePlan((MachineOutage("M1", 1, 3),)) if mb else None,
        transport_enabled=agv,
        buffers_enabled=buffers,
        holding_buffer_enabled=holding,
        quality=q,
    )


COMBINATIONS = [
    (*flags, trigger)
    for flags in product((False, True), repeat=7)
    if not flags[6] or flags[3]
    for trigger in (
        ("dispatch_available", "arrival_event") if flags[0] else ("dispatch_available",)
    )
]


@pytest.mark.parametrize("flags", COMBINATIONS)
def test_every_module_combination_projects_exact_core_actions_and_replays(flags):
    inputs = module_case(*flags)
    validate_capacity(SPEC, inputs.workload, inputs.quality)
    sim = Simulator(inputs.factory, inputs.workload, **inputs.options())
    p = LearningProjection(inputs.factory, SPEC)
    policy = SPTPolicy()
    while (ctx := sim.current_decision) is not None:
        view = p.project(ctx)
        assert len(view.observations) == p.observation_size
        assert {
            a
            for a in view.actions
            if a is not None and not isinstance(a, WaitNextEvent)
        } == set(ctx.feasible_actions)
        action = policy.select_action(ctx)
        assert view.decode(view.actions.index(action)) == action
        result = sim.step(action)
    assert isinstance(result, SimulationResult)
    assert (
        replay(inputs.factory, inputs.workload, result.actions, **inputs.options())
        == result
    )
    schedule = (
        result.execution_schedule
        if inputs.transport_enabled or inputs.buffers_enabled
        else result.schedule
    )
    exact = replay_schedule(
        inputs.factory, inputs.workload, schedule, **inputs.options()
    )
    assert exact.makespan == result.makespan and exact.quality == result.quality


def test_hidden_job_count_and_attributes_do_not_change_any_projection():
    f, w = problem(Job("Z", (op("Z1", "M1", 5),)), Job("A", (op("A1", "M2", 100),)))
    arrivals = ArrivalPlan((JobArrival("Z", 0, 0), JobArrival("A", 10, 10)))
    p = LearningProjection(f, SPEC)
    sim = Simulator(f, w, arrivals=arrivals)
    first = p.project(sim.current_decision)
    _, other = problem(
        Job("Z", (op("Z1", "M1", 5),)),
        Job("B", (op("B1", "M2", 2),)),
        Job("C", (op("C1", "M1", 999), op("C2", "M2", 99, "C1"))),
    )
    other_arrivals = ArrivalPlan(
        (JobArrival("Z", 0, 0), JobArrival("B", 20, 20), JobArrival("C", 30, 30))
    )
    changed = LearningProjection(f, SPEC).project(
        Simulator(f, other, arrivals=other_arrivals).current_decision
    )
    assert first == changed
    assert not visible_future_event(sim.current_decision)
    sim.step(
        first.decode(next(i for i, a in enumerate(first.actions) if a is not None))
    )
    next_view = p.project(sim.current_decision)
    assert tuple(b.job_id for b in next_view.bindings) == ("Z", "A")
    assert next_view.bindings[0] == first.bindings[0]
    with pytest.raises(FrozenInstanceError):
        first.bindings[0].job_id = "changed"


@pytest.mark.parametrize("bad", [True, 1.5, -1, 0])
def test_invalid_dimensions_rejected(bad):
    with pytest.raises(ValueError):
        ProjectionSpec(bad, 2, 3)


def test_capacity_preflight_counts_expanded_quality_modes():
    inputs = module_case(False, False, False, False, False, True, False)
    with pytest.raises(ValueError, match="capacity"):
        validate_capacity(ProjectionSpec(4, 3, 1), inputs.workload, inputs.quality)
    with pytest.raises(ValueError, match="capacity"):
        validate_capacity(ProjectionSpec(1, 3, 6), inputs.workload)
    with pytest.raises(ValueError, match="capacity"):
        validate_capacity(ProjectionSpec(4, 1, 6), inputs.workload)


def test_semantic_order_and_mask_rejection():
    f, w = flexible_case()
    ctx = Simulator(f, w).current_decision
    a = LearningProjection(f, SPEC).project(ctx)
    other = replace(
        ctx,
        jobs=tuple(
            replace(
                j,
                operations=tuple(
                    replace(op, modes=tuple(reversed(op.modes)))
                    for op in reversed(j.operations)
                ),
            )
            for j in reversed(ctx.jobs)
        ),
        operations=tuple(reversed(ctx.operations)),
        candidates=tuple(reversed(ctx.candidates)),
        machines=tuple(reversed(ctx.machines)),
    )
    assert (
        LearningProjection(
            replace(f, machines=tuple(reversed(f.machines))), SPEC
        ).project(other)
        == a
    )
    for bad in (
        True,
        -1,
        len(a.actions),
        1.5,
        next(i for i, x in enumerate(a.actions) if x is None),
    ):
        with pytest.raises(ValueError):
            a.decode(bad)


def test_hidden_probability_and_draws_not_encoded():
    inp = module_case(False, False, False, True, True, True, True)
    options = inp.options() | {"quality_probability_visibility": "hidden"}
    first = LearningProjection(inp.factory, SPEC).project(
        Simulator(inp.factory, inp.workload, **options).current_decision
    )
    factory = replace(
        inp.factory,
        quality_speed=QualitySpeedSpec(
            (QualityMode("M0", "1.2", ".9"), QualityMode("M1", ".8", ".99"))
        ),
    )
    quality = prepare_quality(
        factory,
        inp.workload,
        QualityDrawPlan(
            tuple(QualityDraw(o.operation_id, 0) for o in inp.workload.operations)
        ),
    )
    options["quality"] = quality
    assert (
        LearningProjection(factory, SPEC).project(
            Simulator(factory, inp.workload, **options).current_decision
        )
        == first
    )


def test_public_wait_witness_does_not_require_private_calendar():
    f, w = flexible_case()
    sim = Simulator(f, w)
    assert not visible_future_event(sim.current_decision)
    sim.step(
        next(
            c.action
            for c in sim.current_decision.candidates
            if c.action.processing_mode_id == "slow"
        )
    )
    assert visible_future_event(sim.current_decision)
    assert visible_future_event(
        replace(
            sim.current_decision,
            operations=(),
            machines=(),
            jobs=tuple(replace(j, release_at=100) for j in sim.current_decision.jobs),
        )
    )


def test_projection_import_has_no_optional_or_orchestration_dependencies():
    code = """
import sys
from smartsom.learning import LearningProjection
assert not any(x.split('.')[0] in {'numpy','gymnasium','torch','ray','stable_baselines3','sb3_contrib'} for x in sys.modules)
assert 'smartsom.config' not in sys.modules and 'smartsom.experiments' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_hash_seed_and_working_directory_do_not_change_projection(tmp_path):
    root = Path(__file__).resolve().parents[2]
    code = """
from test_learning_projection import module_case, SPEC
from smartsom.learning import LearningProjection
from smartsom.engine import Simulator
from smartsom.algorithms import SPTPolicy
from smartsom.config.codec import digest
x = module_case(True, True, True, True, True, True, True, 'arrival_event')
s = Simulator(x.factory, x.workload, **x.options())
p = LearningProjection(x.factory, SPEC)
rows = []
while s.current_decision is not None:
    rows.append(p.project(s.current_decision))
    s.step(SPTPolicy().select_action(s.current_decision))
print(digest(rows))
"""
    outputs = []
    for seed, cwd in (("1", root), ("42", tmp_path)):
        env = dict(
            os.environ,
            PYTHONHASHSEED=seed,
            PYTHONPATH=os.pathsep.join((str(root / "src"), str(root / "tests/unit"))),
        )
        outputs.append(
            subprocess.run(
                [sys.executable, "-c", code],
                env=env,
                cwd=cwd,
                text=True,
                capture_output=True,
                check=True,
            ).stdout
        )
    assert outputs[0] == outputs[1]
