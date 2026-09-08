"""Resource ownership, independent joint timelines and public-information checks."""

import os
import subprocess
import sys
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest
from test_fjsp_engine import flexible_case
from test_holding import holding_factory, move
from test_learning_projection import COMBINATIONS, module_case
from test_static_engine import crossing_case, op, problem

from smartsom.algorithms import SPTPolicy
from smartsom.dispatch import Dispatch, WaitNextEvent
from smartsom.domain import ArrivalPlan, Job, JobArrival
from smartsom.engine import SimulationResult, Simulator, replay, replay_schedule
from smartsom.learning.joint import JointActionCoordinator, PolicyStalledError
from smartsom.learning.resources import ResourceProjection, ResourceProjectionSpec

SPEC = ResourceProjectionSpec(4, 3, 6)


def indices(decision, *actions):
    result = {v.agent_id: 0 for v in decision.views}
    for action in actions:
        if isinstance(action, WaitNextEvent):
            continue
        owners = [
            (v.agent_id, i)
            for v in decision.views
            for i, c in enumerate(v.candidates, 1)
            if c.action == action
        ]
        assert len(owners) == 1
        owner, i = owners[0]
        assert not result[owner]
        result[owner] = i
    return result


@pytest.mark.parametrize("flags", COMBINATIONS)
def test_all_module_combinations_preserve_actions_trace_and_replay(flags):
    inp = module_case(*flags)
    core = Simulator(inp.factory, inp.workload, **inp.options())
    direct = Simulator(inp.factory, inp.workload, **inp.options())
    p = ResourceProjection(inp.factory, SPEC, transport_enabled=inp.transport_enabled)
    policy = SPTPolicy()
    while (ctx := core.current_decision) is not None:
        view = p.project(ctx)
        assert {c.action for v in view.views for c in v.candidates} == set(
            ctx.feasible_actions
        )
        assert len({c.action for v in view.views for c in v.candidates}) == sum(
            len(v.candidates) for v in view.views
        )
        for v in view.views:
            assert v.action_mask[0] == 1
            assert len(v.observations) == p.observation_sizes[v.role]
        action = policy.select_action(ctx)
        coordinator = JointActionCoordinator(view, indices(view, action))
        result = coordinator.execute(core)
        expected = direct.step(action)
        assert core.trace == direct.trace
    assert isinstance(result, SimulationResult) and result == expected
    assert replay(inp.factory, inp.workload, result.actions, **inp.options()) == result
    schedule = (
        result.execution_schedule
        if inp.transport_enabled or inp.buffers_enabled
        else result.schedule
    )
    exact = replay_schedule(inp.factory, inp.workload, schedule, **inp.options())
    assert (exact.execution_schedule, exact.makespan, exact.quality) == (
        result.execution_schedule,
        result.makespan,
        result.quality,
    )


def test_independent_same_tick_dispatch_and_clock_boundary():
    f, w, _ = crossing_case()
    sim = Simulator(f, w)
    p = ResourceProjection(f, SPEC)
    view = p.project(sim.current_decision)
    c = JointActionCoordinator(
        view, indices(view, Dispatch("C1", "standard"), Dispatch("D1", "standard"))
    )
    outcome = c.execute(sim)
    assert outcome.simulation_time == 3
    assert [r.disposition for r in c.records] == ["accepted", "accepted"]
    view = p.project(outcome)
    result = JointActionCoordinator(
        view, indices(view, Dispatch("C2", "standard"), Dispatch("D2", "standard"))
    ).execute(sim)
    assert result.makespan == 5
    assert [
        (x.operation_id, x.start_time, x.completion_time) for x in result.schedule
    ] == [
        ("C1", 0, 2),
        ("D1", 0, 3),
        ("C2", 3, 4),
        ("D2", 3, 5),
    ]


def test_machine_conflict_rejected_without_resampling():
    f, w = flexible_case()
    sim = Simulator(f, w)
    p = ResourceProjection(f, SPEC)
    view = p.project(sim.current_decision)
    c = JointActionCoordinator(
        view, indices(view, Dispatch("A1", "slow"), Dispatch("A1", "fast"))
    )
    outcome = c.execute(sim)
    assert outcome.simulation_time == 0
    assert [r.disposition for r in c.records] == ["accepted", "job_claimed"]
    assert c.actions == [Dispatch("A1", "slow")]
    assert Dispatch("B1", "standard") in outcome.feasible_actions


def test_zero_trip_does_not_reinterpret_second_proposal_as_reroute():
    f, w = flexible_case()
    f = holding_factory(f, zero=True, count=2)
    sim = Simulator(f, w, transport_enabled=True)
    p = ResourceProjection(f, SPEC, transport_enabled=True)
    view = p.project(sim.current_decision)
    c = JointActionCoordinator(
        view, indices(view, move("A", "M1", "V1"), move("A", "M2", "V2"))
    )
    outcome = c.execute(sim)
    assert outcome.simulation_time == 0
    assert len(c.actions) == 1
    assert c.records[-1].disposition == "job_claimed"
    # The stale proposal IS now physically legal as a new trip. It still cannot run.
    assert move("A", "M2", "V2") in outcome.feasible_actions
    view = p.project(outcome)
    c = JointActionCoordinator(
        view, indices(view, Dispatch("A1", "slow"), move("A", "M2", "V2"))
    )
    c.execute(sim)
    assert c.actions == [Dispatch("A1", "slow")]
    assert c.records[-1].disposition == "job_claimed"


@pytest.mark.parametrize("bad", [True, 1.0, -1, 10000, "1"])
def test_all_indices_validate_before_first_action(bad):
    f, w, _ = crossing_case()
    sim = Simulator(f, w)
    view = ResourceProjection(f, SPEC).project(sim.current_decision)
    actions = indices(view, Dispatch("C1", "standard"))
    actions["machine:M2"] = bad
    before = sim.trace
    with pytest.raises(ValueError):
        JointActionCoordinator(view, actions)
    assert sim.trace == before


def test_stale_missing_agents_noop_and_public_wait():
    f, w, _ = crossing_case()
    sim = Simulator(f, w)
    p = ResourceProjection(f, SPEC)
    view = p.project(sim.current_decision)
    with pytest.raises(ValueError, match="exactly"):
        JointActionCoordinator(view, {})
    with pytest.raises(PolicyStalledError):
        JointActionCoordinator(view, indices(view)).execute(sim)
    assert not any(type(t).__name__ == "WaitRecord" for t in sim.trace)
    sim.step(Dispatch("C1", "standard"))
    with pytest.raises(ValueError, match="stale"):
        JointActionCoordinator(view, indices(view)).execute(sim)
    view = p.project(sim.current_decision)
    c = JointActionCoordinator(view, indices(view))
    outcome = c.execute(sim)
    assert c.actions == [WaitNextEvent()] and outcome.simulation_time == 2


def test_hidden_job_counts_durations_and_binding_capacity():
    f, w = problem(Job("Z", (op("Z1", "M1", 5),)), Job("A", (op("A1", "M2", 999),)))
    arrivals = ArrivalPlan((JobArrival("Z", 0, 0), JobArrival("A", 10, 10)))
    p = ResourceProjection(f, SPEC)
    a = p.project(Simulator(f, w, arrivals=arrivals).current_decision)
    _, w2 = problem(
        Job("Z", (op("Z1", "M1", 5),)),
        Job("B", (op("B1", "M2", 2),)),
        Job("C", (op("C1", "M1", 99),)),
    )
    b = ResourceProjection(f, SPEC).project(
        Simulator(
            f,
            w2,
            arrivals=ArrivalPlan(
                (
                    JobArrival("Z", 0, 0),
                    JobArrival("B", 20, 20),
                    JobArrival("C", 30, 30),
                )
            ),
        ).current_decision
    )
    assert a == b
    with pytest.raises(FrozenInstanceError):
        a.views[0].agent_id = "changed"
    with pytest.raises(ValueError, match="capacity"):
        ResourceProjection(f, replace(SPEC, max_jobs=1)).project(
            Simulator(f, w).current_decision
        )


def test_actual_duration_and_outage_future_do_not_leak():
    from smartsom.domain import (
        MachineOutage,
        MachineOutagePlan,
        ProcessingTime,
        ProcessingTimePlan,
    )

    f, w = problem(Job("A", (op("A1", "M1", 4),)), Job("B", (op("B1", "M2", 2),)))
    views = []
    for actual, repair in ((9, 12), (30, 20)):
        plan = ProcessingTimePlan(
            (
                ProcessingTime("A1", "standard", 4, actual),
                ProcessingTime("B1", "standard", 2, 2),
            )
        )
        sim = Simulator(
            f,
            w,
            processing_times=plan,
            machine_events=MachineOutagePlan((MachineOutage("M1", 7, repair),)),
        )
        sim.step(Dispatch("A1", "standard"))
        views.append(ResourceProjection(f, SPEC).project(sim.current_decision))
    assert views[0] == views[1]
    # Two machine IDs, present/up, four holding phases, four job slots, then
    # nominal-duration and elapsed-time presence/value pairs.
    assert views[0].for_agent("machine:M1").local_features[12:16] == (1, 0.04, 1, 0)


def test_probability_and_draw_hidden_and_reordered_inputs():
    from smartsom.domain.quality import (
        QualityDraw,
        QualityDrawPlan,
        QualityMode,
        QualitySpeedSpec,
    )
    from smartsom.modules.quality import prepare_quality

    f, w = flexible_case()
    views = []
    for rate, draw in ((".1", 0), (".9", 2**53 - 1)):
        factory = replace(
            f,
            machines=tuple(reversed(f.machines)),
            quality_speed=QualitySpeedSpec((QualityMode("Q", "1.0", rate),)),
        )
        q = prepare_quality(
            factory,
            w,
            QualityDrawPlan(
                tuple(QualityDraw(o.operation_id, draw) for o in w.operations)
            ),
        )
        sim = Simulator(factory, w, quality=q, quality_probability_visibility="hidden")
        views.append(ResourceProjection(factory, SPEC).project(sim.current_decision))
    assert views[0] == views[1]


def test_pre_step_write_failure_retains_no_unexecuted_physical_action():
    f, w, _ = crossing_case()
    sim = Simulator(f, w)
    v = ResourceProjection(f, SPEC).project(sim.current_decision)
    c = JointActionCoordinator(
        v, indices(v, Dispatch("C1", "standard"), Dispatch("D1", "standard"))
    )
    before = sim.trace

    def fail(*args):
        raise OSError("full disk")

    with pytest.raises(OSError, match="full disk"):
        c.execute(sim, on_step=fail)
    assert c.actions == [] and sim.trace == before
    assert all(r.disposition == "execution_failed" for r in c.records)


def test_base_import_independence_and_hash_seed(tmp_path):
    root = Path(__file__).resolve().parents[2]
    script = """
import builtins, hashlib
real = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in ('numpy','torch','ray','gymnasium','pettingzoo'):
        raise ImportError(name)
    return real(name, *args, **kwargs)
builtins.__import__ = guarded
from smartsom.learning.resources import ResourceProjection, ResourceProjectionSpec
from smartsom.learning.joint import JointActionCoordinator
from smartsom.domain import FactorySpec, Machine, WorkloadInstance, Order, Job, Operation, ProcessingMode
from smartsom.engine import Simulator
f = FactorySpec(tuple(Machine(x) for x in {'M1','M2'}))
w = WorkloadInstance((Order('O',(Job('J',(Operation('op',(ProcessingMode('mode','M1',2),)),)),)),))
d = ResourceProjection(f, ResourceProjectionSpec(2,2,2)).project(Simulator(f,w).current_decision)
print(hashlib.sha256(repr(d).encode()).hexdigest())
"""
    outputs = []
    for seed in ("1", "97"):
        r = subprocess.run(
            [sys.executable, "-c", script],
            cwd=tmp_path,
            env={**os.environ, "PYTHONPATH": str(root / "src"), "PYTHONHASHSEED": seed},
            capture_output=True,
            text=True,
        )
        assert r.returncode == 0, r.stderr
        outputs.append(r.stdout)
    assert outputs[0] == outputs[1]
