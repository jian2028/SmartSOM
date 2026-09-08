"""Independent item-10 hand outcomes and composition through the public engine."""

import itertools
from dataclasses import FrozenInstanceError, replace
from decimal import Decimal, localcontext

import pytest

from smartsom.algorithms import FirstFeasiblePolicy, SPTPolicy
from smartsom.config import resolve_run
from smartsom.config.codec import primitive
from smartsom.dispatch import Dispatch, Transfer, WaitUntil
from smartsom.domain import (
    FactorySpec,
    Job,
    Machine,
    Operation,
    Order,
    ProcessingMode,
    WorkloadInstance,
)
from smartsom.domain.buffers import MachineBuffers
from smartsom.domain.machine_events import MachineOutage, MachineOutagePlan
from smartsom.domain.processing_times import ProcessingTime, ProcessingTimePlan
from smartsom.domain.quality import (
    MachineQualityModes,
    QualityDraw,
    QualityDrawPlan,
    QualityMode,
    QualitySpeedSpec,
    quality_mode_id,
)
from smartsom.domain.transport import TransportDestination
from smartsom.engine import Simulator, replay, replay_schedule
from smartsom.engine.simulator import InvalidActionError
from smartsom.modules.quality import prepare_quality, scaled_ticks
from smartsom.trace.records import InspectionRecord, QualityRecord
from smartsom.workloads.quality import generate_quality


def modes():
    return tuple(
        QualityMode(q, scale, rate)
        for q, scale, rate in (
            ("M0", "1.2", ".01"),
            ("M1", "1", ".018"),
            ("M2", ".8", ".03"),
        )
    )


def hand():
    f = FactorySpec(
        (Machine("M1"), Machine("M2")), quality_speed=QualitySpeedSpec(modes())
    )
    w = WorkloadInstance(
        (
            Order(
                "O",
                (
                    Job(
                        "J",
                        (
                            Operation("A1", (ProcessingMode("base", "M1", 10),)),
                            Operation(
                                "A2", (ProcessingMode("base", "M2", 10),), ("A1",)
                            ),
                        ),
                    ),
                ),
            ),
        )
    )
    draws = QualityDrawPlan((QualityDraw("A1", 2**47), QualityDraw("A2", 2**52)))
    return f, w, draws


def action(op, mode):
    return Dispatch(op, quality_mode_id("base", mode))


@pytest.mark.parametrize(
    "mode,end,passed", [("M0", 24, True), ("M1", 20, False), ("M2", 16, False)]
)
def test_hand_and_both_replays(mode, end, passed):
    f, w, d = hand()
    q = prepare_quality(f, w, d)
    simulator = Simulator(f, w, quality=q)
    mid = simulator.step(action("A1", mode))
    assert mid.job_quality[0].passed is None
    assert mid.operations[0].actual_processing_ticks == end // 2
    result = simulator.step(action("A2", mode))
    assert [(x.start_time, x.completion_time) for x in result.schedule] == [
        (0, end // 2),
        (end // 2, end),
    ]
    assert result.makespan == end
    assert result.quality.jobs[0].passed is passed
    assert result.quality.passing_rate == int(passed)
    assert result.quality.operations[1].defective is False
    assert len([x for x in result.trace if isinstance(x, QualityRecord)]) == 2
    assert Simulator(f, w, quality=q).run(SPTPolicy(mode)) == result
    assert Simulator(f, w, quality=q).run(FirstFeasiblePolicy(mode)) == result
    assert replay(f, w, result.actions, quality=q) == result
    assert replay_schedule(f, w, tuple(reversed(result.schedule)), quality=q) == result
    with pytest.raises(FrozenInstanceError):
        mid.job_quality[0].passed = True
    with pytest.raises(FrozenInstanceError):
        q.modes[0].actual_ticks = 42


def test_overrides_replace_whole_table_and_retain_equal_choices():
    f, w, d = hand()
    f = replace(
        f,
        quality_speed=QualitySpeedSpec(
            modes(),
            (
                MachineQualityModes(
                    "M2",
                    (
                        QualityMode("equal-a", "1", "0"),
                        QualityMode("equal-b", "1", "0"),
                    ),
                ),
            ),
        ),
    )
    q = prepare_quality(f, w, d)
    assert {x.mode.quality_mode_id for x in q.modes if x.machine_id == "M2"} == {
        "equal-a",
        "equal-b",
    }
    assert len(q.modes) == 5
    s = Simulator(f, w, quality=q)
    s.step(action("A1", "M2"))
    before = s.trace
    with pytest.raises(InvalidActionError):
        s.step(action("A2", "M0"))
    assert s.trace == before
    result = s.step(action("A2", "equal-b"))
    assert result.schedule[-1].processing_mode_id == quality_mode_id("base", "equal-b")


@pytest.mark.parametrize(
    "rate,draw,failed",
    [
        ("0", 0, False),
        ("1", 2**53 - 1, True),
        (".5", 2**52, False),
        (".5", 2**52 - 1, True),
    ],
)
def test_exact_probability_boundary(rate, draw, failed):
    f, w, _ = hand()
    f = replace(f, quality_speed=QualitySpeedSpec((QualityMode("Q", "1", rate),)))
    d = QualityDrawPlan((QualityDraw("A1", draw), QualityDraw("A2", draw)))
    r = Simulator(f, w, quality=prepare_quality(f, w, d)).run(SPTPolicy())
    assert all(x.defective is failed for x in r.quality.operations)
    assert r.quality.defective_jobs == int(failed)


def test_upt_then_scale_and_no_pause_redraw():
    f, w, d = hand()
    p = ProcessingTimePlan(
        (ProcessingTime("A1", "base", 10, 12), ProcessingTime("A2", "base", 10, 3))
    )
    q = prepare_quality(f, w, d, processing_times=p)
    outages = MachineOutagePlan((MachineOutage("M1", 2, 4), MachineOutage("M1", 6, 8)))
    kw = dict(quality=q, processing_times=p, machine_events=outages)
    r = Simulator(f, w, **kw).run(SPTPolicy("M2"))
    assert [(x.start_time, x.completion_time) for x in r.schedule] == [
        (0, 14),
        (14, 16),
    ]
    assert len([x for x in r.trace if isinstance(x, QualityRecord)]) == 2
    assert len([x for x in r.trace if x.kind == "resume"]) == 2
    assert replay(f, w, r.actions, **kw) == r
    assert replay_schedule(f, w, r.schedule, **kw) == r
    with pytest.raises(ValueError, match="disagrees"):
        Simulator(f, w, quality=q)  # Base UPT may not silently disappear.
    assert scaled_ticks(1, Decimal(".01")) == 1
    assert scaled_ticks(5, Decimal(".5")) == 3
    with localcontext() as ctx:
        ctx.prec = 1
        assert scaled_ticks(12, Decimal("1.2")) == 14


def test_completion_at_breakdown_is_checked_once():
    f, w, d = hand()
    q = prepare_quality(f, w, d)
    r = Simulator(
        f,
        w,
        quality=q,
        machine_events=MachineOutagePlan((MachineOutage("M1", 10, 15),)),
    ).run(SPTPolicy("M1"))
    assert r.quality.operations[0].completion_time == 10
    assert not any(x.kind == "pause" for x in r.trace)


def test_blocked_quality_remains_hidden_until_output():
    f, w, d = hand()
    f = replace(f, buffers=(MachineBuffers("M1", 0, 0), MachineBuffers("M2", 0, 0)))
    q = prepare_quality(f, w, d)
    script = (
        Transfer("J", TransportDestination("machine", "M1")),
        action("A1", "M1"),
        WaitUntil(15),
        Transfer("J", TransportDestination("machine", "M2")),
        action("A2", "M1"),
        WaitUntil(30),
        Transfer("J", TransportDestination("output")),
    )
    s = Simulator(f, w, quality=q, buffers_enabled=True)
    for move in script[:-1]:
        view = s.step(move)
        assert view.job_quality[0].passed is None
    r = s.step(script[-1])
    assert [x.completion_time for x in r.schedule] == [10, 25]
    assert r.quality.jobs[0].inspection_time == 30
    assert r.makespan == 30
    assert len([x for x in r.trace if isinstance(x, InspectionRecord)]) == 1
    assert (
        replay_schedule(
            f, w, r.execution_schedule, quality=q, buffers_enabled=True
        ).quality
        == r.quality
    )


def test_hidden_probabilities_and_draws_cannot_change_preinspection_views():
    f, w, d = hand()
    f2 = replace(
        f,
        quality_speed=QualitySpeedSpec(
            tuple(replace(x, error_rate="1") for x in modes())
        ),
    )
    d2 = QualityDrawPlan((QualityDraw("A1", 0), QualityDraw("A2", 0)))
    s1 = Simulator(
        f, w, quality=prepare_quality(f, w, d), quality_probability_visibility="hidden"
    )
    s2 = Simulator(
        f2,
        w,
        quality=prepare_quality(f2, w, d2),
        quality_probability_visibility="hidden",
    )
    assert s1.current_decision == s2.current_decision
    assert s1.step(action("A1", "M0")) == s2.step(action("A1", "M0"))
    assert all(x.error_rate is None for x in s1.current_decision.quality_modes)
    a = s1.step(action("A2", "M0"))
    b = s2.step(action("A2", "M0"))
    assert a.schedule == b.schedule
    assert a.quality.jobs[0].passed is True and b.quality.jobs[0].passed is False
    public = Simulator(f, w, quality=prepare_quality(f, w, d)).run(SPTPolicy("M0"))
    assert a == public


@pytest.mark.parametrize(
    "agv,ja,mb,upt,buffers", tuple(itertools.product((False, True), repeat=5))
)
def test_all_module_combinations(agv, ja, mb, upt, buffers):
    r = resolve_run("configs/runs/buffers_combined.yaml")
    f = replace(r.factory, quality_speed=QualitySpeedSpec(modes()))
    p = r.processing_times if upt else None
    generated = generate_quality(r.workload, 99)
    draws = replace(
        generated,
        operations=(
            replace(generated.operations[0], draw=0),
            *generated.operations[1:],
        ),
    )
    q = prepare_quality(f, r.workload, draws, processing_times=p)
    for trigger in (
        ("dispatch_available", "arrival_event") if ja else ("dispatch_available",)
    ):
        kw = dict(
            quality=q,
            arrivals=r.arrivals if ja else None,
            decision_trigger=trigger,
            machine_events=r.machine_events if mb else None,
            processing_times=p,
            transport_enabled=agv,
            buffers_enabled=buffers,
        )
        for fixed in (None, "M0"):
            sim = Simulator(f, r.workload, **kw)
            observations = []
            policy = SPTPolicy(fixed)
            while sim.current_decision is not None:
                c = sim.current_decision
                observations.append(c)
                assert {x.job_id for x in c.job_quality} == {j.job_id for j in c.jobs}
                assert {x.operation_id for x in c.quality_modes} == {
                    op.operation_id for op in c.operations
                }
                result = sim.step(policy.select_action(c))
            assert result.quality.total_jobs == len(result.quality.jobs)
            assert len(result.quality.operations) == len(r.workload.operations)
            assert result.quality.defective_jobs >= 1
            assert replay(f, r.workload, result.actions, **kw) == result
            schedule = result.execution_schedule if agv or buffers else result.schedule
            again = replay_schedule(f, r.workload, schedule, **kw)
            assert again.execution_schedule == result.execution_schedule
            assert again.quality == result.quality
            assert again.makespan == result.makespan
            # Views are detached and JSON encodable, without unpublished outcomes.
            assert primitive(observations)


def test_invalid_dispatch_and_mode_changes_are_atomic_during_processing_and_pause():
    f, w, d = hand()
    other = Job("B", (Operation("B1", (ProcessingMode("base", "M2", 20),)),))
    w = replace(w, orders=(replace(w.orders[0], jobs=(*w.orders[0].jobs, other)),))
    d = replace(d, operations=(*d.operations, QualityDraw("B1", 2**52)))
    q = prepare_quality(f, w, d)
    s = Simulator(
        f, w, quality=q, machine_events=MachineOutagePlan((MachineOutage("M1", 2, 4),))
    )
    s.step(action("A1", "M1"))
    for move in (action("A1", "M2"), action("A2", "M0"), action("B1", "missing")):
        before, context = s.trace, s.current_decision
        with pytest.raises(InvalidActionError):
            s.step(move)
        assert s.trace == before and s.current_decision == context
    s.step(WaitUntil(2))
    before = s.trace
    with pytest.raises(InvalidActionError):
        s.step(action("A1", "M0"))
    assert s.trace == before
    assert not any(x.kind == "quality_check" for x in before)


def test_tampered_plan_and_schedule_reject_wrong_mapping_and_duration():
    f, w, d = hand()
    q = prepare_quality(f, w, d)
    bad = replace(q, modes=(replace(q.modes[0], machine_id="M2"), *q.modes[1:]))
    with pytest.raises(ValueError, match="disagrees"):
        Simulator(f, w, quality=bad)
    r = Simulator(f, w, quality=q).run(SPTPolicy("M0"))
    for first in (
        replace(r.schedule[0], processing_mode_id="unknown"),
        replace(r.schedule[0], completion_time=11),
    ):
        with pytest.raises(ValueError):
            replay_schedule(f, w, (first, *r.schedule[1:]), quality=q)
    with pytest.raises(ValueError):
        Simulator(f, w, quality=q, quality_probability_visibility="unknown")
