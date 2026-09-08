"""Independent holding-buffer timelines, capacity ownership and exact replay."""

from dataclasses import FrozenInstanceError, replace
from itertools import product

import pytest
from test_static_engine import op, problem

from smartsom.algorithms import ScriptedPolicy, SPTPolicy
from smartsom.config.codec import primitive
from smartsom.dispatch import Dispatch, Transport, WaitUntil
from smartsom.domain import (
    AGV,
    ArrivalPlan,
    Job,
    JobArrival,
    Machine,
    MachineBuffers,
    MachineLocation,
    MachineOutage,
    MachineOutagePlan,
    ProcessingTime,
    ProcessingTimePlan,
    TransportDestination,
    TransportSpec,
    TravelTime,
)
from smartsom.domain.buffers import HoldingBuffer
from smartsom.domain.quality import (
    QualityDraw,
    QualityDrawPlan,
    QualityMode,
    QualitySpeedSpec,
)
from smartsom.engine import (
    InvalidActionError,
    ReplayError,
    Simulator,
    replay,
    replay_schedule,
)
from smartsom.modules.quality import prepare_quality


def holding_factory(factory, *, capacity=None, zero=False, count=1, overrides=None):
    nodes = ("I", "O", "H", *(m.machine_id for m in factory.machines))
    overrides = overrides or {}
    return replace(
        factory,
        transport=TransportSpec(
            nodes,
            tuple(
                MachineLocation(m.machine_id, m.machine_id) for m in factory.machines
            ),
            "I",
            "O",
            tuple(AGV(f"V{i}", "I") for i in range(1, count + 1)),
            tuple(
                TravelTime(
                    a, b, 0 if a == b else overrides.get((a, b), 0 if zero else 1)
                )
                for a in nodes
                for b in nodes
            ),
        ),
        holding_buffer=HoldingBuffer("b-hold", "H", capacity),
    )


def move(job, target, agv="V1"):
    destination = (
        TransportDestination("holding", buffer_id="b-hold")
        if target == "H"
        else TransportDestination("output")
        if target == "O"
        else TransportDestination("machine", target)
    )
    return Transport(agv, job, destination)


def hand_case(capacity=None):
    f, w = problem(Job("J", (op("J1", "M1", 2), op("J2", "M2", 3, "J1"))))
    f = holding_factory(f, capacity=capacity)
    actions = (
        move("J", "M1"),
        Dispatch("J1", "standard"),
        move("J", "H"),
        WaitUntil(6),
        move("J", "M2"),
        Dispatch("J2", "standard"),
        move("J", "O"),
    )
    return f, w, actions


def exact(f, w, result, **kwargs):
    settings = dict(transport_enabled=True, holding_buffer_enabled=True, **kwargs)
    assert replay(f, w, result.actions, **settings) == result
    r = replay_schedule(f, w, result.execution_schedule, **settings)
    assert r.execution_schedule == result.execution_schedule
    assert r.quality == result.quality
    assert r.makespan == result.makespan
    scrambled = replace(
        result.execution_schedule,
        operations=tuple(reversed(result.schedule)),
        transports=tuple(reversed(result.transport_schedule)),
        arrivals=tuple(reversed(result.execution_schedule.arrivals)),
        action_order=tuple(reversed(result.execution_schedule.action_order)),
    )
    assert replay_schedule(f, w, scrambled, **settings) == r


@pytest.mark.parametrize("capacity", [None, 1, 999999])
def test_hand_calculated_eleven(capacity):
    f, w, actions = hand_case(capacity)
    sim = Simulator(f, w, transport_enabled=True, holding_buffer_enabled=True)
    result = sim.run(ScriptedPolicy(actions))
    assert [
        (s.operation_id, s.start_time, s.completion_time) for s in result.schedule
    ] == [("J1", 1, 3), ("J2", 7, 10)]
    assert [
        (t.start_time, t.pickup_time, t.delivery_time)
        for t in result.transport_schedule
    ] == [(0, 0, 1), (3, 3, 4), (6, 6, 7), (10, 10, 11)]
    assert result.makespan == 11 and result.schedule_version == 2
    exact(f, w, result)


def competition(*, delayed=False):
    f, w = problem(
        Job("A", (op("A1", "M1", 1), op("A2", "M3", 1, "A1"))),
        Job("B", (op("B1", "M2", 1), op("B2", "M3", 1, "B1"))),
    )
    f = replace(f, machines=(*f.machines, Machine("M3")))
    f = holding_factory(
        f,
        capacity=1,
        zero=True,
        count=3 if delayed else 2,
        overrides={("M1", "H"): 2, ("M2", "H"): 1, ("I", "H"): 2},
    )
    sim = Simulator(f, w, transport_enabled=True, holding_buffer_enabled=True)
    for a in (
        move("A", "M1"),
        Dispatch("A1", "standard"),
        move("B", "M2", "V2"),
        Dispatch("B1", "standard"),
        move("A", "H"),
        move("B", "H", "V2"),
    ):
        sim.step(a)
    return f, w, sim


@pytest.mark.parametrize("delayed", [False, True])
def test_exclusive_reservation_and_actual_pickup_capacity(delayed):
    f, w, sim = competition(delayed=delayed)
    view = sim.current_decision
    assert view.simulation_time == 3
    assert view.holding_buffer.jobs == ("A",)
    assert next(a for a in view.agvs if a.agv_id == "V2").phase == "waiting"
    assert [
        (r.simulation_time, r.trip.job_id)
        for r in sim.trace
        if r.kind == "wait_for_unload"
    ] == [(2, "B")]
    with pytest.raises(FrozenInstanceError):
        view.holding_buffer.capacity = 2
    sim.step(move("A", "M3", "V3" if delayed else "V1"))
    if delayed:
        # No legal move exists during empty travel: automatic advance reaches pickup.
        assert sim.current_decision.simulation_time == 5
        assert sim.current_decision.holding_buffer.jobs == ("B",)
    result = sim.run(SPTPolicy())
    trips = {
        t.job_id: t
        for t in result.transport_schedule
        if t.destination.kind == "holding"
    }
    assert trips["A"].delivery_time == 3
    assert trips["B"].delivery_time == (5 if delayed else 3)
    exact(f, w, result)


def test_baseline_falls_back_only_when_downstream_capacity_unavailable():
    f, w = problem(
        Job("A", (op("A1", "M1", 1), op("A2", "M2", 1, "A1"))),
        Job("B", (op("B", "M2", 1),)),
        Job("C", (op("C", "M2", 10),)),
    )
    f = holding_factory(
        replace(f, buffers=(MachineBuffers("M1", 1, 0), MachineBuffers("M2", 1, 0))),
        capacity=1,
        zero=True,
        count=3,
    )
    sim = Simulator(
        f, w, transport_enabled=True, buffers_enabled=True, holding_buffer_enabled=True
    )
    for a in (
        move("C", "M2", "V3"),
        Dispatch("C", "standard"),
        move("B", "M2", "V2"),
        move("A", "M1"),
        Dispatch("A1", "standard"),
    ):
        sim.step(a)
    assert sim.current_decision.simulation_time == 1
    assert SPTPolicy().select_action(sim.current_decision).destination.kind == "holding"
    result = sim.run(SPTPolicy())
    held = [t for t in result.transport_schedule if t.destination.kind == "holding"]
    assert len(held) == 1 and held[0].job_id == "A" and held[0].source.kind == "machine"
    assert result.makespan == 12
    exact(f, w, result, buffers_enabled=True)


def test_no_proactive_holding_no_pending_or_paused_movement_and_atomic_rejection():
    f, w, _ = hand_case(1)
    sim = Simulator(
        f,
        w,
        transport_enabled=True,
        holding_buffer_enabled=True,
        machine_events=MachineOutagePlan((MachineOutage("M1", 2, 8),)),
    )

    def reject(action):
        before, trace = sim.current_decision, sim.trace
        with pytest.raises(InvalidActionError):
            sim.step(action)
        assert sim.current_decision is before and sim.trace == trace

    reject(move("J", "H"))
    sim.step(move("J", "M1"))
    reject(move("J", "H"))
    # Dispatch auto-advances across the pause; no extra decision notification exists.
    sim.step(Dispatch("J1", "standard"))
    sim.step(move("J", "H"))
    reject(move("J", "H"))
    reject(Transport("V1", "J", TransportDestination("holding", buffer_id="unknown")))
    result = sim.run(SPTPolicy())
    assert len([r for r in result.trace if r.kind == "pause"]) == 1
    exact(f, w, result, machine_events=MachineOutagePlan((MachineOutage("M1", 2, 8),)))


@pytest.mark.parametrize("capacity", [0, 1])
def test_real_holding_deadlock(capacity):
    f, w = problem(
        Job("A", (op("A1", "M1", 1), op("A2", "M2", 1, "A1"))),
        Job("B", (op("B1", "M1", 1), op("B2", "M2", 1, "B1"))),
    )
    f = holding_factory(f, capacity=capacity, zero=True)
    sim = Simulator(f, w, transport_enabled=True, holding_buffer_enabled=True)
    sim.step(move("A", "M1"))
    sim.step(Dispatch("A1", "standard"))
    sim.step(WaitUntil(1))
    if capacity:
        sim.step(move("A", "H"))
        sim.step(move("B", "M1"))
        sim.step(Dispatch("B1", "standard"))
        sim.step(WaitUntil(2))
    from smartsom.engine import DeadlockError

    with pytest.raises(DeadlockError, match="holding="):
        sim.step(move("B" if capacity else "A", "H"))


@pytest.mark.parametrize(
    "fault", ["missing", "arrival", "delivery", "destination", "source", "sequence"]
)
def test_holding_schedule_rejects_inconsistent_records(fault):
    f, w, actions = hand_case(1)
    result = Simulator(f, w, transport_enabled=True, holding_buffer_enabled=True).run(
        ScriptedPolicy(actions)
    )
    schedule = result.execution_schedule
    trips = list(schedule.transports)
    if fault == "missing":
        schedule = replace(schedule, transports=tuple(trips[:1] + trips[2:]))
    elif fault == "arrival":
        rows = list(schedule.arrivals)
        rows[1] = replace(rows[1], arrival_time=5)
        schedule = replace(schedule, arrivals=tuple(rows))
    else:
        from smartsom.domain import JobLocation

        changes = {
            "delivery": {"delivery_time": 5},
            "destination": {
                "destination": TransportDestination("holding", buffer_id="unknown")
            },
            "source": {"source": JobLocation("holding", "b-hold")},
            "sequence": {"transport_sequence": 1},
        }[fault]
        trips[1] = replace(trips[1], **changes)
        schedule = replace(schedule, transports=tuple(trips))
    with pytest.raises(ReplayError):
        replay_schedule(
            f, w, schedule, transport_enabled=True, holding_buffer_enabled=True
        )


def test_off_preserves_entire_old_trace_and_observation_encoding():
    f, w, _ = hand_case()
    a = Simulator(f, w, transport_enabled=True)
    b = Simulator(replace(f, holding_buffer=None), w, transport_enabled=True)
    assert primitive(a.current_decision) == primitive(b.current_decision)
    assert "holding_buffer" not in primitive(a.current_decision)
    assert a.run(SPTPolicy()) == b.run(SPTPolicy())


@pytest.mark.parametrize("capacity", [-1, True, 1.5, float("inf")])
def test_invalid_capacity(capacity):
    with pytest.raises(ValueError):
        HoldingBuffer("H", "H", capacity)


@pytest.mark.parametrize(
    "ja,mb,upt,quality,buffers", tuple(product((False, True), repeat=5))
)
def test_all_module_combinations_through_holding(ja, mb, upt, quality, buffers):
    for trigger in (
        ("dispatch_available", "arrival_event") if ja else ("dispatch_available",)
    ):
        f, w, _ = hand_case(1)
        f = replace(
            f,
            buffers=(MachineBuffers("M1", 0, 0), MachineBuffers("M2", 1, 0)),
            quality_speed=QualitySpeedSpec((QualityMode("M0", "1.2", "0.1"),)),
        )
        arrivals = ArrivalPlan((JobArrival("J", 3, 1),)) if ja else None
        outages = MachineOutagePlan((MachineOutage("M1", 5, 7),)) if mb else None
        processing = (
            ProcessingTimePlan(
                tuple(
                    ProcessingTime(
                        op.operation_id,
                        m.processing_mode_id,
                        m.nominal_ticks,
                        m.nominal_ticks * 2,
                    )
                    for op in w.operations
                    for m in op.modes
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
                    tuple(QualityDraw(op.operation_id, 0) for op in w.operations)
                ),
                processing_times=processing,
            )
            if quality
            else None
        )
        settings = dict(
            arrivals=arrivals,
            decision_trigger=trigger,
            machine_events=outages,
            processing_times=processing,
            quality=q,
            buffers_enabled=buffers,
        )
        sim = Simulator(
            f, w, transport_enabled=True, holding_buffer_enabled=True, **settings
        )
        used = False
        while (context := sim.current_decision) is not None:
            if quality:
                assert all(j.passed is None for j in context.job_quality)
            holding = next(
                (
                    c.action
                    for c in context.transport_candidates
                    if c.action.destination.kind == "holding"
                ),
                None,
            )
            action = (
                holding
                if holding is not None and not used
                else SPTPolicy().select_action(context)
            )
            if action == holding:
                used = True
            outcome = sim.step(action)
            from smartsom.engine import SimulationResult

            if isinstance(outcome, SimulationResult):
                break
        assert used and outcome.makespan > 0
        exact(f, w, outcome, **settings)
        assert sum(r.kind == "quality_check" for r in outcome.trace) == (
            2 if quality else 0
        )
        assert sum(r.kind == "inspection" for r in outcome.trace) == (
            1 if quality else 0
        )


def test_hidden_repair_end_and_future_outages_do_not_leak_into_holding_view():
    f, w, _ = hand_case(1)
    common = (move("J", "M1"), Dispatch("J1", "standard"), move("J", "H"))
    views = []
    for end in (20, 30):
        sim = Simulator(
            f,
            w,
            transport_enabled=True,
            holding_buffer_enabled=True,
            machine_events=MachineOutagePlan((MachineOutage("M2", 0, end),)),
        )
        for action in common:
            sim.step(action)
        views.append(sim.current_decision)
    assert primitive(views[0]) == primitive(views[1])
    assert views[0].holding_buffer.jobs == ("J",)


def test_input_reordering_keeps_fixed_action_trace():
    f, w, actions = hand_case(1)
    expected = replay(
        f, w, actions, transport_enabled=True, holding_buffer_enabled=True
    )
    f = replace(
        f,
        machines=tuple(reversed(f.machines)),
        transport=replace(
            f.transport,
            nodes=tuple(reversed(f.transport.nodes)),
            travel_times=tuple(reversed(f.transport.travel_times)),
        ),
    )
    order = w.orders[0]
    job = order.jobs[0]
    w = replace(
        w,
        orders=(
            replace(
                order, jobs=(replace(job, operations=tuple(reversed(job.operations))),)
            ),
        ),
    )
    assert (
        replay(f, w, actions, transport_enabled=True, holding_buffer_enabled=True)
        == expected
    )


def test_paused_job_cannot_be_parked_while_other_jobs_offer_decisions():
    f, w = problem(
        Job("J", (op("J1", "M1", 2), op("J2", "M2", 1, "J1"))),
        Job("C", (op("C", "M2", 1),)),
    )
    f = holding_factory(f, capacity=1)
    outage = MachineOutagePlan((MachineOutage("M1", 2, 8),))
    sim = Simulator(
        f, w, transport_enabled=True, holding_buffer_enabled=True, machine_events=outage
    )
    sim.step(move("J", "M1"))
    sim.step(Dispatch("J1", "standard"))
    sim.step(WaitUntil(2))
    before, trace = sim.current_decision, sim.trace
    assert (
        next(op for op in before.operations if op.operation_id == "J1").status.value
        == "paused"
    )
    with pytest.raises(InvalidActionError):
        sim.step(move("J", "H"))
    assert sim.current_decision is before and sim.trace == trace
    result = sim.run(SPTPolicy())
    exact(f, w, result, machine_events=outage)
