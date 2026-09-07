"""Capacity handoff boundaries and malformed v2 replay inputs."""

from dataclasses import FrozenInstanceError, replace

import pytest
from test_buffers import advance, assert_exact, dispatch, move, vehicle_case, vehicles
from test_static_engine import op, problem

from smartsom.algorithms import SPTPolicy
from smartsom.dispatch import WaitUntil
from smartsom.domain import Job, MachineBuffers, MachineOutage, MachineOutagePlan
from smartsom.engine import InvalidActionError, ReplayError, Simulator, replay_schedule


def test_bound_blocked_source_stays_until_pickup_even_when_post_space_opens():
    f, w = problem(
        *(
            Job(x, (op(x, "M1" if x != "D" else "M2", n),))
            for x, n in [("A", 2), ("B", 1), ("C", 1), ("D", 1)]
        )
    )
    f = vehicles(
        replace(f, buffers=(MachineBuffers("M1", None, 1),)),
        starts={"V1": "M2"},
        extra_times={("I", "M1"): 0, ("M1", "I"): 0, ("M2", "M1"): 2},
    )
    outage = MachineOutagePlan((MachineOutage("M1", 4, 7),))
    sim = Simulator(
        f, w, buffers_enabled=True, transport_enabled=True, machine_events=outage
    )
    for job in "ABC":
        sim.step(move(job, "M1", "V2"))
    sim.step(dispatch("A"))
    advance(sim, 2)
    sim.step(dispatch("B"))
    advance(sim, 3)
    assert sim.current_decision.machine_holdings[0].phase == "blocked"
    sim.step(move("B", agv="V1"))  # Source fixed on machine, pickup at 5.
    sim.step(move("A", agv="V2"))  # Post space opens now, but B is bound.
    view = sim.current_decision
    assert (
        view.simulation_time == 4
    )  # Both vehicles busy: next decision follows A's exit.
    assert view.buffers[0].post_jobs == ()
    assert view.machine_holdings[0].job_id == "B"
    assert (
        next(p for p in view.job_positions if p.job_id == "B").location.kind
        == "machine"
    )
    assert not any(x.kind == "unblock" and x.job_id == "B" for x in sim.trace)
    advance(sim, 5)
    assert all(h.job_id != "B" for h in sim.current_decision.machine_holdings)
    assert [
        (x.kind, x.simulation_time)
        for x in sim.trace
        if x.kind in ("pause", "resume") and x.operation_id == "B"
    ] == []
    r = sim.run(SPTPolicy())
    b = next(
        t
        for t in r.transport_schedule
        if t.job_id == "B" and t.destination.kind == "output"
    )
    assert b.source.kind == "machine" and b.pickup_time == 5
    assert next(x for x in r.schedule if x.operation_id == "B").completion_time == 3
    assert_exact(f, w, r, transport_enabled=True, machine_events=outage)


def test_positive_prebuffer_never_bypasses_queue_and_snapshots_are_frozen():
    f, w, _ = vehicle_case()
    f = replace(f, buffers=(MachineBuffers("M1", 1, 1),))
    sim = Simulator(f, w, buffers_enabled=True, transport_enabled=True)
    sim.step(move("A", "M1", "V1"))
    sim.step(move("B", "M1", "V2"))
    view = sim.current_decision
    assert view.simulation_time == 2
    assert view.machines[0].operation_id is None
    assert view.buffers[0].pre_jobs == ("A",)
    assert next(a for a in view.agvs if a.agv_id == "V2").phase == "waiting"
    for obj, field, value in (
        (view.buffers[0], "pre_capacity", 100),
        (view.agvs[1], "phase", "idle"),
        (view.buffers[0], "pre_jobs", ()),
    ):
        with pytest.raises(FrozenInstanceError):
            setattr(obj, field, value)
    trace = sim.trace
    with pytest.raises(InvalidActionError):
        sim.step(dispatch("B"))
    assert sim.current_decision is view and sim.trace == trace
    sim.step(dispatch("A"))
    assert sim.current_decision.buffers[0].pre_jobs == ("B",)
    assert view.buffers[0].pre_jobs == ("A",)  # Snapshot has not changed.
    r = sim.run(SPTPolicy())
    assert_exact(f, w, r, transport_enabled=True)


@pytest.mark.parametrize("field", ["action_order", "arrivals", "transports"])
@pytest.mark.parametrize("bad", [-1, True, 1.5, "1"])
def test_replay_rejects_bad_sequence_before_sorting(field, bad):
    f, w, _ = vehicle_case()
    r = Simulator(f, w, buffers_enabled=True, transport_enabled=True).run(SPTPolicy())
    items = list(getattr(r.execution_schedule, field))
    key = "sequence" if field == "action_order" else "transport_sequence"
    items[0] = replace(items[0], **{key: bad})
    with pytest.raises(ReplayError, match="sequence"):
        replay_schedule(
            f,
            w,
            replace(r.execution_schedule, **{field: items}),
            buffers_enabled=True,
            transport_enabled=True,
        )


@pytest.mark.parametrize(
    "fault",
    [
        "missing_action",
        "duplicate_action",
        "arrival",
        "delivery",
        "source",
        "missing_trip",
    ],
)
def test_v2_rejects_incomplete_or_inconsistent_physical_records(fault):
    from smartsom.domain import JobLocation

    f, w, actions = vehicle_case()
    from smartsom.engine import replay

    r = replay(f, w, actions, buffers_enabled=True, transport_enabled=True)
    s = r.execution_schedule
    if fault == "missing_action":
        s = replace(s, action_order=s.action_order[:-1])
    if fault == "duplicate_action":
        s = replace(s, action_order=(*s.action_order, s.action_order[0]))
    if fault == "arrival":
        s = replace(
            s, arrivals=(replace(s.arrivals[0], arrival_time=3), *s.arrivals[1:])
        )
    if fault == "delivery":
        s = replace(
            s, transports=(replace(s.transports[0], delivery_time=3), *s.transports[1:])
        )
    if fault == "source":
        s = replace(
            s,
            transports=(
                replace(s.transports[0], source=JobLocation("postbuffer", "M2")),
                *s.transports[1:],
            ),
        )
    if fault == "missing_trip":
        s = replace(s, transports=s.transports[:-1])
    with pytest.raises(ReplayError):
        replay_schedule(f, w, s, buffers_enabled=True, transport_enabled=True)


def test_same_arrival_tie_uses_vehicle_ids_not_booking_order():
    f, w, _ = vehicle_case()
    sim = Simulator(f, w, buffers_enabled=True, transport_enabled=True)
    sim.step(move("B", "M1", "V2"))
    sim.step(move("A", "M1", "V1"))
    view = sim.current_decision
    assert view.simulation_time == 2
    assert view.machine_holdings[0].job_id == "A"
    assert next(a for a in view.agvs if a.agv_id == "V2").phase == "waiting"
    assert_exact(f, w, sim.run(SPTPolicy()), transport_enabled=True)


def test_pre_slot_stays_occupied_until_reroute_pickup():
    from smartsom.domain import Operation, ProcessingMode

    f, w = problem(
        Job(
            "A",
            (
                Operation(
                    "A", (ProcessingMode("x", "M1", 1), ProcessingMode("y", "M2", 1))
                ),
            ),
        ),
        Job("B", (op("B", "M1", 1),)),
    )
    f = vehicles(
        replace(
            f, buffers=(MachineBuffers("M1", 1, None), MachineBuffers("M2", 1, None))
        )
    )
    sim = Simulator(f, w, buffers_enabled=True, transport_enabled=True)
    sim.step(move("A", "M1", "V1"))
    advance(sim, 2)
    sim.step(move("A", "M2", "V2"))  # V2 must first drive I->M1 for 2 ticks.
    view = sim.current_decision
    assert view.buffers[0].pre_jobs == ("A",)
    assert [r.job_id for r in view.buffers[1].reservations] == ["A"]
    trace = sim.trace
    for bad in (move("A", "M2", "V1"), dispatch("A")):
        with pytest.raises(InvalidActionError):
            sim.step(bad)
    assert sim.trace == trace and sim.current_decision is view
    sim.step(move("B", "M1", "V1"))  # Full source, so no reservation for B.
    assert not any(x.kind == "reserve" and x.job_id == "B" for x in sim.trace)
    while sim.current_decision.simulation_time < 4:
        sim.step(WaitUntil(4))
    assert "A" not in sim.current_decision.buffers[0].pre_jobs
    assert_exact(f, w, sim.run(SPTPolicy()), transport_enabled=True)


def test_conservative_policy_exhaustion_is_distinct_from_physical_deadlock():
    f, w = problem(
        Job("A", (op("A1", "M1", 2), op("A2", "M2", 1, "A1"))),
        Job("B", (op("B1", "M2", 2), op("B2", "M1", 1, "B1"))),
    )
    f = vehicles(
        replace(f, buffers=(MachineBuffers("M1", 0, 0), MachineBuffers("M2", 0, 0))),
        zero=True,
    )
    sim = Simulator(f, w, buffers_enabled=True, transport_enabled=True)
    sim.step(move("A", "M1", "V1"))
    sim.step(move("B", "M2", "V2"))
    sim.step(dispatch("A1"))
    sim.step(dispatch("B1"))
    context, trace = sim.current_decision, sim.trace
    assert context.simulation_time == 2 and context.transport_candidates
    with pytest.raises(InvalidActionError, match="future event"):
        sim.run(SPTPolicy())
    assert sim.current_decision is context and sim.trace == trace
    # A legal early loaded trip frees M1 so the other trip can proceed.
    sim.step(move("A", "M2", "V1"))
    sim.step(move("B", "M1", "V2"))
    result = sim.run(SPTPolicy())
    assert result.makespan == 3
    assert_exact(f, w, result, transport_enabled=True)
