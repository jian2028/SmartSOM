"""Independent hand cases fixed before the buffer implementation."""

import json
from dataclasses import replace
from pathlib import Path

import pytest
from test_static_engine import op, problem

from smartsom.algorithms import SPTPolicy
from smartsom.dispatch import Dispatch, Transfer, Transport, WaitUntil
from smartsom.domain import (
    AGV,
    Job,
    MachineBuffers,
    MachineLocation,
    TransportDestination,
    TransportSpec,
    TravelTime,
)
from smartsom.engine import DeadlockError, Simulator, replay, replay_schedule

EXPECTED = json.loads(
    (
        Path(__file__).parents[2] / "data/reference/buffers/hand_expectations.json"
    ).read_text()
)


def move(job, target=None, agv=None):
    destination = (
        TransportDestination("output")
        if target is None
        else TransportDestination("machine", target)
    )
    return (
        Transfer(job, destination) if agv is None else Transport(agv, job, destination)
    )


def dispatch(op):
    return Dispatch(op, "standard")


def direct_case():
    f, w = problem(
        Job("A", (op("A1", "M1", 2), op("A2", "M2", 1, "A1"))),
        Job("B", (op("B1", "M1", 1),)),
        Job("C", (op("C1", "M2", 5),)),
    )
    f = replace(f, buffers=(MachineBuffers("M1", 0, 0), MachineBuffers("M2", 0, 0)))
    actions = (
        move("C", "M2"),
        dispatch("C1"),
        move("A", "M1"),
        dispatch("A1"),
        move("C"),
        move("A", "M2"),
        dispatch("A2"),
        move("B", "M1"),
        dispatch("B1"),
        move("A"),
        move("B"),
    )
    return f, w, actions


def vehicles(factory, count=2, zero=False, extra_times=None, starts=None):
    nodes = ("I", "M1", "M2", "O")
    times = {("I", "M1"): 2, ("M1", "O"): 1}
    times.update(extra_times or {})
    return replace(
        factory,
        transport=TransportSpec(
            nodes,
            tuple(
                MachineLocation(m.machine_id, m.machine_id) for m in factory.machines
            ),
            "I",
            "O",
            tuple(
                AGV(f"V{i}", (starts or {}).get(f"V{i}", "I"))
                for i in range(1, count + 1)
            ),
            tuple(
                TravelTime(a, b, 0 if zero or a == b else times.get((a, b), 1))
                for a in nodes
                for b in nodes
            ),
        ),
    )


def vehicle_case():
    f, w = problem(Job("A", (op("A", "M1", 4),)), Job("B", (op("B", "M1", 1),)))
    f = vehicles(replace(f, buffers=(MachineBuffers("M1", 0, 1),)))
    actions = (
        move("A", "M1", "V1"),
        move("B", "M1", "V2"),
        dispatch("A"),
        dispatch("B"),
        move("A", agv="V1"),
        move("B", agv="V2"),
    )
    return f, w, actions


def post_case():
    f, w = problem(
        *(Job(x, (op(x, "M1", n),)) for x, n in (("A", 2), ("B", 1), ("C", 1)))
    )
    f = replace(f, buffers=(MachineBuffers("M1", None, 1),))
    actions = (
        move("A", "M1"),
        move("B", "M1"),
        move("C", "M1"),
        dispatch("A"),
        dispatch("B"),
        WaitUntil(5),
        WaitUntil(5),
        move("A"),
        dispatch("C"),
        move("B"),
        move("C"),
    )
    return f, w, actions


@pytest.mark.parametrize(
    "build,name,agv",
    [
        (direct_case, "direct_zero", False),
        (vehicle_case, "two_vehicles_zero_pre", True),
        (post_case, "direct_post_one", False),
    ],
)
def test_hand_processing_and_full_action_replay(build, name, agv):
    f, w, actions = build()
    r = replay(f, w, actions, buffers_enabled=True, transport_enabled=agv)
    assert {
        x.operation_id: [x.start_time, x.completion_time] for x in r.schedule
    } == EXPECTED[name]["processing"]
    assert r.makespan == EXPECTED[name]["makespan"]
    assert r.schedule_version == 2
    if agv:
        assert [x.arrival_time for x in r.transport_arrivals[:2]] == [2, 2]
        assert [x.delivery_time for x in r.transport_schedule[:2]] == [2, 6]
        assert [
            (x.simulation_time, x.trip.job_id)
            for x in r.trace
            if x.kind == "wait_for_unload"
        ] == [(2, "B")]
    else:
        blocked = {}
        unblocked = {}
        for x in r.trace:
            if x.kind == "block":
                blocked.setdefault(x.job_id, x.simulation_time)
            if x.kind == "unblock":
                unblocked.setdefault(x.job_id, x.simulation_time)
        expected_job = "A" if name == "direct_zero" else "B"
        assert blocked[expected_job] == (2 if name == "direct_zero" else 3)
        assert unblocked[expected_job] == 5
    assert replay(f, w, r.actions, buffers_enabled=True, transport_enabled=agv) == r
    actual = replay_schedule(
        f, w, r.execution_schedule, buffers_enabled=True, transport_enabled=agv
    )
    assert actual.execution_schedule == r.execution_schedule
    assert actual.makespan == r.makespan


def test_real_deadlock_vs_conservative_policy():
    f, w, _ = vehicle_case()
    f = replace(
        f,
        buffers=(MachineBuffers("M1", 0, 0),),
        transport=replace(f.transport, agvs=f.transport.agvs[:1]),
    )
    sim = Simulator(f, w, buffers_enabled=True, transport_enabled=True)
    sim.step(move("A", "M1", "V1"))
    sim.step(dispatch("A"))
    with pytest.raises(DeadlockError, match="vehicles=.*waiting"):
        sim.step(move("B", "M1", "V1"))
    r = Simulator(f, w, buffers_enabled=True, transport_enabled=True).run(SPTPolicy())
    assert len(r.schedule) == 2 and r.makespan > 0


def advance(sim, tick):
    while sim.current_decision.simulation_time < tick:
        sim.step(WaitUntil(tick))
    assert sim.current_decision.simulation_time == tick


def assert_exact(f, w, r, **kwargs):
    assert replay(f, w, r.actions, buffers_enabled=True, **kwargs) == r
    actual = replay_schedule(f, w, r.execution_schedule, buffers_enabled=True, **kwargs)
    assert actual.execution_schedule == r.execution_schedule
    shuffled = replace(
        r.execution_schedule,
        operations=tuple(reversed(r.schedule)),
        transports=tuple(reversed(r.transport_schedule)),
        arrivals=tuple(reversed(r.transport_arrivals)),
        transfers=tuple(reversed(r.transfer_schedule)),
        action_order=tuple(reversed(r.action_order)),
    )
    assert replay_schedule(f, w, shuffled, buffers_enabled=True, **kwargs) == actual


def reservation_case():
    f, w = problem(
        *(Job(x, (op(x, "M1", n),)) for x, n in (("A", 2), ("B", 1), ("C", 1)))
    )
    f = vehicles(
        replace(f, buffers=(MachineBuffers("M1", 1, None),)),
        count=3,
        extra_times={("M2", "I"): 2},
        starts={"V2": "M2"},
    )
    sim = Simulator(f, w, buffers_enabled=True, transport_enabled=True)
    sim.step(move("A", "M1", "V3"))
    advance(sim, 2)
    sim.step(move("B", "M1", "V1"))  # Full: no reservation.
    sim.step(dispatch("A"))  # Frees capacity while B is still travelling.
    sim.step(move("C", "M1", "V2"))  # C exclusively reserves it.
    return f, w, sim


def test_exclusive_reservation_arrival_fifo_and_same_tick_order():
    from smartsom.engine import ReplayError

    f, w, sim = reservation_case()
    advance(sim, 4)
    view = sim.current_decision
    assert next(a for a in view.agvs if a.agv_id == "V1").phase == "waiting"
    b = next(b for b in view.buffers if b.machine_id == "M1")
    assert b.pre_jobs == () and [r.job_id for r in b.reservations] == ["C"]
    advance(sim, 6)
    b = next(b for b in sim.current_decision.buffers if b.machine_id == "M1")
    assert b.pre_jobs == ("C",) and not b.reservations
    sim.step(dispatch("C"))
    assert next(
        b for b in sim.current_decision.buffers if b.machine_id == "M1"
    ).pre_jobs == ("B",)
    r = sim.run(SPTPolicy())
    inbound = {
        t.job_id: t for t in r.transport_schedule if t.destination.kind == "machine"
    }
    assert inbound["B"].delivery_time == inbound["C"].delivery_time == 6
    assert_exact(f, w, r, transport_enabled=True)
    rows = list(r.action_order)
    i = next(i for i, x in enumerate(rows) if x.action == move("B", "M1", "V1"))
    j = next(i for i, x in enumerate(rows) if x.action == dispatch("A"))
    rows[i], rows[j] = (
        replace(rows[j], sequence=rows[i].sequence),
        replace(rows[i], sequence=rows[j].sequence),
    )
    with pytest.raises(ReplayError):
        replay_schedule(
            f,
            w,
            replace(r.execution_schedule, action_order=rows),
            buffers_enabled=True,
            transport_enabled=True,
        )


@pytest.mark.parametrize("pre,post", [(0, 0), (0, 1), (1, 0), (1, 1), (None, 1)])
@pytest.mark.parametrize("agv", [False, True])
def test_same_machine_successor_and_conservative_baseline(pre, post, agv):
    f, w = problem(Job("J", (op("A", "M1", 2), op("B", "M1", 3, "A"))))
    f = replace(f, buffers=(MachineBuffers("M1", pre, post),))
    if agv:
        f = vehicles(f, zero=True, count=1)
    r = Simulator(f, w, buffers_enabled=True, transport_enabled=agv).run(SPTPolicy())
    assert [(x.start_time, x.completion_time) for x in r.schedule] == [(0, 2), (2, 5)]
    assert r.makespan == 5
    assert len(r.transport_schedule if agv else r.transfer_schedule) == 3
    assert_exact(f, w, r, transport_enabled=agv)


@pytest.mark.parametrize("agv", [False, True])
def test_all_dynamic_buffer_combinations(agv):
    from itertools import product

    from smartsom.algorithms import FirstFeasiblePolicy
    from smartsom.domain import (
        ArrivalPlan,
        JobArrival,
        MachineOutage,
        MachineOutagePlan,
        ProcessingTime,
        ProcessingTimePlan,
    )

    f, w = problem(
        Job("J", (op("A", "M1", 2), op("B", "M2", 3, "A"))),
        Job("K", (op("K", "M2", 2),)),
    )
    f = replace(f, buffers=(MachineBuffers("M1", 0, 0), MachineBuffers("M2", 1, 1)))
    if agv:
        f = vehicles(f, count=2)
    for ja, mb, upt in product((False, True), repeat=3):
        for trigger in (
            ("dispatch_available", "arrival_event") if ja else ("dispatch_available",)
        ):
            kwargs = dict(
                transport_enabled=agv,
                arrivals=ArrivalPlan((JobArrival("J", 0, 0), JobArrival("K", 4, 2)))
                if ja
                else None,
                decision_trigger=trigger,
                machine_events=MachineOutagePlan(
                    (MachineOutage("M1", 3, 5), MachineOutage("M2", 6, 8))
                )
                if mb
                else None,
                processing_times=ProcessingTimePlan(
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
                else None,
            )
            for Policy in (SPTPolicy, FirstFeasiblePolicy):
                r = Simulator(f, w, buffers_enabled=True, **kwargs).run(Policy())
                assert len(r.schedule) == 3
                assert_exact(f, w, r, **kwargs)


def test_repair_controls_zero_unload_and_hides_unknown_time():
    from smartsom.config.codec import primitive
    from smartsom.domain import MachineOutage, MachineOutagePlan

    f, w, _ = vehicle_case()
    f = replace(f, buffers=(MachineBuffers("M1", 0, 1),))
    sims = [
        Simulator(
            f,
            w,
            buffers_enabled=True,
            transport_enabled=True,
            machine_events=MachineOutagePlan((MachineOutage("M1", 0, end),)),
        )
        for end in (5, 9)
    ]
    for sim in sims:
        sim.step(move("A", "M1", "V1"))
        advance(sim, 2)
    assert sims[0].current_decision == sims[1].current_decision
    view = sims[0].current_decision
    assert next(a for a in view.agvs if a.agv_id == "V1").phase == "waiting"
    assert "delivery_time" not in json.dumps(primitive(view.agvs))
    assert not view.candidates
    advance(sims[0], 5)
    assert (
        next(a for a in sims[0].current_decision.agvs if a.agv_id == "V1").phase
        == "idle"
    )
    assert sims[0].current_decision.machine_holdings[0].phase == "awaiting_dispatch"


def test_direct_pending_load_reroutes_from_down_machine_and_dispatch_is_atomic():
    from smartsom.domain import (
        MachineOutage,
        MachineOutagePlan,
        Operation,
        ProcessingMode,
    )
    from smartsom.engine import InvalidActionError

    flexible = Operation(
        "A", (ProcessingMode("x", "M1", 2), ProcessingMode("y", "M2", 1))
    )
    f, w = problem(Job("J", (flexible,)), Job("K", (op("K", "M2", 10),)))
    f = replace(f, buffers=(MachineBuffers("M1", 0, 0), MachineBuffers("M2", 0, 0)))
    outage = MachineOutagePlan((MachineOutage("M1", 1, 4),))
    sim = Simulator(f, w, buffers_enabled=True, machine_events=outage)
    sim.step(move("J", "M1"))
    advance(sim, 1)
    before, trace = sim.current_decision, sim.trace
    for bad in (Dispatch("A", "x"), move("J", "M1"), move("J")):
        with pytest.raises(InvalidActionError):
            sim.step(bad)
        assert sim.current_decision is before and sim.trace == trace
    assert sim.current_decision.operations[0].processing_mode_id is None
    sim.step(move("J", "M2"))
    sim.step(Dispatch("A", "y"))
    r = sim.run(SPTPolicy())
    assert (
        next(x for x in r.schedule if x.operation_id == "A").processing_mode_id == "y"
    )
    assert_exact(f, w, r, machine_events=outage)


@pytest.mark.parametrize("bad", [-1, True, 1.5, "1"])
def test_strict_capacities(bad):
    with pytest.raises(ValueError):
        MachineBuffers("M1", bad, 1)
    with pytest.raises(ValueError):
        MachineBuffers("M1", 1, bad)


def test_unlimited_and_disabled_keep_old_transport_trace():
    from test_transport import hand_case

    from smartsom.config.codec import digest

    f, w = hand_case()
    old = Simulator(f, w, transport_enabled=True).run(SPTPolicy())
    unlimited = replace(f, buffers=(MachineBuffers("M1"), MachineBuffers("M2")))
    assert digest(f) == digest(unlimited)
    assert (
        Simulator(unlimited, w, transport_enabled=True, buffers_enabled=True).run(
            SPTPolicy()
        )
        == old
    )
    limited = replace(f, buffers=(MachineBuffers("M1", 0, 0),))
    assert Simulator(limited, w, transport_enabled=True).run(SPTPolicy()) == old
