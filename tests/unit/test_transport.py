import json
from dataclasses import FrozenInstanceError, replace
from itertools import product

import pytest
from test_static_engine import action, op, problem

from smartsom.algorithms import FirstFeasiblePolicy, SPTPolicy
from smartsom.config.codec import primitive
from smartsom.dispatch import Dispatch, Transport, WaitUntil
from smartsom.domain import (
    AGV,
    ArrivalPlan,
    ExecutionSchedule,
    Job,
    JobArrival,
    JobLocation,
    MachineLocation,
    MachineOutage,
    MachineOutagePlan,
    Operation,
    ProcessingMode,
    ProcessingTime,
    ProcessingTimePlan,
    TransportDestination,
    TransportSpec,
    TravelTime,
)
from smartsom.engine import (
    InvalidActionError,
    ReplayError,
    SimulationFinishedError,
    Simulator,
    replay,
    replay_schedule,
)


def logistics(factory, *, zero=False, count=1, overrides=None):
    nodes = ("I", "M1", "M2", "O")
    times = {("M2", "I"): 2, ("I", "M1"): 3, ("M1", "M2"): 4}
    times.update(overrides or {})
    return replace(
        factory,
        transport=TransportSpec(
            nodes,
            tuple(
                MachineLocation(m.machine_id, m.machine_id) for m in factory.machines
            ),
            "I",
            "O",
            tuple(AGV(f"V{i}", "M2") for i in range(1, count + 1)),
            tuple(
                TravelTime(a, b, 0 if zero or a == b else times.get((a, b), 1))
                for a in nodes
                for b in nodes
            ),
        ),
    )


def hand_case(**kwargs):
    f, w = problem(Job("J", (op("A", "M1", 2), op("B", "M2", 3, "A"))))
    return logistics(f, **kwargs), w


def move(job="J", destination="M1", agv="V1"):
    return Transport(
        agv,
        job,
        TransportDestination("output")
        if destination is None
        else TransportDestination("machine", destination),
    )


def assert_replays(f, w, result, **kwargs):
    assert replay(f, w, result.actions, transport_enabled=True, **kwargs) == result
    actual = replay_schedule(
        f, w, result.execution_schedule, transport_enabled=True, **kwargs
    )
    assert actual.schedule == result.schedule
    assert actual.transport_schedule == result.transport_schedule
    assert actual.makespan == result.makespan
    assert (
        replay_schedule(
            f,
            w,
            replace(
                result.execution_schedule,
                operations=tuple(reversed(result.schedule)),
                transports=tuple(reversed(result.transport_schedule)),
            ),
            transport_enabled=True,
            **kwargs,
        )
        == actual
    )


def test_hand_timetable_actions_phases_and_termination():
    f, w = hand_case()
    r = Simulator(f, w, transport_enabled=True).run(SPTPolicy())
    assert [(x.start_time, x.completion_time) for x in r.schedule] == [(5, 7), (11, 14)]
    assert [
        (x.start_time, x.pickup_time, x.delivery_time) for x in r.transport_schedule
    ] == [(0, 2, 5), (7, 7, 11), (14, 14, 15)]
    assert r.makespan == 15
    assert [x.kind for x in r.trace if hasattr(x, "trip")] == [
        "empty_start",
        "pickup",
        "loaded_start",
        "delivery",
    ] * 3
    assert_replays(f, w, r)
    with pytest.raises(ReplayError):
        replay_schedule(f, w, r.schedule, transport_enabled=True)


def test_binding_atomicity_busy_agv_and_mode_deferred_until_dispatch():
    f, w = hand_case(count=2)
    a = Operation(
        "A",
        (
            ProcessingMode("fast", "M1", 2),
            ProcessingMode("slow", "M1", 4),
            ProcessingMode("other", "M2", 1),
        ),
    )
    w = replace(
        w, orders=(replace(w.orders[0], jobs=(Job("J", (a, w.operations[1])),)),)
    )
    sim = Simulator(f, w, transport_enabled=True)
    assert len(sim.current_decision.transport_candidates) == 4  # Per machine, not mode.
    with pytest.raises(InvalidActionError):
        sim.step(Dispatch("A", "fast"))
    sim.step(move())  # No other ready job: auto advances all the way to delivery.
    view = sim.current_decision
    assert view.job_positions[0].location == JobLocation("prebuffer", "M1")
    assert view.operations[0].processing_mode_id is None
    assert {
        x.processing_mode_id for x in view.feasible_actions if isinstance(x, Dispatch)
    } == {"fast", "slow"}
    sim.step(Dispatch("A", "slow"))
    result = sim.run(SPTPolicy())
    assert result.schedule[0].processing_mode_id == "slow"
    with pytest.raises(SimulationFinishedError):
        sim.step(move())
    assert_replays(f, w, result)


def test_two_agvs_race_for_one_job_and_bound_queue_cannot_process():
    f, w = problem(
        Job(
            "J",
            (
                Operation(
                    "A", (ProcessingMode("x", "M1", 2), ProcessingMode("y", "M2", 2))
                ),
            ),
        ),
        Job("K", (op("K", "M1", 1),)),
    )
    f = logistics(f, count=2)
    sim = Simulator(f, w, transport_enabled=True)
    sim.step(move())
    before, trace = sim.current_decision, sim.trace
    for cmd in (move(agv="V2"), move("K"), Dispatch("A", "x")):
        with pytest.raises(InvalidActionError):
            sim.step(cmd)
        assert sim.current_decision is before and sim.trace == trace
    assert sim.step(WaitUntil(5)).simulation_time == 2  # Pickup interrupts this wait.
    sim.step(WaitUntil(5))
    assert sim.current_decision.job_positions[0].location.kind == "prebuffer"
    sim.step(move(destination="M2", agv="V2"))
    before, trace = sim.current_decision, sim.trace
    assert before.job_positions[0].bound_agv_id == "V2"
    for cmd in (Dispatch("A", "x"), move(agv="V1")):
        with pytest.raises(InvalidActionError):
            sim.step(cmd)
        assert sim.current_decision is before and sim.trace == trace
    r = sim.run(SPTPolicy())
    assert_replays(f, w, r)


def test_same_machine_successors_require_agv_and_zero_time_keeps_processing():
    f, w = problem(Job("J", (op("A", "M1", 2), op("B", "M1", 3, "A"))))
    old = Simulator(f, w).run(SPTPolicy())
    f = logistics(f, zero=True)
    assert Simulator(f, w).run(SPTPolicy()) == old
    r = Simulator(f, w, transport_enabled=True).run(SPTPolicy())
    assert (r.schedule, r.makespan) == (old.schedule, old.makespan)
    assert r.trace != old.trace
    assert len(r.transport_schedule) == 3
    assert r.transport_schedule[1].source == JobLocation("postbuffer", "M1")
    assert_replays(f, w, r)


def test_down_queue_reroutes_to_idle_machine_and_does_not_lock_mode():
    f, w = problem(
        Job(
            "J",
            (
                Operation(
                    "A",
                    (
                        ProcessingMode("x", "M1", 2),
                        ProcessingMode("fast", "M2", 2),
                        ProcessingMode("slow", "M2", 6),
                    ),
                ),
            ),
        )
    )
    f = logistics(f, overrides={("I", "M1"): 1, ("I", "M2"): 3, ("M1", "M2"): 1})
    outages = MachineOutagePlan((MachineOutage("M1", 0, 20),))
    sim = Simulator(f, w, transport_enabled=True, machine_events=outages)
    sim.step(move())
    assert SPTPolicy().select_action(sim.current_decision) == move(destination="M2")
    r = sim.run(SPTPolicy())
    assert r.schedule[0].processing_mode_id == "fast"
    assert r.transport_schedule[1].source == JobLocation("prebuffer", "M1")
    assert r.makespan == 7
    assert_replays(f, w, r, machine_events=outages)


@pytest.mark.parametrize(
    "ja,mb,upt,trigger",
    [
        (ja, mb, upt, trigger)
        for ja, mb, upt in product((False, True), repeat=3)
        for trigger in (
            ("dispatch_available", "arrival_event") if ja else ("dispatch_available",)
        )
    ],
)
def test_all_dynamic_switch_combinations(ja, mb, upt, trigger):
    f, w = problem(
        Job("J", (op("A", "M1", 2), op("B", "M2", 3, "A"))),
        Job("K", (op("K", "M2", 2),)),
    )
    f = logistics(f, count=2)
    kwargs = dict(
        arrivals=ArrivalPlan((JobArrival("J", 0, 0), JobArrival("K", 4, 2)))
        if ja
        else None,
        decision_trigger=trigger,
        machine_events=MachineOutagePlan(
            (MachineOutage("M1", 6, 9), MachineOutage("M2", 8, 10))
        )
        if mb
        else None,
        processing_times=ProcessingTimePlan(
            tuple(
                ProcessingTime(
                    op.operation_id,
                    m.processing_mode_id,
                    m.nominal_ticks,
                    m.nominal_ticks + 1,
                )
                for op in w.operations
                for m in op.modes
            )
        )
        if upt
        else None,
    )
    for policy in (SPTPolicy(), FirstFeasiblePolicy()):
        r = Simulator(f, w, transport_enabled=True, **kwargs).run(policy)
        assert sum(t.destination.kind == "output" for t in r.transport_schedule) == 2
        assert_replays(f, w, r, **kwargs)


@pytest.mark.parametrize(
    "field,value",
    [("ticks", -1), ("ticks", True), ("ticks", 1.5), ("from_node_id", " ")],
)
def test_invalid_matrix_scalars(field, value):
    with pytest.raises(ValueError):
        replace(TravelTime("I", "M1", 1), **{field: value})


def test_input_normalization_coverage_and_snapshot_immutability():
    f, w = hand_case()
    spec = f.transport
    for changes in (
        dict(nodes=spec.nodes + ("I",)),
        dict(travel_times=spec.travel_times[:-1]),
        dict(travel_times=spec.travel_times + spec.travel_times[:1]),
        dict(agvs=(AGV("V", "missing"),)),
        dict(machine_locations=spec.machine_locations[:1]),
    ):
        with pytest.raises(ValueError):
            replace(f, transport=replace(spec, **changes))
    reversed_spec = replace(
        spec,
        nodes=list(reversed(spec.nodes)),
        agvs=list(reversed(spec.agvs)),
        machine_locations=list(reversed(spec.machine_locations)),
        travel_times=list(reversed(spec.travel_times)),
    )
    assert reversed_spec == spec
    sim = Simulator(f, w, transport_enabled=True)
    for obj, field, value in (
        (sim.current_decision, "agvs", ()),
        (spec, "nodes", ()),
        (sim.current_decision.job_positions[0], "bound_agv_id", "bad"),
    ):
        with pytest.raises(FrozenInstanceError):
            setattr(obj, field, value)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "duplicate",
        "duration",
        "order",
        "vehicle",
        "node",
        "source",
        "early",
        "no_output",
    ],
)
def test_bad_execution_timetables(mutation):
    f, w = hand_case()
    r = Simulator(f, w, transport_enabled=True).run(SPTPolicy())
    trips = list(r.transport_schedule)
    if mutation == "missing":
        trips.pop(0)
    elif mutation == "duplicate":
        trips.append(trips[0])
    elif mutation == "duration":
        trips[0] = replace(trips[0], pickup_time=3)
    elif mutation == "order":
        trips[1] = replace(trips[1], transport_sequence=4)
    elif mutation == "vehicle":
        trips[0] = replace(trips[0], agv_id="bad")
    elif mutation == "node":
        trips[0] = replace(trips[0], from_node_id="I")
    elif mutation == "source":
        trips[1] = replace(trips[1], source=JobLocation("prebuffer", "M1"))
    elif mutation == "early":
        trips[1] = replace(trips[1], start_time=6, pickup_time=6, delivery_time=10)
    else:
        trips.pop()
    with pytest.raises(ReplayError):
        replay_schedule(
            f, w, ExecutionSchedule(r.schedule, trips), transport_enabled=True
        )


def test_hidden_arrivals_repairs_and_processing_values():
    f, w = hand_case()
    arrivals = ArrivalPlan((JobArrival("J", 4, 2),))
    sims = [
        Simulator(
            f,
            w,
            transport_enabled=True,
            arrivals=arrivals,
            decision_trigger="arrival_event",
            machine_events=MachineOutagePlan((MachineOutage("M1", 0, end),)),
            processing_times=ProcessingTimePlan(
                (
                    ProcessingTime("A", "standard", 2, duration),
                    ProcessingTime("B", "standard", 3, 3),
                )
            ),
        )
        for end, duration in ((10, 2), (15, 8))
    ]
    assert sims[0].current_decision == sims[1].current_decision
    view = sims[0].current_decision
    assert (
        view.simulation_time == 2
        and view.job_positions[0].location.kind == "unreleased"
    )
    assert not view.transport_candidates
    data = json.dumps(primitive(view))
    assert "repair" not in data and "actual_ticks" not in data


@pytest.mark.parametrize(
    "bad",
    [
        move("missing"),
        move(agv="missing"),
        move(destination="missing"),
        move(destination=None),
        Transport([], "J", TransportDestination("machine", "M1")),
        Transport("V1", "J", None),
    ],
)
def test_transport_invalid_actions_are_atomic(bad):
    f, w = hand_case()
    sim = Simulator(f, w, transport_enabled=True)
    before, trace = sim.current_decision, sim.trace
    with pytest.raises(InvalidActionError):
        sim.step(bad)
    assert sim.current_decision is before and sim.trace == trace


def test_zero_reroutes_keep_occurrence_order_and_machine_selects_non_fifo():
    flexible = Operation(
        "A", (ProcessingMode("x", "M1", 2), ProcessingMode("y", "M2", 3))
    )
    f, w = problem(Job("J", (flexible,)), Job("K", (op("K", "M1", 1),)))
    f = logistics(f, zero=True, count=2)
    sim = Simulator(f, w, transport_enabled=True)
    for cmd in (move(), move(destination="M2"), move(), move("K", agv="V2")):
        sim.step(cmd)
        assert sim.current_decision.simulation_time == 0
    with pytest.raises(InvalidActionError):
        sim.step(move())  # Already in this prebuffer.
    assert {c.action.operation_id for c in sim.current_decision.candidates} == {
        "A",
        "K",
    }
    assert SPTPolicy().select_action(sim.current_decision) == action("K")
    sim.step(action("K"))  # K arrived later, processes first; unlimited waiting.
    result = sim.run(SPTPolicy())
    assert next(a for a in result.actions if isinstance(a, Dispatch)) == action("K")
    assert next(o for o in result.schedule if o.operation_id == "K").start_time == 0
    assert [
        (t.transport_sequence, t.start_time) for t in result.transport_schedule[:4]
    ] == [(1, 0), (2, 0), (3, 0), (4, 0)]
    assert_replays(f, w, result)
    trips = list(result.transport_schedule)
    trips[0], trips[1] = (
        replace(trips[0], transport_sequence=2),
        replace(trips[1], transport_sequence=1),
    )
    with pytest.raises(ReplayError):
        replay_schedule(
            f, w, ExecutionSchedule(result.schedule, trips), transport_enabled=True
        )


def test_paused_processing_cannot_move_and_waiting_job_reroutes_away_from_busy():
    flexible = Operation(
        "A", (ProcessingMode("x", "M1", 2), ProcessingMode("y", "M2", 2))
    )
    f, w = problem(Job("J", (flexible,)), Job("K", (op("K", "M1", 5),)))
    f = logistics(f, zero=True)
    outages = MachineOutagePlan((MachineOutage("M1", 1, 4),))
    sim = Simulator(f, w, transport_enabled=True, machine_events=outages)
    sim.step(move("K"))
    sim.step(action("K"))
    sim.step(move())
    assert SPTPolicy().select_action(sim.current_decision) == move(destination="M2")
    sim.step(WaitUntil(1))
    before, trace = sim.current_decision, sim.trace
    assert (
        next(o for o in before.operations if o.operation_id == "K").status.value
        == "paused"
    )
    for cmd in (move("K", destination=None), move("K", destination="M1"), action("K")):
        with pytest.raises(InvalidActionError):
            sim.step(cmd)
        assert sim.current_decision is before and sim.trace == trace
    result = sim.run(SPTPolicy())
    assert_replays(f, w, result, machine_events=outages)


def test_same_tick_calendar_phases_include_all_transports_before_arrivals():
    from smartsom.engine.calendar import CompletionEvent, EventCalendar
    from smartsom.modules.arrivals import ArrivalEvent
    from smartsom.modules.machine_events import MachineEvent
    from smartsom.modules.transport import TransportEvent

    expected = [
        CompletionEvent(4, "A", "standard", "M1"),
        MachineEvent(4, "M1", "breakdown"),
        MachineEvent(4, "M2", "repair"),
        TransportEvent(4, "V1", "J", 1, "pickup"),
        TransportEvent(4, "V2", "K", 2, "pickup"),
        TransportEvent(4, "V1", "J", 1, "delivery"),
        ArrivalEvent(4, "J", "reveal"),
        ArrivalEvent(4, "J", "release"),
    ]
    calendar = EventCalendar()
    for event in reversed(expected):
        calendar.schedule(event)
    assert [calendar.pop() for _ in expected] == expected


@pytest.mark.parametrize(
    "changes",
    [
        dict(source=None),
        dict(destination="M1"),
        dict(agv_id=[]),
        dict(start_time=True),
        dict(delivery_time=5.0),
    ],
)
def test_timetable_rejects_malformed_typed_records(changes):
    f, w = hand_case()
    result = Simulator(f, w, transport_enabled=True).run(SPTPolicy())
    trips = (
        replace(result.transport_schedule[0], **changes),
        *result.transport_schedule[1:],
    )
    with pytest.raises(ReplayError):
        replay_schedule(
            f, w, ExecutionSchedule(result.schedule, trips), transport_enabled=True
        )


def test_invariant_detects_idle_vehicle_teleport():
    from smartsom.engine.invariants import InvariantViolation

    f, w = hand_case()
    sim = Simulator(f, w, transport_enabled=True)
    sim._transport.agvs["V1"] = replace(sim._transport.agvs["V1"], node_id="I")
    with pytest.raises(InvariantViolation, match="last delivery node"):
        sim._check_invariants()
