"""Arrival semantics checked independently of configuration and solver code."""

from dataclasses import FrozenInstanceError, replace
from itertools import permutations

import pytest
from test_static_engine import (
    action,
    assert_schedule_is_legal,
    competition_case,
    op,
    problem,
)

from smartsom.algorithms import FirstFeasiblePolicy, SPTPolicy
from smartsom.dispatch import WaitNextEvent, WaitUntil
from smartsom.domain import (
    ArrivalPlan,
    Job,
    JobArrival,
    Order,
    ScheduledOperation,
    WorkloadInstance,
)
from smartsom.engine import (
    InvalidActionError,
    ReplayError,
    SimulationFinishedError,
    Simulator,
    replay,
    replay_schedule,
)
from smartsom.engine.calendar import CompletionEvent, EventCalendar
from smartsom.modules.arrivals import ArrivalEvent, ArrivalModule

TRIGGERS = ("dispatch_available", "arrival_event")


def arrival_case(reveal=1):
    factory, workload = problem(
        Job("A", (op("A1", "M1", 3), op("A2", "M2", 2, "A1"))),
        Job("B", (op("B1", "M2", 2), op("B2", "M1", 1, "B1"))),
    )
    return (
        factory,
        workload,
        ArrivalPlan((JobArrival("A", 0, 0), JobArrival("B", 2, reveal))),
    )


def test_release_index_covers_hidden_jobs_and_cannot_be_mutated():
    _, workload, plan = arrival_case()
    module = ArrivalModule(workload, plan)
    assert [job.job_id for job in module.visible_jobs(0)] == ["A"]
    assert [module.release_at(key) for key in ("A1", "A2", "B1", "B2")] == [0, 0, 2, 2]
    with pytest.raises(TypeError):
        module._release_by_operation["B1"] = 0
    with pytest.raises(FrozenInstanceError):
        module._release_by_operation = {}
    assert ArrivalModule(workload, None).release_at("B1") == 0


REFERENCE = (
    ScheduledOperation("A1", "standard", "M1", 0, 3),
    ScheduledOperation("B1", "standard", "M2", 2, 4),
    ScheduledOperation("A2", "standard", "M2", 4, 6),
    ScheduledOperation("B2", "standard", "M1", 4, 5),
)


@pytest.mark.parametrize("trigger", TRIGGERS)
@pytest.mark.parametrize("reveal", [0, 1, 2])
def test_hand_schedule_and_action_replay(trigger, reveal):
    factory, workload, arrivals = arrival_case(reveal)
    result = Simulator(
        factory, workload, arrivals=arrivals, decision_trigger=trigger
    ).run(SPTPolicy())
    assert result.makespan == 6
    assert result.schedule == REFERENCE
    assert_schedule_is_legal(workload, result)
    assert result == replay(
        factory, workload, result.actions, arrivals=arrivals, decision_trigger=trigger
    )
    exact = replay_schedule(
        factory,
        workload,
        reversed(REFERENCE),
        arrivals=arrivals,
        decision_trigger=trigger,
    )
    assert exact.schedule == REFERENCE
    assert exact.makespan == 6
    assert any(isinstance(a, WaitNextEvent) for a in result.actions) == (
        trigger == "arrival_event" and reveal == 1
    )


def test_reveal_is_full_but_release_is_physical_and_rejection_is_atomic():
    factory, workload, arrivals = arrival_case()
    sim = Simulator(
        factory, workload, arrivals=arrivals, decision_trigger="arrival_event"
    )
    initial = sim.current_decision
    assert [job.job_id for job in initial.jobs] == ["A"]
    assert [state.operation_id for state in initial.operations] == ["A1", "A2"]
    for hidden in ("B1", "absent"):
        with pytest.raises(InvalidActionError) as error:
            sim.step(action(hidden))
        assert error.value.reason == "unknown operation ID"
        assert sim.current_decision is initial
    view = sim.step(action("A1"))
    assert view.simulation_time == 1 and not view.candidates
    assert [job.job_id for job in view.jobs] == ["A", "B"]
    assert view.jobs[1].operations == workload.orders[0].jobs[1].operations
    assert view.jobs[1].release_at == 2 and view.jobs[1].reveal_at == 1
    before = sim.trace
    with pytest.raises(InvalidActionError, match="not been released"):
        sim.step(action("B1"))
    assert sim.trace == before and sim.current_decision is view
    with pytest.raises(FrozenInstanceError):
        view.jobs[1].release_at = 1
    with pytest.raises(FrozenInstanceError):
        view.jobs[1].operations[0].modes[0].nominal_ticks = 1
    assert sim.step(WaitNextEvent()).simulation_time == 2


@pytest.mark.parametrize("trigger", TRIGGERS)
def test_hidden_world_changes_do_not_change_visible_prefix(trigger):
    factory, workload, arrivals = arrival_case()
    first = Simulator(factory, workload, arrivals=arrivals, decision_trigger=trigger)
    hidden = (
        Job("different", (op("secret", "M2", 70),)),
        Job("another", (op("other", "M1", 30),)),
    )
    other = WorkloadInstance((Order("order", (workload.orders[0].jobs[0], *hidden)),))
    timing = ArrivalPlan(
        (
            JobArrival("A", 0, 0),
            JobArrival("different", 50, 20),
            JobArrival("another", 30, 10),
        )
    )
    second = Simulator(factory, other, arrivals=timing, decision_trigger=trigger)
    assert first.current_decision == second.current_decision
    assert first.trace == second.trace
    for policy in (SPTPolicy(), FirstFeasiblePolicy()):
        assert policy.select_action(first.current_decision) == policy.select_action(
            second.current_decision
        )


@pytest.mark.parametrize("trigger", TRIGGERS)
def test_zero_plan_exactly_preserves_old_complete_trace(trigger):
    factory, workload, actions = competition_case()
    original = replay(factory, workload, actions)
    zeros = ArrivalPlan(
        tuple(
            JobArrival(job.job_id, 0, 0)
            for order in workload.orders
            for job in order.jobs
        )
    )
    assert (
        replay(factory, workload, actions, arrivals=zeros, decision_trigger=trigger)
        == original
    )


@pytest.mark.parametrize("trigger", TRIGGERS)
@pytest.mark.parametrize("reveal", [0, 2, 5])
def test_no_initial_work_and_initial_notice(trigger, reveal):
    factory, workload = problem(Job("A", (op("A1", "M1", 2),)))
    arrivals = ArrivalPlan((JobArrival("A", 5, reveal),))
    sim = Simulator(factory, workload, arrivals=arrivals, decision_trigger=trigger)
    view = sim.current_decision
    assert view.simulation_time == (reveal if trigger == "arrival_event" else 5)
    assert bool(view.candidates) == (view.simulation_time == 5)
    result = sim.run(FirstFeasiblePolicy())
    assert result.makespan == 7 and result.schedule[0].start_time == 5
    with pytest.raises(SimulationFinishedError):
        sim.step(WaitNextEvent())


def test_wait_rules_and_completion_only_empty_views():
    factory, workload, arrivals = arrival_case()
    sim = Simulator(
        factory, workload, arrivals=arrivals, decision_trigger="arrival_event"
    )
    assert sim.step(WaitUntil(20)).simulation_time == 1
    assert sim.step(WaitNextEvent()).simulation_time == 2
    static = Simulator(factory, workload)
    before = static.trace
    with pytest.raises(InvalidActionError, match="no future event"):
        static.step(WaitNextEvent())
    assert static.trace == before and static.current_decision.simulation_time == 0
    # A completion alone at tick 1 must not create an empty decision.
    factory, workload = problem(
        Job("A", (op("A1", "M1", 1),)), Job("B", (op("B1", "M2", 1),))
    )
    plan = ArrivalPlan((JobArrival("A", 0, 0), JobArrival("B", 5, 5)))
    sim = Simulator(factory, workload, arrivals=plan, decision_trigger="arrival_event")
    assert sim.step(action("A1")).simulation_time == 5
    assert not any(
        record.kind == "decision" and record.simulation_time == 1
        for record in sim.trace
    )


def test_same_tick_phase_order_and_calendar_insertion_independence():
    events = (
        ArrivalEvent(3, "B", "release"),
        ArrivalEvent(3, "B", "reveal"),
        CompletionEvent(3, "Z", "standard", "M2"),
        CompletionEvent(3, "A", "standard", "M1"),
    )
    expected = (events[3], events[2], events[1], events[0])
    for ordering in permutations(events):
        calendar = EventCalendar()
        for event in ordering:
            calendar.schedule(event)
        assert calendar.pending == expected
    factory, workload = problem(
        Job("A", (op("A1", "M1", 3),)), Job("B", (op("B1", "M2", 1),))
    )
    sim = Simulator(
        factory,
        workload,
        arrivals=ArrivalPlan((JobArrival("A", 0, 0), JobArrival("B", 3, 3))),
        decision_trigger="arrival_event",
    )
    sim.step(action("A1"))
    assert [record.kind for record in sim.trace if record.simulation_time == 3] == [
        "complete",
        "reveal",
        "release",
        "decision",
    ]


@pytest.mark.parametrize(
    "release,reveal",
    [(True, 0), (1.0, 0), (1, False), (1, 0.0), (-1, 0), (1, -1), (1, 2)],
)
def test_invalid_timing(release, reveal):
    with pytest.raises(ValueError):
        JobArrival("A", release, reveal)


def test_invalid_coverage_and_schedule_release():
    factory, workload, _ = arrival_case()
    for rows in [
        (),
        (JobArrival("A", 0, 0),) * 2,
        (JobArrival("A", 0, 0),),
        (JobArrival("A", 0, 0), JobArrival("unknown", 1, 1)),
    ]:
        with pytest.raises(ValueError):
            Simulator(factory, workload, arrivals=ArrivalPlan(rows))
    with pytest.raises(ValueError, match="requires arrivals"):
        Simulator(factory, workload, decision_trigger="arrival_event")
    with pytest.raises(ValueError, match="unknown decision_trigger"):
        Simulator(factory, workload, decision_trigger="anything")
    plan = ArrivalPlan((JobArrival("A", 1, 0), JobArrival("B", 2, 1)))
    with pytest.raises(ReplayError, match="before job release"):
        replay_schedule(factory, workload, REFERENCE, arrivals=plan)


@pytest.mark.parametrize("trigger", TRIGGERS)
def test_container_reordering_and_intentional_idle(trigger):
    factory, workload, plan = arrival_case()
    changed = replace(
        workload,
        orders=tuple(
            replace(
                order,
                jobs=tuple(
                    replace(job, operations=tuple(reversed(job.operations)))
                    for job in reversed(order.jobs)
                ),
            )
            for order in reversed(workload.orders)
        ),
    )
    expected = Simulator(
        factory, workload, arrivals=plan, decision_trigger=trigger
    ).run(SPTPolicy())
    actual = Simulator(
        replace(factory, machines=tuple(reversed(factory.machines))),
        changed,
        arrivals=ArrivalPlan(tuple(reversed(plan.jobs))),
        decision_trigger=trigger,
    ).run(SPTPolicy())
    assert expected == actual
    delayed = tuple(
        replace(
            entry,
            start_time=entry.start_time + 10,
            completion_time=entry.completion_time + 10,
        )
        for entry in REFERENCE
    )
    assert (
        replay_schedule(
            factory, changed, delayed, arrivals=plan, decision_trigger=trigger
        ).schedule
        == delayed
    )


def test_arrivals_preserve_flexible_mode_selection_and_early_notice():
    from test_fjsp_engine import flexible_case

    factory, workload = flexible_case()
    plan = ArrivalPlan((JobArrival("A", 2, 0), JobArrival("B", 2, 1)))
    result = Simulator(
        factory, workload, arrivals=plan, decision_trigger="arrival_event"
    ).run(SPTPolicy())
    assert all(entry.start_time >= 2 for entry in result.schedule)
    assert (
        next(
            entry.processing_mode_id
            for entry in result.schedule
            if entry.operation_id == "A1"
        )
        == "fast"
    )
    assert (
        replay(
            factory,
            workload,
            result.actions,
            arrivals=plan,
            decision_trigger="arrival_event",
        )
        == result
    )
    assert (
        replay_schedule(factory, workload, result.schedule, arrivals=plan).schedule
        == result.schedule
    )


def test_empty_arrival_notification_is_once_per_tick_and_all_events_settle():
    factory, workload = problem(
        Job("A", (op("A1", "M1", 4),)),
        Job("B", (op("B1", "M1", 1),)),
        Job("C", (op("C1", "M1", 1),)),
    )
    plan = ArrivalPlan(
        (JobArrival("A", 0, 0), JobArrival("B", 2, 2), JobArrival("C", 2, 2))
    )
    sim = Simulator(factory, workload, arrivals=plan, decision_trigger="arrival_event")
    view = sim.step(action("A1"))
    assert view.simulation_time == 2 and not view.candidates
    assert [job.job_id for job in view.jobs] == ["A", "B", "C"]
    assert [row.kind for row in sim.trace if row.simulation_time == 2] == [
        "reveal",
        "reveal",
        "release",
        "release",
        "decision",
    ]
    assert sim.step(WaitNextEvent()).simulation_time == 4
    assert (
        sum(row.kind == "decision" and row.simulation_time == 2 for row in sim.trace)
        == 1
    )


def test_missing_arrival_event_is_an_invariant_failure():
    from smartsom.engine import InvariantViolation

    factory, workload, plan = arrival_case()
    sim = Simulator(factory, workload, arrivals=plan)
    sim._calendar.pop()  # Fault injection, not a supported caller operation.
    with pytest.raises(InvariantViolation, match="arrival calendar"):
        sim._check_invariants()
