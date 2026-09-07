from dataclasses import FrozenInstanceError, replace

import pytest
from test_static_engine import (
    action,
    assert_schedule_is_legal,
    competition_case,
    crossing_case,
    op,
    problem,
)

from smartsom.algorithms import SPTPolicy
from smartsom.dispatch import WaitUntil
from smartsom.domain import Job, ScheduledOperation
from smartsom.engine import (
    InvalidActionError,
    ReplayError,
    SimulationFinishedError,
    Simulator,
    replay,
    replay_schedule,
)
from smartsom.engine.schedule import ScheduleReplayPolicy
from smartsom.trace import CompletionRecord, DecisionRecord, WaitRecord


def interval(name, machine, start, end):
    return ScheduledOperation(name, "standard", machine, start, end)


def test_initial_wait_without_events_and_action_replay():
    factory, workload = problem(Job("A", (op("A", "M1", 2),)))
    simulator = Simulator(factory, workload)
    initial = simulator.current_decision
    context = simulator.step(WaitUntil(5))
    assert context.simulation_time == 5
    assert context.feasible_actions == initial.feasible_actions == (action("A"),)
    with pytest.raises(FrozenInstanceError):
        context.simulation_time = 10
    result = simulator.step(action("A"))
    assert result.schedule == (interval("A", "M1", 5, 7),)
    assert result.actions == (WaitUntil(5), action("A"))
    assert result.trace[1] == WaitRecord(1, 0, WaitUntil(5))
    assert replay(factory, workload, result.actions) == result
    with pytest.raises(SimulationFinishedError):
        simulator.step(WaitUntil(10))


@pytest.mark.parametrize("target", [-1, 0, 5, True, False, 6.0, "6", None])
def test_invalid_wait_is_atomic(target):
    factory, workload, _ = crossing_case()
    simulator = Simulator(factory, workload)
    simulator.step(WaitUntil(5))
    before = simulator.current_decision, simulator.trace
    with pytest.raises(InvalidActionError, match="tick 5.*integer greater"):
        simulator.step(WaitUntil(target))
    assert (simulator.current_decision, simulator.trace) == before


def test_wait_interrupted_and_not_persisted():
    factory, workload, _ = crossing_case()
    simulator = Simulator(factory, workload)
    simulator.step(action("C1"))
    context = simulator.step(WaitUntil(10))
    assert context.simulation_time == 2
    assert context.feasible_actions == (action("C2"), action("D1"))
    context = simulator.step(action("D1"))
    assert context.simulation_time == 5  # No legal dispatch while M2 is busy.
    context = simulator.step(action("C2"))
    result = simulator.step(action("D2"))
    assert result.makespan == 7  # The abandoned wait to 10 has no effect.


def test_target_before_completion_and_all_same_tick_events_settle():
    factory, workload = problem(
        Job("A", (op("A", "M1", 3),)),
        Job("B", (op("B", "M2", 3),)),
        Job("C", (op("C", "M1", 1),)),
    )
    simulator = Simulator(factory, workload)
    simulator.step(action("A"))
    context = simulator.step(WaitUntil(1))
    assert context.simulation_time == 1
    # Separate case: retain a dispatch choice on a third machine at tick 0.
    from smartsom.domain import FactorySpec, Machine

    factory = FactorySpec((*factory.machines, Machine("M3")))
    workload = replace(
        workload,
        orders=(
            replace(
                workload.orders[0],
                jobs=(*workload.orders[0].jobs[:2], Job("C", (op("C", "M3", 1),))),
            ),
        ),
    )
    simulator = Simulator(factory, workload)
    simulator.step(action("B"))
    simulator.step(action("A"))
    context = simulator.step(WaitUntil(8))
    assert context.simulation_time == 3
    at_three = [record for record in simulator.trace if record.simulation_time == 3]
    assert [type(record) for record in at_three] == [
        CompletionRecord,
        CompletionRecord,
        DecisionRecord,
    ]
    assert [record.action.operation_id for record in at_three[:2]] == ["A", "B"]


def test_schedule_retains_idle_and_canonicalizes_simultaneous_starts():
    factory, workload, _ = crossing_case()
    schedule = (
        interval("C1", "M1", 5, 7),
        interval("D1", "M2", 5, 8),
        interval("C2", "M2", 10, 11),
        interval("D2", "M1", 10, 12),
    )
    result = replay_schedule(factory, workload, reversed(schedule))
    assert result.schedule == schedule
    assert result.makespan == 12
    assert replay_schedule(factory, workload, schedule) == result
    assert replay(factory, workload, result.actions) == result
    reordered = replace(
        workload,
        orders=(
            replace(
                workload.orders[0],
                jobs=tuple(
                    replace(job, operations=tuple(reversed(job.operations)))
                    for job in reversed(workload.orders[0].jobs)
                ),
            ),
        ),
    )
    assert (
        replay_schedule(
            replace(factory, machines=tuple(reversed(factory.machines))),
            reordered,
            schedule,
        )
        == result
    )
    assert_schedule_is_legal(workload, result)


@pytest.mark.parametrize(
    "mutation,reason",
    [
        (lambda s: s[:-1], "missing"),
        (lambda s: (*s, s[0]), "duplicate"),
        (lambda s: (replace(s[0], operation_id="unknown"), *s[1:]), "unknown"),
        (lambda s: (replace(s[0], processing_mode_id="wrong"), *s[1:]), "mode"),
        (lambda s: (replace(s[0], machine_id="M2"), *s[1:]), "machine"),
        (lambda s: (replace(s[0], start_time=True), *s[1:]), "integer"),
        (lambda s: (replace(s[0], completion_time=1.0), *s[1:]), "integer"),
        (lambda s: (replace(s[0], start_time=-1), *s[1:]), "integer"),
        (lambda s: (replace(s[0], completion_time=2), *s[1:]), "duration"),
        (
            lambda s: (s[0], replace(s[1], start_time=0, completion_time=3), *s[2:]),
            "overlap",
        ),
        (
            lambda s: (*s[:-1], replace(s[-1], start_time=3, completion_time=5)),
            "precedence",
        ),
        (lambda s: (None, *s[1:]), "ScheduledOperation"),
    ],
)
def test_invalid_schedules_are_rejected_before_simulator(mutation, reason, monkeypatch):
    factory, workload, actions = competition_case()
    schedule = replay(factory, workload, actions).schedule

    def forbidden(*args):
        pytest.fail("invalid timetable constructed a Simulator")

    monkeypatch.setattr("smartsom.engine.schedule.Simulator", forbidden)
    with pytest.raises(ReplayError, match=reason):
        replay_schedule(factory, workload, mutation(schedule))


def test_replay_postcheck_rejects_interval_and_objective_deviation():
    factory, workload, actions = competition_case()
    result = replay(factory, workload, actions)
    policy = ScheduleReplayPolicy(factory, workload, result.schedule)
    with pytest.raises(ReplayError, match="makespan"):
        policy.verify_result(replace(result, makespan=7))
    with pytest.raises(ReplayError, match="schedule"):
        policy.verify_result(replace(result, schedule=result.schedule[:-1]))


def test_spt_uses_current_duration_then_semantic_identity():
    factory, workload = problem(
        Job("A", (op("A", "M1", 3),)),
        Job("B", (op("B", "M1", 1),)),
        Job("C", (op("C", "M2", 1),)),
    )
    simulator = Simulator(factory, workload)
    policy = SPTPolicy()
    assert policy.select_action(simulator.current_decision) == action("B")
    simulator.step(action("B"))
    assert policy.select_action(simulator.current_decision) == action("C")
    assert_schedule_is_legal(workload, simulator.run(policy))
