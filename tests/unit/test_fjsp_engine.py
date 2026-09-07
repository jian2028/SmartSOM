from dataclasses import FrozenInstanceError, replace
from itertools import permutations

import pytest
from test_static_engine import action, assert_schedule_is_legal, op, problem

from smartsom.algorithms import SPTPolicy
from smartsom.dispatch import Dispatch
from smartsom.domain import (
    Job,
    Operation,
    OperationStatus,
    ProcessingMode,
    ScheduledOperation,
)
from smartsom.engine import (
    InvalidActionError,
    InvariantViolation,
    Simulator,
    replay,
    replay_schedule,
)


def flexible_case():
    return problem(
        Job(
            "A",
            (
                Operation(
                    "A1",
                    (ProcessingMode("slow", "M1", 4), ProcessingMode("fast", "M2", 1)),
                ),
                op("A2", "M1", 2, "A1"),
            ),
        ),
        Job("B", (op("B1", "M2", 2), op("B2", "M1", 1, "B1"))),
    )


FAST = (
    ScheduledOperation("A1", "fast", "M2", 0, 1),
    ScheduledOperation("A2", "standard", "M1", 1, 3),
    ScheduledOperation("B1", "standard", "M2", 1, 3),
    ScheduledOperation("B2", "standard", "M1", 3, 4),
)
SLOW = (
    ScheduledOperation("A1", "slow", "M1", 0, 4),
    ScheduledOperation("B1", "standard", "M2", 0, 2),
    ScheduledOperation("B2", "standard", "M1", 4, 5),
    ScheduledOperation("A2", "standard", "M1", 5, 7),
)


@pytest.mark.parametrize(("schedule", "makespan"), [(FAST, 4), (SLOW, 7)])
def test_hand_calculated_selected_modes(schedule, makespan):
    factory, workload = flexible_case()
    result = replay_schedule(factory, workload, schedule)
    assert result.schedule == schedule
    assert result.makespan == makespan
    assert replay(factory, workload, result.actions) == result
    assert_schedule_is_legal(workload, result)
    for machines in permutations(factory.machines):
        for modes in permutations(workload.operations[0].modes):
            changed = replace(
                workload,
                orders=tuple(
                    replace(
                        order,
                        jobs=tuple(
                            replace(
                                job,
                                operations=tuple(
                                    replace(
                                        operation,
                                        modes=modes
                                        if operation.operation_id == "A1"
                                        else operation.modes,
                                    )
                                    for operation in reversed(job.operations)
                                ),
                            )
                            for job in reversed(order.jobs)
                        ),
                    )
                    for order in workload.orders
                ),
            )
            assert (
                replay(replace(factory, machines=machines), changed, result.actions)
                == result
            )
            assert (
                replay_schedule(
                    replace(factory, machines=machines), changed, reversed(schedule)
                )
                == result
            )


def test_selected_mode_is_fixed_and_all_alternatives_disappear():
    factory, workload = flexible_case()
    sim = Simulator(factory, workload)
    initial = sim.current_decision
    assert initial.feasible_actions == (
        Dispatch("A1", "fast"),
        Dispatch("A1", "slow"),
        action("B1"),
    )
    assert all(state.processing_mode_id is None for state in initial.operations)
    context = sim.step(Dispatch("A1", "slow"))
    assert context.feasible_actions == (action("B1"),)
    state = next(state for state in context.operations if state.operation_id == "A1")
    assert state.processing_mode_id == "slow"
    assert state.status == OperationStatus.PROCESSING
    with pytest.raises(FrozenInstanceError):
        state.processing_mode_id = "fast"
    for invalid, reason in [
        (Dispatch("A1", "fast"), "already started"),
        (Dispatch("A1", "nope"), "unknown processing mode"),
        (action("A2"), "predecessor"),
    ]:
        trace = sim.trace
        with pytest.raises(InvalidActionError, match=reason):
            sim.step(invalid)
        assert sim.trace == trace and sim.current_decision is context
    context = sim.step(action("B1"))
    state = next(state for state in context.operations if state.operation_id == "A1")
    assert (
        state.processing_mode_id == "slow" and state.status == OperationStatus.COMPLETED
    )


def test_busy_mode_is_excluded_but_another_mode_remains_legal():
    factory, workload = flexible_case()
    sim = Simulator(factory, workload)
    context = sim.step(action("B1"))
    assert context.feasible_actions == (Dispatch("A1", "slow"),)
    trace = sim.trace
    with pytest.raises(InvalidActionError, match="machine is busy"):
        sim.step(Dispatch("A1", "fast"))
    assert sim.trace == trace and sim.current_decision is context
    assert SPTPolicy().select_action(context) == Dispatch("A1", "slow")


def test_same_machine_modes_remain_distinct_and_script_can_choose_slow():
    modes = (
        ProcessingMode("slow", "M1", 5),
        ProcessingMode("z", "M1", 2),
        ProcessingMode("a", "M1", 2),
    )
    factory, workload = problem(Job("A", (Operation("A1", modes),)))
    for ordered in permutations(modes):
        altered = replace(
            workload,
            orders=(
                replace(
                    workload.orders[0], jobs=(Job("A", (Operation("A1", ordered),)),)
                ),
            ),
        )
        context = Simulator(factory, altered).current_decision
        assert context.feasible_actions == tuple(
            Dispatch("A1", key) for key in ("a", "slow", "z")
        )
        assert SPTPolicy().select_action(context) == Dispatch("A1", "a")
        for mode in modes:
            result = replay(
                factory, altered, (Dispatch("A1", mode.processing_mode_id),)
            )
            assert result.makespan == mode.nominal_ticks
            assert result.schedule[0].processing_mode_id == mode.processing_mode_id


def test_invariants_check_the_selected_mode_against_pending_event():
    factory, workload = flexible_case()
    sim = Simulator(factory, workload)
    sim.step(Dispatch("A1", "slow"))
    sim._state.operations["A1"] = replace(
        sim._state.operations["A1"], processing_mode_id="fast"
    )
    with pytest.raises(InvariantViolation):
        sim._check_invariants()
