"""Immutable matrix reference fixtures for source-data and solver import checks.

These describe historical benchmark inputs; they do not run a simulator.
"""

from collections import Counter, defaultdict

from smartsom.dispatch import Dispatch
from smartsom.domain import (
    FactorySpec,
    Job,
    Machine,
    Operation,
    Order,
    ProcessingMode,
    ScheduledOperation,
    WorkloadInstance,
)
from smartsom.trace import CompletionRecord, DispatchRecord, TerminationRecord


def op(name, machine, duration, predecessor=None):
    return Operation(
        name,
        (ProcessingMode("standard", machine, duration),),
        (predecessor,) if predecessor else (),
    )


def action(operation_id):
    return Dispatch(operation_id, "standard")


def problem(*jobs):
    return FactorySpec((Machine("M1"), Machine("M2"))), WorkloadInstance(
        (Order("order", jobs),)
    )


def competition_case():
    factory, workload = problem(
        Job("A", (op("A1", "M1", 3), op("A2", "M2", 2, "A1"))),
        Job("B", (op("B1", "M1", 1), op("B2", "M2", 3, "B1"))),
    )
    return factory, workload, tuple(map(action, ("B1", "A1", "B2", "A2")))


def crossing_case():
    factory, workload = problem(
        Job("C", (op("C1", "M1", 2), op("C2", "M2", 1, "C1"))),
        Job("D", (op("D1", "M2", 3), op("D2", "M1", 2, "D1"))),
    )
    return factory, workload, tuple(map(action, ("C1", "D1", "C2", "D2")))


class SequencePolicy:
    def __init__(self, actions):
        self.actions = iter(actions)

    def select_action(self, context):
        return next(self.actions)


class DurationPolicy:
    def __init__(self, longest=False):
        self.longest = longest

    def select_action(self, context):
        return min(
            context.candidates,
            key=lambda candidate: (
                -candidate.nominal_ticks if self.longest else candidate.nominal_ticks,
                candidate.action.operation_id,
            ),
        ).action


def assert_schedule_is_legal(workload, result):
    # Independently recompute feasibility from completed intervals, not engine state.
    expected = {operation.operation_id: operation for operation in workload.operations}
    counts = Counter(entry.operation_id for entry in result.schedule)
    assert counts == Counter({key: 1 for key in expected})
    by_operation = {entry.operation_id: entry for entry in result.schedule}
    by_machine = defaultdict(list)
    for entry in result.schedule:
        operation = expected[entry.operation_id]
        mode = next(
            mode
            for mode in operation.modes
            if mode.processing_mode_id == entry.processing_mode_id
        )
        assert (entry.processing_mode_id, entry.machine_id) == (
            mode.processing_mode_id,
            mode.machine_id,
        )
        assert entry.start_time >= 0
        assert entry.completion_time - entry.start_time == mode.nominal_ticks
        for predecessor in operation.predecessor_ids:
            assert by_operation[predecessor].completion_time <= entry.start_time
        by_machine[entry.machine_id].append(entry)
    for entries in by_machine.values():
        entries.sort(key=lambda entry: entry.start_time)
        assert all(
            left.completion_time <= right.start_time
            for left, right in zip(entries, entries[1:])
        )
    assert result.makespan == max(entry.completion_time for entry in result.schedule)
    assert [record.sequence for record in result.trace] == list(
        range(len(result.trace))
    )
    starts = [
        record.action.operation_id
        for record in result.trace
        if isinstance(record, DispatchRecord)
    ]
    finishes = [
        record.action.operation_id
        for record in result.trace
        if isinstance(record, CompletionRecord)
    ]
    assert Counter(starts) == Counter(finishes) == counts
    assert sum(isinstance(record, TerminationRecord) for record in result.trace) == 1
    assert result.trace[-1].simulation_time == result.makespan


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
