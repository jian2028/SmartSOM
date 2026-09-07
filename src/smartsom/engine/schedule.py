"""Validate and replay a complete semantic timetable through the same step API."""

from collections.abc import Iterable

from smartsom.dispatch import DecisionContext, Dispatch, SemanticAction, WaitUntil
from smartsom.domain import (
    FactorySpec,
    OperationStatus,
    ScheduledOperation,
    WorkloadInstance,
    validate_problem,
)
from smartsom.engine.replay import ReplayError
from smartsom.engine.result import SimulationResult
from smartsom.engine.simulator import Simulator


def validate_schedule(
    factory: FactorySpec,
    workload: WorkloadInstance,
    schedule: Iterable[ScheduledOperation],
) -> tuple[ScheduledOperation, ...]:
    """Return canonical intervals only after validating the entire timetable."""
    validate_problem(factory, workload)
    operations = {op.operation_id: op for op in workload.operations}
    entries = {}
    machines: dict[str, list[ScheduledOperation]] = {}
    for entry in schedule:
        if not isinstance(entry, ScheduledOperation):
            raise ReplayError("schedule entries must be ScheduledOperation values")
        if (
            not isinstance(entry.operation_id, str)
            or entry.operation_id not in operations
        ):
            raise ReplayError(f"unknown operation: {entry.operation_id!r}")
        if entry.operation_id in entries:
            raise ReplayError(f"duplicate scheduled operation: {entry.operation_id}")
        mode = operations[entry.operation_id].modes[0]
        if (entry.processing_mode_id, entry.machine_id) != (
            mode.processing_mode_id,
            mode.machine_id,
        ):
            raise ReplayError(f"incorrect machine or mode: {entry.operation_id}")
        if any(
            type(tick) is not int or tick < 0
            for tick in (entry.start_time, entry.completion_time)
        ):
            raise ReplayError(
                f"schedule times must be nonnegative integers: {entry.operation_id}"
            )
        if entry.completion_time - entry.start_time != mode.nominal_ticks:
            raise ReplayError(f"incorrect duration: {entry.operation_id}")
        entries[entry.operation_id] = entry
        machines.setdefault(entry.machine_id, []).append(entry)
    missing = sorted(operations.keys() - entries.keys())
    if missing:
        raise ReplayError(f"schedule is missing operations: {missing}")
    for key, entry in entries.items():
        if any(
            entries[pred].completion_time > entry.start_time
            for pred in operations[key].predecessor_ids
        ):
            raise ReplayError(f"precedence violation: {key}")
    for machine, intervals in machines.items():
        ordered = sorted(
            intervals, key=lambda entry: (entry.start_time, entry.operation_id)
        )
        if any(
            left.completion_time > right.start_time
            for left, right in zip(ordered, ordered[1:])
        ):
            raise ReplayError(f"resource overlap on machine: {machine}")
    return tuple(
        sorted(
            entries.values(), key=lambda entry: (entry.start_time, entry.operation_id)
        )
    )


class ScheduleReplayPolicy:
    """One schedule-to-action conversion shared by standalone replay and runner."""

    def __init__(
        self,
        factory: FactorySpec,
        workload: WorkloadInstance,
        schedule: Iterable[ScheduledOperation],
    ) -> None:
        self._schedule = validate_schedule(factory, workload, schedule)

    def select_action(self, context: DecisionContext) -> SemanticAction:
        pending = {
            op.operation_id
            for op in context.operations
            if op.status == OperationStatus.PENDING
        }
        entry = next(
            (entry for entry in self._schedule if entry.operation_id in pending), None
        )
        if entry is None:
            raise ReplayError("decision remains after all scheduled operations started")
        if context.simulation_time < entry.start_time:
            return WaitUntil(entry.start_time)
        action = Dispatch(entry.operation_id, entry.processing_mode_id)
        if (
            context.simulation_time != entry.start_time
            or action not in context.feasible_actions
        ):
            raise ReplayError(
                f"cannot start {entry.operation_id} exactly at tick {entry.start_time}"
            )
        return action

    def verify_result(self, result: SimulationResult) -> None:
        actual = tuple(
            sorted(
                result.schedule,
                key=lambda entry: (entry.start_time, entry.operation_id),
            )
        )
        if actual != self._schedule:
            raise ReplayError("actual schedule differs from the requested schedule")
        if result.makespan != max(entry.completion_time for entry in self._schedule):
            raise ReplayError("actual makespan differs from the requested schedule")


def replay_schedule(
    factory: FactorySpec,
    workload: WorkloadInstance,
    schedule: Iterable[ScheduledOperation],
) -> SimulationResult:
    policy = ScheduleReplayPolicy(factory, workload, schedule)
    result = Simulator(factory, workload).run(policy)
    policy.verify_result(result)
    return result
