"""Pure interval validation for frozen external solver references.

This checks historical benchmark data; it does not execute factory physics.
"""

from collections.abc import Iterable

from smartsom.domain import (
    FactorySpec,
    ScheduledOperation,
    WorkloadInstance,
    validate_problem,
)
from smartsom.domain.arrivals import ArrivalPlan
from smartsom.domain.machine_events import MachineOutagePlan
from smartsom.domain.processing_times import ProcessingTimePlan
from smartsom.modules.arrivals import ArrivalModule
from smartsom.modules.machine_events import MachineEventModule
from smartsom.modules.processing_times import ProcessingTimeModule


class ScheduleValidationError(ValueError):
    pass


def validate_schedule(
    factory: FactorySpec,
    workload: WorkloadInstance,
    schedule: Iterable[ScheduledOperation],
    *,
    arrivals: ArrivalPlan | None = None,
    processing_times: ProcessingTimePlan | None = None,
    machine_events: MachineOutagePlan | None = None,
) -> tuple[ScheduledOperation, ...]:
    """Return canonical intervals only after validating the entire timetable."""
    validate_problem(factory, workload)
    timing = ArrivalModule(workload, arrivals)
    execution = ProcessingTimeModule(workload, processing_times)
    outages = MachineEventModule(factory, machine_events)
    operations = {op.operation_id: op for op in workload.operations}
    entries = {}
    machines: dict[str, list[ScheduledOperation]] = {}
    for entry in schedule:
        if not isinstance(entry, ScheduledOperation):
            raise ScheduleValidationError(
                "schedule entries must be ScheduledOperation values"
            )
        if (
            not isinstance(entry.operation_id, str)
            or entry.operation_id not in operations
        ):
            raise ScheduleValidationError(f"unknown operation: {entry.operation_id!r}")
        if entry.operation_id in entries:
            raise ScheduleValidationError(
                f"duplicate scheduled operation: {entry.operation_id}"
            )
        mode = operations[entry.operation_id].mode(entry.processing_mode_id)
        if mode is None or entry.machine_id != mode.machine_id:
            raise ScheduleValidationError(
                f"incorrect machine or mode: {entry.operation_id}"
            )
        if any(
            type(tick) is not int or tick < 0
            for tick in (entry.start_time, entry.completion_time)
        ):
            raise ScheduleValidationError(
                f"schedule times must be nonnegative integers: {entry.operation_id}"
            )
        if (
            entry.completion_time <= entry.start_time
            or outages.by_machine[entry.machine_id].is_down(entry.start_time)
            or outages.by_machine[entry.machine_id].processing_between(
                entry.start_time, entry.completion_time
            )
            != execution.durations[(entry.operation_id, entry.processing_mode_id)]
        ):
            raise ScheduleValidationError(
                f"incorrect duration or down-machine start: {entry.operation_id}"
            )
        # A completion cannot be postponed through a downtime interval after all
        # work was already done at its left boundary (completion-first).
        if outages.by_machine[entry.machine_id].is_down(entry.completion_time - 1):
            raise ScheduleValidationError(
                f"completion delayed after processing finished: {entry.operation_id}"
            )
        if entry.start_time < timing.release_at(entry.operation_id):
            raise ScheduleValidationError(
                f"start before job release: {entry.operation_id}"
            )
        entries[entry.operation_id] = entry
        machines.setdefault(entry.machine_id, []).append(entry)
    missing = sorted(operations.keys() - entries.keys())
    if missing:
        raise ScheduleValidationError(f"schedule is missing operations: {missing}")
    for key, entry in entries.items():
        if any(
            entries[pred].completion_time > entry.start_time
            for pred in operations[key].predecessor_ids
        ):
            raise ScheduleValidationError(f"precedence violation: {key}")
    for machine, intervals in machines.items():
        ordered = sorted(
            intervals, key=lambda entry: (entry.start_time, entry.operation_id)
        )
        if any(
            left.completion_time > right.start_time
            for left, right in zip(ordered, ordered[1:])
        ):
            raise ScheduleValidationError(f"resource overlap on machine: {machine}")
    return tuple(
        sorted(
            entries.values(), key=lambda entry: (entry.start_time, entry.operation_id)
        )
    )
