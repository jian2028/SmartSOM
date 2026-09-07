"""Cross-check actual intervals, runtime state, resources, and pending events."""

from collections import defaultdict
from collections.abc import Mapping, Sequence

from smartsom.domain import FactorySpec, Operation, OperationStatus, ScheduledOperation
from smartsom.engine.calendar import CompletionEvent
from smartsom.engine.state import RuntimeState


class InvariantViolation(RuntimeError):
    """Canonical simulator state has become inconsistent."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise InvariantViolation(message)


def check_invariants(
    factory: FactorySpec,
    operations: Mapping[str, Operation],
    state: RuntimeState,
    events: tuple[CompletionEvent, ...],
    schedule: Sequence[ScheduledOperation],
    *,
    durations: Mapping[tuple[str, str], int] | None = None,
) -> None:
    _require(set(state.operations) == set(operations), "operation identity changed")
    _require(
        set(state.machine_occupants)
        == {machine.machine_id for machine in factory.machines},
        "machine identity changed",
    )
    _require(
        type(state.simulation_time) is int and state.simulation_time >= 0,
        "invalid clock",
    )
    processing = set()
    completed = set()
    intervals: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for operation_id, current in state.operations.items():
        _require(
            current.operation_id == operation_id, "operation state identity changed"
        )
        _require(current.status in OperationStatus, "unknown operation status")
        if current.status == OperationStatus.PENDING:
            _require(
                current.start_time is None and current.completion_time is None,
                "pending operation has execution times",
            )
            _require(
                current.processing_mode_id is None,
                "pending operation has a selected mode",
            )
            continue
        start = current.start_time
        _require(
            type(start) is int and 0 <= start <= state.simulation_time,
            "invalid start time",
        )
        operation = operations[operation_id]
        mode = operation.mode(current.processing_mode_id)
        _require(mode is not None, "unknown selected processing mode")
        ticks = (
            mode.nominal_ticks
            if durations is None
            else durations[(operation_id, mode.processing_mode_id)]
        )
        _require(type(ticks) is int and ticks > 0, "invalid execution duration")
        expected_end = start + ticks
        intervals[mode.machine_id].append((start, expected_end))
        for predecessor_id in operation.predecessor_ids:
            predecessor = state.operations[predecessor_id]
            _require(
                predecessor.status == OperationStatus.COMPLETED
                and type(predecessor.completion_time) is int
                and predecessor.completion_time <= start,
                f"precedence violated for {operation_id!r}",
            )
        if current.status == OperationStatus.PROCESSING:
            processing.add(operation_id)
            _require(
                current.completion_time is None,
                "processing operation already completed",
            )
            _require(
                state.machine_occupants[mode.machine_id] == operation_id,
                "processing operation is not held by its machine",
            )
        else:
            completed.add(operation_id)
            _require(
                type(current.completion_time) is int
                and current.completion_time == expected_end <= state.simulation_time,
                "invalid actual completion time",
            )

    occupants = [
        value for value in state.machine_occupants.values() if value is not None
    ]
    _require(
        len(occupants) == len(set(occupants)) and set(occupants) == processing,
        "machine occupancy does not match processing operations",
    )
    _require(
        len(events) == len(processing)
        and {event.operation_id for event in events} == processing,
        "completion events do not match processing operations",
    )
    for event in events:
        mode = operations[event.operation_id].mode(
            state.operations[event.operation_id].processing_mode_id
        )
        _require(
            event.machine_id == mode.machine_id
            and event.processing_mode_id == mode.processing_mode_id
            and event.simulation_time
            == state.operations[event.operation_id].start_time
            + (
                mode.nominal_ticks
                if durations is None
                else durations[(event.operation_id, mode.processing_mode_id)]
            )
            and event.simulation_time >= state.simulation_time,
            "invalid pending completion event",
        )
    _require(
        len(schedule) == len(completed)
        and {entry.operation_id for entry in schedule} == completed,
        "actual schedule has missing or duplicate completions",
    )
    for entry in schedule:
        current = state.operations[entry.operation_id]
        mode = operations[entry.operation_id].mode(current.processing_mode_id)
        _require(
            entry.processing_mode_id == mode.processing_mode_id
            and entry.machine_id == mode.machine_id
            and entry.start_time == current.start_time
            and entry.completion_time == current.completion_time,
            "actual interval disagrees with runtime state",
        )
    for machine_intervals in intervals.values():
        machine_intervals.sort()
        _require(
            all(
                left[1] <= right[0]
                for left, right in zip(machine_intervals, machine_intervals[1:])
            ),
            "machine processing intervals overlap",
        )
