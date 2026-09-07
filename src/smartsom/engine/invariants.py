"""Cross-check actual intervals, runtime state, resources, and pending events."""

from collections import defaultdict
from collections.abc import Mapping, Sequence

from smartsom.domain import FactorySpec, Operation, OperationStatus, ScheduledOperation
from smartsom.engine.calendar import CompletionEvent
from smartsom.engine.state import RuntimeState
from smartsom.modules.machine_events import MachineEventModule


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
    held_operations: frozenset[str] = frozenset(),
    durations: Mapping[tuple[str, str], int] | None = None,
    machine_events: MachineEventModule | None = None,
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
    paused = set()
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
            _require(
                current.actual_processing_ticks is None
                and operation_id not in state.progress,
                "pending operation has processing progress",
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
        _require(operation_id in state.progress, "started operation lacks progress")
        progress = state.progress[operation_id]
        _require(
            type(progress.processed_ticks) is int
            and 0 <= progress.processed_ticks <= ticks,
            "invalid accumulated processing",
        )
        active = current.status == OperationStatus.PROCESSING
        if active:
            _require(
                type(progress.segment_start) is int
                and start <= progress.segment_start <= state.simulation_time,
                "invalid active segment",
            )
            elapsed = (
                progress.processed_ticks
                + state.simulation_time
                - progress.segment_start
            )
        else:
            _require(
                progress.segment_start is None,
                "inactive operation has an active segment",
            )
            elapsed = progress.processed_ticks
        end = (
            current.completion_time
            if current.status == OperationStatus.COMPLETED
            else state.simulation_time
        )
        _require(
            type(end) is int and start <= end <= state.simulation_time,
            "invalid interval end",
        )
        expected_work = (
            end - start
            if machine_events is None
            else machine_events.by_machine[mode.machine_id].processing_between(
                start, end
            )
        )
        _require(
            elapsed == expected_work and 0 <= elapsed <= ticks,
            "processing progress is not conserved",
        )
        intervals[mode.machine_id].append((start, end))
        for predecessor_id in operation.predecessor_ids:
            predecessor = state.operations[predecessor_id]
            _require(
                predecessor.status == OperationStatus.COMPLETED
                and type(predecessor.completion_time) is int
                and predecessor.completion_time <= start,
                f"precedence violated for {operation_id!r}",
            )
        if current.status in (OperationStatus.PROCESSING, OperationStatus.PAUSED):
            (processing if active else paused).add(operation_id)
            _require(
                (mode.machine_id in state.down_machines) == (not active),
                "processing status disagrees with machine availability",
            )
            _require(
                current.actual_processing_ticks is None,
                "unfinished operation discloses actual processing time",
            )
            if not active:
                _require(elapsed < ticks, "paused operation has no remaining work")
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
                and current.completion_time <= state.simulation_time
                and elapsed == ticks
                and type(current.actual_processing_ticks) is int
                and current.actual_processing_ticks == ticks,
                "invalid actual completion time",
            )

    _require(
        all(
            state.operations[x].status
            in (OperationStatus.PENDING, OperationStatus.COMPLETED)
            for x in held_operations
        ),
        "invalid nonprocessing hold",
    )
    occupants = [
        value for value in state.machine_occupants.values() if value is not None
    ]
    _require(
        len(occupants) == len(set(occupants))
        and set(occupants) == processing | paused | held_operations,
        "machine occupancy does not match processing operations",
    )
    _require(
        len(events) == len(processing)
        and {event.operation_id for event in events} == processing,
        "completion events do not match processing operations",
    )
    _require(
        set(state.progress) == processing | paused | completed,
        "progress identities disagree with started operations",
    )
    for event in events:
        progress = state.progress[event.operation_id]
        mode = operations[event.operation_id].mode(
            state.operations[event.operation_id].processing_mode_id
        )
        _require(
            event.machine_id == mode.machine_id
            and event.processing_mode_id == mode.processing_mode_id
            and event.simulation_time
            == progress.segment_start
            + (
                mode.nominal_ticks
                if durations is None
                else durations[(event.operation_id, mode.processing_mode_id)]
            )
            - progress.processed_ticks
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
