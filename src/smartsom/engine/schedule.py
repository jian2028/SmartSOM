"""Validate and replay a complete semantic timetable through the same step API."""

from collections.abc import Iterable

from smartsom.dispatch import (
    DecisionContext,
    Dispatch,
    SemanticAction,
    Transport,
    WaitNextEvent,
    WaitUntil,
)
from smartsom.domain import (
    ExecutionSchedule,
    FactorySpec,
    OperationStatus,
    ScheduledOperation,
    WorkloadInstance,
    validate_problem,
)
from smartsom.domain.arrivals import ArrivalPlan, DecisionTrigger
from smartsom.domain.machine_events import MachineOutagePlan
from smartsom.domain.processing_times import ProcessingTimePlan
from smartsom.domain.quality import ProbabilityVisibility, QualityPlan
from smartsom.engine.execution_schedule import ExecutionReplay
from smartsom.engine.replay import ReplayError
from smartsom.engine.result import SimulationResult
from smartsom.engine.simulator import Simulator
from smartsom.engine.transport_schedule import validate_transports
from smartsom.modules.arrivals import ArrivalModule
from smartsom.modules.machine_events import MachineEventModule
from smartsom.modules.processing_times import ProcessingTimeModule
from smartsom.modules.quality import QualityModule


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
            raise ReplayError("schedule entries must be ScheduledOperation values")
        if (
            not isinstance(entry.operation_id, str)
            or entry.operation_id not in operations
        ):
            raise ReplayError(f"unknown operation: {entry.operation_id!r}")
        if entry.operation_id in entries:
            raise ReplayError(f"duplicate scheduled operation: {entry.operation_id}")
        mode = operations[entry.operation_id].mode(entry.processing_mode_id)
        if mode is None or entry.machine_id != mode.machine_id:
            raise ReplayError(f"incorrect machine or mode: {entry.operation_id}")
        if any(
            type(tick) is not int or tick < 0
            for tick in (entry.start_time, entry.completion_time)
        ):
            raise ReplayError(
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
            raise ReplayError(
                f"incorrect duration or down-machine start: {entry.operation_id}"
            )
        # A completion cannot be postponed through a downtime interval after all
        # work was already done at its left boundary (completion-first).
        if outages.by_machine[entry.machine_id].is_down(entry.completion_time - 1):
            raise ReplayError(
                f"completion delayed after processing finished: {entry.operation_id}"
            )
        if entry.start_time < timing.release_at(entry.operation_id):
            raise ReplayError(f"start before job release: {entry.operation_id}")
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
        schedule: Iterable[ScheduledOperation] | ExecutionSchedule,
        *,
        arrivals: ArrivalPlan | None = None,
        processing_times: ProcessingTimePlan | None = None,
        machine_events: MachineOutagePlan | None = None,
        transport_enabled: bool = False,
        buffers_enabled: bool = False,
        holding_buffer_enabled: bool = False,
        quality: QualityPlan | None = None,
    ) -> None:
        if quality is not None:
            module = QualityModule(factory, workload, processing_times, quality)
            workload, processing_times = module.workload, module.processing_times
        self._execution = None
        detailed = (
            holding_buffer_enabled
            or buffers_enabled
            and (bool(factory.buffers) or not transport_enabled)
        )
        if (transport_enabled or detailed) and not isinstance(
            schedule, ExecutionSchedule
        ):
            raise ReplayError("transport replay requires an ExecutionSchedule")
        if (
            not transport_enabled
            and isinstance(schedule, ExecutionSchedule)
            and schedule.transports
        ):
            raise ReplayError(
                "transport timetable supplied while transport is disabled"
            )
        operations = (
            schedule.operations if isinstance(schedule, ExecutionSchedule) else schedule
        )
        self._schedule = validate_schedule(
            factory,
            workload,
            operations,
            arrivals=arrivals,
            processing_times=processing_times,
            machine_events=machine_events,
        )

        if detailed:
            self._execution = ExecutionReplay(
                factory,
                workload,
                schedule,
                self._schedule,
                transport_enabled=transport_enabled,
                buffers_enabled=buffers_enabled,
                holding_buffer_enabled=holding_buffer_enabled,
            )
            return
        if isinstance(schedule, ExecutionSchedule) and schedule.version != 1:
            raise ReplayError("v2 schedule requires finite buffers or direct logistics")
        self._transports = (
            validate_transports(factory, workload, schedule, self._schedule, arrivals)
            if transport_enabled
            else ()
        )
        self._next_transport = 0
        # A zero-time visit may be followed by another recorded reroute at this
        # same tick. Do not process on an intermediate visit to the same machine.
        operation_jobs = {
            op.operation_id: job.job_id
            for order in workload.orders
            for job in order.jobs
            for op in job.operations
        }
        self._before_processing = {
            entry.operation_id: max(
                (
                    trip.transport_sequence
                    for trip in self._transports
                    if trip.job_id == operation_jobs[entry.operation_id]
                    and trip.start_time <= entry.start_time
                ),
                default=0,
            )
            for entry in self._schedule
        }

    def select_action(self, context: DecisionContext) -> SemanticAction:
        if self._execution is not None:
            return self._execution.select_action(context)
        if not context.feasible_actions:
            return WaitNextEvent()
        started = {
            op.operation_id
            for op in context.operations
            if op.status != OperationStatus.PENDING
        }
        pending = [
            entry for entry in self._schedule if entry.operation_id not in started
        ]
        tick = context.simulation_time
        for entry in pending:
            if entry.start_time < tick:
                raise ReplayError(
                    f"missed start for {entry.operation_id} at {entry.start_time}"
                )
            action = Dispatch(entry.operation_id, entry.processing_mode_id)
            if (
                entry.start_time == tick
                and action in context.feasible_actions
                and self._next_transport >= self._before_processing[entry.operation_id]
            ):
                return action
        trip = (
            self._transports[self._next_transport]
            if self._next_transport < len(self._transports)
            else None
        )
        if trip is not None and trip.start_time == tick:
            action = Transport(trip.agv_id, trip.job_id, trip.destination)
            if action in context.feasible_actions:
                self._next_transport += 1
                return action
        targets = [entry.start_time for entry in pending]
        if trip is not None:
            targets.append(trip.start_time)
        if not targets or min(targets) <= tick:
            raise ReplayError(f"cannot execute timetable exactly at tick {tick}")
        return WaitUntil(min(targets))

    def verify_result(self, result: SimulationResult) -> None:
        if self._execution is not None:
            return self._execution.verify_result(result)
        actual = tuple(
            sorted(
                result.schedule,
                key=lambda entry: (entry.start_time, entry.operation_id),
            )
        )
        if actual != self._schedule:
            raise ReplayError("actual schedule differs from the requested schedule")
        if result.transport_schedule != self._transports:
            raise ReplayError(
                "actual transport schedule differs from requested schedule"
            )
        expected_end = (
            max(
                t.delivery_time
                for t in self._transports
                if t.destination.kind == "output"
            )
            if self._transports
            else max(entry.completion_time for entry in self._schedule)
        )
        if result.makespan != expected_end:
            raise ReplayError("actual makespan differs from the requested schedule")


def replay_schedule(
    factory: FactorySpec,
    workload: WorkloadInstance,
    schedule: Iterable[ScheduledOperation] | ExecutionSchedule,
    *,
    arrivals: ArrivalPlan | None = None,
    decision_trigger: DecisionTrigger = "dispatch_available",
    processing_times: ProcessingTimePlan | None = None,
    machine_events: MachineOutagePlan | None = None,
    transport_enabled: bool = False,
    buffers_enabled: bool = False,
    holding_buffer_enabled: bool = False,
    quality: QualityPlan | None = None,
    quality_probability_visibility: ProbabilityVisibility = "public",
) -> SimulationResult:
    policy = ScheduleReplayPolicy(
        factory,
        workload,
        schedule,
        arrivals=arrivals,
        processing_times=processing_times,
        machine_events=machine_events,
        transport_enabled=transport_enabled,
        buffers_enabled=buffers_enabled,
        holding_buffer_enabled=holding_buffer_enabled,
        quality=quality,
    )
    result = Simulator(
        factory,
        workload,
        arrivals=arrivals,
        decision_trigger=decision_trigger,
        processing_times=processing_times,
        machine_events=machine_events,
        transport_enabled=transport_enabled,
        buffers_enabled=buffers_enabled,
        holding_buffer_enabled=holding_buffer_enabled,
        quality=quality,
        quality_probability_visibility=quality_probability_visibility,
    ).run(policy)
    policy.verify_result(result)
    return result
