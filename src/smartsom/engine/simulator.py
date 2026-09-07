"""One canonical step transition shared by policies and semantic replay."""

from dataclasses import replace

from smartsom.dispatch import (
    DecisionContext,
    Dispatch,
    OnlinePolicy,
    SemanticAction,
    WaitNextEvent,
    WaitUntil,
)
from smartsom.dispatch.feasibility import build_decision
from smartsom.domain import (
    ArrivalPlan,
    FactorySpec,
    MachineOutagePlan,
    MachineState,
    OperationState,
    OperationStatus,
    ScheduledOperation,
    WorkloadInstance,
    validate_problem,
)
from smartsom.domain.arrivals import DecisionTrigger
from smartsom.domain.processing_times import ProcessingTimePlan
from smartsom.engine.calendar import CompletionEvent, EventCalendar
from smartsom.engine.invariants import InvariantViolation, check_invariants
from smartsom.engine.result import SimulationResult
from smartsom.engine.state import ProcessingProgress, RuntimeState
from smartsom.modules.arrivals import ArrivalEvent, ArrivalModule
from smartsom.modules.machine_events import MachineEvent, MachineEventModule
from smartsom.modules.processing_times import ProcessingTimeModule
from smartsom.trace import (
    ArrivalRecord,
    CompletionRecord,
    DecisionRecord,
    DispatchRecord,
    MachineRecord,
    ProcessingRecord,
    TerminationRecord,
    TraceRecord,
    WaitRecord,
)


class InvalidActionError(ValueError):
    """A rejected action; canonical state remains unchanged."""

    def __init__(self, simulation_time: int, action: object, reason: str) -> None:
        self.simulation_time = simulation_time
        self.action = action
        self.reason = reason
        super().__init__(
            f"invalid action at tick {simulation_time}: {action!r}: {reason}"
        )


class SimulationFinishedError(RuntimeError):
    """A caller tried to advance an already completed simulator."""


class DeadlockError(RuntimeError):
    """Unfinished work has neither feasible actions nor future events."""


class Simulator:
    def __init__(
        self,
        factory: FactorySpec,
        workload: WorkloadInstance,
        *,
        arrivals: ArrivalPlan | None = None,
        decision_trigger: DecisionTrigger = "dispatch_available",
        processing_times: ProcessingTimePlan | None = None,
        machine_events: MachineOutagePlan | None = None,
    ) -> None:
        validate_problem(factory, workload)
        self._processing_times = ProcessingTimeModule(workload, processing_times)
        self._machine_events = MachineEventModule(factory, machine_events)
        self._handled_machine_events: set[MachineEvent] = set()
        if decision_trigger not in ("dispatch_available", "arrival_event"):
            raise ValueError("unknown decision_trigger")
        if decision_trigger == "arrival_event" and arrivals is None:
            raise ValueError("arrival_event decision trigger requires arrivals")
        self._arrivals = ArrivalModule(workload, arrivals)
        self._arrival_events = frozenset(self._arrivals.events)
        self._decision_trigger = decision_trigger
        self._arrival_notice = bool(arrivals and self._arrivals.visible_jobs(0))
        self._handled_arrivals: set[ArrivalEvent] = set()
        self._factory = factory
        self._operations = {op.operation_id: op for op in workload.operations}
        self._state = RuntimeState(
            0,
            {key: OperationState(key) for key in sorted(self._operations)},
            {machine.machine_id: None for machine in factory.machines},
            set(),
            {},
        )
        self._calendar = EventCalendar()
        for event in self._arrivals.events:
            self._calendar.schedule(event)
        for event in self._machine_events.events:
            self._calendar.schedule(event)
        self._schedule: list[ScheduledOperation] = []
        self._actions: list[SemanticAction] = []
        self._trace: list[TraceRecord] = []
        self._current_decision: DecisionContext | None = None
        self._result: SimulationResult | None = None
        if self._calendar.next_time == 0:
            self._advance_to(0)
        self._settle()

    @property
    def current_decision(self) -> DecisionContext | None:
        return self._current_decision

    @property
    def trace(self) -> tuple[TraceRecord, ...]:
        return tuple(self._trace)

    def trace_since(self, cursor: int) -> tuple[TraceRecord, ...]:
        """Read an immutable suffix without copying or consuming earlier records."""
        if type(cursor) is not int or not 0 <= cursor <= len(self._trace):
            raise ValueError("trace cursor must be an integer in [0, trace length]")
        return tuple(self._trace[cursor:])

    def step(self, action: SemanticAction) -> DecisionContext | SimulationResult:
        if self._result is not None:
            raise SimulationFinishedError("simulation already completed")
        self._validate_action(action)
        if isinstance(action, (WaitUntil, WaitNextEvent)):
            self._actions.append(action)
            self._trace.append(
                WaitRecord(len(self._trace), self._state.simulation_time, action)
            )
            self._current_decision = None
            next_time = self._calendar.next_time
            target = next_time if isinstance(action, WaitNextEvent) else action.until
            self._advance_to(
                min(target, next_time) if next_time is not None else target
            )
            return self._settle()
        operation = self._operations[action.operation_id]
        mode = operation.mode(action.processing_mode_id)
        start = self._state.simulation_time
        self._state.operations[action.operation_id] = OperationState(
            action.operation_id,
            OperationStatus.PROCESSING,
            start,
            processing_mode_id=action.processing_mode_id,
        )
        self._state.machine_occupants[mode.machine_id] = action.operation_id
        self._state.progress[action.operation_id] = ProcessingProgress(
            segment_start=start
        )
        self._calendar.schedule(
            CompletionEvent(
                start
                + self._processing_times.durations[
                    (action.operation_id, action.processing_mode_id)
                ],
                action.operation_id,
                action.processing_mode_id,
                mode.machine_id,
            )
        )
        self._actions.append(action)
        self._trace.append(
            DispatchRecord(
                len(self._trace), start, action, mode.machine_id, mode.nominal_ticks
            )
        )
        self._current_decision = None
        return self._settle()

    def run(self, policy: OnlinePolicy) -> SimulationResult:
        """Continue this episode by repeatedly invoking the public step API."""
        context = self._current_decision
        if context is None:
            raise SimulationFinishedError("simulation already completed")
        while True:
            outcome = self.step(policy.select_action(context))
            if isinstance(outcome, SimulationResult):
                return outcome
            context = outcome

    def _validate_action(self, action: SemanticAction) -> None:
        def reject(reason: str) -> None:
            raise InvalidActionError(self._state.simulation_time, action, reason)

        context = self._current_decision
        if isinstance(action, WaitNextEvent):
            if self._calendar.next_time is None:
                reject("no future event to wait for")
            return
        if isinstance(action, WaitUntil):
            if (
                type(action.until) is not int
                or action.until <= self._state.simulation_time
            ):
                reject("wait target must be an integer greater than the current tick")
            return
        if (
            isinstance(action, Dispatch)
            and context is not None
            and action in context.feasible_actions
        ):
            return
        # The published legal view is authoritative; the checks below only
        # explain rejection and never authorize an action excluded by it.
        if not isinstance(action, Dispatch):
            reject("expected Dispatch or WaitUntil or WaitNextEvent")
        if not isinstance(action.operation_id, str) or action.operation_id not in {
            op.operation_id for op in context.operations
        }:
            reject("unknown operation ID")
        operation = self._operations[action.operation_id]
        mode = operation.mode(action.processing_mode_id)
        if mode is None:
            reject("unknown processing mode for this operation")
        if self._arrivals.release_at(action.operation_id) > self._state.simulation_time:
            reject("job has not been released")
        if (
            self._state.operations[action.operation_id].status
            != OperationStatus.PENDING
        ):
            reject("operation has already started")
        if any(
            self._state.operations[key].status != OperationStatus.COMPLETED
            for key in operation.predecessor_ids
        ):
            reject("predecessor has not completed")
        if self._state.machine_occupants[mode.machine_id] is not None:
            reject("machine is busy")
        if mode.machine_id in self._state.down_machines:
            reject("machine is down")
        reject("action is excluded from the current feasible action view")

    def _check_invariants(self) -> None:
        pending_events = self._calendar.pending
        check_invariants(
            self._factory,
            self._operations,
            self._state,
            tuple(
                event for event in pending_events if isinstance(event, CompletionEvent)
            ),
            self._schedule,
            durations=self._processing_times.durations,
            machine_events=self._machine_events,
        )
        expected_machine = self._machine_events.events - self._handled_machine_events
        pending_machine = tuple(
            event for event in pending_events if isinstance(event, MachineEvent)
        )
        if (
            set(pending_machine) != expected_machine
            or len(pending_machine) != len(expected_machine)
            or not self._handled_machine_events <= self._machine_events.events
            or any(
                event.simulation_time < self._state.simulation_time
                for event in pending_machine
            )
        ):
            raise InvariantViolation(
                "machine calendar disagrees with materialized plan"
            )
        # During a same-tick phase, availability follows consumed facts, not
        # unprocessed machine events at that same clock tick.
        balances = {machine.machine_id: 0 for machine in self._factory.machines}
        for event in self._handled_machine_events:
            balances[event.machine_id] += 1 if event.kind == "breakdown" else -1
        if any(
            value not in (0, 1) for value in balances.values()
        ) or self._state.down_machines != {
            key for key, value in balances.items() if value
        }:
            raise InvariantViolation(
                "machine availability disagrees with consumed events"
            )
        expected = self._arrival_events - self._handled_arrivals
        pending = tuple(
            event for event in pending_events if isinstance(event, ArrivalEvent)
        )
        if (
            set(pending) != expected
            or len(pending) != len(expected)
            or not self._handled_arrivals <= self._arrival_events
            or any(
                event.simulation_time < self._state.simulation_time for event in pending
            )
        ):
            raise InvariantViolation(
                "arrival calendar disagrees with materialized plan"
            )
        for state in self._state.operations.values():
            if (
                state.start_time is not None
                and state.start_time < self._arrivals.release_at(state.operation_id)
            ):
                raise InvariantViolation("operation started before job release")

    def _settle(self) -> DecisionContext | SimulationResult:
        self._check_invariants()
        while True:
            if len(self._schedule) == len(self._operations):
                self._trace.append(
                    TerminationRecord(len(self._trace), self._state.simulation_time)
                )
                self._result = SimulationResult(
                    max(entry.completion_time for entry in self._schedule),
                    tuple(
                        sorted(
                            self._schedule,
                            key=lambda entry: (entry.start_time, entry.operation_id),
                        )
                    ),
                    tuple(self._actions),
                    self.trace,
                )
                return self._result
            visible_jobs = self._arrivals.visible_jobs(self._state.simulation_time)
            visible_operations = tuple(
                op for job in visible_jobs for op in job.operations
            )
            context = build_decision(
                self._state.simulation_time,
                visible_operations,
                tuple(
                    self._state.operations[op.operation_id] for op in visible_operations
                ),
                tuple(
                    MachineState(
                        key, value, "down" if key in self._state.down_machines else "up"
                    )
                    for key, value in self._state.machine_occupants.items()
                ),
                released_operations=self._arrivals.released_operations(
                    self._state.simulation_time
                ),
            )
            context = replace(context, jobs=visible_jobs)
            if context.candidates or (
                self._decision_trigger == "arrival_event" and self._arrival_notice
            ):
                self._arrival_notice = False
                self._current_decision = context
                self._trace.append(
                    DecisionRecord(
                        len(self._trace),
                        context.simulation_time,
                        context.feasible_actions,
                    )
                )
                return context
            next_time = self._calendar.next_time
            if next_time is None:
                unfinished = sorted(
                    key
                    for key, value in self._state.operations.items()
                    if value.status != OperationStatus.COMPLETED
                )
                raise DeadlockError(
                    f"deadlock at tick {self._state.simulation_time}; unfinished={unfinished}"
                )
            self._advance_to(next_time)

    def _advance_to(self, tick: int) -> None:
        self._state.simulation_time = tick
        # All completion/machine/arrival phases settle before a policy can run.
        while self._calendar.next_time == tick:
            event = self._calendar.pop()
            if isinstance(event, CompletionEvent):
                self._complete(event)
            elif isinstance(event, MachineEvent):
                self._handle_machine_event(event)
            else:
                self._handled_arrivals.add(event)
                self._arrival_notice = True
                self._trace.append(
                    ArrivalRecord(len(self._trace), tick, event.job_id, event.kind)
                )
            self._check_invariants()

    def _complete(self, event: CompletionEvent) -> None:
        current = self._state.operations[event.operation_id]
        progress = self._state.progress[event.operation_id]
        processed = (
            progress.processed_ticks + event.simulation_time - progress.segment_start
        )
        self._state.progress[event.operation_id] = ProcessingProgress(processed)
        self._state.operations[event.operation_id] = replace(
            current,
            status=OperationStatus.COMPLETED,
            completion_time=event.simulation_time,
            actual_processing_ticks=processed,
        )
        self._state.machine_occupants[event.machine_id] = None
        self._schedule.append(
            ScheduledOperation(
                event.operation_id,
                event.processing_mode_id,
                event.machine_id,
                current.start_time,
                event.simulation_time,
            )
        )
        self._trace.append(
            CompletionRecord(
                len(self._trace),
                event.simulation_time,
                Dispatch(event.operation_id, event.processing_mode_id),
                event.machine_id,
            )
        )

    def _handle_machine_event(self, event: MachineEvent) -> None:
        tick, machine = event.simulation_time, event.machine_id
        self._handled_machine_events.add(event)
        self._trace.append(MachineRecord(len(self._trace), tick, machine, event.kind))
        if event.kind == "breakdown":
            self._state.down_machines.add(machine)
        else:
            self._state.down_machines.remove(machine)
        operation_id = self._state.machine_occupants[machine]
        if operation_id is None:
            return
        current = self._state.operations[operation_id]
        progress = self._state.progress[operation_id]
        action = Dispatch(operation_id, current.processing_mode_id)
        if event.kind == "breakdown":
            self._state.progress[operation_id] = ProcessingProgress(
                progress.processed_ticks + tick - progress.segment_start
            )
            self._state.operations[operation_id] = replace(
                current, status=OperationStatus.PAUSED
            )
            self._calendar.cancel_completion(operation_id)
            kind = "pause"
        else:
            self._state.progress[operation_id] = replace(progress, segment_start=tick)
            self._state.operations[operation_id] = replace(
                current, status=OperationStatus.PROCESSING
            )
            remaining = (
                self._processing_times.durations[
                    (operation_id, current.processing_mode_id)
                ]
                - progress.processed_ticks
            )
            self._calendar.schedule(
                CompletionEvent(
                    tick + remaining, operation_id, current.processing_mode_id, machine
                )
            )
            kind = "resume"
        self._trace.append(
            ProcessingRecord(len(self._trace), tick, action, machine, kind)
        )
