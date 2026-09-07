"""One canonical step transition shared by policies and semantic replay."""

from dataclasses import replace

from smartsom.dispatch import (
    DecisionContext,
    Dispatch,
    OnlinePolicy,
    SemanticAction,
    WaitUntil,
)
from smartsom.dispatch.feasibility import build_decision
from smartsom.domain import (
    FactorySpec,
    MachineState,
    OperationState,
    OperationStatus,
    ScheduledOperation,
    WorkloadInstance,
    validate_problem,
)
from smartsom.engine.calendar import CompletionEvent, EventCalendar
from smartsom.engine.invariants import check_invariants
from smartsom.engine.result import SimulationResult
from smartsom.engine.state import RuntimeState
from smartsom.trace import (
    CompletionRecord,
    DecisionRecord,
    DispatchRecord,
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
    def __init__(self, factory: FactorySpec, workload: WorkloadInstance) -> None:
        validate_problem(factory, workload)
        self._factory = factory
        self._operations = {op.operation_id: op for op in workload.operations}
        self._state = RuntimeState(
            0,
            {key: OperationState(key) for key in sorted(self._operations)},
            {machine.machine_id: None for machine in factory.machines},
        )
        self._calendar = EventCalendar()
        self._schedule: list[ScheduledOperation] = []
        self._actions: list[SemanticAction] = []
        self._trace: list[TraceRecord] = []
        self._current_decision: DecisionContext | None = None
        self._result: SimulationResult | None = None
        self._settle()

    @property
    def current_decision(self) -> DecisionContext | None:
        return self._current_decision

    @property
    def trace(self) -> tuple[TraceRecord, ...]:
        return tuple(self._trace)

    def step(self, action: SemanticAction) -> DecisionContext | SimulationResult:
        if self._result is not None:
            raise SimulationFinishedError("simulation already completed")
        self._validate_action(action)
        if isinstance(action, WaitUntil):
            self._actions.append(action)
            self._trace.append(
                WaitRecord(len(self._trace), self._state.simulation_time, action)
            )
            self._current_decision = None
            next_time = self._calendar.next_time
            self._advance_to(
                min(action.until, next_time) if next_time is not None else action.until
            )
            return self._settle()
        operation = self._operations[action.operation_id]
        mode = operation.modes[0]
        start = self._state.simulation_time
        self._state.operations[action.operation_id] = OperationState(
            action.operation_id, OperationStatus.PROCESSING, start
        )
        self._state.machine_occupants[mode.machine_id] = action.operation_id
        self._calendar.schedule(
            CompletionEvent(
                start + mode.nominal_ticks,
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
            reject("expected Dispatch or WaitUntil")
        if (
            not isinstance(action.operation_id, str)
            or action.operation_id not in self._operations
        ):
            reject("unknown operation ID")
        operation = self._operations[action.operation_id]
        mode = operation.modes[0]
        if action.processing_mode_id != mode.processing_mode_id:
            reject("unknown processing mode for this operation")
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
        reject("action is excluded from the current feasible action view")

    def _check_invariants(self) -> None:
        check_invariants(
            self._factory,
            self._operations,
            self._state,
            self._calendar.pending,
            self._schedule,
        )

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
            context = build_decision(
                self._state.simulation_time,
                tuple(self._operations.values()),
                tuple(self._state.operations.values()),
                tuple(
                    MachineState(key, value)
                    for key, value in self._state.machine_occupants.items()
                ),
            )
            if context.candidates:
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
        # All completions at this tick settle before any policy can run.
        while self._calendar.next_time == tick:
            self._complete(self._calendar.pop())
            self._check_invariants()

    def _complete(self, event: CompletionEvent) -> None:
        current = self._state.operations[event.operation_id]
        self._state.operations[event.operation_id] = replace(
            current,
            status=OperationStatus.COMPLETED,
            completion_time=event.simulation_time,
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
