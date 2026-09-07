"""Immutable completed-episode output."""

from dataclasses import dataclass, field
from typing import Literal

from smartsom.dispatch import SemanticAction
from smartsom.domain import ExecutionSchedule, ScheduledOperation, ScheduledTransport
from smartsom.trace import TraceRecord


@dataclass(frozen=True, slots=True)
class SimulationResult:
    makespan: int
    schedule: tuple[ScheduledOperation, ...]
    actions: tuple[SemanticAction, ...]
    trace: tuple[TraceRecord, ...]
    transport_schedule: tuple[ScheduledTransport, ...] = ()

    @property
    def execution_schedule(self) -> ExecutionSchedule:
        return ExecutionSchedule(self.schedule, self.transport_schedule)

    end_reason: Literal["completed"] = field(default="completed", init=False)
