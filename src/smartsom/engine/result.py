"""Immutable completed-episode output."""

from dataclasses import dataclass, field
from typing import Literal

from smartsom.dispatch import SemanticAction
from smartsom.domain import (
    ExecutionSchedule,
    ScheduledOperation,
    ScheduledTransfer,
    ScheduledTransport,
    TimedAction,
    TransportArrival,
)
from smartsom.trace import TraceRecord


@dataclass(frozen=True, slots=True)
class SimulationResult:
    makespan: int
    schedule: tuple[ScheduledOperation, ...]
    actions: tuple[SemanticAction, ...]
    trace: tuple[TraceRecord, ...]
    transport_schedule: tuple[ScheduledTransport, ...] = ()

    transport_arrivals: tuple[TransportArrival, ...] = ()
    transfer_schedule: tuple[ScheduledTransfer, ...] = ()
    action_order: tuple[TimedAction, ...] = ()
    schedule_version: int = 1

    @property
    def execution_schedule(self) -> ExecutionSchedule:
        return ExecutionSchedule(
            self.schedule,
            self.transport_schedule,
            self.transport_arrivals,
            self.transfer_schedule,
            self.action_order,
            self.schedule_version,
        )

    end_reason: Literal["completed"] = field(default="completed", init=False)
