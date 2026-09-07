"""Immutable completed-episode output."""

from dataclasses import dataclass, field
from typing import Literal

from smartsom.dispatch import SemanticAction
from smartsom.domain import ScheduledOperation
from smartsom.trace import TraceRecord


@dataclass(frozen=True, slots=True)
class SimulationResult:
    makespan: int
    schedule: tuple[ScheduledOperation, ...]
    actions: tuple[SemanticAction, ...]
    trace: tuple[TraceRecord, ...]
    end_reason: Literal["completed"] = field(default="completed", init=False)
