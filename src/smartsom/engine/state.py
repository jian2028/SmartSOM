"""Mutable runtime storage, owned exclusively by one simulator."""

from dataclasses import dataclass

from smartsom.domain import OperationState


@dataclass(frozen=True, slots=True)
class ProcessingProgress:
    processed_ticks: int = 0
    segment_start: int | None = None


@dataclass(slots=True)
class RuntimeState:
    simulation_time: int
    operations: dict[str, OperationState]
    machine_occupants: dict[str, str | None]
    down_machines: set[str]
    progress: dict[str, ProcessingProgress]
