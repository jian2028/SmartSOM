"""Mutable runtime storage, owned exclusively by one simulator."""

from dataclasses import dataclass

from smartsom.domain import OperationState


@dataclass(slots=True)
class RuntimeState:
    simulation_time: int
    operations: dict[str, OperationState]
    machine_occupants: dict[str, str | None]
