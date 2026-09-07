"""Independent waiting capacities and immutable public occupancy views."""

from dataclasses import dataclass
from typing import Literal

from smartsom.domain.validation import DomainValidationError, _identifier


@dataclass(frozen=True, slots=True)
class MachineBuffers:
    machine_id: str
    pre_capacity: int | None = None
    post_capacity: int | None = None

    def __post_init__(self):
        _identifier(self.machine_id, "buffer machine_id")
        for capacity in (self.pre_capacity, self.post_capacity):
            if capacity is not None and (type(capacity) is not int or capacity < 0):
                raise DomainValidationError(
                    "buffer capacity must be a nonnegative integer or null"
                )


@dataclass(frozen=True, slots=True)
class BufferReservation:
    machine_id: str
    agv_id: str
    job_id: str
    transport_sequence: int


@dataclass(frozen=True, slots=True)
class BufferState:
    machine_id: str
    pre_capacity: int | None
    post_capacity: int | None
    pre_jobs: tuple[str, ...]
    post_jobs: tuple[str, ...]
    reservations: tuple[BufferReservation, ...]


@dataclass(frozen=True, slots=True)
class MachineHolding:
    machine_id: str
    job_id: str
    operation_id: str
    phase: Literal["awaiting_dispatch", "processing", "paused", "blocked"]
