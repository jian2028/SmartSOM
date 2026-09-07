"""Immutable snapshots and actual execution intervals."""

from dataclasses import dataclass
from enum import StrEnum


class OperationStatus(StrEnum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"


@dataclass(frozen=True, slots=True)
class OperationState:
    operation_id: str
    status: OperationStatus = OperationStatus.PENDING
    start_time: int | None = None
    completion_time: int | None = None
    processing_mode_id: str | None = None


@dataclass(frozen=True, slots=True)
class MachineState:
    machine_id: str
    operation_id: str | None


@dataclass(frozen=True, slots=True)
class ScheduledOperation:
    operation_id: str
    processing_mode_id: str
    machine_id: str
    start_time: int
    completion_time: int
