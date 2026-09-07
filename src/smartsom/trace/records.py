"""Canonical semantic records, independent of files and provenance."""

from dataclasses import dataclass, field
from typing import Literal

from smartsom.dispatch import Dispatch, Transfer, Transport, WaitNextEvent, WaitUntil
from smartsom.domain.transport import (
    ActiveTransport,
    ScheduledTransfer,
    ScheduledTransport,
)


@dataclass(frozen=True, slots=True)
class DecisionRecord:
    sequence: int
    simulation_time: int
    feasible_actions: tuple[Dispatch | Transport | Transfer, ...]
    kind: Literal["decision"] = field(default="decision", init=False)


@dataclass(frozen=True, slots=True)
class DispatchRecord:
    sequence: int
    simulation_time: int
    action: Dispatch
    machine_id: str
    nominal_ticks: int
    kind: Literal["dispatch"] = field(default="dispatch", init=False)


@dataclass(frozen=True, slots=True)
class WaitRecord:
    sequence: int
    simulation_time: int
    action: WaitUntil | WaitNextEvent
    kind: Literal["wait"] = field(default="wait", init=False)


@dataclass(frozen=True, slots=True)
class CompletionRecord:
    sequence: int
    simulation_time: int
    action: Dispatch
    machine_id: str
    kind: Literal["complete"] = field(default="complete", init=False)


@dataclass(frozen=True, slots=True)
class TerminationRecord:
    sequence: int
    simulation_time: int
    end_reason: Literal["completed"] = field(default="completed", init=False)
    kind: Literal["terminate"] = field(default="terminate", init=False)


@dataclass(frozen=True, slots=True)
class ArrivalRecord:
    sequence: int
    simulation_time: int
    job_id: str
    kind: Literal["reveal", "release"]


@dataclass(frozen=True, slots=True)
class MachineRecord:
    sequence: int
    simulation_time: int
    machine_id: str
    kind: Literal["breakdown", "repair"]


@dataclass(frozen=True, slots=True)
class ProcessingRecord:
    sequence: int
    simulation_time: int
    action: Dispatch
    machine_id: str
    kind: Literal["pause", "resume"]


@dataclass(frozen=True, slots=True)
class TransportRecord:
    sequence: int
    simulation_time: int
    trip: ScheduledTransport
    kind: Literal["empty_start", "pickup", "loaded_start", "delivery"]


@dataclass(frozen=True, slots=True)
class VehicleRecord:
    sequence: int
    simulation_time: int
    trip: ActiveTransport
    kind: Literal[
        "empty_start",
        "pickup",
        "loaded_start",
        "arrival",
        "wait_for_unload",
        "delivery",
    ]


@dataclass(frozen=True, slots=True)
class TransferRecord:
    sequence: int
    simulation_time: int
    transfer: ScheduledTransfer
    kind: Literal["transfer"] = field(default="transfer", init=False)


@dataclass(frozen=True, slots=True)
class BufferRecord:
    sequence: int
    simulation_time: int
    machine_id: str
    job_id: str
    kind: Literal["reserve", "consume_reservation", "block", "unblock"]
    transport_sequence: int | None = None
    reason: Literal["postbuffer", "pickup", "transfer"] | None = None


type TraceRecord = (
    DecisionRecord
    | VehicleRecord
    | TransferRecord
    | BufferRecord
    | TransportRecord
    | DispatchRecord
    | WaitRecord
    | CompletionRecord
    | TerminationRecord
    | ArrivalRecord
    | MachineRecord
    | ProcessingRecord
)
