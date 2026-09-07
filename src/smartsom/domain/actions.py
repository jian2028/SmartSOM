"""Immutable semantic actions, independent of policies and execution."""

from dataclasses import dataclass
from typing import Literal

from smartsom.domain.destinations import TransportDestination


@dataclass(frozen=True, slots=True)
class Dispatch:
    operation_id: str
    processing_mode_id: str


@dataclass(frozen=True, slots=True)
class Transport:
    agv_id: str
    job_id: str
    destination: TransportDestination


@dataclass(frozen=True, slots=True)
class WaitUntil:
    until: int


@dataclass(frozen=True, slots=True, init=False)
class WaitNextEvent:
    kind: Literal["wait_next_event"]

    def __init__(self) -> None:
        object.__setattr__(self, "kind", "wait_next_event")


@dataclass(frozen=True, slots=True)
class Transfer:
    job_id: str
    destination: TransportDestination


@dataclass(frozen=True, slots=True)
class TimedAction:
    sequence: int
    simulation_time: int
    action: Dispatch | Transport | Transfer


type SemanticAction = Dispatch | Transport | Transfer | WaitUntil | WaitNextEvent
