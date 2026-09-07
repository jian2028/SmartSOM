"""Entity-based decisions shared by direct callers and online policies."""

from dataclasses import dataclass
from typing import Literal, Protocol

from smartsom.domain import MachineState, OperationState, VisibleJob
from smartsom.domain.transport import AGVState, JobPosition, TransportDestination


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


type SemanticAction = Dispatch | Transport | WaitUntil | WaitNextEvent


@dataclass(frozen=True, slots=True)
class DispatchCandidate:
    action: Dispatch
    machine_id: str
    nominal_ticks: int


@dataclass(frozen=True, slots=True)
class TransportCandidate:
    action: Transport
    empty_ticks: int
    loaded_ticks: int
    source_machine_id: str | None = None  # Set only for prebuffer reroutes.


@dataclass(frozen=True, slots=True)
class DecisionContext:
    simulation_time: int
    operations: tuple[OperationState, ...]
    machines: tuple[MachineState, ...]
    candidates: tuple[DispatchCandidate, ...]
    jobs: tuple[VisibleJob, ...] = ()

    agvs: tuple[AGVState, ...] = ()
    job_positions: tuple[JobPosition, ...] = ()
    transport_candidates: tuple[TransportCandidate, ...] = ()

    @property
    def feasible_actions(self) -> tuple[Dispatch | Transport, ...]:
        return tuple(
            candidate.action
            for candidate in (*self.candidates, *self.transport_candidates)
        )


class OnlinePolicy(Protocol):
    def select_action(self, context: DecisionContext) -> SemanticAction:
        """Select one semantic action without changing simulator state."""
