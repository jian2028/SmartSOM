"""Entity-based decisions shared by direct callers and online policies."""

from dataclasses import dataclass
from typing import Literal, Protocol

from smartsom.domain import MachineState, OperationState, VisibleJob


@dataclass(frozen=True, slots=True)
class Dispatch:
    operation_id: str
    processing_mode_id: str


@dataclass(frozen=True, slots=True)
class WaitUntil:
    until: int


@dataclass(frozen=True, slots=True, init=False)
class WaitNextEvent:
    kind: Literal["wait_next_event"]

    def __init__(self) -> None:
        object.__setattr__(self, "kind", "wait_next_event")


type SemanticAction = Dispatch | WaitUntil | WaitNextEvent


@dataclass(frozen=True, slots=True)
class DispatchCandidate:
    action: Dispatch
    machine_id: str
    nominal_ticks: int


@dataclass(frozen=True, slots=True)
class DecisionContext:
    simulation_time: int
    operations: tuple[OperationState, ...]
    machines: tuple[MachineState, ...]
    candidates: tuple[DispatchCandidate, ...]
    jobs: tuple[VisibleJob, ...] = ()

    @property
    def feasible_actions(self) -> tuple[Dispatch, ...]:
        return tuple(candidate.action for candidate in self.candidates)


class OnlinePolicy(Protocol):
    def select_action(self, context: DecisionContext) -> SemanticAction:
        """Select one semantic action without changing simulator state."""
