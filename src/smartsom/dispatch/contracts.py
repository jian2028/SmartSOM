"""Entity-based decisions shared by direct callers and online policies."""

from dataclasses import dataclass
from typing import Protocol

from smartsom.domain import MachineState, OperationState


@dataclass(frozen=True, slots=True)
class Dispatch:
    operation_id: str
    processing_mode_id: str


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

    @property
    def feasible_actions(self) -> tuple[Dispatch, ...]:
        return tuple(candidate.action for candidate in self.candidates)


class OnlinePolicy(Protocol):
    def select_action(self, context: DecisionContext) -> Dispatch:
        """Select one semantic action without changing simulator state."""
