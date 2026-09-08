"""Entity-based decisions shared by direct callers and online policies."""

from dataclasses import dataclass
from typing import Protocol

from smartsom.domain import MachineState, OperationState, VisibleJob
from smartsom.domain.actions import (
    Dispatch,
    SemanticAction,
    Transfer,
    Transport,
)
from smartsom.domain.actions import WaitNextEvent as WaitNextEvent
from smartsom.domain.actions import WaitUntil as WaitUntil
from smartsom.domain.buffers import BufferState, MachineHolding
from smartsom.domain.quality import JobQualityView, QualityModeView
from smartsom.domain.transport import AGVState, JobPosition


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
    source_machine_id: str | None = None  # Set for pending-job reroutes.


@dataclass(frozen=True, slots=True)
class TransferCandidate:
    action: Transfer
    source_machine_id: str | None = None


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

    transfer_candidates: tuple[TransferCandidate, ...] = ()
    buffers: tuple[BufferState, ...] = ()
    machine_holdings: tuple[MachineHolding, ...] = ()

    quality_modes: tuple[QualityModeView, ...] = ()
    job_quality: tuple[JobQualityView, ...] = ()

    @property
    def feasible_actions(self) -> tuple[Dispatch | Transport | Transfer, ...]:
        return tuple(
            candidate.action
            for candidate in (
                *self.candidates,
                *self.transport_candidates,
                *self.transfer_candidates,
            )
        )


class OnlinePolicy(Protocol):
    def select_action(self, context: DecisionContext) -> SemanticAction:
        """Select one semantic action without changing simulator state."""
