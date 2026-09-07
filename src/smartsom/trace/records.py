"""Canonical semantic records, independent of files and provenance."""

from dataclasses import dataclass, field
from typing import Literal

from smartsom.dispatch import Dispatch


@dataclass(frozen=True, slots=True)
class DecisionRecord:
    sequence: int
    simulation_time: int
    feasible_actions: tuple[Dispatch, ...]
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


type TraceRecord = (
    DecisionRecord | DispatchRecord | CompletionRecord | TerminationRecord
)
