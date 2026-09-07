"""Immutable offline solver contracts, independent of external solver packages."""

from dataclasses import dataclass
from enum import StrEnum
from math import isfinite
from typing import Literal, Protocol

from smartsom.domain import (
    FactorySpec,
    ScheduledOperation,
    WorkloadInstance,
    validate_problem,
)


@dataclass(frozen=True, slots=True)
class SolveRequest:
    factory: FactorySpec
    workload: WorkloadInstance
    objective: Literal["makespan"]
    solver_time_limit_seconds: float
    solver_seed: int

    def __post_init__(self) -> None:
        validate_problem(self.factory, self.workload)
        if self.objective != "makespan":
            raise ValueError("only makespan is supported")
        budget = self.solver_time_limit_seconds
        if type(budget) not in (int, float) or not isfinite(budget) or budget <= 0:
            raise ValueError("solver time limit must be finite and positive")
        if type(self.solver_seed) is not int or not 0 <= self.solver_seed < 2**64:
            raise ValueError("solver seed must be an unsigned 64-bit integer")

    @property
    def backend_seed(self) -> int:
        return self.solver_seed % 2**31


class SolverStatus(StrEnum):
    OPTIMAL = "OPTIMAL"
    FEASIBLE = "FEASIBLE"
    INFEASIBLE = "INFEASIBLE"
    TIME_LIMIT = "TIME_LIMIT"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True, slots=True)
class ScheduleSolution:
    schedule: tuple[ScheduledOperation, ...]
    status: SolverStatus
    objective: float | None
    bound: float | None
    runtime_seconds: float | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "schedule", tuple(self.schedule))
        if not all(isinstance(entry, ScheduledOperation) for entry in self.schedule):
            raise TypeError("solver schedule must contain ScheduledOperation values")
        if not isinstance(self.status, SolverStatus):
            raise TypeError("solver status must be SolverStatus")
        for field in ("objective", "bound", "runtime_seconds"):
            value = getattr(self, field)
            if value is not None:
                if type(value) not in (int, float):
                    raise TypeError(f"{field} must be numeric or None")
                if not isfinite(value):
                    object.__setattr__(self, field, None)

    @property
    def gap(self) -> float | None:
        if self.objective is None or self.bound is None or self.objective <= 0:
            return None
        gap = (self.objective - self.bound) / self.objective
        return gap if isfinite(gap) else None

    def require_incumbent(self) -> None:
        """Check reporting consistency; schedule feasibility is checked by replay."""
        if (
            self.status not in (SolverStatus.OPTIMAL, SolverStatus.FEASIBLE)
            or not self.schedule
        ):
            raise ValueError(f"solver returned no feasible incumbent ({self.status})")
        makespan = max(entry.completion_time for entry in self.schedule)
        if self.objective != makespan:
            raise ValueError("solver objective differs from schedule makespan")
        if self.bound is not None and not 0 <= self.bound <= makespan:
            raise ValueError("solver bound is outside [0, objective]")
        if self.status == SolverStatus.OPTIMAL and self.bound != self.objective:
            raise ValueError(
                "OPTIMAL solver result requires matching objective and bound"
            )


class SolverAdapter(Protocol):
    def solve(self, request: SolveRequest) -> ScheduleSolution:
        """Solve a fully visible static instance and return semantic intervals."""
