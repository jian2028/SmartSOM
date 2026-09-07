"""Materialized job timing, independent of workload structure and configuration."""

from dataclasses import dataclass
from typing import Literal

from smartsom.domain.models import (
    DomainValidationError,
    Operation,
    WorkloadInstance,
    _identifier,
    _items,
    _unique,
)

type DecisionTrigger = Literal["dispatch_available", "arrival_event"]


@dataclass(frozen=True, slots=True)
class JobArrival:
    job_id: str
    release_at: int
    reveal_at: int

    def __post_init__(self) -> None:
        _identifier(self.job_id, "job_id")
        if any(
            type(tick) is not int or tick < 0
            for tick in (self.release_at, self.reveal_at)
        ):
            raise DomainValidationError("arrival times must be nonnegative integers")
        if self.reveal_at > self.release_at:
            raise DomainValidationError("reveal_at must not exceed release_at")


@dataclass(frozen=True, slots=True)
class ArrivalPlan:
    jobs: tuple[JobArrival, ...]

    def __post_init__(self) -> None:
        jobs = _items(self.jobs, JobArrival, "arrivals")
        _unique(tuple(job.job_id for job in jobs), "arrival job ID")
        object.__setattr__(
            self, "jobs", tuple(sorted(jobs, key=lambda job: job.job_id))
        )

    def validate(self, workload: WorkloadInstance) -> None:
        expected = {job.job_id for order in workload.orders for job in order.jobs}
        actual = {job.job_id for job in self.jobs}
        if expected != actual:
            raise DomainValidationError(
                f"arrival coverage mismatch: missing={sorted(expected - actual)}, unknown={sorted(actual - expected)}"
            )


@dataclass(frozen=True, slots=True)
class VisibleJob:
    job_id: str
    order_id: str
    release_at: int
    reveal_at: int
    operations: tuple[Operation, ...]
