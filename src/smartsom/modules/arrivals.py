"""Pure timing and visibility projection for the online arrivals module."""

from dataclasses import dataclass, replace
from typing import Literal

from smartsom.domain import WorkloadInstance
from smartsom.domain.arrivals import ArrivalPlan, JobArrival, VisibleJob


@dataclass(frozen=True, slots=True, order=True)
class ArrivalEvent:
    simulation_time: int
    job_id: str
    kind: Literal["reveal", "release"]


@dataclass(frozen=True, slots=True, init=False)
class ArrivalModule:
    jobs: tuple[VisibleJob, ...]
    events: tuple[ArrivalEvent, ...]

    def __init__(self, workload: WorkloadInstance, plan: ArrivalPlan | None):
        if plan is not None:
            if not isinstance(plan, ArrivalPlan):
                raise ValueError("expected ArrivalPlan")
            plan.validate(workload)
        timings = {entry.job_id: entry for entry in plan.jobs} if plan else {}
        jobs = []
        events = []
        for order in workload.orders:
            for job in order.jobs:
                timing = timings.get(job.job_id, JobArrival(job.job_id, 0, 0))
                operations = tuple(
                    replace(
                        op,
                        modes=tuple(
                            sorted(op.modes, key=lambda mode: mode.processing_mode_id)
                        ),
                    )
                    for op in sorted(job.operations, key=lambda op: op.operation_id)
                )
                jobs.append(
                    VisibleJob(
                        job.job_id,
                        order.order_id,
                        timing.release_at,
                        timing.reveal_at,
                        operations,
                    )
                )
                for kind, tick in (
                    ("reveal", timing.reveal_at),
                    ("release", timing.release_at),
                ):
                    if (
                        tick > 0
                    ):  # Zero-time facts are initialized without extra trace events.
                        events.append(ArrivalEvent(tick, job.job_id, kind))
        object.__setattr__(
            self, "jobs", tuple(sorted(jobs, key=lambda job: job.job_id))
        )
        object.__setattr__(self, "events", tuple(sorted(events)))

    def visible_jobs(self, tick: int) -> tuple[VisibleJob, ...]:
        return tuple(job for job in self.jobs if job.reveal_at <= tick)

    def released_operations(self, tick: int) -> frozenset[str]:
        return frozenset(
            op.operation_id
            for job in self.jobs
            if job.release_at <= tick
            for op in job.operations
        )

    def release_at(self, operation_id: str) -> int:
        return next(
            job.release_at
            for job in self.jobs
            if any(op.operation_id == operation_id for op in job.operations)
        )
