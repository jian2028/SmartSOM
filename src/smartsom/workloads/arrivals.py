"""Materialize independent release/reveal timing before simulation."""

import random
from dataclasses import dataclass

from smartsom.domain import ArrivalPlan, JobArrival, WorkloadInstance
from smartsom.workloads.static_jsp import IntegerRange


@dataclass(frozen=True, slots=True)
class UniformReleaseProfile:
    initial_job_count: int
    release_window: IntegerRange
    notice_ticks: int = 0

    def __post_init__(self) -> None:
        for value in (self.initial_job_count, self.notice_ticks):
            if type(value) is not int or value < 0:
                raise ValueError(
                    "initial_job_count and notice_ticks must be nonnegative integers"
                )
        if not isinstance(self.release_window, IntegerRange):
            raise ValueError("release_window must be IntegerRange")


def generate_arrivals(
    workload: WorkloadInstance, profile: UniformReleaseProfile, seed: int
) -> ArrivalPlan:
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise ValueError("seed must be an unsigned 64-bit integer")
    jobs = sorted(job.job_id for order in workload.orders for job in order.jobs)
    if profile.initial_job_count > len(jobs):
        raise ValueError("initial_job_count exceeds the number of jobs")
    rng = random.Random(seed)
    timings = []
    for index, job_id in enumerate(jobs):
        release = (
            0
            if index < profile.initial_job_count
            else rng.randint(profile.release_window.min, profile.release_window.max)
        )
        timings.append(
            JobArrival(job_id, release, max(0, release - profile.notice_ticks))
        )
    return ArrivalPlan(tuple(timings))
