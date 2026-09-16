"""Arrival semantics checked independently of configuration and solver code."""

from dataclasses import FrozenInstanceError

import pytest
from reference_cases import (
    op,
    problem,
)

from smartsom.domain import (
    ArrivalPlan,
    Job,
    JobArrival,
    ScheduledOperation,
)
from smartsom.modules.arrivals import ArrivalModule

TRIGGERS = ("dispatch_available", "arrival_event")


def arrival_case(reveal=1):
    factory, workload = problem(
        Job("A", (op("A1", "M1", 3), op("A2", "M2", 2, "A1"))),
        Job("B", (op("B1", "M2", 2), op("B2", "M1", 1, "B1"))),
    )
    return (
        factory,
        workload,
        ArrivalPlan((JobArrival("A", 0, 0), JobArrival("B", 2, reveal))),
    )


def test_release_index_covers_hidden_jobs_and_cannot_be_mutated():
    _, workload, plan = arrival_case()
    module = ArrivalModule(workload, plan)
    assert [job.job_id for job in module.visible_jobs(0)] == ["A"]
    assert [module.release_at(key) for key in ("A1", "A2", "B1", "B2")] == [0, 0, 2, 2]
    with pytest.raises(TypeError):
        module._release_by_operation["B1"] = 0
    with pytest.raises(FrozenInstanceError):
        module._release_by_operation = {}
    assert ArrivalModule(workload, None).release_at("B1") == 0


REFERENCE = (
    ScheduledOperation("A1", "standard", "M1", 0, 3),
    ScheduledOperation("B1", "standard", "M2", 2, 4),
    ScheduledOperation("A2", "standard", "M2", 4, 6),
    ScheduledOperation("B2", "standard", "M1", 4, 5),
)


@pytest.mark.parametrize(
    "release,reveal",
    [(True, 0), (1.0, 0), (1, False), (1, 0.0), (-1, 0), (1, -1), (1, 2)],
)
def test_invalid_timing(release, reveal):
    with pytest.raises(ValueError):
        JobArrival("A", release, reveal)
