"""Version 1 static JSP generator; no runtime sampling or global RNG."""

import random
from dataclasses import dataclass

from smartsom.domain import (
    FactorySpec,
    Job,
    Operation,
    Order,
    ProcessingMode,
    WorkloadInstance,
    validate_problem,
)

GENERATOR = "static_jsp_v1"
GENERATOR_VERSION = "1"


@dataclass(frozen=True, slots=True)
class IntegerRange:
    min: int
    max: int

    def __post_init__(self) -> None:
        if any(type(value) is not int or value < 1 for value in (self.min, self.max)):
            raise ValueError("range bounds must be positive integers")
        if self.min > self.max:
            raise ValueError("range min must not exceed max")


@dataclass(frozen=True, slots=True)
class StaticJSPProfile:
    order_count: int
    jobs_per_order: int
    operations_per_job: IntegerRange
    nominal_ticks: IntegerRange

    def __post_init__(self) -> None:
        for value in (self.order_count, self.jobs_per_order):
            if type(value) is not int or value < 1:
                raise ValueError(
                    "order_count and jobs_per_order must be positive integers"
                )
        if not isinstance(self.operations_per_job, IntegerRange) or not isinstance(
            self.nominal_ticks, IntegerRange
        ):
            raise ValueError(
                "operations_per_job and nominal_ticks must be IntegerRange"
            )


def generate(
    factory: FactorySpec, profile: StaticJSPProfile, seed: int
) -> WorkloadInstance:
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise ValueError("seed must be an unsigned 64-bit integer")
    if profile.operations_per_job.max > len(factory.machines):
        raise ValueError("operations_per_job.max exceeds the number of machines")
    rng = random.Random(seed)
    machines = sorted(machine.machine_id for machine in factory.machines)
    orders = []
    # Draw order is part of v1: count, route, then durations, in entity order.
    for order_index in range(1, profile.order_count + 1):
        order_id = f"order_{order_index}"
        jobs = []
        for job_index in range(1, profile.jobs_per_order + 1):
            job_id = f"{order_id}/job_{job_index}"
            count = rng.randint(
                profile.operations_per_job.min, profile.operations_per_job.max
            )
            route = rng.sample(machines, count)
            operations = []
            for operation_index, machine_id in enumerate(route, 1):
                operation_id = f"{job_id}/op_{operation_index}"
                operations.append(
                    Operation(
                        operation_id,
                        (
                            ProcessingMode(
                                "standard",
                                machine_id,
                                rng.randint(
                                    profile.nominal_ticks.min, profile.nominal_ticks.max
                                ),
                            ),
                        ),
                        (operations[-1].operation_id,) if operations else (),
                    )
                )
            jobs.append(Job(job_id, tuple(operations)))
        orders.append(Order(order_id, tuple(jobs)))
    workload = WorkloadInstance(tuple(orders))
    validate_problem(factory, workload)
    return workload
