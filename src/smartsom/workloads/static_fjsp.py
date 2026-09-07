"""Version 1 FJSP sampling: independent candidate sets and fixed nominal ticks."""

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
from smartsom.workloads.static_jsp import IntegerRange


@dataclass(frozen=True, slots=True)
class StaticFJSPProfile:
    order_count: int
    jobs_per_order: int
    operations_per_job: IntegerRange
    eligible_machines_per_operation: IntegerRange
    nominal_ticks: IntegerRange

    def __post_init__(self) -> None:
        if any(
            type(value) is not int or value < 1
            for value in (self.order_count, self.jobs_per_order)
        ):
            raise ValueError("order_count and jobs_per_order must be positive integers")
        if any(
            not isinstance(value, IntegerRange)
            for value in (
                self.operations_per_job,
                self.eligible_machines_per_operation,
                self.nominal_ticks,
            )
        ):
            raise ValueError(
                "operation, candidate, and duration ranges must be IntegerRange"
            )


def generate_fjsp(
    factory: FactorySpec, profile: StaticFJSPProfile, seed: int
) -> WorkloadInstance:
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise ValueError("seed must be an unsigned 64-bit integer")
    if profile.eligible_machines_per_operation.max > len(factory.machines):
        raise ValueError(
            "eligible_machines_per_operation.max exceeds the number of machines"
        )
    rng = random.Random(seed)
    machines = sorted(machine.machine_id for machine in factory.machines)
    orders = []
    for order_index in range(1, profile.order_count + 1):
        order_id = f"order_{order_index}"
        jobs = []
        for job_index in range(1, profile.jobs_per_order + 1):
            job_id = f"{order_id}/job_{job_index}"
            operations = []
            count = rng.randint(
                profile.operations_per_job.min, profile.operations_per_job.max
            )
            for op_index in range(1, count + 1):
                eligible = rng.randint(
                    profile.eligible_machines_per_operation.min,
                    profile.eligible_machines_per_operation.max,
                )
                selected = sorted(rng.sample(machines, eligible))
                modes = tuple(
                    ProcessingMode(
                        f"machine/{machine}",
                        machine,
                        rng.randint(
                            profile.nominal_ticks.min, profile.nominal_ticks.max
                        ),
                    )
                    for machine in selected
                )
                operations.append(
                    Operation(
                        f"{job_id}/op_{op_index}",
                        modes,
                        (operations[-1].operation_id,) if operations else (),
                    )
                )
            jobs.append(Job(job_id, tuple(operations)))
        orders.append(Order(order_id, tuple(jobs)))
    workload = WorkloadInstance(tuple(orders))
    validate_problem(factory, workload)
    return workload
