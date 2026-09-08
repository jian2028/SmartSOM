"""Pure quality expansion and immutable runtime lookups; no RNG or state writes."""

from dataclasses import replace
from fractions import Fraction
from types import MappingProxyType

from smartsom.domain import (
    FactorySpec,
    ProcessingMode,
    WorkloadInstance,
    validate_problem,
)
from smartsom.domain.processing_times import ProcessingTime, ProcessingTimePlan
from smartsom.domain.quality import (
    QualityDrawPlan,
    QualityExecutionMode,
    QualityPlan,
    quality_mode_id,
)


def scaled_ticks(ticks: int, scale) -> int:
    value = ticks * Fraction(scale) + Fraction(1, 2)
    return max(1, value.numerator // value.denominator)


def prepare_quality(
    factory: FactorySpec,
    workload: WorkloadInstance,
    draws: QualityDrawPlan,
    *,
    processing_times: ProcessingTimePlan | None = None,
) -> QualityPlan:
    """Combine original inputs once; never accept already scaled execution inputs."""
    validate_problem(factory, workload)
    if factory.quality_speed is None:
        raise ValueError("enabled quality requires factory quality_speed")
    if not isinstance(draws, QualityDrawPlan) or {
        x.operation_id for x in draws.operations
    } != {x.operation_id for x in workload.operations}:
        raise ValueError("quality draws must cover every operation exactly once")
    actual = {}
    if processing_times is not None:
        processing_times.validate(workload)
        actual = {
            (x.operation_id, x.processing_mode_id): x.actual_ticks
            for x in processing_times.modes
        }
    modes = []
    for op in workload.operations:
        for base in op.modes:
            table = factory.quality_speed.for_machine(base.machine_id)
            if not table:
                raise ValueError(
                    f"missing quality mode table for machine {base.machine_id!r}"
                )
            for mode in table:
                modes.append(
                    QualityExecutionMode(
                        op.operation_id,
                        base.processing_mode_id,
                        quality_mode_id(base.processing_mode_id, mode.quality_mode_id),
                        base.machine_id,
                        mode,
                        scaled_ticks(base.nominal_ticks, mode.time_scale),
                        scaled_ticks(
                            actual.get(
                                (op.operation_id, base.processing_mode_id),
                                base.nominal_ticks,
                            ),
                            mode.time_scale,
                        ),
                    )
                )
    return QualityPlan(draws, tuple(modes))


class QualityModule:
    """Validated execution projection and readonly lookup from base inputs."""

    def __init__(
        self,
        factory: FactorySpec,
        workload: WorkloadInstance,
        processing_times: ProcessingTimePlan | None,
        plan: QualityPlan,
    ) -> None:
        if not isinstance(plan, QualityPlan):
            raise ValueError("quality must be a QualityPlan")
        if plan != prepare_quality(
            factory, workload, plan.draws, processing_times=processing_times
        ):
            raise ValueError(
                "quality plan disagrees with base inputs or factory capabilities"
            )
        self.plan = plan
        self.modes = MappingProxyType(
            {(x.operation_id, x.processing_mode_id): x for x in plan.modes}
        )
        self.draws = MappingProxyType(
            {x.operation_id: x.draw for x in plan.draws.operations}
        )
        by_operation = {}
        for row in plan.modes:
            by_operation.setdefault(row.operation_id, []).append(
                ProcessingMode(
                    row.processing_mode_id, row.machine_id, row.nominal_ticks
                )
            )
        self.workload = replace(
            workload,
            orders=tuple(
                replace(
                    order,
                    jobs=tuple(
                        replace(
                            job,
                            operations=tuple(
                                replace(op, modes=tuple(by_operation[op.operation_id]))
                                for op in job.operations
                            ),
                        )
                        for job in order.jobs
                    ),
                )
                for order in workload.orders
            ),
        )
        self.processing_times = ProcessingTimePlan(
            tuple(
                ProcessingTime(
                    x.operation_id,
                    x.processing_mode_id,
                    x.nominal_ticks,
                    x.actual_ticks,
                )
                for x in plan.modes
            )
        )
