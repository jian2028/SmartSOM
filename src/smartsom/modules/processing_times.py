"""Private execution-time lookup; never included in policy observations."""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from smartsom.domain import WorkloadInstance
from smartsom.domain.processing_times import ProcessingTimePlan


@dataclass(frozen=True, slots=True, init=False)
class ProcessingTimeModule:
    durations: Mapping[tuple[str, str], int]

    def __init__(self, workload: WorkloadInstance, plan: ProcessingTimePlan | None):
        if plan is not None:
            if not isinstance(plan, ProcessingTimePlan):
                raise ValueError("expected ProcessingTimePlan")
            plan.validate(workload)
            durations = {
                (row.operation_id, row.processing_mode_id): row.actual_ticks
                for row in plan.modes
            }
        else:
            durations = {
                (op.operation_id, mode.processing_mode_id): mode.nominal_ticks
                for op in workload.operations
                for mode in op.modes
            }
        object.__setattr__(self, "durations", MappingProxyType(durations))
