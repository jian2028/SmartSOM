"""Domain specifications and immutable state views; no framework dependencies."""

from smartsom.domain.arrivals import ArrivalPlan, JobArrival, VisibleJob
from smartsom.domain.machine_events import MachineOutage, MachineOutagePlan
from smartsom.domain.models import (
    DomainValidationError,
    FactorySpec,
    Job,
    Machine,
    Operation,
    Order,
    ProcessingMode,
    WorkloadInstance,
    validate_problem,
)
from smartsom.domain.processing_times import ProcessingTime, ProcessingTimePlan
from smartsom.domain.state import (
    MachineState,
    OperationState,
    OperationStatus,
    ScheduledOperation,
)

__all__ = [
    "ArrivalPlan",
    "JobArrival",
    "VisibleJob",
    "DomainValidationError",
    "FactorySpec",
    "Job",
    "Machine",
    "MachineOutage",
    "MachineOutagePlan",
    "MachineState",
    "Operation",
    "OperationState",
    "OperationStatus",
    "Order",
    "ProcessingMode",
    "ProcessingTime",
    "ProcessingTimePlan",
    "ScheduledOperation",
    "WorkloadInstance",
    "validate_problem",
]
