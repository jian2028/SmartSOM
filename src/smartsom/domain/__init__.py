"""Domain specifications and immutable state views; no framework dependencies."""

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
from smartsom.domain.state import (
    MachineState,
    OperationState,
    OperationStatus,
    ScheduledOperation,
)

__all__ = [
    "DomainValidationError",
    "FactorySpec",
    "Job",
    "Machine",
    "MachineState",
    "Operation",
    "OperationState",
    "OperationStatus",
    "Order",
    "ProcessingMode",
    "ScheduledOperation",
    "WorkloadInstance",
    "validate_problem",
]
