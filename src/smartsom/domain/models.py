"""Immutable domain inputs for the serial, single-mode static slice."""

from dataclasses import dataclass


class DomainValidationError(ValueError):
    """An input violates a supported domain rule."""


def _identifier(value: str, label: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise DomainValidationError(f"{label} must be a non-empty string")


def _unique(values: tuple[str, ...], label: str) -> None:
    seen: set[str] = set()
    for value in values:
        if value in seen:
            raise DomainValidationError(f"duplicate {label}: {value!r}")
        seen.add(value)


def _items[T](values: tuple[T, ...], item_type: type[T], label: str) -> tuple[T, ...]:
    # Copy caller-owned lists before storing them in a frozen object.
    if not isinstance(values, (tuple, list)) or not values:
        raise DomainValidationError(f"{label} must be a non-empty tuple or list")
    result = tuple(values)
    if any(not isinstance(value, item_type) for value in result):
        raise DomainValidationError(
            f"{label} must contain {item_type.__name__} objects"
        )
    return result


@dataclass(frozen=True, slots=True)
class Machine:
    machine_id: str

    def __post_init__(self) -> None:
        _identifier(self.machine_id, "machine_id")


@dataclass(frozen=True, slots=True)
class FactorySpec:
    machines: tuple[Machine, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "machines", _items(self.machines, Machine, "machines"))
        _unique(tuple(machine.machine_id for machine in self.machines), "machine ID")


@dataclass(frozen=True, slots=True)
class ProcessingMode:
    processing_mode_id: str
    machine_id: str
    nominal_ticks: int

    def __post_init__(self) -> None:
        _identifier(self.processing_mode_id, "processing_mode_id")
        _identifier(self.machine_id, "machine_id")
        if type(self.nominal_ticks) is not int or self.nominal_ticks <= 0:
            raise DomainValidationError("nominal_ticks must be a positive integer")


@dataclass(frozen=True, slots=True)
class Operation:
    operation_id: str
    modes: tuple[ProcessingMode, ...]
    predecessor_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _identifier(self.operation_id, "operation_id")
        object.__setattr__(self, "modes", _items(self.modes, ProcessingMode, "modes"))
        _unique(tuple(mode.processing_mode_id for mode in self.modes), "mode ID")
        if len(self.modes) != 1:
            raise DomainValidationError("exactly one processing mode is supported")
        if not isinstance(self.predecessor_ids, (tuple, list)):
            raise DomainValidationError("predecessor_ids must be a tuple or list")
        object.__setattr__(self, "predecessor_ids", tuple(self.predecessor_ids))
        for predecessor in self.predecessor_ids:
            _identifier(predecessor, "predecessor ID")
        _unique(self.predecessor_ids, "predecessor ID")
        if len(self.predecessor_ids) > 1:
            raise DomainValidationError(
                "serial operations have at most one predecessor"
            )


@dataclass(frozen=True, slots=True)
class Job:
    job_id: str
    operations: tuple[Operation, ...]

    def __post_init__(self) -> None:
        _identifier(self.job_id, "job_id")
        object.__setattr__(
            self, "operations", _items(self.operations, Operation, "operations")
        )
        _unique(tuple(op.operation_id for op in self.operations), "operation ID")
        known = {op.operation_id for op in self.operations}
        successors: dict[str, str] = {}
        roots = []
        for operation in self.operations:
            if not operation.predecessor_ids:
                roots.append(operation.operation_id)
                continue
            predecessor = operation.predecessor_ids[0]
            if predecessor not in known:
                raise DomainValidationError(
                    f"predecessor {predecessor!r} is outside job {self.job_id!r}"
                )
            if predecessor == operation.operation_id:
                raise DomainValidationError("an operation cannot precede itself")
            if predecessor in successors:
                raise DomainValidationError("branching is not a serial job chain")
            successors[predecessor] = operation.operation_id
        if len(roots) != 1:
            raise DomainValidationError("a serial job chain must have exactly one root")
        visited: set[str] = set()
        current: str | None = roots[0]
        while current is not None and current not in visited:
            visited.add(current)
            current = successors.get(current)
        if visited != known or current is not None:
            raise DomainValidationError("job chain is disconnected or contains a cycle")


@dataclass(frozen=True, slots=True)
class Order:
    order_id: str
    jobs: tuple[Job, ...]

    def __post_init__(self) -> None:
        _identifier(self.order_id, "order_id")
        object.__setattr__(self, "jobs", _items(self.jobs, Job, "jobs"))
        _unique(tuple(job.job_id for job in self.jobs), "job ID")


@dataclass(frozen=True, slots=True)
class WorkloadInstance:
    orders: tuple[Order, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "orders", _items(self.orders, Order, "orders"))
        _unique(tuple(order.order_id for order in self.orders), "order ID")
        _unique(
            tuple(job.job_id for order in self.orders for job in order.jobs), "job ID"
        )
        _unique(tuple(op.operation_id for op in self.operations), "operation ID")

    @property
    def operations(self) -> tuple[Operation, ...]:
        return tuple(
            operation
            for order in self.orders
            for job in order.jobs
            for operation in job.operations
        )


def validate_problem(factory: FactorySpec, workload: WorkloadInstance) -> None:
    """Validate references across the two independent domain owners."""
    if not isinstance(factory, FactorySpec) or not isinstance(
        workload, WorkloadInstance
    ):
        raise DomainValidationError("expected FactorySpec and WorkloadInstance")
    machines = {machine.machine_id for machine in factory.machines}
    for operation in workload.operations:
        for mode in operation.modes:
            if mode.machine_id not in machines:
                raise DomainValidationError(
                    f"operation {operation.operation_id!r} references unknown machine "
                    f"{mode.machine_id!r}"
                )
