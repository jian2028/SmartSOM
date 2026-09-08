"""Immutable domain inputs for static serial JSP and FJSP."""

from dataclasses import dataclass

from smartsom.domain.buffers import MachineBuffers
from smartsom.domain.quality import QualitySpeedSpec
from smartsom.domain.transport import TransportSpec
from smartsom.domain.validation import (
    DomainValidationError,
    _identifier,
    _items,
    _unique,
)


@dataclass(frozen=True, slots=True)
class Machine:
    machine_id: str

    def __post_init__(self) -> None:
        _identifier(self.machine_id, "machine_id")


@dataclass(frozen=True, slots=True)
class FactorySpec:
    machines: tuple[Machine, ...]
    transport: TransportSpec | None = None
    buffers: tuple[MachineBuffers, ...] = ()
    quality_speed: QualitySpeedSpec | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "machines", _items(self.machines, Machine, "machines"))
        _unique(tuple(machine.machine_id for machine in self.machines), "machine ID")
        if not isinstance(self.buffers, (tuple, list)) or any(
            not isinstance(x, MachineBuffers) for x in self.buffers
        ):
            raise DomainValidationError("buffers must contain MachineBuffers")
        _unique(tuple(x.machine_id for x in self.buffers), "buffer machine ID")
        if any(
            x.machine_id not in {m.machine_id for m in self.machines}
            for x in self.buffers
        ):
            raise DomainValidationError("buffer references unknown machine")
        object.__setattr__(
            self,
            "buffers",
            tuple(
                sorted(
                    (
                        x
                        for x in self.buffers
                        if x.pre_capacity is not None or x.post_capacity is not None
                    ),
                    key=lambda x: x.machine_id,
                )
            ),
        )
        if self.quality_speed is not None:
            if not isinstance(self.quality_speed, QualitySpeedSpec) or any(
                x.machine_id not in {m.machine_id for m in self.machines}
                for x in self.quality_speed.machine_modes
            ):
                raise DomainValidationError("quality_speed references unknown machine")
        if self.transport is not None:
            if not isinstance(self.transport, TransportSpec) or {
                x.machine_id for x in self.transport.machine_locations
            } != {x.machine_id for x in self.machines}:
                raise DomainValidationError(
                    "transport must map every factory machine exactly once"
                )


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

    def mode(self, processing_mode_id: str | None) -> ProcessingMode | None:
        """Look up a semantic mode within its owning operation."""
        return next(
            (
                mode
                for mode in self.modes
                if mode.processing_mode_id == processing_mode_id
            ),
            None,
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
