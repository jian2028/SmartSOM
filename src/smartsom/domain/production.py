"""Immutable inputs and semantic commands for grid production execution."""

from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from typing import Literal

from smartsom.domain.factory_design import FactoryDesign
from smartsom.domain.validation import _identifier


@dataclass(frozen=True, slots=True)
class ProductionStep:
    operation_id: str
    operation_type: str
    nominal_ticks: int
    machine_nominal_ticks: tuple[tuple[str, int], ...] = ()

    def __post_init__(self):
        _identifier(self.operation_id, "operation_id")
        _identifier(self.operation_type, "operation_type")
        if type(self.nominal_ticks) is not int or self.nominal_ticks < 1:
            raise ValueError("nominal_ticks must be a positive integer")
        values = self.machine_nominal_ticks
        pairs = tuple(values.items()) if isinstance(values, Mapping) else tuple(values)
        if len({key for key, _ in pairs}) != len(pairs):
            raise ValueError("duplicate machine in machine_nominal_ticks")
        for key, ticks in pairs:
            _identifier(key, "machine_id")
            if type(ticks) is not int or ticks < 1:
                raise ValueError("machine_nominal_ticks requires positive integers")
        object.__setattr__(self, "machine_nominal_ticks", tuple(sorted(pairs)))

    def ticks_on(self, machine_id: str) -> int:
        return dict(self.machine_nominal_ticks).get(machine_id, self.nominal_ticks)


@dataclass(frozen=True, slots=True)
class Demand:
    demand_id: str
    steps: tuple[ProductionStep, ...]
    release_at: int = 0
    due_at: int = 100
    priority: int = 1
    input_id: str | None = None
    reveal_at: int | None = None

    def __post_init__(self):
        _identifier(self.demand_id, "demand_id")
        object.__setattr__(self, "steps", tuple(self.steps))
        if any(not isinstance(step, ProductionStep) for step in self.steps):
            raise ValueError("demand steps must be ProductionStep values")
        if self.input_id is not None:
            _identifier(self.input_id, "input_id")
        if not self.steps or len({s.operation_id for s in self.steps}) != len(
            self.steps
        ):
            raise ValueError("demand requires uniquely identified production steps")
        for name in ("release_at", "due_at", "priority"):
            value = getattr(self, name)
            if type(value) is not int or value < (1 if name == "priority" else 0):
                raise ValueError(f"invalid {name}")
        reveal = self.release_at if self.reveal_at is None else self.reveal_at
        if type(reveal) is not int or not 0 <= reveal <= self.release_at:
            raise ValueError("reveal_at must be between zero and release_at")
        object.__setattr__(self, "reveal_at", reveal)


@dataclass(frozen=True, slots=True)
class Outage:
    machine_id: str
    start: int
    end: int

    def __post_init__(self):
        _identifier(self.machine_id, "machine_id")
        if type(self.start) is not int or type(self.end) is not int:
            raise ValueError("outage endpoints must be integer ticks")
        if not 0 <= self.start < self.end:
            raise ValueError("outage must be a nonempty half-open interval")


@dataclass(frozen=True, slots=True)
class ProcessingSample:
    demand_id: str
    operation_id: str
    machine_id: str
    actual_ticks: int
    attempt: int = 1

    def __post_init__(self):
        for name in ("demand_id", "operation_id", "machine_id"):
            _identifier(getattr(self, name), name)
        for name in ("actual_ticks", "attempt"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True, slots=True)
class QualitySample:
    demand_id: str
    operation_id: str
    draw: int
    attempt: int = 1

    def __post_init__(self):
        for name in ("demand_id", "operation_id"):
            _identifier(getattr(self, name), name)
        if type(self.attempt) is not int or self.attempt < 1:
            raise ValueError("attempt must be a positive integer")
        if type(self.draw) is not int or not 0 <= self.draw < 2**53:
            raise ValueError("quality draw must be an integer in [0, 2**53)")


@dataclass(frozen=True, slots=True)
class ProductionScenario:
    factory: FactoryDesign
    demands: tuple[Demand, ...]
    mode: Literal["static", "dynamic"] = "static"
    tick_limit: int = 1000
    seed: int = 42
    outages: tuple[Outage, ...] = ()
    processing_low: Decimal = Decimal(1)
    processing_high: Decimal = Decimal(1)
    reward_time_scale: int = 100
    processing_samples: tuple[ProcessingSample, ...] = ()
    quality_samples: tuple[QualitySample, ...] = ()
    quality_probability_visibility: Literal["public", "hidden"] = "public"

    def __post_init__(self):
        if not isinstance(self.factory, FactoryDesign):
            raise ValueError("production requires the current grid FactoryDesign")
        object.__setattr__(self, "demands", tuple(self.demands))
        object.__setattr__(self, "outages", tuple(self.outages))
        for name, cls in (("demands", Demand), ("outages", Outage)):
            if any(not isinstance(row, cls) for row in getattr(self, name)):
                raise ValueError(f"{name} must contain {cls.__name__} values")
        for name, cls in (
            ("processing_samples", ProcessingSample),
            ("quality_samples", QualitySample),
        ):
            rows = tuple(getattr(self, name))
            if any(not isinstance(row, cls) for row in rows):
                raise ValueError(f"{name} must contain {cls.__name__} values")
            object.__setattr__(self, name, rows)
            keys = [
                (
                    row.demand_id,
                    row.operation_id,
                    row.attempt,
                    getattr(row, "machine_id", None),
                )
                for row in rows
            ]
            if len(set(keys)) != len(keys):
                raise ValueError(f"duplicate identity in {name}")
        if self.quality_probability_visibility not in ("public", "hidden"):
            raise ValueError("quality_probability_visibility must be public or hidden")
        if self.mode not in ("static", "dynamic"):
            raise ValueError("mode must be static or dynamic")
        for name in ("tick_limit", "reward_time_scale"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.seed) is not int or not 0 <= self.seed < 2**64:
            raise ValueError("seed must be an unsigned 64-bit integer")
        ids = [d.demand_id for d in self.demands]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate demand identity")
        if self.mode == "static" and any(d.release_at for d in self.demands):
            raise ValueError("static demands must be released at tick zero")
        for name in ("processing_low", "processing_high"):
            value = Decimal(str(getattr(self, name)))
            if not value.is_finite() or value <= 0:
                raise ValueError("processing multipliers must be finite and positive")
            object.__setattr__(self, name, value)
        if self.processing_low > self.processing_high:
            raise ValueError("processing multiplier bounds are reversed")


@dataclass(frozen=True, slots=True)
class MachineCommand:
    job_id: str | None = None
    mode_id: str | None = None

    def __post_init__(self):
        if (self.job_id is None) != (self.mode_id is None):
            raise ValueError("machine START requires both job_id and mode_id")


@dataclass(frozen=True, slots=True)
class JointCommand:
    """Missing commands mean WAIT; rankings contain stable job identities."""

    agvs: tuple[tuple[str, str], ...] = ()
    machines: tuple[tuple[str, MachineCommand], ...] = ()
    quality: tuple[tuple[str, str], ...] = ()
    rankings: tuple[tuple[str, tuple[str, ...]], ...] = ()

    def __post_init__(self):
        for name in ("agvs", "machines", "quality", "rankings"):
            rows = tuple(getattr(self, name))
            if len({key for key, _ in rows}) != len(rows):
                raise ValueError(f"duplicate resource in {name}")
            object.__setattr__(self, name, rows)


MOVES = {"UP": (0, -1), "DOWN": (0, 1), "LEFT": (-1, 0), "RIGHT": (1, 0)}
AGV_ACTIONS = (*MOVES, "INTERACT", "WAIT")


def validate_production_scenario(scenario):
    """Validate executable grid contracts without allocating or advancing state."""
    from dataclasses import asdict

    from smartsom.domain.factory_design import validate_factory_design

    factory = scenario.factory
    machines = {m.machine_id: m for m in factory.machines}
    demands = {d.demand_id: d for d in scenario.demands}
    roles = {b.buffer_id: b.role for b in factory.buffers}
    owners = [(b.role, b.machine_id) for b in factory.buffers if b.machine_id]
    if len(set(owners)) != len(owners):
        raise ValueError("a machine can own at most one buffer per role")
    errors = [
        x.message for x in validate_factory_design(factory) if x.severity == "error"
    ]
    if errors:
        raise ValueError("; ".join(errors))
    if not factory.agvs:
        raise ValueError("production requires at least one AGV")
    for buffer in factory.buffers:
        if (
            buffer.role in ("machine_pre", "machine_post")
            and buffer.machine_id not in machines
        ):
            raise ValueError(f"buffer {buffer.buffer_id} requires a known machine")
    for agv in factory.agvs:
        if agv.job_capacity != 1 or agv.move_cells_per_tick != 1:
            raise ValueError("AGVs require capacity one and speed one cell/tick")
    for p in factory.ports:
        services = {
            next(
                v
                for k, v in asdict(b.target).items()
                if k.endswith("_id") and k != "slot_id"
            )
            for b in p.bindings
        }
        if len(services) > 1:
            raise ValueError(f"port {p.port_id} must serve one facility")
        active = any("charge" not in b.operations for b in p.bindings)
        if active and set(p.allowed_headings) != {"north", "east", "south", "west"}:
            raise ValueError(f"port {p.port_id} must allow all headings")
    inputs = [k for k, role in roles.items() if role == "system_input"]
    if not inputs or "system_output" not in roles.values():
        raise ValueError("production requires input and output facilities")
    for d in demands.values():
        if d.input_id is None and len(inputs) != 1:
            raise ValueError("demand input_id required with multiple input facilities")
        if d.input_id is not None and d.input_id not in inputs:
            raise ValueError("demand input_id must identify a system input")
        for step in d.steps:
            for machine_id, _ in step.machine_nominal_ticks:
                machine = machines.get(machine_id)
                if (
                    machine is None
                    or step.operation_type not in machine.operation_types
                ):
                    raise ValueError(
                        f"machine_nominal_ticks references unknown or incapable machine {machine_id}"
                    )
            if not any(
                step.operation_type in m.operation_types for m in machines.values()
            ):
                raise ValueError(f"no capable machine for {step.operation_type}")
    if any(x.machine_id not in machines for x in scenario.outages):
        raise ValueError("outage references an unknown machine")
    for sample in (
        *scenario.processing_samples,
        *scenario.quality_samples,
    ):
        demand = demands.get(sample.demand_id)
        operation = (
            next(
                (
                    step
                    for step in demand.steps
                    if step.operation_id == sample.operation_id
                ),
                None,
            )
            if demand
            else None
        )
        if operation is None:
            raise ValueError("fixed sample references an unknown demand or operation")
        if hasattr(sample, "machine_id"):
            machine = machines.get(sample.machine_id)
            if (
                machine is None
                or operation.operation_type not in machine.operation_types
            ):
                raise ValueError(
                    "processing sample references an unknown or incapable machine"
                )
