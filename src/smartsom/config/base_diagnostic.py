"""Opt-in four-file inputs for the reservation Base development diagnostics."""

import hashlib
import math
import random
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from smartsom.config.factory_design import parse_factory_design_file
from smartsom.domain.base_observation import ObservationVersion
from smartsom.domain.production import (
    Demand,
    Outage,
    ProductionScenario,
    ProductionStep,
)
from smartsom.engine.base_routes import StaticRoutes


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TruncatedExponential(Strict):
    scale: float = Field(gt=0)
    lower: float = Field(ge=0)
    upper: float = Field(gt=0)

    @model_validator(mode="after")
    def ordered(self):
        if self.lower >= self.upper:
            raise ValueError("truncation bounds must be ordered")
        return self

    def draw(self, rng):
        # Conditional exponential on [lower, upper], not clipping and not a
        # shifted unbounded exponential. Stable even with large lower/scale.
        mass = -math.expm1(-(self.upper - self.lower) / self.scale)
        return max(
            1, math.ceil(self.lower - self.scale * math.log1p(-rng.random() * mass))
        )


class BaseWorkload(Strict):
    schema_id: Literal["smartsom.base-workload/v1"] = Field(alias="schema")
    orders: int = Field(gt=0)
    initial_orders: int = Field(ge=0)
    mean_interarrival: float = Field(gt=0)
    families: tuple[tuple[str, ...], ...] = (
        ("A", "B", "C", "D"),
        ("B", "C", "D", "A"),
        ("C", "D", "A", "B"),
        ("D", "A", "B", "C"),
    )
    optional_repeat_probability: float = Field(default=0.5, ge=0, le=1)
    base_ticks: dict[str, float] = {"A": 30, "B": 40, "C": 50, "D": 40}
    workload_rho: float = Field(default=0.2, gt=0, lt=1)
    urgent_probability: float = Field(default=0.2, ge=0, le=1)
    normal_due: tuple[float, float] = (2.0, 3.0)
    urgent_due: tuple[float, float] = (1.5, 2.5)
    fault_intervals: dict[str, TruncatedExponential] = {
        "low": TruncatedExponential(scale=600, lower=60, upper=1800),
        "medium": TruncatedExponential(scale=300, lower=30, upper=900),
        "high": TruncatedExponential(scale=150, lower=15, upper=450),
    }
    repair: TruncatedExponential = TruncatedExponential(scale=60, lower=10, upper=180)

    @model_validator(mode="after")
    def valid(self):
        if not 0 < self.initial_orders < self.orders:
            raise ValueError("finite arrivals require 0 < initial_orders < orders")
        if set(self.fault_intervals) != {"low", "medium", "high"}:
            raise ValueError("three named fault conditions required")
        for bounds in (self.normal_due, self.urgent_due):
            if not 1 <= bounds[0] <= bounds[1]:
                raise ValueError("due factors must be ordered and at least one")
        if not self.families or any(
            not f or any(t not in self.base_ticks for t in f) for f in self.families
        ):
            raise ValueError("families require declared operation work")
        if any(x <= 0 for x in self.base_ticks.values()):
            raise ValueError("work must be positive")
        return self


class PolicyProvider(Strict):
    path: str
    factory: str = "create_policy"
    sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")


class BaseAlgorithm(Strict):
    schema_id: Literal["smartsom.base-algorithm/v1"] = Field(alias="schema")
    method: Literal["ppo", "rule", "random"] = "ppo"
    observation_schema: ObservationVersion = "native-v1"
    allow_stop: bool = True
    allow_empty_wait: bool = False
    bindings: dict[str, str] = Field(default_factory=dict)
    providers: dict[str, PolicyProvider] = Field(default_factory=dict)
    width: int = Field(default=32, ge=8)
    learning_rate: float = Field(default=0.0003, gt=0)
    epochs: int = Field(default=3, gt=0)
    batch_size: int = Field(default=256, gt=0)
    clip: float = Field(default=0.2, gt=0, lt=1)
    entropy: float = Field(default=0.01, ge=0)
    reevaluate_ticks: int = Field(default=10, gt=0)
    time_scale: float = Field(default=200.0, gt=0)
    distance_scale: float = Field(default=50.0, gt=0)
    count_scale: float = Field(default=20.0, gt=0)
    quality_penalty: float = Field(default=1000.0, ge=0)
    learning_reward_scale: float = Field(default=1.0, gt=0, allow_inf_nan=False)
    critic_initial_scale: float = Field(default=1.0, gt=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def valid_bindings(self):
        from smartsom.algorithms.base_policy import BUILTINS, DECISION_POINTS

        if set(self.bindings) - DECISION_POINTS:
            raise ValueError("unknown Base role.stage policy binding")
        if set(self.providers) & BUILTINS:
            raise ValueError("custom providers cannot replace builtin identifiers")
        if set(self.bindings.values()) - BUILTINS - set(self.providers):
            raise ValueError("policy binding names an undeclared provider")
        if set(self.providers) - set(self.bindings.values()):
            raise ValueError("declared policy provider is unused")
        return self


class ReservationRun(Strict):
    """Execution settings, separate from factory physics and learner parameters."""

    task: Literal["smoke", "calibrate", "train", "evaluate"] = "smoke"
    output: str = "runs/reservation-smoke"
    seed: int = 101
    seconds: float = Field(default=120, gt=0)
    smoke_ticks: int = Field(default=300, gt=0)
    calibration: str | None = None
    calibration_cases: int = Field(default=1, gt=0)
    max_drain: int = Field(default=10000, gt=0)
    checkpoint: str | None = None
    methods: tuple[Literal["rule", "random", "ppo", "configured"], ...] = (
        "rule",
        "random",
        "ppo",
    )
    selection: Literal["sample", "argmax"] = "sample"
    split: Literal["validation", "test"] = "validation"
    action_seeds: int = Field(default=1, gt=0)
    background: bool = False
    diagnostics: bool = False
    tune: Literal["off", "recommend", "auto"] = "off"
    resource_execution: Literal["adaptive", "fixed"] = "adaptive"
    max_concurrent: int = Field(default=2, gt=0)
    max_failures: int = Field(default=1, ge=0)
    calibration_seconds: float = Field(default=60, gt=0)
    probe_ticks: int = Field(default=200, gt=0)

    @model_validator(mode="after")
    def valid_methods(self):
        if not self.methods or len(set(self.methods)) != len(self.methods):
            raise ValueError("methods must be nonempty and unique")
        return self


class ReservationValidation(Strict):
    milestones: tuple[int, ...] = ()
    modes: tuple[Literal["sample", "argmax"], ...] = ("sample", "argmax")
    sample_action_seeds: int = Field(default=2, gt=0)
    selection: Literal["last", "sample_complete_cost", "sample_delivery_cost"] = "last"
    select_initial: bool = False

    @model_validator(mode="after")
    def valid_schedule(self):
        if (
            any(x < 0 for x in self.milestones)
            or tuple(sorted(set(self.milestones))) != self.milestones
        ):
            raise ValueError(
                "validation milestones must be strictly increasing nonnegative episodes"
            )
        if not self.modes or len(set(self.modes)) != len(self.modes):
            raise ValueError("validation modes must be nonempty and unique")
        if self.selection != "last" and (
            not self.milestones or "sample" not in self.modes
        ):
            raise ValueError("sample selection requires sample validation milestones")
        return self


class ReservationBatch(Strict):
    stage: int = Field(default=0, ge=0)
    parallel_files: int = Field(default=1, gt=0)


class BaseExperiment(Strict):
    schema_id: Literal["smartsom.base-experiment/v1"] = Field(alias="schema")
    factory: str
    workload: str
    algorithm: str
    drain_ticks: int = Field(default=2000, gt=0)
    drain_frozen: bool = False
    train_ticks: int = Field(default=100000, gt=0)
    train_episodes: int | None = Field(default=None, gt=0)
    budget_boundary: Literal["cut", "complete_episode"] = "cut"
    condition_sampling: Literal["random", "balanced_cycle"] = "random"
    checkpoint_every_episodes: int | None = Field(default=None, gt=0)
    validation: ReservationValidation = Field(default_factory=ReservationValidation)
    train_seeds: tuple[int, ...] = (101, 102, 103)
    validation_cases: int = Field(default=32, gt=0)
    test_cases: int = Field(default=64, gt=0)
    record_first_case: bool = False
    batch: ReservationBatch = Field(default_factory=ReservationBatch)
    run: ReservationRun = Field(default_factory=ReservationRun)


def seed_for(seed, *parts):
    return int.from_bytes(
        hashlib.sha256(repr((seed, parts)).encode()).digest()[:8], "big"
    )


def load_inputs(path):
    path = Path(path).resolve()
    document = yaml.safe_load(path.read_text())
    experiment = BaseExperiment.model_validate(document)
    paths = {
        k: (path.parent / getattr(experiment, k)).resolve()
        for k in ("factory", "workload", "algorithm")
    }
    documents = {
        "experiment": document,
        **{name: yaml.safe_load(source.read_text()) for name, source in paths.items()},
    }
    ex, factory, workload, algorithm = parse_inputs(documents)
    algorithm = algorithm.model_copy(
        update={
            "providers": {
                name: spec.model_copy(
                    update={
                        "path": str((paths["algorithm"].parent / spec.path).resolve())
                    }
                )
                for name, spec in algorithm.providers.items()
            }
        }
    )
    return ex, factory, workload, algorithm, {"experiment": path, **paths}


def parse_inputs(documents):
    """One typed parser shared by file loading and detached compilation."""
    experiment = BaseExperiment.model_validate(documents["experiment"])
    factory = parse_factory_design_file(documents["factory"]).factory
    if (
        factory.inspection_stations
        or factory.scrap_bins
        or set(factory.operation_types) != set("ABCD")
    ):
        raise ValueError(
            "Base diagnostic uses A/B/C/D and final Output inspection only"
        )
    for m in factory.machines:
        if len(m.operation_types) != 1 or "normal" not in {
            mode.quality_mode_id for mode in m.quality_modes
        }:
            raise ValueError(
                "Base requires single-type machines with a normal reference mode"
            )
        owned = [b for b in factory.buffers if b.machine_id == m.machine_id]
        if {b.role for b in owned} != {"machine_pre", "machine_post"} or len(
            owned
        ) != 2:
            raise ValueError("each Base station requires pre and post storage")
        if any(getattr(b.storage, "capacity", None) is None for b in owned):
            raise ValueError("Base pre/post use finite pooled storage")
    workload = BaseWorkload.model_validate(documents["workload"])
    algorithm = BaseAlgorithm.model_validate(documents["algorithm"])
    return experiment, factory, workload, algorithm


def nominal_baseline(factory, steps, routes):
    pre = {
        b.machine_id: b.buffer_id for b in factory.buffers if b.role == "machine_pre"
    }
    post = {
        b.machine_id: b.buffer_id for b in factory.buffers if b.role == "machine_post"
    }
    source = next(b.buffer_id for b in factory.buffers if b.role == "system_input")
    output = next(b.buffer_id for b in factory.buffers if b.role == "system_output")
    costs = {None: 0.0}
    previous_type = None
    for step in steps:
        next_costs = {}
        for machine in factory.machines:
            if step.operation_type not in machine.operation_types:
                continue
            processing = max(
                1,
                math.ceil(
                    float(step.work_ticks_on(machine.machine_id))
                    / float(machine.processing_rate_multiplier)
                ),
            )
            values = []
            for previous, cost in costs.items():
                if (
                    previous_type == step.operation_type
                    and previous != machine.machine_id
                ):
                    continue  # Mandatory same-machine continuation, not optional hauling.
                travel = (
                    0
                    if previous == machine.machine_id
                    else routes.between(
                        source if previous is None else post[previous],
                        pre[machine.machine_id],
                    )
                    + 2
                )
                values.append(cost + travel + processing)
            if values:
                next_costs[machine.machine_id] = min(values)
        if not next_costs:
            raise ValueError("no feasible operation chain")
        costs, previous_type = next_costs, step.operation_type
    return min(
        cost + routes.between(post[machine], output) + 2
        for machine, cost in costs.items()
    )


def generate_scenario(factory, workload, seed, condition, drain_ticks):
    routes = StaticRoutes(factory)
    rng = random.Random(seed_for(seed, "orders"))
    arrivals = random.Random(seed_for(seed, "arrivals"))
    clock, demands = 0.0, []
    for index in range(workload.orders):
        if index >= workload.initial_orders:
            clock += arrivals.expovariate(1 / workload.mean_interarrival)
        release = math.ceil(clock)
        family = list(rng.choice(workload.families))
        if rng.random() < workload.optional_repeat_probability:
            family.insert(1, family[0])
        steps = []
        for i, kind in enumerate(family):
            work = workload.base_ticks[kind] * rng.uniform(
                1 - workload.workload_rho, 1 + workload.workload_rho
            )
            steps.append(
                ProductionStep(
                    f"op_{i}", kind, math.ceil(work), reference_ticks=str(work)
                )
            )
        urgent = rng.random() < workload.urgent_probability
        factor = rng.uniform(*(workload.urgent_due if urgent else workload.normal_due))
        due = release + math.ceil(factor * nominal_baseline(factory, steps, routes))
        demands.append(
            Demand(f"order_{index:04d}", tuple(steps), release, due, rush=urgent)
        )
    horizon = demands[-1].release_at + drain_ticks
    outages = []
    if condition != "normal":
        distribution = workload.fault_intervals[condition]
        for machine in factory.machines:
            rng_fault = random.Random(seed_for(seed, "faults", machine.machine_id))
            t = 0
            while t < horizon:
                t += distribution.draw(rng_fault)
                repair = workload.repair.draw(rng_fault)
                if t < horizon:
                    outages.append(Outage(machine.machine_id, t, t + repair))
                t += repair
    return ProductionScenario(
        factory,
        tuple(demands),
        mode="finite",
        tick_limit=horizon,
        seed=seed,
        outages=tuple(outages),
        processing_rounding="ceil",
    )
