"""Opt-in four-file inputs for the reservation Base development diagnostics."""

import hashlib
import math
import random
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from smartsom.config.factory_design import load_factory_design
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


class BaseAlgorithm(Strict):
    schema_id: Literal["smartsom.base-algorithm/v1"] = Field(alias="schema")
    method: Literal["ppo", "rule", "random"] = "ppo"
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


class BaseExperiment(Strict):
    schema_id: Literal["smartsom.base-experiment/v1"] = Field(alias="schema")
    factory: str
    workload: str
    algorithm: str
    drain_ticks: int = Field(default=2000, gt=0)
    drain_frozen: bool = False
    train_ticks: int = Field(default=100000, gt=0)
    train_seeds: tuple[int, ...] = (101, 102, 103)
    validation_cases: int = Field(default=32, gt=0)
    test_cases: int = Field(default=64, gt=0)
    record_first_case: bool = False


def seed_for(seed, *parts):
    return int.from_bytes(
        hashlib.sha256(repr((seed, parts)).encode()).digest()[:8], "big"
    )


def load_inputs(path):
    path = Path(path).resolve()
    experiment = BaseExperiment.model_validate(yaml.safe_load(path.read_text()))
    paths = {
        k: (path.parent / getattr(experiment, k)).resolve()
        for k in ("factory", "workload", "algorithm")
    }
    factory, _ = load_factory_design(paths["factory"])
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
    workload = BaseWorkload.model_validate(
        yaml.safe_load(paths["workload"].read_text())
    )
    algorithm = BaseAlgorithm.model_validate(
        yaml.safe_load(paths["algorithm"].read_text())
    )
    return experiment, factory, workload, algorithm, {"experiment": path, **paths}


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
