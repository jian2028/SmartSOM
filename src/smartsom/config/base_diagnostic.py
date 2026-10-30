"""Opt-in four-file inputs for the reservation Base development diagnostics."""

import hashlib
import json
import math
import random
from functools import cached_property
from pathlib import Path
from typing import Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    field_validator,
    model_validator,
)

from smartsom.config.evaluation_protocol import EvaluationProtocol
from smartsom.config.factory_design import parse_factory_design_file
from smartsom.config.launch import Concurrency, Display, Launch, Resources, Tracking
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


class Component(Strict):
    """An explicit callable, optionally backed by a frozen external source bundle."""

    target: str | None = None
    path: str | None = None
    factory: str | None = None
    source_root: str | None = None
    sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    params: dict = Field(default_factory=dict)

    @model_validator(mode="after")
    def reference(self):
        if bool(self.target) == bool(self.path):
            raise ValueError("component requires exactly one target or path")
        if self.path and (not self.factory or not self.factory.isidentifier()):
            raise ValueError("file component requires an explicit factory function")
        if self.target and (self.factory or self.source_root):
            raise ValueError(
                "module target cannot also declare a file factory/source_root"
            )
        return self


class AcceptActions(Strict):
    allow_stop: bool = True


class BufferActions(Strict):
    accept: AcceptActions = Field(default_factory=AcceptActions)


class PickupActions(Strict):
    allow_empty_wait: bool = False


class DispatcherActions(Strict):
    pickup: PickupActions = Field(default_factory=PickupActions)


class DecisionProtocol(Strict):
    pre: BufferActions | None = None
    post: BufferActions | None = None
    dispatcher: DispatcherActions | None = None
    reevaluate_ticks: int = Field(default=10, gt=0)
    time_scale: float = Field(default=200.0, gt=0)
    distance_scale: float = Field(default=50.0, gt=0)
    count_scale: float = Field(default=20.0, gt=0)
    quality_penalty: float = Field(default=1000.0, ge=0)


class LearnerParameters(Strict):
    width: int = Field(default=32, ge=8)
    learning_rate: float = Field(default=0.0003, gt=0)
    epochs: int = Field(default=3, gt=0)
    batch_size: int = Field(default=256, gt=0)
    clip: float = Field(default=0.2, gt=0, lt=1)
    entropy: float = Field(default=0.01, ge=0)
    replay_capacity: int = Field(default=10000, gt=0)
    target_update_interval: int = Field(default=100, gt=0)
    epsilon: float = Field(default=0.1, ge=0, le=1)
    learning_reward_scale: float = Field(default=1.0, gt=0, allow_inf_nan=False)
    critic_initial_scale: float = Field(default=1.0, gt=0, allow_inf_nan=False)


Role = Literal["machine", "dispatcher", "pre", "post", "input"]


class ParameterSharing(Strict):
    default: Literal["shared", "independent"] = "shared"
    roles: dict[Role, Literal["shared", "independent"]] = Field(default_factory=dict)

    def mode(self, role):
        return self.roles.get(role, self.default)


class PolicyDefinition(Strict):
    kind: Literal["learning", "policy"]
    parameter_sharing: ParameterSharing = Field(default_factory=ParameterSharing)
    role_params: dict[Role, dict] = Field(default_factory=dict)
    implementation: Component | None = None
    learner: Component | None = None
    observation: Component | None = None
    encoder: Component | None = None
    network: Component | None = None

    @model_validator(mode="after")
    def contract(self):
        if self.kind == "policy":
            if self.role_params or self.parameter_sharing != ParameterSharing():
                raise ValueError("network sharing and role_params require learning")
            if self.implementation is None or any(
                (self.learner, self.encoder, self.network)
            ):
                raise ValueError(
                    "policy requires implementation, optional observation, and no learner/network"
                )
        elif (
            self.implementation
            or self.learner is None
            or self.observation is None
            or bool(self.encoder) == bool(self.network)
        ):
            raise ValueError(
                "learning policy requires learner, observation and exactly one encoder or network"
            )
        if self.learner and self.learner.target in {
            "smartsom.learning.ppo.create_learner",
            "smartsom.learning.dqn.create_learner",
        }:
            LearnerParameters.model_validate(self.learner.params)
            for overrides in self.role_params.values():
                LearnerParameters.model_validate({**self.learner.params, **overrides})
        return self


class ProductionPolicies(Strict):
    default: str
    bindings: dict[str, str] = Field(default_factory=dict)
    policies: dict[str, PolicyDefinition]

    @model_validator(mode="after")
    def valid(self):
        if any(not key.isidentifier() for key in self.policies):
            raise ValueError("policy names must be identifiers")
        used = {self.default, *self.bindings.values()}
        if used - self.policies.keys():
            raise ValueError("policy binding names an undeclared provider")
        if self.policies.keys() - used:
            raise ValueError("declared policy provider is unused")
        if sum(p.kind == "learning" for p in self.policies.values()) > 1:
            raise ValueError("one experiment supports one learning policy instance")
        return self


class Traffic(Strict):
    mover: Component = Field(
        default_factory=lambda: Component(
            target="smartsom.traffic.rule_mover.create_mover"
        )
    )


class BaseAlgorithm(Strict):
    _owner_ids: dict = PrivateAttr(default_factory=dict)
    schema_id: Literal["smartsom.algorithm-components/v1"] = Field(alias="schema")
    control: Literal["base", "centralized"] = "base"
    decision_protocol: DecisionProtocol | None = None
    production: ProductionPolicies
    traffic: Traffic = Field(default_factory=Traffic)

    @model_validator(mode="after")
    def valid(self):
        from smartsom.algorithms.base_policy import DECISION_POINTS

        points = (
            {"central.assign"} if self.control == "centralized" else DECISION_POINTS
        )
        if set(self.production.bindings) - points:
            raise ValueError("unknown role.stage policy binding")
        protocol = self.decision_protocol or DecisionProtocol()
        if self.control == "centralized":
            if any((protocol.pre, protocol.post, protocol.dispatcher)):
                raise ValueError(
                    "centralized control rejects distributed decision_protocol"
                )
            if self.learning_policy is not None:
                raise ValueError("centralized DRL training is not supported")
        else:
            protocol = protocol.model_copy(
                update={
                    "pre": protocol.pre or BufferActions(),
                    "post": protocol.post or BufferActions(),
                    "dispatcher": protocol.dispatcher or DispatcherActions(),
                }
            )
        object.__setattr__(self, "decision_protocol", protocol)
        return self

    @property
    def learning_policy(self):
        return next(
            (p for p in self.production.policies.values() if p.kind == "learning"), None
        )

    @property
    def method(self):
        spec = self.production.policies[self.production.default]
        entry = spec.learner if spec.kind == "learning" else spec.implementation
        return (
            (entry.target or self.production.default).split(".")[-2]
            if entry.target
            else self.production.default
        )

    @property
    def observation_schema(self):
        return "native-v2"

    @property
    def observation(self):
        return self.learning_policy.observation if self.learning_policy else None

    @property
    def network(self):
        return self.learning_policy.network if self.learning_policy else None

    @property
    def encoder(self):
        return self.learning_policy.encoder if self.learning_policy else None

    @property
    def social_arm(self):
        if (
            self.observation
            and self.observation.target
            == "smartsom.representations.social.observation.create_observation"
        ):
            return self.observation.params.get("arm", "all")
        return "base"

    @property
    def social_pool(self):
        return (
            self.encoder.params.get("pool", "attention")
            if self.encoder
            else "attention"
        )

    @property
    def allow_empty_wait(self):
        return bool(
            self.decision_protocol
            and self.decision_protocol.dispatcher
            and self.decision_protocol.dispatcher.pickup.allow_empty_wait
        )

    @cached_property
    def learner_parameters(self):
        return LearnerParameters.model_validate(
            self.learning_policy.learner.params if self.learning_policy else {}
        )

    def model_copy(self, *, update=None, deep=False):
        result = super().model_copy(update=update, deep=deep)
        result.__dict__.pop("learner_parameters", None)
        return result

    def for_role(self, role):
        policy = self.learning_policy
        if policy is None or not policy.role_params.get(role):
            return self
        name = next(n for n, p in self.production.policies.items() if p is policy)
        changed = policy.model_copy(
            update={
                "learner": policy.learner.model_copy(
                    update={
                        "params": {**policy.learner.params, **policy.role_params[role]}
                    }
                )
            }
        )
        return self.model_copy(
            update={
                "production": self.production.model_copy(
                    update={"policies": {**self.production.policies, name: changed}}
                )
            }
        )

    def component_items(self):
        yield "traffic.mover", self.traffic.mover
        for name, policy in self.production.policies.items():
            for field in (
                "implementation",
                "learner",
                "observation",
                "encoder",
                "network",
            ):
                spec = getattr(policy, field)
                if spec is not None:
                    yield f"production.policies.{name}.{field}", spec

    def science_identity(self):
        data = self.model_dump(mode="json", by_alias=True)
        for key, spec in self.component_items():
            node = data
            parts = key.split(".")
            for part in parts:
                node = node[part]
            if spec.sha256:
                if spec.path:
                    node["path"] = (
                        Path(spec.path).relative_to(spec.source_root).as_posix()
                        if spec.source_root
                        else Path(spec.path).name
                    )
                node.pop("source_root", None)
        if self.learning_policy and any(
            self.learning_policy.parameter_sharing.mode(g) == "independent"
            for g in self._owner_ids
        ):
            data["entity_owners"] = self._owner_ids
        return data


# Read-only accessors keep algorithm math independent of the YAML nesting.
for _name in LearnerParameters.model_fields:
    setattr(
        BaseAlgorithm,
        _name,
        property(lambda self, name=_name: getattr(self.learner_parameters, name)),
    )
for _name in (
    "reevaluate_ticks",
    "time_scale",
    "distance_scale",
    "count_scale",
    "quality_penalty",
):
    setattr(
        BaseAlgorithm,
        _name,
        property(
            lambda self, name=_name: getattr(
                self.decision_protocol or DecisionProtocol(), name
            )
        ),
    )


class ReservationRun(Strict):
    tags: tuple[str, ...] = ()
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
    methods: tuple[Literal["rule", "random", "ppo", "dqn", "configured"], ...] = (
        "rule",
        "random",
        "ppo",
    )
    selection: Literal["sample", "argmax"] = "sample"
    split: Literal["validation", "test"] = "validation"
    action_seeds: int = Field(default=1, gt=0)
    diagnostics: bool = False
    metrics: Literal["auto", "off", "summary", "detailed"] = "auto"
    metric_interval: int = Field(default=50, gt=0)
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
    unit: Literal["episodes", "physical_ticks"] = "episodes"
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
                "validation milestones must be strictly increasing nonnegative nodes"
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


TrackingOptions = Tracking


class BaseExperiment(Strict):
    schema_id: Literal["smartsom.base-experiment/v1"] = Field(alias="schema")
    factory: str
    workload: str
    algorithm: str
    drain_ticks: int = Field(default=2000, gt=0)
    drain_frozen: bool = False
    final_evaluation: bool = False
    training_conditions: tuple[Literal["normal", "low", "medium", "high"], ...] = (
        "normal",
        "low",
        "medium",
        "high",
    )
    training_world_domain: str = "train"
    training_world_prefix: tuple[str, ...] = ()
    training_seed_modulus: int | None = Field(default=2**31, gt=0)
    train_ticks: int = Field(default=100000, gt=0)
    train_episodes: int | None = Field(default=None, gt=0)
    budget_boundary: Literal["cut", "complete_episode"] = "cut"
    condition_sampling: Literal["random", "balanced_cycle"] = "random"
    checkpoint_every_episodes: int | None = Field(default=None, gt=0)
    validation: ReservationValidation = Field(default_factory=ReservationValidation)
    evaluation: EvaluationProtocol = Field(default_factory=EvaluationProtocol)
    train_seeds: tuple[int, ...] = (101, 102, 103)
    case_sets: dict[Literal["validation", "test"], str] = Field(default_factory=dict)
    case_set_sha256: dict[Literal["validation", "test"], str] = Field(
        default_factory=dict
    )
    validation_cases: int = Field(default=32, gt=0)
    test_cases: int = Field(default=64, gt=0)
    record_first_case: bool = False
    batch: ReservationBatch = Field(default_factory=ReservationBatch)
    launch: Launch = Field(default_factory=Launch)
    concurrency: Concurrency = Field(default_factory=Concurrency)
    resources: Resources = Field(default_factory=Resources)
    display: Display = Field(default_factory=Display)
    output_root: str | None = None
    tracking: TrackingOptions = Field(default_factory=TrackingOptions)
    run: ReservationRun = Field(default_factory=ReservationRun)

    @field_validator("train_seeds", mode="before")
    @classmethod
    def expand_seeds(cls, value):
        if type(value) is int:
            if value < 1:
                raise ValueError("train_seeds count must be positive")
            return tuple(range(101, 101 + value))
        if (
            not isinstance(value, (list, tuple))
            or not value
            or any(type(seed) is not int or seed < 0 for seed in value)
            or len(set(value)) != len(value)
        ):
            raise ValueError(
                "train_seeds requires a positive count or unique nonnegative integer list"
            )
        return value

    @model_validator(mode="after")
    def valid_drain_limit(self):
        if not self.training_conditions or len(set(self.training_conditions)) != len(
            self.training_conditions
        ):
            raise ValueError("training_conditions must be nonempty and unique")
        if self.validation.unit == "physical_ticks":
            if self.train_episodes is not None or self.budget_boundary != "cut":
                raise ValueError(
                    "physical tick validation requires a tick budget and budget_boundary: cut"
                )
            if any(n > self.train_ticks for n in self.validation.milestones):
                raise ValueError("validation node exceeds training tick budget")
        elif self.train_episodes is not None and any(
            n > self.train_episodes for n in self.validation.milestones
        ):
            raise ValueError("validation node exceeds training episode budget")
        if self.run.task == "calibrate" and self.run.max_drain < self.drain_ticks:
            raise ValueError("max_drain must be at least the initial drain_ticks")
        return self


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
    execution = path.parent.parent / "execution.json"
    if path.parent.name == "inputs" and execution.is_file():
        from smartsom.config.launch import Profile

        settings = Profile.model_validate(json.loads(execution.read_text())["settings"])
        ex = ex.model_copy(
            update={
                "launch": settings.launch,
                "tracking": settings.tracking,
                "concurrency": settings.concurrency,
                "resources": settings.resources,
                "display": settings.display,
                "run": ex.run.model_copy(
                    update={
                        "max_concurrent": settings.concurrency.max_concurrent,
                        "resource_execution": "adaptive"
                        if settings.concurrency.mode == "adaptive"
                        else "fixed",
                    }
                ),
            }
        )
    from smartsom.config.components import absolute_components

    algorithm = absolute_components(algorithm, paths["algorithm"].parent)
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
        normal = next(
            mode for mode in m.quality_modes if mode.quality_mode_id == "normal"
        )
        if normal.time_scale != 1:
            raise ValueError("Base normal reference mode requires time_scale=1")
        owned = [b for b in factory.buffers if b.machine_id == m.machine_id]
        if {b.role for b in owned} != {"machine_pre", "machine_post"} or len(
            owned
        ) != 2:
            raise ValueError("each Base station requires pre and post storage")
        if any(getattr(b.storage, "capacity", None) is None for b in owned):
            raise ValueError("Base pre/post use finite pooled storage")
    workload = BaseWorkload.model_validate(documents["workload"])
    if documents["algorithm"].get("schema") != "smartsom.algorithm-components/v1":
        raise ValueError(
            "old algorithm configuration: run smartsom config migrate SOURCE --output DEST; resume historical experiments with archived source"
        )
    algorithm = BaseAlgorithm.model_validate(documents["algorithm"])
    algorithm._owner_ids = {
        "machine": tuple(m.machine_id for m in factory.machines),
        "dispatcher": tuple(a.agv_id for a in factory.agvs),
        **{
            group: tuple(b.buffer_id for b in factory.buffers if b.role == role)
            for group, role in (
                ("pre", "machine_pre"),
                ("post", "machine_post"),
                ("input", "system_input"),
            )
        },
    }
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
