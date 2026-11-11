"""Opt-in four-file inputs for the reservation Base development diagnostics."""

import hashlib
import json
import math
import random
from copy import deepcopy
from functools import cached_property, lru_cache
from pathlib import Path
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    field_validator,
    model_serializer,
    model_validator,
)

from smartsom.config.codec import safe_yaml_load
from smartsom.config.evaluation_cases import FixedTestPanel
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


class MachineFaults(Strict):
    enabled: bool = True
    intervals: dict[str, TruncatedExponential] = {
        "low": TruncatedExponential(scale=600, lower=60, upper=1800),
        "medium": TruncatedExponential(scale=300, lower=30, upper=900),
        "high": TruncatedExponential(scale=150, lower=15, upper=450),
    }
    repair: TruncatedExponential = TruncatedExponential(scale=60, lower=10, upper=180)

    @model_validator(mode="after")
    def conditions(self):
        if set(self.intervals) != {"low", "medium", "high"}:
            raise ValueError("three named fault conditions required")
        return self


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
    due_mixture: tuple[tuple[float, float, float], ...] = ()
    fault_intervals: dict[str, TruncatedExponential] = {
        "low": TruncatedExponential(scale=600, lower=60, upper=1800),
        "medium": TruncatedExponential(scale=300, lower=30, upper=900),
        "high": TruncatedExponential(scale=150, lower=15, upper=450),
    }
    repair: TruncatedExponential = TruncatedExponential(scale=60, lower=10, upper=180)
    machine_faults: dict[str, MachineFaults] = Field(default_factory=dict)

    @model_serializer(mode="wrap")
    def omit_empty_fault_overrides(self, handler):
        data = handler(self)
        if not self.machine_faults:
            data.pop("machine_faults", None)
        return data

    @model_validator(mode="after")
    def valid(self):
        if not 0 < self.initial_orders < self.orders:
            raise ValueError("finite arrivals require 0 < initial_orders < orders")
        if set(self.fault_intervals) != {"low", "medium", "high"}:
            raise ValueError("three named fault conditions required")
        for bounds in (self.normal_due, self.urgent_due):
            if not 1 <= bounds[0] <= bounds[1]:
                raise ValueError("due factors must be ordered and at least one")
        if self.due_mixture and (
            not math.isclose(sum(weight for weight, _, _ in self.due_mixture), 1.0)
            or any(
                not (weight > 0 and 1 <= lower <= upper)
                for weight, lower, upper in self.due_mixture
            )
        ):
            raise ValueError(
                "due mixture requires positive weights summing to one and ordered bounds >= 1"
            )
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
    auto_accept_uncontested: bool = False
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
    target_update_clock: Literal["global_gradient_steps", "network_gradient_steps"] = (
        "network_gradient_steps"
    )
    epsilon: float = Field(default=0.1, ge=0, le=1)
    update_trigger: Literal["episode", "transitions", "fragment"] = "episode"
    update_every: int = Field(default=256, gt=0)
    update_amount: Literal["fixed", "ratio"] = "fixed"
    updates_per_transition: float = Field(default=1.0, gt=0, allow_inf_nan=False)
    warmup_transitions: int = Field(default=0, ge=0)
    exploration: Literal["constant", "linear"] = "constant"
    epsilon_final: float = Field(default=0.1, ge=0, le=1)
    epsilon_decay_transitions: int = Field(default=100000, gt=0)
    learning_reward_scale: float = Field(default=1.0, gt=0, allow_inf_nan=False)
    critic_initial_scale: float = Field(default=1.0, gt=0, allow_inf_nan=False)
    initialization: Literal["native"] = "native"
    return_estimator: Literal["episode_return", "gae"] = "episode_return"
    gamma: float = Field(default=1.0, gt=0, le=1)
    gae_lambda: float = Field(default=0.95, ge=0, le=1)


Role = Literal["machine", "dispatcher", "pre", "post", "input"]


class ActionRepresentation(Strict):
    """Explicit action space, with role then stage overrides."""

    default: Literal["direct", "rules"]
    roles: dict[Role, Literal["direct", "rules"]] = Field(default_factory=dict)
    stages: dict[str, Literal["direct", "rules"]] = Field(default_factory=dict)
    rules: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    custom_rules: dict[str, Component] = Field(default_factory=dict)
    merge_aliases: bool = True

    def mode(self, point):
        return self.stages.get(point, self.roles.get(point.split(".")[0], self.default))

    def selected_rules(self, point):
        return self.rules.get(point, self.rules.get(point.split(".")[0]))

    @model_validator(mode="after")
    def valid(self):
        from smartsom.algorithms.base_policy import DECISION_POINTS
        from smartsom.algorithms.rule_actions import RULE_CATALOG

        roles = {p.split(".")[0] for p in DECISION_POINTS}
        if set(self.stages) - DECISION_POINTS or set(self.rules) - (
            DECISION_POINTS | roles
        ):
            raise ValueError("unknown role.stage action representation")
        builtin = {r for rules in RULE_CATALOG.values() for r in rules}
        if any(not n.isidentifier() or n in builtin for n in self.custom_rules):
            raise ValueError(
                "custom rule names must be identifiers and cannot replace builtins"
            )
        used = set()
        for binding, names in self.rules.items():
            points = (
                [binding]
                if binding in DECISION_POINTS
                else [
                    p
                    for p in DECISION_POINTS
                    if p.startswith(binding + ".")
                    and p not in self.rules
                    and self.mode(p) == "rules"
                ]
            )
            if not points or any(self.mode(p) != "rules" for p in points):
                raise ValueError(
                    f"unused rule subset or direct decision point: {binding}"
                )
            if not names or len(names) != len(set(names)):
                raise ValueError(
                    "rule subsets must be nonempty and unique; order is alias priority"
                )
            for point in points:
                allowed = (
                    set(RULE_CATALOG[tuple(point.split("."))])
                    | self.custom_rules.keys()
                )
                if set(names) - allowed:
                    raise ValueError(f"unknown rule for {point}")
            used.update(set(names) & self.custom_rules.keys())
        if used != self.custom_rules.keys():
            raise ValueError(
                "custom rules must be explicitly included in a stage subset"
            )
        return self


class ParameterSharing(Strict):
    default: Literal["shared", "independent"] = "shared"
    roles: dict[Role, Literal["shared", "independent"]] = Field(default_factory=dict)

    def mode(self, role):
        return self.roles.get(role, self.default)


class PolicyDefinition(Strict):
    kind: Literal["learning", "policy"]
    parameter_sharing: ParameterSharing = Field(default_factory=ParameterSharing)
    split_stages: tuple[Role, ...] = ()
    frozen_stages: tuple[str, ...] = ()
    role_params: dict[Role, dict] = Field(default_factory=dict)
    implementation: Component | None = None
    learner: Component | None = None
    observation: Component
    critic_observation: Component = Field(
        default_factory=lambda: Component(
            target="smartsom.observations.critic.create_observation"
        )
    )
    encoder: Component | None = None
    network: Component | None = None

    @model_validator(mode="after")
    def contract(self):
        if len(set(self.split_stages)) != len(self.split_stages):
            raise ValueError("split_stages must be unique")
        if self.split_stages and (self.kind != "learning" or self.network is None):
            raise ValueError("split_stages requires a full learning network")
        from smartsom.algorithms.base_policy import DECISION_POINTS

        if len(set(self.frozen_stages)) != len(self.frozen_stages) or any(
            point not in DECISION_POINTS or point.split(".")[0] not in self.split_stages
            for point in self.frozen_stages
        ):
            raise ValueError("frozen_stages must name unique, separately owned stages")
        if "machine" in self.split_stages:
            raise ValueError("machine has only one decision stage")
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
            settings = LearnerParameters.model_validate(self.learner.params)
            dqn = self.learner.target == "smartsom.learning.dqn.create_learner"
            if settings.update_trigger == ("fragment" if dqn else "transitions"):
                raise ValueError("update_trigger does not apply to this learner")
            if not dqn and (
                settings.update_amount != "fixed"
                or settings.warmup_transitions
                or settings.exploration != "constant"
            ):
                raise ValueError(
                    "replay update ratio, warmup and exploration are DQN settings"
                )
            for overrides in self.role_params.values():
                if "update_trigger" in overrides:
                    raise ValueError(
                        "update_trigger is global and cannot be overridden per role"
                    )
                if "initialization" in overrides:
                    raise ValueError(
                        "initialization is global and cannot be overridden per role"
                    )
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


class CentralSettings(Strict):
    # CONWIP release limit: input releases are withheld while this many jobs
    # are between system input and output. Without it, centralized release can
    # fill every station and lock the cross-station buffer cycle.
    wip_limit: int = Field(gt=0)


class BaseAlgorithm(Strict):
    _owner_ids: dict = PrivateAttr(default_factory=dict)
    schema_id: Literal["smartsom.algorithm-components/v2"] = Field(alias="schema")
    control: Literal["base", "centralized"] = "base"
    central: CentralSettings | None = None
    decision_protocol: DecisionProtocol | None = None
    production: ProductionPolicies
    action_representation: ActionRepresentation
    traffic: Traffic = Field(default_factory=Traffic)
    reward: Component = Field(
        default_factory=lambda: Component(
            target="smartsom.rewards.default.create_reward"
        )
    )

    @model_serializer(mode="wrap")
    def omit_unset_central(self, handler):
        # Unset keeps scientific identities of runs and checkpoints made before.
        data = handler(self)
        if self.central is None:
            data.pop("central", None)
        return data

    @model_validator(mode="after")
    def valid(self):
        from smartsom.algorithms.base_policy import DECISION_POINTS

        points = (
            {"central.assign"} if self.control == "centralized" else DECISION_POINTS
        )
        if set(self.production.bindings) - points:
            raise ValueError("unknown role.stage policy binding")
        protocol = self.decision_protocol or DecisionProtocol()
        if self.central is not None and self.control != "centralized":
            raise ValueError("central settings require centralized control")
        if self.control == "centralized":
            if (
                self.action_representation.model_dump()
                != ActionRepresentation(default="direct").model_dump()
            ):
                raise ValueError(
                    "centralized control requires explicit direct actions without distributed overrides"
                )
            if any((protocol.pre, protocol.post, protocol.dispatcher)):
                raise ValueError(
                    "centralized control rejects distributed decision_protocol"
                )
            if self.learning_policy and (
                self.learning_policy.parameter_sharing != ParameterSharing()
                or self.learning_policy.role_params
                or self.learning_policy.split_stages
            ):
                raise ValueError("centralized learning has one shared factory subject")
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
        for name, spec in self.action_representation.custom_rules.items():
            yield f"action_representation.custom_rules.{name}", spec
        yield "traffic.mover", self.traffic.mover
        yield "reward", self.reward
        for name, policy in self.production.policies.items():
            for field in (
                "implementation",
                "learner",
                "observation",
                "critic_observation",
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
    seconds: float | None = Field(default=120, gt=0, allow_inf_nan=False)
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
    split: Literal["periodic_evaluation", "test"] = "periodic_evaluation"
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
        if self.seconds is None and (
            self.task not in {"train", "evaluate"} or self.tune != "off"
        ):
            raise ValueError(
                "seconds=null supports native train/evaluate only; smoke, calibration and Tune require a finite wall budget"
            )
        if not self.methods or len(set(self.methods)) != len(self.methods):
            raise ValueError("methods must be nonempty and unique")
        return self


class PeriodicEvaluation(Strict):
    unit: Literal["episodes", "physical_ticks"] = "episodes"
    milestones: tuple[int, ...] = ()
    modes: tuple[Literal["sample", "argmax"], ...] = ("sample", "argmax")
    sample_action_seeds: int = Field(default=2, gt=0)
    selection: Literal[
        "last", "sample_complete_cost", "sample_delivery_cost", "argmax_complete_cost"
    ] = "last"
    select_initial: bool = False

    @model_validator(mode="after")
    def valid_schedule(self):
        if (
            any(x < 0 for x in self.milestones)
            or tuple(sorted(set(self.milestones))) != self.milestones
        ):
            raise ValueError(
                "periodic_evaluation milestones must be strictly increasing nonnegative nodes"
            )
        if not self.modes or len(set(self.modes)) != len(self.modes):
            raise ValueError("periodic_evaluation modes must be nonempty and unique")
        decoder = "argmax" if self.selection == "argmax_complete_cost" else "sample"
        if self.selection != "last" and (
            not self.milestones or decoder not in self.modes
        ):
            raise ValueError(
                f"{decoder} selection requires {decoder} periodic_evaluation milestones"
            )
        return self


class ReservationBatch(Strict):
    stage: int = Field(default=0, ge=0)
    parallel_files: int = Field(default=1, gt=0)


TrackingOptions = Tracking


class ReplaySettings(Strict):
    @field_validator("train", "periodic_evaluation", "evaluation", mode="before")
    @classmethod
    def yaml_off(cls, value):
        # PyYAML's YAML 1.1 loader reads an unquoted `off` as False.
        return "off" if value is False else value

    train: Literal["off", "first", "all"] = "off"
    periodic_evaluation: Literal["off", "first", "all"] = "off"
    evaluation: Literal["off", "first", "all"] = "off"


class SamplingSettings(Strict):
    num_envs: int = Field(default=1, strict=True, gt=0, le=64)
    processes: int = Field(default=0, strict=True, ge=0, le=64)
    fragment_ticks: int = Field(default=256, strict=True, gt=0)

    @model_validator(mode="after")
    def layout(self):
        if self.num_envs == 1 and self.processes:
            raise ValueError("single-environment sampling uses processes: 0")
        if self.processes not in (0, self.num_envs):
            raise ValueError("sampling.processes must be zero or equal num_envs")
        if self.num_envs > 1 and not self.processes:
            raise ValueError("multiple sampling environments require processes")
        return self


class BaseExperiment(Strict):
    sampling: SamplingSettings = Field(default_factory=SamplingSettings)
    schema_id: Literal["smartsom.resolved-experiment/v1"] = Field(alias="schema")
    factory: str
    workload: str
    algorithm: str
    condition_scenarios: dict[Literal["normal", "low", "medium", "high"], str] = Field(
        default_factory=dict
    )
    _condition_inputs: dict = PrivateAttr(default_factory=dict)

    @field_validator("condition_scenarios")
    @classmethod
    def four_condition_scenarios(cls, value):
        if value and (
            set(value) != {"normal", "low", "medium", "high"} or not all(value.values())
        ):
            raise ValueError("condition_scenarios requires all four fault conditions")
        return value

    @model_serializer(mode="wrap")
    def omit_empty_condition_scenarios(self, handler):
        data = handler(self)
        if not self.condition_scenarios:
            data.pop("condition_scenarios", None)
        if self.test is None:
            data.pop("test", None)
        if self.checkpoint_retention == "all":
            data.pop("checkpoint_retention", None)
        return data

    def inputs_for(self, condition, factory, workload):
        if self.condition_scenarios:
            return self._condition_inputs[condition]
        return factory, workload

    def with_execution(self, settings):
        """Validate the complete effective config, retaining parsed scene inputs."""
        values = self.model_dump(mode="python", by_alias=True)
        for key in ("launch", "tracking", "concurrency", "resources", "display"):
            values[key] = getattr(settings, key).model_dump(mode="python")
        values["run"].update(
            max_concurrent=settings.concurrency.max_concurrent,
            resource_execution="adaptive"
            if settings.concurrency.mode == "adaptive"
            else "fixed",
        )
        effective = type(self).model_validate(values)
        effective._condition_inputs = self._condition_inputs
        return effective

    def validate_execution(self):
        if self.concurrency.evaluation_processes > 1 and (
            self.run.tune != "off" or self.evaluation.recovery.action != "disabled"
        ):
            raise ValueError(
                "parallel evaluation requires native execution and disabled recovery"
            )
        if self.resources.guard is not None and (
            self.run.tune != "off"
            or self.run.task not in {"train", "evaluate", "calibrate"}
        ):
            raise ValueError(
                "resources.guard requires native train/evaluate/calibrate with tune=off"
            )
        return self

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
    checkpoint_retention: Literal["all", "best_last"] = "all"
    periodic_evaluation: PeriodicEvaluation = Field(default_factory=PeriodicEvaluation)
    evaluation: EvaluationProtocol = Field(default_factory=EvaluationProtocol)
    train_seeds: tuple[int, ...] = (101, 102, 103)
    case_sets: dict[Literal["periodic_evaluation", "test"], str] = Field(
        default_factory=dict
    )
    case_set_sha256: dict[Literal["periodic_evaluation", "test"], str] = Field(
        default_factory=dict
    )
    test: FixedTestPanel | None = None

    @model_validator(mode="before")
    @classmethod
    def inline_test_count(cls, data):
        if isinstance(data, dict) and data.get("test") is not None:
            if "test" in data.get("case_sets", {}):
                raise ValueError(
                    "test cannot combine inline cases with a case_sets reference"
                )
            panel = FixedTestPanel.model_validate(data["test"])
            count = len(panel.cases)
            if "test_cases" in data and data["test_cases"] != count:
                raise ValueError("test_cases differs from inline test panel size")
            data = {**data, "test_cases": count}
        return data

    periodic_evaluation_cases: int = Field(default=32, gt=0)
    test_cases: int = Field(default=64, gt=0)
    replay: ReplaySettings = Field(default_factory=ReplaySettings)
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
        if self.test is not None and self.evaluation.recovery.action != "disabled":
            raise ValueError("inline test panels do not support controlled recovery")
        if self.run.seconds is None and self.evaluation.recovery.action != "disabled":
            raise ValueError("recovery protocol requires a finite wall budget")
        if self.evaluation.checkpoints and (
            not self.final_evaluation
            or self.train_episodes is None
            or any(c.episode > self.train_episodes for c in self.evaluation.checkpoints)
        ):
            raise ValueError(
                "evaluation checkpoints require final_evaluation and episode nodes within train_episodes"
            )
        if self.checkpoint_retention == "best_last":
            if self.periodic_evaluation.selection == "last":
                raise ValueError(
                    "best_last retention requires periodic_evaluation best selection"
                )
            if any(
                c.episode != self.train_episodes for c in self.evaluation.checkpoints
            ):
                raise ValueError(
                    "best_last retains only the final prescribed evaluation checkpoint"
                )
        if not self.training_conditions or len(set(self.training_conditions)) != len(
            self.training_conditions
        ):
            raise ValueError("training_conditions must be nonempty and unique")
        if self.periodic_evaluation.unit == "physical_ticks":
            if self.train_episodes is not None or self.budget_boundary != "cut":
                raise ValueError(
                    "physical tick periodic_evaluation requires a tick budget and budget_boundary: cut"
                )
            if any(n > self.train_ticks for n in self.periodic_evaluation.milestones):
                raise ValueError(
                    "periodic_evaluation node exceeds training tick budget"
                )
        elif self.train_episodes is not None and any(
            n > self.train_episodes for n in self.periodic_evaluation.milestones
        ):
            raise ValueError("periodic_evaluation node exceeds training episode budget")
        if self.run.task == "calibrate" and self.run.max_drain < self.drain_ticks:
            raise ValueError("max_drain must be at least the initial drain_ticks")
        return self.validate_execution()


def seed_for(seed, *parts):
    return int.from_bytes(
        hashlib.sha256(repr((seed, parts)).encode()).digest()[:8], "big"
    )


def load_inputs(path, *, execution_settings=None):
    path = Path(path).resolve()
    document = safe_yaml_load(path.read_text())
    resolved = document.get("schema") == "smartsom.resolved-experiment/v1"
    if not resolved and (
        "scenario" not in document or {"factory", "workload"} & document.keys()
    ):
        raise ValueError(
            "authored experiments require Scenario; old inputs require archived source"
        )
    scene_input = not resolved
    paths = {
        k: (path.parent / document[k]).resolve()
        for k in (
            ("factory", "workload", "algorithm")
            if resolved
            else ("scenario", "algorithm")
        )
    }
    paths.update(
        {
            f"condition_{condition}": (path.parent / reference).resolve()
            for condition, reference in document.get("condition_scenarios", {}).items()
        }
    )
    documents = {
        "experiment": document,
        **{name: safe_yaml_load(source.read_text()) for name, source in paths.items()},
    }
    if scene_input:
        from smartsom.config.scenario_authoring import expand_documents

        expand_documents(documents)
    ex, factory, workload, algorithm = parse_inputs(documents)
    execution = path.parent.parent / "execution.json"
    if execution_settings is not None or (
        path.parent.name == "inputs" and execution.is_file()
    ):
        from smartsom.config.launch import Profile

        settings = Profile.model_validate(
            execution_settings
            if execution_settings is not None
            else json.loads(execution.read_text())["settings"]
        )
        ex = ex.with_execution(settings)
    from smartsom.config.components import absolute_components

    algorithm = absolute_components(algorithm, paths["algorithm"].parent)
    return ex, factory, workload, algorithm, {"experiment": path, **paths}


@lru_cache(maxsize=32)
def _cached_condition_scene(content):
    from smartsom.config.scenario_authoring import ScenarioFile

    envelope, workload = ScenarioFile.model_validate_json(content).expand()
    return envelope.factory, workload


def _expand_condition_scene(content):
    # Pydantic frozen models still contain mutable dictionaries. Never expose
    # cache-owned objects to callers or share them between condition bindings.
    return deepcopy(_cached_condition_scene(content))


def parse_inputs(documents):
    """One typed parser shared by file loading and detached compilation."""
    experiment = BaseExperiment.model_validate(documents["experiment"])
    factory = parse_factory_design_file(documents["factory"]).factory
    validate_base_factory(factory)
    workload = BaseWorkload.model_validate(documents["workload"])
    if set(workload.machine_faults) - {m.machine_id for m in factory.machines}:
        raise ValueError("machine fault override references an unknown machine")
    if documents["algorithm"].get("schema") != "smartsom.algorithm-components/v2":
        raise ValueError(
            "old algorithm configuration requires archived source; author current component definitions"
        )
    algorithm = BaseAlgorithm.model_validate(documents["algorithm"])
    learning = algorithm.learning_policy
    active_sampling = experiment.sampling.num_envs > 1
    if learning:
        active_sampling |= (
            learning.learner.params.get("update_trigger", "episode") != "episode"
        )
    if active_sampling:
        if learning is None or learning.learner.target not in {
            "smartsom.learning.ppo.create_learner",
            "smartsom.learning.dqn.create_learner",
        }:
            raise ValueError(
                "training sampling controls require a native PPO or DQN learner"
            )
        if experiment.run.tune != "off":
            raise ValueError(
                "training sampling currently requires native execution with tune=off"
            )
        if (
            experiment.resources.cpus is not None
            and experiment.sampling.processes + 1 > experiment.resources.cpus
        ):
            raise ValueError(
                "sampling workers plus learner exceed experiment CPU budget"
            )
    if experiment.condition_scenarios:
        from smartsom.config.codec import primitive

        jobs = workload.model_dump(
            mode="json", exclude={"fault_intervals", "repair", "machine_faults"}
        )
        for condition in experiment.condition_scenarios:
            selected_factory, selected_workload = _expand_condition_scene(
                json.dumps(documents[f"condition_{condition}"], sort_keys=True)
            )
            if (
                primitive(selected_factory) != primitive(factory)
                or selected_workload.model_dump(
                    mode="json", exclude={"fault_intervals", "repair", "machine_faults"}
                )
                != jobs
            ):
                raise ValueError(
                    "condition_scenarios may differ only in machine breakdown parameters; factory and jobs must match"
                )
            experiment._condition_inputs[condition] = (
                selected_factory,
                selected_workload,
            )
        if experiment.evaluation.recovery.action != "disabled":
            raise ValueError("condition_scenarios does not support controlled recovery")
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


def validate_base_factory(factory):
    """Current eligibility boundary, shared by execution and case previews."""
    if factory.inspection_stations or factory.scrap_bins:
        raise ValueError("Base supports final Output inspection only")
    for m in factory.machines:
        if "normal" not in {mode.quality_mode_id for mode in m.quality_modes}:
            raise ValueError("Base requires machines with a normal reference mode")
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


def _nominal_layout(factory):
    pre = {
        b.machine_id: b.buffer_id for b in factory.buffers if b.role == "machine_pre"
    }
    post = {
        b.machine_id: b.buffer_id for b in factory.buffers if b.role == "machine_post"
    }
    source = next(b.buffer_id for b in factory.buffers if b.role == "system_input")
    output = next(b.buffer_id for b in factory.buffers if b.role == "system_output")
    machines = {}
    for machine in factory.machines:
        for kind in machine.operation_types:
            machines.setdefault(kind, []).append(machine)
    return pre, post, source, output, machines


def nominal_baseline(factory, steps, routes, *, layout=None):
    pre, post, source, output, machines = (
        _nominal_layout(factory) if layout is None else layout
    )
    costs = {None: 0.0}
    previous_type = None
    for step in steps:
        next_costs = {}
        for machine in machines.get(step.operation_type, ()):
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
    layout = _nominal_layout(factory)
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
        draw = rng.random()
        if workload.due_mixture:
            urgent = False
            cumulative = 0.0
            bounds = workload.due_mixture[-1][1:]
            for weight, lower, upper in workload.due_mixture:
                cumulative += weight
                if draw < cumulative:
                    bounds = lower, upper
                    break
        else:
            urgent = draw < workload.urgent_probability
            bounds = workload.urgent_due if urgent else workload.normal_due
        factor = rng.uniform(*bounds)
        due = release + math.ceil(
            factor * nominal_baseline(factory, steps, routes, layout=layout)
        )
        demands.append(
            Demand(f"order_{index:04d}", tuple(steps), release, due, rush=urgent)
        )
    horizon = demands[-1].release_at + drain_ticks
    outages = []
    if condition != "normal":
        for machine in factory.machines:
            fault = workload.machine_faults.get(machine.machine_id)
            if fault is not None and not fault.enabled:
                continue
            distribution = (fault.intervals if fault else workload.fault_intervals)[
                condition
            ]
            repair_distribution = fault.repair if fault else workload.repair
            rng_fault = random.Random(seed_for(seed, "faults", machine.machine_id))
            t = 0
            while t < horizon:
                t += distribution.draw(rng_fault)
                repair = repair_distribution.draw(rng_fault)
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
