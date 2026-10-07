"""Strict four-file authoring contracts; no framework imports or execution."""

from typing import Annotated, Literal

from pydantic import Field, model_validator

from smartsom.config.diagnostics import DiagnosticsOptions
from smartsom.config.experiment import (
    CheckpointOptions,
    EditableModel,
    LoggingOptions,
    OutputOptions,
    RuntimeOptions,
    Seed,
    ValidationOptions,
)
from smartsom.config.experiment_v3 import EvaluationOptionsV3
from smartsom.config.extensions import ExtensionRef, RewardSpec
from smartsom.config.policies import (
    ModelImplementation,
    NewImplementation,
    RuleImplementation,
)
from smartsom.config.travel_time import TransportSettings

AgentPolicy = Annotated[
    RuleImplementation | NewImplementation | ModelImplementation,
    Field(discriminator="kind"),
]
ROLES = {"machine", "buffer", "dispatcher", "mover"}


class EntityAgent(EditableModel):
    group: str
    policy: AgentPolicy


class RoleAgents(EditableModel):
    default: AgentPolicy
    group: str | None = None
    overrides: dict[str, EntityAgent] = Field(default_factory=dict)
    reward: ExtensionRef | None = None


class Learner(EditableModel):
    backend: Literal["sb3", "rllib"]
    algorithm: Literal["ppo", "dqn"]
    gamma: Annotated[float, Field(gt=0, le=1)] = 0.99
    parameters: dict = Field(default_factory=dict)
    reward: RewardSpec = Field(default_factory=RewardSpec)

    @model_validator(mode="after")
    def parameters_contract(self):
        from smartsom.config.experiment_v3 import DQNParameters, PPOParameters

        cls = PPOParameters if self.algorithm == "ppo" else DQNParameters
        from smartsom.config.codec import canonical_json, primitive

        object.__setattr__(
            self,
            "parameters",
            primitive(cls.model_validate_json(canonical_json(self.parameters))),
        )
        return self


class AlgorithmV2(EditableModel):
    schema_id: Literal["smartsom.algorithm/v2"] = Field(alias="schema")
    mode: Literal["rules", "central", "resource"]
    learner: Learner | None = None
    agents: dict[str, RoleAgents] = Field(default_factory=dict)
    controller: AgentPolicy | None = None
    pickup_matching: Literal["global_optimal", "priority_greedy"] = "global_optimal"

    @model_validator(mode="after")
    def mode_contract(self):
        if self.mode == "central":
            if self.controller is None or self.agents:
                raise ValueError("central requires an exclusive controller")
            if self.controller.kind == "rule":
                raise ValueError("central controller requires a PPO model or new_model")
            if self.learner is None or self.learner.algorithm != "ppo":
                raise ValueError("central requires an explicit PPO learner/backend")
        else:
            if self.controller is not None or set(self.agents) != ROLES:
                raise ValueError("rules/resource require exactly four agent roles")
            if self.mode == "resource" and (
                self.learner is None or self.learner.backend != "rllib"
            ):
                raise ValueError("resource requires an explicit RLlib learner")
        if self.mode == "rules":
            if self.learner is not None:
                raise ValueError("rules cannot declare a learner")
            if any(role.reward is not None for role in self.agents.values()):
                raise ValueError("role reward extensions require a learner")
            if any(
                p.kind != "rule"
                for role in self.agents.values()
                for p in [role.default, *(e.policy for e in role.overrides.values())]
            ):
                raise ValueError("rules mode accepts rule policies only")
        return self


class TrainingV4(EditableModel):
    total_ticks: Annotated[int, Field(gt=0)] = 4096
    ticks_per_update: Annotated[int, Field(gt=0)] = 256
    max_ticks: Annotated[int, Field(gt=0)] = 10000
    record_initial: bool = False


class EnvironmentV4(EditableModel):
    mode: Literal["static", "dynamic", "finite"] = "finite"
    tick_limit: Annotated[int, Field(gt=0)] | None = None
    processing_low: Annotated[float, Field(gt=0)] = 1.0
    processing_high: Annotated[float, Field(gt=0)] = 1.0
    reward_time_scale: Annotated[int, Field(gt=0)] = 100
    quality_probability_visibility: Literal["hidden", "public"] = "public"
    transport: TransportSettings = Field(default_factory=TransportSettings)
    processing_rounding: Literal["half_up", "ceil"] = "ceil"


class RuntimeV4(RuntimeOptions):
    # Concurrency belongs to execution, not the scientific sampler contract.
    max_concurrent: Literal[1] = 1
    environment: EnvironmentV4 = Field(default_factory=EnvironmentV4)


class ExecutionV4(EditableModel):
    executor: Literal["native", "tune"] = "native"
    background: bool = False
    max_concurrent: Annotated[int, Field(gt=0)] = 1
    tuning: Literal["off", "recommend", "auto"] = "off"
    mode: Literal["balanced", "performance"] = "balanced"
    scheduling: Literal["adaptive", "fixed"] = "adaptive"
    calibration_level: Literal["off", "quick", "full"] = "quick"
    calibration_candidate: Annotated[str, Field(min_length=1)] = "latest"
    calibration_seconds: Annotated[float, Field(gt=0)] | None = None
    preflight: Literal["quick", "full"] = "quick"
    preflight_coverage: Literal["each", "representative"] = "each"


class MatrixV4(EditableModel):
    factories: Annotated[tuple[str, ...], Field(min_length=1)]
    workloads: Annotated[tuple[str, ...], Field(min_length=1)]
    algorithms: Annotated[tuple[str, ...], Field(min_length=1)] | None = None
    seeds: Annotated[tuple[Seed, ...], Field(min_length=1)] | None = None


class BatchGateV4(EditableModel):
    """Execution gate applied to this file after every evaluation case commits."""

    min_cases: Annotated[int, Field(gt=0)]
    min_deliveries_each: Annotated[int, Field(ge=0)] = 0
    require_first_pickup: bool = False


class BatchV4(EditableModel):
    """Directory-run ordering only; it does not change a scientific entry."""

    stage: Annotated[int, Field(ge=0)]
    parallel_files: Annotated[int, Field(gt=0)] = 1
    gate: BatchGateV4 | None = None


class ValidationV4(ValidationOptions):
    seed: Seed = Field(default=303, exclude=True)
    best_mode: Literal[
        "completion_first", "all_complete", "custom", "completion_delivery_return"
    ] = "completion_first"

    @model_validator(mode="after")
    def fixed_completion_rank(self):
        if self.best_mode == "completion_delivery_return" and (
            self.metric != "makespan"
            or self.direction != "min"
            or self.failure_policy is not None
            or self.min_delta != 0
        ):
            raise ValueError(
                "completion_delivery_return uses a fixed ranking without "
                "metric, direction, failure_policy or min_delta overrides"
            )
        return self


class EvaluationV4(EvaluationOptionsV3):
    seed: Seed = Field(default=202, exclude=True)


class ExperimentV4(EditableModel):
    schema_id: Literal["smartsom.experiment-config/v4"] = Field(alias="schema")
    task: Literal["train", "evaluate", "train-evaluate"]
    factory: str | None = None
    workload: str | None = None
    algorithm: str | None = None
    matrix: MatrixV4 | None = None
    batch: BatchV4 | None = None
    seed: Seed = 101
    data_seed: Seed = 0
    training: TrainingV4 | None = None
    runtime: RuntimeV4 = Field(default_factory=RuntimeV4)
    execution: ExecutionV4 = Field(default_factory=ExecutionV4)
    diagnostics: DiagnosticsOptions = Field(default_factory=DiagnosticsOptions)
    logging: LoggingOptions = Field(default_factory=LoggingOptions)
    validation: ValidationV4 = Field(default_factory=ValidationV4)
    evaluation: EvaluationV4 = Field(default_factory=EvaluationV4)
    checkpointing: CheckpointOptions = Field(default_factory=CheckpointOptions)
    output: OutputOptions = Field(default_factory=OutputOptions)

    @model_validator(mode="after")
    def data_roots(self):
        for options in (self.validation, self.evaluation):
            if "seed" in options.model_fields_set:
                raise ValueError(
                    "v4 data seeds are derived from data_seed; phase seed overrides are unsupported"
                )
        return self

    @model_validator(mode="after")
    def explicit_inputs(self):
        if self.matrix is None:
            if self.factory is None or self.workload is None:
                raise ValueError("single experiment needs factory and workload")
        elif self.factory is not None or self.workload is not None:
            raise ValueError("matrix and single factory/workload references conflict")
        has_algorithm = self.algorithm is not None
        has_algorithm_axis = (
            self.matrix is not None and self.matrix.algorithms is not None
        )
        if has_algorithm == has_algorithm_axis:
            raise ValueError("provide exactly one of algorithm or matrix.algorithms")
        if self.task != "evaluate" and self.training is None:
            raise ValueError("training tasks require an explicit training budget")
        if self.task == "evaluate" and self.matrix and self.matrix.seeds is not None:
            raise ValueError("evaluation cannot declare a training-seed axis")
        if self.validation.scenarios or self.evaluation.scenarios:
            raise ValueError(
                "v4 cases come from the selected workload; no scenario files"
            )
        return self
