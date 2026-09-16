"""Strict authoring envelopes; domain objects retain their stdlib contracts."""

from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    model_serializer,
    model_validator,
)

from smartsom.config.extensions import ExtensionSpec
from smartsom.dispatch import SemanticAction
from smartsom.domain import ExecutionSchedule, FactorySpec, WorkloadInstance
from smartsom.domain.arrivals import DecisionTrigger
from smartsom.domain.machine_events import MachineOutagePlan
from smartsom.domain.processing_times import ProcessingTimePlan
from smartsom.domain.production import JointCommand
from smartsom.domain.quality import ProbabilityVisibility, QualityDrawPlan
from smartsom.learning.episode import EpisodeLimits
from smartsom.learning.projection import ProjectionSpec
from smartsom.learning.resources import ResourceProjectionSpec
from smartsom.workloads import StaticFJSPProfile, StaticJSPProfile
from smartsom.workloads.arrivals import UniformReleaseProfile
from smartsom.workloads.fjs import ImportProvenance
from smartsom.workloads.machine_events import MachineEventProfile
from smartsom.workloads.processing_times import (
    GENERATOR_VERSION,
    ProcessingDraw,
    UniformMultiplierProfile,
    actual_ticks,
)
from smartsom.workloads.quality import operation_draw

Seed = Annotated[int, Field(ge=0, lt=2**64)]
Reference = Annotated[str, StringConstraints(min_length=1, pattern=r"\S")]
SHA256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]


class StrictModel(BaseModel):
    model_config = ConfigDict(
        strict=True, extra="forbid", frozen=True, validate_default=True
    )


class FactoryFile(StrictModel):
    schema_id: Literal["smartsom.factory/v1"] = Field(alias="schema")
    factory: FactorySpec


class ProfileFile(StrictModel):
    schema_id: Literal["smartsom.workload-profile/v1"] = Field(alias="schema")
    generator: Literal["static_jsp_v1", "static_fjsp_v1"]
    profile: StaticJSPProfile | StaticFJSPProfile

    @model_validator(mode="after")
    def matching_profile(self):
        expected = (
            StaticJSPProfile if self.generator == "static_jsp_v1" else StaticFJSPProfile
        )
        if not isinstance(self.profile, expected):
            raise ValueError("profile must match its generator")
        return self


class GenerationProvenance(StrictModel):
    generator: Literal["static_jsp_v1", "static_fjsp_v1"]
    generator_version: Literal["1"]
    profile_sha256: SHA256
    effective_seed: Seed


class InstanceFile(StrictModel):
    schema_id: Literal["smartsom.workload-instance/v1"] = Field(alias="schema")
    workload: WorkloadInstance
    content_sha256: SHA256 | None = None
    provenance: GenerationProvenance | ImportProvenance | None = None


class WorkloadSource(StrictModel):
    kind: Literal["profile", "instance"]
    path: Reference


class FixedArrivals(StrictModel):
    kind: Literal["fixed"]
    path: Reference


class GeneratedArrivals(StrictModel):
    kind: Literal["uniform_release_v1"]
    profile: UniformReleaseProfile


class ArrivalProvenance(StrictModel):
    generator: Literal["uniform_release_v1"] = "uniform_release_v1"
    generator_version: Literal["1"] = "1"
    profile_sha256: SHA256
    effective_seed: Seed


class FixedProcessingTimes(StrictModel):
    kind: Literal["fixed"]
    path: Reference


class GeneratedProcessingTimes(StrictModel):
    kind: Literal["uniform_multiplier"]
    profile: UniformMultiplierProfile = Field(default_factory=UniformMultiplierProfile)


class ProcessingProvenance(StrictModel):
    generator: Literal["uniform_multiplier"] = "uniform_multiplier"
    generator_version: Literal["smartsom.processing-time/v1"] = GENERATOR_VERSION
    profile: UniformMultiplierProfile
    profile_sha256: SHA256
    effective_seed: Seed
    draws: tuple[ProcessingDraw, ...]


class ProcessingTimeFile(StrictModel):
    schema_id: Literal["smartsom.processing-times/v1"] = Field(alias="schema")
    processing_times: ProcessingTimePlan
    content_sha256: SHA256 | None = None
    provenance: ProcessingProvenance | None = None

    @model_validator(mode="after")
    def verified_content(self):
        from smartsom.config.codec import digest

        if self.content_sha256 is not None and self.content_sha256 != digest(
            self.processing_times
        ):
            raise ValueError("processing time content_sha256 mismatch")
        provenance = self.provenance
        if provenance is not None:
            if provenance.profile_sha256 != digest(provenance.profile):
                raise ValueError("processing profile_sha256 mismatch")
            draws = {
                (row.operation_id, row.processing_mode_id): row.draw
                for row in provenance.draws
            }
            expected = {
                (row.operation_id, row.processing_mode_id)
                for row in self.processing_times.modes
            }
            if len(draws) != len(provenance.draws) or draws.keys() != expected:
                raise ValueError("processing draw coverage mismatch")
            for row in self.processing_times.modes:
                draw = draws[(row.operation_id, row.processing_mode_id)]
                if (
                    actual_ticks(row.nominal_ticks, provenance.profile, draw)
                    != row.actual_ticks
                ):
                    raise ValueError(
                        "processing provenance disagrees with realized duration"
                    )
        return self


class FixedMachineEvents(StrictModel):
    kind: Literal["fixed"]
    path: Reference


class GeneratedMachineEvents(StrictModel):
    kind: Literal["exponential_uptime_v1"]
    profile: MachineEventProfile


class MachineEventProvenance(StrictModel):
    generator: Literal["exponential_uptime_v1"] = "exponential_uptime_v1"
    generator_version: Literal["smartsom.machine-events/v1"] = (
        "smartsom.machine-events/v1"
    )
    profile: MachineEventProfile
    profile_sha256: SHA256
    effective_seed: Seed


class MachineEventFile(StrictModel):
    schema_id: Literal["smartsom.machine-events/v1"] = Field(alias="schema")
    machine_events: MachineOutagePlan
    content_sha256: SHA256 | None = None
    provenance: MachineEventProvenance | None = None

    @model_validator(mode="after")
    def verified_content(self):
        from smartsom.config.codec import digest

        if self.content_sha256 is not None and self.content_sha256 != digest(
            self.machine_events
        ):
            raise ValueError("machine event content_sha256 mismatch")
        if self.provenance is not None and self.provenance.profile_sha256 != digest(
            self.provenance.profile
        ):
            raise ValueError("machine event profile_sha256 mismatch")
        return self


class GeneratedQuality(StrictModel):
    kind: Literal["independent_operation_v1"]
    probability_visibility: ProbabilityVisibility = "public"


class FixedQuality(StrictModel):
    kind: Literal["fixed"]
    path: Reference
    probability_visibility: ProbabilityVisibility = "public"


class QualityProvenance(StrictModel):
    generator: Literal["independent_operation_v1"] = "independent_operation_v1"
    generator_version: Literal["smartsom.quality/v1"] = "smartsom.quality/v1"
    effective_seed: Seed


class QualityFile(StrictModel):
    schema_id: Literal["smartsom.quality-draws/v1"] = Field(alias="schema")
    draws: QualityDrawPlan
    content_sha256: SHA256 | None = None
    provenance: QualityProvenance | None = None

    @model_validator(mode="after")
    def verified_content(self):
        from smartsom.config.codec import digest

        if self.content_sha256 is not None and self.content_sha256 != digest(
            self.draws
        ):
            raise ValueError("quality content_sha256 mismatch")
        if self.provenance is not None and any(
            row.draw != operation_draw(self.provenance.effective_seed, row.operation_id)
            for row in self.draws.operations
        ):
            raise ValueError("quality provenance disagrees with realized draws")
        return self


class FixedMatrixTransport(StrictModel):
    kind: Literal["fixed_matrix"]


class LimitedBuffers(StrictModel):
    kind: Literal["limited"]


class ExecutionScheduleFile(StrictModel):
    schema_id: Literal[
        "smartsom.execution-schedule/v1", "smartsom.execution-schedule/v2"
    ] = Field(alias="schema")
    execution_schedule: ExecutionSchedule

    @model_validator(mode="after")
    def matching_version(self):
        if (
            self.schema_id
            != f"smartsom.execution-schedule/v{self.execution_schedule.version}"
        ):
            raise ValueError("execution schedule version disagrees with schema")
        return self

    @model_serializer
    def serialize(self):
        from smartsom.config.codec import primitive

        return {
            "schema": self.schema_id,
            "execution_schedule": primitive(self.execution_schedule),
        }


class SharedHoldingBuffer(StrictModel):
    kind: Literal["shared"]


class ScenarioFile(StrictModel):
    schema_id: Literal["smartsom.scenario/v1"] = Field(alias="schema")
    factory: Reference
    workload: WorkloadSource
    modules: Annotated[tuple[str, ...], Field(max_length=0)] = ()
    transport: FixedMatrixTransport | None = None
    buffers: LimitedBuffers | None = None
    holding_buffer: SharedHoldingBuffer | None = Field(
        default=None, exclude_if=lambda v: v is None
    )
    quality: (
        Annotated[GeneratedQuality | FixedQuality, Field(discriminator="kind")] | None
    ) = None
    visibility: Literal["decision_context", "full_static"] = "decision_context"
    termination: Literal["all_jobs_complete"] = "all_jobs_complete"
    arrivals: (
        Annotated[FixedArrivals | GeneratedArrivals, Field(discriminator="kind")] | None
    ) = None
    decision_trigger: DecisionTrigger = "dispatch_available"
    processing_time: (
        Annotated[
            FixedProcessingTimes | GeneratedProcessingTimes, Field(discriminator="kind")
        ]
        | None
    ) = None
    machine_events: (
        Annotated[
            FixedMachineEvents | GeneratedMachineEvents, Field(discriminator="kind")
        ]
        | None
    ) = None

    @model_validator(mode="after")
    def information_contract(self):
        if self.holding_buffer is not None and self.transport is None:
            raise ValueError("holding buffer requires AGV transport")
        if self.quality is not None and self.visibility != "decision_context":
            raise ValueError("quality requires decision_context visibility")
        if self.machine_events is not None and self.visibility != "decision_context":
            raise ValueError("machine events require decision_context visibility")
        if self.arrivals is None and self.decision_trigger != "dispatch_available":
            raise ValueError("arrival_event requires arrivals")
        if self.arrivals is not None and self.visibility != "decision_context":
            raise ValueError("arrivals require decision_context visibility")
        if self.processing_time is not None and self.visibility != "decision_context":
            raise ValueError(
                "processing uncertainty requires decision_context visibility"
            )
        return self


class ScriptParameters(StrictModel):
    actions: tuple[SemanticAction, ...] | None = None
    commands: tuple[JointCommand, ...] | None = None

    @model_validator(mode="after")
    def one_source(self):
        if (self.actions is None) == (self.commands is None):
            raise ValueError("provide commands or historical actions, exactly one")
        return self


class EmptyParameters(StrictModel):
    pass


class ScriptedAlgorithm(StrictModel):
    provider: Literal["builtin.scripted"]
    parameters: ScriptParameters
    interface_kind: Literal["online_policy"] = "online_policy"
    required_information: Literal["decision_context"] = "decision_context"


class DispatchRuleParameters(StrictModel):
    quality_mode: Reference | None = None
    transport_rule: Literal["shortest_trip"] = "shortest_trip"
    rerouting_rule: Literal["idle_destination"] = "idle_destination"
    buffer_admission_rule: Literal["immediate_capacity"] = "immediate_capacity"


class DispatchRuleAlgorithm(StrictModel):
    provider: Literal["builtin.first_feasible", "builtin.spt", "builtin.coordinated"]
    parameters: DispatchRuleParameters = Field(default_factory=DispatchRuleParameters)
    interface_kind: Literal["online_policy"] = "online_policy"
    required_information: Literal["decision_context"] = "decision_context"


class CPSatAlgorithm(StrictModel):
    provider: Literal["pyjobshop.cp_sat"]
    parameters: EmptyParameters = Field(default_factory=EmptyParameters)
    interface_kind: Literal["offline_solver"]
    required_information: Literal["full_static"]


class PPOParameters(StrictModel):
    hidden_sizes: tuple[Annotated[int, Field(gt=0)], ...] = (64, 64)
    activation: Literal["tanh"] = "tanh"
    learning_rate: Annotated[float, Field(gt=0, allow_inf_nan=False)] = 0.0003
    gamma: Annotated[float, Field(gt=0, le=1)] = 1.0
    gae_lambda: Annotated[float, Field(gt=0, le=1)] = 0.95
    clip_range: Annotated[float, Field(gt=0, lt=1)] = 0.2
    entropy_coefficient: Annotated[float, Field(ge=0, allow_inf_nan=False)] = 0.0
    n_steps: Annotated[int, Field(gt=1)] = 256
    batch_size: Annotated[int, Field(gt=1)] = 64
    n_epochs: Annotated[int, Field(gt=0)] = 10

    @model_validator(mode="after")
    def valid_rollout(self):
        if not self.hidden_sizes or self.n_steps % self.batch_size:
            raise ValueError("PPO needs a nonempty network and whole minibatches")
        return self


class LearningAlgorithm(StrictModel):
    provider: Literal["rllib.ppo", "sb3.maskable_ppo"]
    projection: ProjectionSpec
    parameters: PPOParameters = Field(default_factory=PPOParameters)
    extensions: ExtensionSpec | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    checkpoint: Reference | None = None
    checkpoint_sha256: SHA256 | None = Field(
        default=None, exclude_if=lambda v: v is None
    )
    interface_kind: Literal["online_policy"] = "online_policy"
    required_information: Literal["decision_context"] = "decision_context"


class ResourcePPOParameters(PPOParameters):
    # Optimization units only. Environment rewards and replay remain in ticks.
    learner_reward_scale: Annotated[float, Field(gt=0, allow_inf_nan=False)] = Field(
        default=1.0, exclude_if=lambda value: value == 1.0
    )


class ResourceLearningAlgorithm(LearningAlgorithm):
    provider: Literal["rllib.resource_ppo"]
    projection: ResourceProjectionSpec
    parameters: ResourcePPOParameters = Field(default_factory=ResourcePPOParameters)


class AlgorithmFile(StrictModel):
    schema_id: Literal["smartsom.algorithm/v1"] = Field(alias="schema")
    algorithm: Annotated[
        ScriptedAlgorithm
        | DispatchRuleAlgorithm
        | CPSatAlgorithm
        | LearningAlgorithm
        | ResourceLearningAlgorithm,
        Field(discriminator="provider"),
    ]


class RunBudget(StrictModel):
    solver_time_limit_seconds: Annotated[float, Field(gt=0, allow_inf_nan=False)] = 60.0


class RecordingSpec(StrictModel):
    observations: Literal["full", "hash"] = "full"
    debug: bool = False


class EpisodeBudget(StrictModel):
    max_decisions: Annotated[int, Field(gt=0)] = 1024
    max_ticks: Annotated[int, Field(gt=0)] = 10000

    def limits(self) -> EpisodeLimits:
        return EpisodeLimits(self.max_decisions, self.max_ticks)


class TrainingBudget(EpisodeBudget):
    environment_steps: Annotated[int, Field(gt=0)]


class TrainingRunSpec(StrictModel):
    schema_id: Literal["smartsom.training-run/v1"] = Field(alias="schema")
    scenario: Reference
    algorithm: Reference
    seed: Seed
    output_root: Reference
    objective: Literal["makespan"] = "makespan"
    budget: TrainingBudget
    recording: RecordingSpec = Field(
        default_factory=lambda: RecordingSpec(observations="hash")
    )


class RunSpec(StrictModel):
    schema_id: Literal["smartsom.run/v1"] = Field(alias="schema")
    scenario: Reference
    algorithm: Reference
    seed: Seed
    output_root: Reference
    objective: Literal["makespan"] = "makespan"
    budget: RunBudget | EpisodeBudget | None = None
    recording: RecordingSpec | None = Field(
        default=None, exclude_if=lambda v: v is None
    )
