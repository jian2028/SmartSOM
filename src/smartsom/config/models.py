"""Strict authoring envelopes; domain objects retain their stdlib contracts."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from smartsom.dispatch import Dispatch
from smartsom.domain import FactorySpec, WorkloadInstance
from smartsom.workloads import StaticJSPProfile

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
    generator: Literal["static_jsp_v1"]
    profile: StaticJSPProfile


class GenerationProvenance(StrictModel):
    generator: Literal["static_jsp_v1"]
    generator_version: Literal["1"]
    profile_sha256: SHA256
    effective_seed: Seed


class InstanceFile(StrictModel):
    schema_id: Literal["smartsom.workload-instance/v1"] = Field(alias="schema")
    workload: WorkloadInstance
    content_sha256: SHA256 | None = None
    provenance: GenerationProvenance | None = None


class WorkloadSource(StrictModel):
    kind: Literal["profile", "instance"]
    path: Reference


class ScenarioFile(StrictModel):
    schema_id: Literal["smartsom.scenario/v1"] = Field(alias="schema")
    factory: Reference
    workload: WorkloadSource
    modules: Annotated[tuple[str, ...], Field(max_length=0)] = ()
    visibility: Literal["decision_context"] = "decision_context"
    termination: Literal["all_jobs_complete"] = "all_jobs_complete"


class ScriptParameters(StrictModel):
    actions: tuple[Dispatch, ...]


class EmptyParameters(StrictModel):
    pass


class ScriptedAlgorithm(StrictModel):
    provider: Literal["builtin.scripted"]
    parameters: ScriptParameters
    interface_kind: Literal["online_policy"] = "online_policy"
    required_information: Literal["decision_context"] = "decision_context"


class FirstFeasibleAlgorithm(StrictModel):
    provider: Literal["builtin.first_feasible"]
    parameters: EmptyParameters = Field(default_factory=EmptyParameters)
    interface_kind: Literal["online_policy"] = "online_policy"
    required_information: Literal["decision_context"] = "decision_context"


class AlgorithmFile(StrictModel):
    schema_id: Literal["smartsom.algorithm/v1"] = Field(alias="schema")
    algorithm: Annotated[
        ScriptedAlgorithm | FirstFeasibleAlgorithm, Field(discriminator="provider")
    ]


class RunSpec(StrictModel):
    schema_id: Literal["smartsom.run/v1"] = Field(alias="schema")
    scenario: Reference
    algorithm: Reference
    seed: Seed
    output_root: Reference
    objective: Literal["makespan"] = "makespan"
