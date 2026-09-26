"""Typed per-role rule, fresh network and model artifact selectors."""

from typing import Annotated, Literal

from pydantic import Field, model_validator

from smartsom.config.extensions import (
    ActorCriticSpec,
    ExtensionModel,
    ExtensionRef,
    NetworkBranch,
)


class QNetwork(ExtensionModel):
    q: NetworkBranch = Field(default_factory=NetworkBranch)


class PolicyExtensions(ExtensionModel):
    observation: ExtensionRef | None = None
    network: ActorCriticSpec | QNetwork | None = None


class ProjectionSettings(ExtensionModel):
    max_jobs: Annotated[int, Field(gt=0)] = 64
    time_scale: Annotated[float, Field(gt=0)] = 100.0
    count_scale: Annotated[float, Field(gt=0)] = 100.0


class RuleImplementation(ExtensionModel):
    kind: Literal["rule"]
    name: str
    parameters: dict = Field(default_factory=dict)


class NewImplementation(ExtensionModel):
    kind: Literal["new_model"]
    extensions: PolicyExtensions = Field(default_factory=PolicyExtensions)
    projection: ProjectionSettings = Field(default_factory=ProjectionSettings)


class ModelSelector(ExtensionModel):
    source: str
    checkpoint: str | None = None
    group: str | None = None


class ModelImplementation(ExtensionModel):
    kind: Literal["model"]
    model: ModelSelector


class PolicyFile(ExtensionModel):
    schema_id: Literal["smartsom.policy/v1"] = Field(alias="schema")
    role: Literal["machine", "buffer", "dispatcher", "mover", "central"]
    implementation: Annotated[
        RuleImplementation | NewImplementation | ModelImplementation,
        Field(discriminator="kind"),
    ]

    @model_validator(mode="after")
    def valid_rule(self):
        if isinstance(self.implementation, RuleImplementation):
            from smartsom.algorithms.production_rules import RULES

            if self.implementation.name not in RULES.get(self.role, set()):
                raise ValueError(f"unknown {self.role} rule")
        return self
