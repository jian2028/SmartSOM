"""Typed per-role rule, fresh network and model artifact selectors."""

from typing import Annotated, Literal

from pydantic import Field, model_validator

from smartsom.config.extensions import (
    ActorCriticSpec,
    CodeDigest,
    ExtensionModel,
    ExtensionName,
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
    name: ExtensionName
    version: str | None = Field(default=None, exclude_if=lambda value: value is None)
    code_sha256: CodeDigest | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
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
    schema_id: Literal["smartsom.policy/v1", "smartsom.frozen-policy/v1"] = Field(
        alias="schema"
    )
    role: Literal["machine", "buffer", "dispatcher", "mover", "central"]
    implementation: Annotated[
        RuleImplementation | NewImplementation | ModelImplementation,
        Field(discriminator="kind"),
    ]

    @model_validator(mode="after")
    def valid_rule(self):
        if isinstance(self.implementation, RuleImplementation):
            from smartsom.algorithms.rule_registry import freeze_rule

            impl = self.implementation
            freeze_rule(
                self.role, impl.name, impl.version, impl.parameters, impl.code_sha256
            )
        return self
