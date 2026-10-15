"""Named research extensions, with no import paths or framework dependencies."""

import json
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StringConstraints,
    model_validator,
)

type BackendName = Literal[
    "rllib.ppo", "sb3.maskable_ppo", "rllib.resource_ppo", "rllib.resource_dqn"
]
type ResourceRole = Literal[
    "machine_policy", "agv_policy", "buffer_policy", "dispatcher_policy", "mover_policy"
]
type ExtensionName = Annotated[
    str, StringConstraints(pattern=r"^[A-Za-z][A-Za-z0-9_.-]*$")
]
type CodeDigest = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]


class ExtensionModel(BaseModel):
    model_config = ConfigDict(
        strict=True,
        extra="forbid",
        frozen=True,
        validate_default=True,
        allow_inf_nan=False,
    )


class ExtensionRef(ExtensionModel):
    """An explicitly registered implementation and JSON-only experiment parameters."""

    name: ExtensionName
    version: Annotated[
        str, StringConstraints(min_length=1, pattern=r"^[A-Za-z0-9_.-]+$")
    ]
    parameters: dict[str, JsonValue] = Field(default_factory=dict)
    code_sha256: CodeDigest | None = Field(
        default=None, exclude_if=lambda value: value is None
    )

    @model_validator(mode="before")
    @classmethod
    def current_scoped_interface(cls, value):
        # These built-ins have one maintained implementation. Keep its internal
        # identity in frozen metadata without requiring a user version selector.
        if (
            isinstance(value, dict)
            and value.get("name")
            in ("builtin.public_fields", "builtin.scoped_candidate")
            and "version" not in value
        ):
            return {**value, "version": "compact"}
        return value

    @model_validator(mode="after")
    def finite_json(self):
        json.dumps(self.parameters, allow_nan=False)
        return self


class NetworkBranch(ExtensionModel):
    """A registered feed-forward encoder followed by its own actor or critic head."""

    encoder: ExtensionRef = Field(
        default_factory=lambda: ExtensionRef(name="builtin.flatten", version="1")
    )
    hidden_sizes: tuple[Annotated[int, Field(gt=0)], ...] = (64, 64)
    activation: Literal["tanh", "relu", "gelu"] = "tanh"


class ActorCriticSpec(ExtensionModel):
    actor: NetworkBranch = Field(default_factory=NetworkBranch)
    critic: NetworkBranch = Field(default_factory=NetworkBranch)


class NetworkSpec(ActorCriticSpec):
    """Role overrides share parameters within a role, never across resource roles."""

    roles: dict[ResourceRole, ActorCriticSpec] = Field(
        default_factory=dict, exclude_if=lambda value: not value
    )

    def for_role(self, role: str | None) -> ActorCriticSpec:
        return self.roles.get(
            role, ActorCriticSpec(actor=self.actor, critic=self.critic)
        )


class PassingRateSmoothing(ExtensionModel):
    schema_id: Literal["smartsom.passing-rate-training/v1"] = Field(
        default="smartsom.passing-rate-training/v1", alias="schema"
    )
    pseudocount: Annotated[float, Field(gt=0, allow_inf_nan=False)]
    terminal_compensation: Literal["objective_episode"] = "objective_episode"


class RewardSpec(ExtensionModel):
    """Research transforms precede positive numerical scaling for the learner."""

    team: ExtensionRef | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    roles: dict[ResourceRole, ExtensionRef] = Field(
        default_factory=dict, exclude_if=lambda value: not value
    )
    learner_scale: Annotated[float, Field(gt=0, allow_inf_nan=False)] = 1.0
    passing_rate_smoothing: PassingRateSmoothing | None = Field(
        default=None, exclude_if=lambda value: value is None
    )


class ExtensionSpec(ExtensionModel):
    observation: ExtensionRef | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    role_observations: dict[ResourceRole, ExtensionRef] = Field(
        default_factory=dict, exclude_if=lambda value: not value
    )
    network: NetworkSpec | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    reward: RewardSpec | None = Field(
        default=None, exclude_if=lambda value: value is None
    )

    @model_validator(mode="after")
    def smoothing_requires_task_reward(self):
        if (
            self.reward is not None
            and self.reward.passing_rate_smoothing is not None
            and getattr(self.reward, "task", None) is None
        ):
            raise ValueError(
                "passing_rate_smoothing requires a V3 shipment task reward"
            )
        return self

    def observation_for_role(self, role: str | None) -> ExtensionRef | None:
        return self.role_observations.get(role, self.observation)
