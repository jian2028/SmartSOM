"""Named parameter-sharing groups and stable semantic resource bindings."""

from typing import Literal

from pydantic import Field, model_validator

from smartsom.config.extensions import ExtensionModel


class Group(ExtensionModel):
    policy: str


class Binding(ExtensionModel):
    default: str
    overrides: dict[str, str] = Field(default_factory=dict)


class CompositionFile(ExtensionModel):
    schema_id: Literal["smartsom.composition/v1", "smartsom.frozen-composition/v1"] = (
        Field(alias="schema")
    )
    pickup_matching: str
    groups: dict[str, Group] = Field(default_factory=dict)
    bindings: dict[str, Binding] = Field(default_factory=dict)
    controller: Group | None = None

    @model_validator(mode="after")
    def controller_or_groups(self):
        if self.controller is not None:
            if self.groups or self.bindings:
                raise ValueError(
                    "central controller cannot have independent role groups"
                )
        elif set(self.bindings) != {"machine", "buffer", "dispatcher", "mover"}:
            raise ValueError("resource composition requires all four role bindings")
        for binding in self.bindings.values():
            if set((binding.default, *binding.overrides.values())) - self.groups.keys():
                raise ValueError("binding references an unknown strategy group")
        return self
