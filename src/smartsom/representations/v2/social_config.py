"""Social-learning type and fusion contracts, independent of Torch."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ConcatSlots(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    stations: int = Field(gt=0)
    vehicles: int = Field(gt=0)
    input_jobs: int = Field(gt=0)
    post_jobs: int = Field(gt=0)


class SocialConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    kind: Literal["none", "implicit", "explicit"] = "none"
    fusion: Literal["concat", "attention"] | None = None
    slots: ConcatSlots | None = None

    @model_validator(mode="after")
    def supported(self):
        if self.kind == "explicit":
            raise ValueError("explicit social learning is reserved, not implemented")
        if self.kind == "none":
            if self.fusion is not None or self.slots is not None:
                raise ValueError("social kind none cannot select fusion or slots")
        elif self.fusion is None:
            raise ValueError("implicit social learning requires an explicit fusion")
        elif (self.fusion == "concat") != (self.slots is not None):
            raise ValueError("fixed slots are required only for concat fusion")
        return self


def social_config(social=None, social_enabled=None):
    """Retain historical boolean inputs without silently mixing two interfaces."""
    if social is not None and social_enabled is not None:
        raise ValueError("use social or legacy social_enabled, not both")
    if social is None:
        if social_enabled is not None and type(social_enabled) is not bool:
            raise ValueError("social_enabled must be boolean")
        social = (
            {"kind": "implicit", "fusion": "attention"}
            if social_enabled
            else {"kind": "none"}
        )
    return SocialConfig.model_validate(social)
