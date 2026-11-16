"""Explicit scientific search inputs; validation never imports Ray or Optuna."""

from typing import Annotated, Literal

from pydantic import Field, model_validator

from smartsom.config.codec import ConfigurationError, StrictModel, read_document


class Choice(StrictModel):
    distribution: Literal["choice"]
    values: list[str | int | float | bool] = Field(min_length=1)

    @model_validator(mode="after")
    def finite_choices(self):
        import math

        if any(
            isinstance(value, float) and not math.isfinite(value)
            for value in self.values
        ):
            raise ValueError("search choices must be finite")
        return self


class Range(StrictModel):
    distribution: Literal["uniform", "loguniform", "randint"]
    lower: float
    upper: float

    @model_validator(mode="after")
    def bounds(self):
        import math

        if not all(math.isfinite(value) for value in (self.lower, self.upper)):
            raise ValueError("search bounds must be finite")
        if self.lower >= self.upper:
            raise ValueError("search lower bound must be below upper bound")
        if self.distribution == "loguniform" and self.lower <= 0:
            raise ValueError("loguniform requires a positive lower bound")
        if self.distribution == "randint" and any(
            not value.is_integer() for value in (self.lower, self.upper)
        ):
            raise ValueError("randint bounds must be integers; upper is exclusive")
        return self


class Objective(StrictModel):
    panel: Literal["sample", "argmax"]
    metric: str = Field(pattern=r"^configured\.(all|normal|low|medium|high)\.[a-z_]+$")
    mode: Literal["min", "max"]


class Search(StrictModel):
    algorithm: Literal["optuna"]
    num_samples: int = Field(gt=0)
    seed: int = Field(ge=0)
    train_seed: int = Field(ge=0)
    objectives: list[Objective] = Field(min_length=1)
    parameters: dict[
        str, Annotated[Choice | Range, Field(discriminator="distribution")]
    ]

    @model_validator(mode="after")
    def parameter_paths(self):
        identities = [(o.panel, o.metric) for o in self.objectives]
        if len(set(identities)) != len(identities):
            raise ValueError("search objectives must be distinct")
        if not self.parameters:
            raise ValueError("search requires at least one parameter")
        for path in self.parameters:
            parts = path.split(".")
            if (
                len(parts) < 2
                or parts[0] != "algorithm"
                or any(not part or "seed" in part.lower() for part in parts)
            ):
                raise ValueError(
                    "search parameter paths must name algorithm settings, excluding seeds"
                )
        return self


def load_search(path):
    return Search.model_validate(read_document(path))


def validate_paths(search, algorithm):
    """Reject absent leaves before creating a Tuner or freezing search evidence."""
    for path in search.parameters:
        value = algorithm
        for key in path.split(".")[1:]:
            if not isinstance(value, dict) or key not in value:
                raise ConfigurationError(f"unknown search parameter: {path}")
            value = value[key]
        if isinstance(value, (dict, list)):
            raise ConfigurationError(f"search parameter must be a scalar leaf: {path}")


def reject_failed_retry(retry_failed):
    if retry_failed:
        raise ConfigurationError(
            "failed Optuna search trials cannot be retried in place; "
            "preserve this search and start a new one, or use resume for unfinished trials"
        )
