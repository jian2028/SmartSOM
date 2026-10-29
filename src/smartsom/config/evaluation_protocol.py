"""Scientific evaluation protocols, distinct from telemetry/display settings."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Protocol(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RecoveryFault(Protocol):
    machine_id: str
    start_tick: int = Field(ge=1)
    duration_ticks: int = Field(gt=0)


class RecoveryCandidate(Protocol):
    name: str
    load: Literal["normal", "saturated"]
    orders: int = Field(gt=1)
    initial_orders: int = Field(gt=0)
    mean_interarrival: float = Field(gt=0)
    horizon_ticks: int = Field(gt=0)
    window_ticks: int = Field(gt=0)
    baseline_windows: int = Field(ge=3)
    minimum_window_deliveries: int = Field(gt=0)
    maximum_cv: float = Field(ge=0)
    maximum_relative_trend: float = Field(ge=0)
    faults: tuple[RecoveryFault, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def valid(self):
        if self.initial_orders >= self.orders:
            raise ValueError("recovery requires initial stock and continuing arrivals")
        for fault in self.faults:
            if fault.start_tick < self.window_ticks * self.baseline_windows:
                raise ValueError("insufficient pre-fault baseline windows")
            if (
                fault.start_tick + fault.duration_ticks + 3 * self.window_ticks
                > self.horizon_ticks
            ):
                raise ValueError("recovery horizon must cover repair and three windows")
        return self


class RecoveryEvaluation(Protocol):
    action: Literal["disabled", "calibrate", "evaluate"] = "disabled"
    candidates: tuple[RecoveryCandidate, ...] = ()
    selected: str | None = None
    calibration: str | None = None
    calibration_sha256: str | None = None
    maximum_relative_capacity_gain: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def valid(self):
        names = [c.name for c in self.candidates]
        if len(names) != len(set(names)):
            raise ValueError("recovery candidate names must be unique")
        if self.action != "disabled" and not self.candidates:
            raise ValueError("explicit recovery candidates required")
        if self.action == "evaluate" and (
            self.selected not in names or not self.calibration
        ):
            raise ValueError(
                "recovery evaluation requires selected candidate and frozen calibration"
            )
        if any(c.load == "saturated" for c in self.candidates):
            if self.maximum_relative_capacity_gain is None:
                raise ValueError("saturation requires explicit capacity-gain tolerance")
        return self


class LearningTargetConfig(Protocol):
    reference: str
    rule_cost: float | None = Field(default=None, ge=0)
    reference_sha256: str | None = None
    cost_ratio: float = Field(default=1, ge=0)
    consecutive: Literal[3] = 3


class EvaluationProtocol(Protocol):
    schema_id: Literal["smartsom.evaluation/v1"] = Field(
        default="smartsom.evaluation/v1", alias="schema"
    )
    mode: Literal["limited", "completion"] = "limited"
    common_window_end_tick: int | None = Field(default=None, gt=0)
    completion_drain_multiplier: float = Field(default=5, ge=1)
    recovery: RecoveryEvaluation = Field(default_factory=RecoveryEvaluation)
    learning_target: LearningTargetConfig | None = None
