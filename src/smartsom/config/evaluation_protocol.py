"""Scientific evaluation protocols, distinct from telemetry/display settings."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_serializer, model_validator


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
    consecutive_windows: int = Field(default=3, gt=0, strict=True)
    threshold_fraction: float = Field(
        default=0.95, gt=0, le=1, allow_inf_nan=False, strict=True
    )
    duration_from: Literal["fault", "repair"] = "fault"
    attainment_point: Literal["first", "confirmation"] = "first"
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
                fault.start_tick
                + fault.duration_ticks
                + self.consecutive_windows * self.window_ticks
                > self.horizon_ticks
            ):
                raise ValueError(
                    "recovery horizon must cover repair and configured consecutive windows"
                )
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


class EvaluationCheckpoint(Protocol):
    episode: int = Field(ge=0)
    modes: tuple[Literal["sample", "argmax"], ...] = ("sample", "argmax")

    @model_validator(mode="after")
    def distinct_modes(self):
        if not self.modes or len(set(self.modes)) != len(self.modes):
            raise ValueError("checkpoint decoders must be nonempty and distinct")
        return self


class AutoPlateau(Protocol):
    plateau_windows: int = Field(ge=2, le=10000, strict=True)
    preset_windows: int = Field(ge=1, le=10000, strict=True)
    maximum_cv: float = Field(ge=0, allow_inf_nan=False)
    maximum_relative_trend: float = Field(ge=0, allow_inf_nan=False)
    maximum_upward_trend: float = Field(ge=0, allow_inf_nan=False)
    search_order: Literal["latest", "earliest"]


class SameRunAnalysis(Protocol):
    mode: Literal["manual", "auto"]
    warmup_end_tick: int = Field(ge=0, strict=True)
    baseline_windows: int | None = Field(default=None, ge=1, le=10000, strict=True)
    fraction: float = Field(gt=0, le=1, allow_inf_nan=False)
    consecutive_windows: int = Field(ge=1, le=10000, strict=True)
    duration_from: Literal["fault", "repair"]
    maximum_fault_events: int = Field(ge=1, le=10000, strict=True)
    auto: AutoPlateau | None = None

    @model_validator(mode="after")
    def validate_mode(self):
        if (self.mode == "auto") != (self.auto is not None):
            raise ValueError("auto settings are required only in auto mode")
        if (self.mode == "manual") != (self.baseline_windows is not None):
            raise ValueError("baseline_windows is required only in manual mode")
        return self


class WindowAnalysis(Protocol):
    throughput_width_ticks: int = Field(gt=0, strict=True)
    throughput_step_ticks: int = Field(gt=0, strict=True)
    recovery_window_ticks: int = Field(gt=0, strict=True)
    same_run: SameRunAnalysis | None = None

    @model_serializer(mode="wrap")
    def omit_absent_same_run(self, handler):
        result = handler(self)
        if self.same_run is None:
            result.pop("same_run", None)
        return result


class EvaluationProtocol(Protocol):
    schema_id: Literal["smartsom.evaluation/v1"] = Field(
        default="smartsom.evaluation/v1", alias="schema"
    )
    mode: Literal["limited", "completion"] = "limited"
    checkpoints: tuple[EvaluationCheckpoint, ...] = ()
    case_seconds: dict[Literal["sample", "argmax"], float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def distinct_checkpoints(self):
        import math

        if any(not math.isfinite(v) or v <= 0 for v in self.case_seconds.values()):
            raise ValueError("case_seconds must be finite positive limits")
        nodes = [c.episode for c in self.checkpoints]
        if len(nodes) != len(set(nodes)):
            raise ValueError("evaluation checkpoint episodes must be unique")
        if self.window_analysis is not None and self.recovery.action != "disabled":
            if any(
                c.window_ticks != self.window_analysis.recovery_window_ticks
                for c in self.recovery.candidates
            ):
                raise ValueError(
                    "recovery candidate window differs from window_analysis"
                )
        return self

    common_window_end_tick: int | None = Field(default=None, gt=0)
    completion_drain_multiplier: float = Field(default=5, ge=1)
    recovery: RecoveryEvaluation = Field(default_factory=RecoveryEvaluation)
    learning_target: LearningTargetConfig | None = None
    window_analysis: WindowAnalysis | None = None

    @model_serializer(mode="wrap")
    def omit_absent_window_analysis(self, handler):
        result = handler(self)
        if self.window_analysis is None:
            result.pop("window_analysis", None)
        return result
