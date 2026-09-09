"""Opt-in training lifecycle controls; legacy training recipes remain unchanged."""

from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Literal


def _positive(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True, slots=True)
class ValidationControls:
    every_updates: int = 4
    seed: int = 303
    replications: int = 5
    scenarios: tuple[str, ...] = ()
    case_id: str = "learning"
    deterministic: bool = True
    full_replay: bool = False
    best_mode: Literal["completion_first", "all_complete", "custom"] = (
        "completion_first"
    )
    metric: Literal["makespan", "return", "passing_rate"] = "makespan"
    direction: Literal["min", "max"] = "min"
    failure_policy: Literal["ineligible", "successful_only"] | None = None
    patience: int | None = None
    min_delta: float = 0.0

    def __post_init__(self):
        for name in ("every_updates", "replications"):
            _positive(getattr(self, name), name)
        if type(self.seed) is not int or not 0 <= self.seed < 2**64:
            raise ValueError("validation seed must be an unsigned 64-bit integer")
        if not isinstance(self.case_id, str) or not self.case_id:
            raise ValueError("validation case_id must be nonempty")
        if not isinstance(self.scenarios, tuple) or any(
            not isinstance(value, str) or not value for value in self.scenarios
        ):
            raise ValueError("validation scenarios must be a tuple of nonempty paths")
        if self.best_mode not in ("completion_first", "all_complete", "custom"):
            raise ValueError("unknown best selection mode")
        if self.metric not in ("makespan", "return", "passing_rate"):
            raise ValueError("unsupported validation metric")
        if self.direction not in ("min", "max"):
            raise ValueError("validation direction must be min or max")
        if self.best_mode == "custom" and self.failure_policy is None:
            raise ValueError("custom selection requires an explicit failure policy")
        if self.failure_policy not in (None, "ineligible", "successful_only"):
            raise ValueError("unknown validation failure policy")
        if self.patience is not None:
            _positive(self.patience, "patience")
        if isinstance(self.min_delta, bool) or not isfinite(self.min_delta):
            raise ValueError("min_delta must be finite and nonnegative")
        if self.min_delta < 0:
            raise ValueError("min_delta must be finite and nonnegative")
        for name in ("deterministic", "full_replay"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a boolean")


@dataclass(frozen=True, slots=True)
class TrainingControls:
    checkpoint_every_updates: int | None = 4
    keep_last: int = 2
    save_last: bool = True
    save_best: bool = True
    resume_from: Path | None = None
    initialize_from: Path | None = None
    validation: ValidationControls | None = ValidationControls()
    device: Literal["cpu", "cuda"] = "cpu"
    numerical_threads: int = 1
    num_envs: int = 1
    sampling_processes: int = 0
    validation_inputs_json: str | None = None
    # An explicit boundary stop is also useful in unattended jobs and recovery tests.
    stop_after_updates: int | None = None

    def __post_init__(self):
        if self.checkpoint_every_updates is not None:
            _positive(self.checkpoint_every_updates, "checkpoint_every_updates")
        _positive(self.keep_last, "keep_last")
        if self.resume_from is not None and self.initialize_from is not None:
            raise ValueError(
                "resume and independent weights initialization are exclusive"
            )
        for name in ("resume_from", "initialize_from"):
            value = getattr(self, name)
            if value is not None:
                if not isinstance(value, (str, Path)):
                    raise ValueError(f"{name} must be a checkpoint path")
                object.__setattr__(self, name, Path(value).resolve())
        if self.device not in ("cpu", "cuda"):
            raise ValueError("device must be cpu or cuda")
        _positive(self.numerical_threads, "numerical_threads")
        _positive(self.num_envs, "num_envs")
        if (
            type(self.sampling_processes) is not int
            or not 0 <= self.sampling_processes <= self.num_envs
        ):
            raise ValueError("sampling_processes must be between zero and num_envs")
        if self.stop_after_updates is not None:
            _positive(self.stop_after_updates, "stop_after_updates")
        for name in ("save_last", "save_best"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a boolean")
        if self.validation is not None and not isinstance(
            self.validation, ValidationControls
        ):
            raise TypeError("validation requires ValidationControls or None")
        if bool(self.validation and self.validation.scenarios) != (
            self.validation_inputs_json is not None
        ):
            raise ValueError(
                "external validation scenarios require frozen validation inputs"
            )
        if self.validation_inputs_json is not None:
            from smartsom.config.codec import canonical_json
            from smartsom.config.validation import validate_frozen_cases

            inputs = validate_frozen_cases(self.validation_inputs_json, self.validation)
            object.__setattr__(self, "validation_inputs_json", canonical_json(inputs))
