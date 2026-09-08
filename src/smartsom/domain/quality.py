"""Immutable quality capabilities, realized inputs and inspected outcomes."""

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Literal
from urllib.parse import quote

from smartsom.domain.validation import _identifier, _items, _unique

QUALITY_VERSION = "smartsom.quality/v1"
DRAW_DENOMINATOR = 2**53
type ProbabilityVisibility = Literal["public", "hidden"]


def _decimal(value, name: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (str, int, float, Decimal)):
        raise ValueError(f"{name} must be a finite decimal")
    try:
        result = Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError(f"invalid {name}") from exc
    if not result.is_finite():
        raise ValueError(f"{name} must be finite")
    sign, digits, exponent = result.as_tuple()
    digits = list(digits)
    while len(digits) > 1 and digits[-1] == 0:
        digits.pop()
        exponent += 1
    return Decimal((sign if result else 0, tuple(digits), exponent if result else 0))


@dataclass(frozen=True, slots=True)
class QualityMode:
    quality_mode_id: str
    time_scale: Decimal
    error_rate: Decimal

    def __post_init__(self):
        _identifier(self.quality_mode_id, "quality mode ID")
        for name in ("time_scale", "error_rate"):
            object.__setattr__(self, name, _decimal(getattr(self, name), name))
        if self.time_scale <= 0 or not 0 <= self.error_rate <= 1:
            raise ValueError("time_scale must be positive and error_rate in [0,1]")


def _modes(items):
    rows = _items(items, QualityMode, "quality modes")
    _unique(tuple(x.quality_mode_id for x in rows), "quality mode ID")
    return tuple(sorted(rows, key=lambda x: x.quality_mode_id))


@dataclass(frozen=True, slots=True)
class MachineQualityModes:
    machine_id: str
    modes: tuple[QualityMode, ...]

    def __post_init__(self):
        _identifier(self.machine_id, "machine ID")
        object.__setattr__(self, "modes", _modes(self.modes))


@dataclass(frozen=True, slots=True)
class QualitySpeedSpec:
    default_modes: tuple[QualityMode, ...] = ()
    machine_modes: tuple[MachineQualityModes, ...] = ()

    def __post_init__(self):
        if not isinstance(self.default_modes, (tuple, list)):
            raise ValueError("default_modes must be a collection")
        if self.default_modes:
            object.__setattr__(self, "default_modes", _modes(self.default_modes))
        else:
            object.__setattr__(self, "default_modes", ())
        if not isinstance(self.machine_modes, (tuple, list)) or any(
            not isinstance(x, MachineQualityModes) for x in self.machine_modes
        ):
            raise ValueError("machine_modes must contain MachineQualityModes")
        _unique(tuple(x.machine_id for x in self.machine_modes), "quality machine ID")
        object.__setattr__(
            self,
            "machine_modes",
            tuple(sorted(self.machine_modes, key=lambda x: x.machine_id)),
        )
        if not self.default_modes and not self.machine_modes:
            raise ValueError("quality_speed must define at least one mode table")

    def for_machine(self, machine_id: str) -> tuple[QualityMode, ...]:
        return next(
            (x.modes for x in self.machine_modes if x.machine_id == machine_id),
            self.default_modes,
        )


@dataclass(frozen=True, slots=True)
class QualityDraw:
    operation_id: str
    draw: int

    def __post_init__(self):
        _identifier(self.operation_id, "operation ID")
        if type(self.draw) is not int or not 0 <= self.draw < DRAW_DENOMINATOR:
            raise ValueError("quality draw must be a 53-bit unsigned integer")


@dataclass(frozen=True, slots=True)
class QualityDrawPlan:
    operations: tuple[QualityDraw, ...]

    def __post_init__(self):
        rows = _items(self.operations, QualityDraw, "quality draws")
        _unique(tuple(x.operation_id for x in rows), "quality operation ID")
        object.__setattr__(
            self, "operations", tuple(sorted(rows, key=lambda x: x.operation_id))
        )


def quality_mode_id(base_mode_id: str, quality_id: str) -> str:
    """Versioned injective encoding, independent of candidate positions."""
    _identifier(base_mode_id, "base mode ID")
    _identifier(quality_id, "quality mode ID")
    return f"quality-v1/{quote(base_mode_id, safe='')}/{quote(quality_id, safe='')}"


@dataclass(frozen=True, slots=True)
class QualityExecutionMode:
    operation_id: str
    base_processing_mode_id: str
    processing_mode_id: str
    machine_id: str
    mode: QualityMode
    nominal_ticks: int
    actual_ticks: int

    def __post_init__(self):
        for name in (
            "operation_id",
            "base_processing_mode_id",
            "processing_mode_id",
            "machine_id",
        ):
            _identifier(getattr(self, name), name)
        if not isinstance(self.mode, QualityMode):
            raise ValueError("execution mode must contain QualityMode")
        if any(
            type(x) is not int or x < 1 for x in (self.nominal_ticks, self.actual_ticks)
        ):
            raise ValueError("quality execution durations must be positive integers")


@dataclass(frozen=True, slots=True)
class QualityPlan:
    draws: QualityDrawPlan
    modes: tuple[QualityExecutionMode, ...]

    def __post_init__(self):
        if not isinstance(self.draws, QualityDrawPlan):
            raise ValueError("quality requires a QualityDrawPlan")
        rows = _items(self.modes, QualityExecutionMode, "quality execution modes")
        keys = tuple((x.operation_id, x.processing_mode_id) for x in rows)
        if len(set(keys)) != len(keys):
            raise ValueError("duplicate quality execution mode")
        object.__setattr__(
            self,
            "modes",
            tuple(sorted(rows, key=lambda x: (x.operation_id, x.processing_mode_id))),
        )


@dataclass(frozen=True, slots=True)
class QualityModeView:
    operation_id: str
    processing_mode_id: str
    base_processing_mode_id: str
    quality_mode_id: str
    machine_id: str
    time_scale: Decimal
    nominal_ticks: int
    error_rate: Decimal | None


@dataclass(frozen=True, slots=True)
class JobQualityView:
    job_id: str
    passed: bool | None = None
    inspection_time: int | None = None


@dataclass(frozen=True, slots=True)
class OperationQuality:
    operation_id: str
    processing_mode_id: str
    job_id: str
    completion_time: int
    defective: bool


@dataclass(frozen=True, slots=True)
class QualityResult:
    operations: tuple[OperationQuality, ...]
    jobs: tuple[JobQualityView, ...]

    @property
    def total_jobs(self) -> int:
        return len(self.jobs)

    @property
    def passed_jobs(self) -> int:
        return sum(x.passed is True for x in self.jobs)

    @property
    def defective_jobs(self) -> int:
        return sum(x.passed is False for x in self.jobs)

    @property
    def passing_rate(self) -> float:
        return self.passed_jobs / self.total_jobs
