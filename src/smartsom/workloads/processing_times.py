"""Versioned entity-local draws and exact half-up duration materialization."""

import hashlib
import json
import random
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction

from smartsom.domain import WorkloadInstance
from smartsom.domain.processing_times import ProcessingTime, ProcessingTimePlan

GENERATOR_VERSION = "smartsom.processing-time/v1"
DRAW_DENOMINATOR = 2**53


@dataclass(frozen=True, slots=True)
class UniformMultiplierProfile:
    low: Decimal = Decimal("0.8")
    high: Decimal = Decimal("1.2")

    def __post_init__(self):
        for name in ("low", "high"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(
                value, (str, int, float, Decimal)
            ):
                raise ValueError("multiplier bounds must be positive finite decimals")
            try:
                value = Decimal(str(value))
            except InvalidOperation as exc:
                raise ValueError("invalid decimal multiplier") from exc
            if not value.is_finite() or value <= 0:
                raise ValueError("multiplier bounds must be positive finite decimals")
            # Canonicalize without consulting the caller's decimal context.
            sign, digits, exponent = value.as_tuple()
            digits = list(digits)
            while len(digits) > 1 and digits[-1] == 0:
                digits.pop()
                exponent += 1
            value = Decimal((sign, tuple(digits), exponent))
            object.__setattr__(self, name, value)
        if self.low > self.high:
            raise ValueError("multiplier low must not exceed high")


@dataclass(frozen=True, slots=True)
class ProcessingDraw:
    operation_id: str
    processing_mode_id: str
    draw: int | None


def mode_draw(seed: int, operation_id: str, processing_mode_id: str) -> int:
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise ValueError("processing_time seed must be an unsigned 64-bit integer")
    identity = json.dumps(
        [GENERATOR_VERSION, seed, operation_id, processing_mode_id],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    derived = int.from_bytes(hashlib.sha256(identity.encode("utf-8")).digest(), "big")
    return random.Random(derived).getrandbits(53)


def actual_ticks(
    nominal: int, profile: UniformMultiplierProfile, draw: int | None
) -> int:
    if type(nominal) is not int or nominal < 1:
        raise ValueError("nominal_ticks must be a positive integer")
    low, high = Fraction(profile.low), Fraction(profile.high)
    if low == high:
        if draw is not None:
            raise ValueError("constant multiplier must not contain a draw")
        multiplier = low
    else:
        if type(draw) is not int or not 0 <= draw < DRAW_DENOMINATOR:
            raise ValueError("draw must be a 53-bit unsigned integer")
        multiplier = low + (high - low) * Fraction(draw, DRAW_DENOMINATOR)
    rounded = nominal * multiplier + Fraction(1, 2)
    return max(1, rounded.numerator // rounded.denominator)


def generate_processing_times(
    workload: WorkloadInstance, profile: UniformMultiplierProfile, seed: int
) -> tuple[ProcessingTimePlan, tuple[ProcessingDraw, ...]]:
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise ValueError("processing_time seed must be an unsigned 64-bit integer")
    entries, draws = [], []
    for op in sorted(workload.operations, key=lambda op: op.operation_id):
        for mode in sorted(op.modes, key=lambda mode: mode.processing_mode_id):
            draw = (
                None
                if profile.low == profile.high
                else mode_draw(seed, op.operation_id, mode.processing_mode_id)
            )
            draws.append(ProcessingDraw(op.operation_id, mode.processing_mode_id, draw))
            entries.append(
                ProcessingTime(
                    op.operation_id,
                    mode.processing_mode_id,
                    mode.nominal_ticks,
                    actual_ticks(mode.nominal_ticks, profile, draw),
                )
            )
    return ProcessingTimePlan(tuple(entries)), tuple(draws)
