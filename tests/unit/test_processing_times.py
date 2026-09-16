"""Independent duration, information, replay and exact sampling checks."""

import hashlib
import json
import math
import random
from dataclasses import replace
from decimal import ROUND_HALF_UP, Decimal, localcontext

import pytest
from reference_cases import op, problem

from smartsom.domain import (
    Job,
    ProcessingTime,
    ProcessingTimePlan,
    ScheduledOperation,
)
from smartsom.workloads.processing_times import (
    UniformMultiplierProfile,
    actual_ticks,
    generate_processing_times,
    mode_draw,
)


def processing_case():
    factory, workload = problem(
        Job("A", (op("A1", "M1", 10), op("A2", "M2", 5, "A1"))),
        Job("B", (op("B1", "M2", 5), op("B2", "M1", 10, "B1"))),
    )
    plan = ProcessingTimePlan(
        tuple(
            ProcessingTime(name, "standard", nominal, actual)
            for name, nominal, actual in [
                ("A1", 10, 12),
                ("A2", 5, 4),
                ("B1", 5, 6),
                ("B2", 10, 8),
            ]
        )
    )
    return factory, workload, plan


REFERENCE = (
    ScheduledOperation("A1", "standard", "M1", 0, 12),
    ScheduledOperation("B1", "standard", "M2", 0, 6),
    ScheduledOperation("A2", "standard", "M2", 12, 16),
    ScheduledOperation("B2", "standard", "M1", 12, 20),
)


@pytest.mark.parametrize(
    "nominal,actual", [(True, 1), (1, True), (1, 1.0), (0, 1), (1, 0), (1, -1)]
)
def test_invalid_duration_scalars(nominal, actual):
    with pytest.raises(ValueError):
        ProcessingTime("A1", "standard", nominal, actual)


@pytest.mark.parametrize(
    "low,high",
    [
        (0, 1),
        (-1, 1),
        (2, 1),
        (True, 1),
        (1, False),
        ("NaN", 1),
        (1, "Infinity"),
        ("bad", 1),
    ],
)
def test_invalid_multiplier_bounds(low, high):
    with pytest.raises(ValueError):
        UniformMultiplierProfile(low, high)


@pytest.mark.parametrize(
    "nominal,low,high,draw,expected",
    [
        (5, "0.9", "0.9", None, 5),
        (
            5,
            "0.899999999999999999999999999999999999999",
            "0.899999999999999999999999999999999999999",
            None,
            4,
        ),
        (1, "0.1", "0.1", None, 1),
        (10, "0.8", "1.2", 0, 8),
        (10, "0.8", "1.2", 2**52, 10),
        (10, "0.8", "1.2", 2**53 - 1, 12),
    ],
)
def test_rounding_fixed_draws_against_independent_decimal(
    nominal, low, high, draw, expected
):
    profile = UniformMultiplierProfile(low, high)
    with localcontext() as context:
        context.prec = 100
        multiplier = Decimal(low) + (Decimal(high) - Decimal(low)) * Decimal(
            draw or 0
        ) / Decimal(2**53)
        reference = max(
            1, int((nominal * multiplier).to_integral_value(rounding=ROUND_HALF_UP))
        )
    assert actual_ticks(nominal, profile, draw) == reference == expected


def test_seed_draw_golden_entity_extension_and_rng_isolation():
    factory, workload, _ = processing_case()
    seed = 3797569027003476775
    profile = UniformMultiplierProfile()
    before = random.getstate()
    plan, draws = generate_processing_times(workload, profile, seed)
    assert [row.draw for row in draws] == GOLDEN_DRAWS
    assert random.getstate() == before
    shuffled = replace(
        workload,
        orders=tuple(
            replace(
                order,
                jobs=tuple(
                    replace(job, operations=tuple(reversed(job.operations)))
                    for job in reversed(order.jobs)
                ),
            )
            for order in workload.orders
        ),
    )
    assert generate_processing_times(shuffled, profile, seed) == (plan, draws)
    extended = replace(
        workload,
        orders=(
            replace(
                workload.orders[0],
                jobs=workload.orders[0].jobs + (Job("0", (op("0", "M1", 8),)),),
            ),
        ),
    )
    extension, newdraws = generate_processing_times(extended, profile, seed)
    assert extension.modes[1:] == plan.modes and newdraws[1:] == draws
    for row in draws:
        identity = json.dumps(
            [
                "smartsom.processing-time/v1",
                seed,
                row.operation_id,
                row.processing_mode_id,
            ],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        independent_seed = int(hashlib.sha256(identity.encode()).hexdigest(), 16)
        assert row.draw == random.Random(independent_seed).getrandbits(53)
    assert mode_draw(seed, "a/b", "c") != mode_draw(seed, "a", "b/c")
    assert len({mode_draw(seed, "A1", mode) for mode in ("fast", "slow", "tie")}) == 3


@pytest.mark.parametrize("nominal", [1, 10, 100])
@pytest.mark.parametrize("low,high", [("0.8", "1.2"), ("0.9", "1.4"), ("0.1", "0.3")])
def test_preregistered_rounded_mean(nominal, low, high):
    n = 10000
    profile = UniformMultiplierProfile(low, high)
    values = [
        actual_ticks(nominal, profile, mode_draw(42, f"sample_{i}", "standard"))
        for i in range(n)
    ]
    # Independent continuous-uniform bin integration, including the clamp at 1.
    a, b = nominal * float(low), nominal * float(high)
    lower = max(1, math.floor(a + 0.5))
    upper = max(1, math.floor(b + 0.5))
    mean = sum(
        k * max(0, min(b, k + 0.5) - max(a, 0 if k == 1 else k - 0.5)) / (b - a)
        for k in range(lower, upper + 1)
    )
    # Union bound over all nine cases: total alpha <= 1e-6. The 53-bit grid
    # correction is below 1e-10 for these bounded profiles.
    tolerance = (upper - lower) * math.sqrt(math.log(2 * 9 / 1e-6) / (2 * n)) + 1e-10
    assert abs(sum(values) / n - mean) <= tolerance
    if upper == lower:
        assert set(values) == {lower}
    else:
        assert len(set(values)) > 1


GOLDEN_DRAWS = [7221518816510873, 8380621278969816, 92466246856231, 8063507948353621]


def test_decimal_context_does_not_change_bounds_or_rounding():
    with localcontext() as context:
        context.prec = 2
        context.Emax = 3
        context.Emin = -3
        profile = UniformMultiplierProfile("10000.00", "10000")
        assert actual_ticks(2, profile, None) == 20000
        small = UniformMultiplierProfile("0.00001", "0.00001")
        assert actual_ticks(1000000, small, None) == 10


@pytest.mark.parametrize("draw", [True, -1, 2**53, 1.0, None])
def test_invalid_draws_rejected(draw):
    with pytest.raises(ValueError):
        actual_ticks(10, UniformMultiplierProfile(), draw)
