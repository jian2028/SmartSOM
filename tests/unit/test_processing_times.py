"""Independent duration, information, replay and exact sampling checks."""

import hashlib
import json
import math
import random
from dataclasses import FrozenInstanceError, replace
from decimal import ROUND_HALF_UP, Decimal, localcontext

import pytest
from test_arrivals import TRIGGERS
from test_static_engine import action, competition_case, op, problem

from smartsom.algorithms import SPTPolicy
from smartsom.config.codec import primitive
from smartsom.dispatch import Dispatch, WaitUntil
from smartsom.domain import (
    ArrivalPlan,
    Job,
    JobArrival,
    Operation,
    ProcessingMode,
    ProcessingTime,
    ProcessingTimePlan,
    ScheduledOperation,
)
from smartsom.engine import (
    InvariantViolation,
    ReplayError,
    Simulator,
    replay,
    replay_schedule,
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


def test_actual_hand_intervals_and_same_engine_replay():
    factory, workload, plan = processing_case()
    result = Simulator(factory, workload, processing_times=plan).run(SPTPolicy())
    assert result.schedule == REFERENCE and result.makespan == 20
    assert replay(factory, workload, result.actions, processing_times=plan) == result
    exact = replay_schedule(
        factory, workload, reversed(REFERENCE), processing_times=plan
    )
    assert exact.schedule == REFERENCE
    assert [
        record.nominal_ticks for record in result.trace if record.kind == "dispatch"
    ] == [5, 10, 5, 10]
    with pytest.raises(ReplayError, match="duration"):
        replay_schedule(factory, workload, REFERENCE)
    for state in Simulator(
        factory, workload, processing_times=plan
    ).current_decision.operations:
        assert state.completion_time is None


def test_same_machine_modes_choose_nominal_and_never_disclose_counterfactuals():
    factory, workload = problem(
        Job(
            "A",
            (
                Operation(
                    "A1",
                    (
                        ProcessingMode("fast", "M1", 1),
                        ProcessingMode("slow", "M1", 2),
                        ProcessingMode("tie", "M1", 1),
                    ),
                ),
            ),
        ),
        Job("B", (op("B1", "M2", 20),)),
    )
    plan = ProcessingTimePlan(
        (
            ProcessingTime("A1", "fast", 1, 10),
            ProcessingTime("A1", "slow", 2, 1),
            ProcessingTime("A1", "tie", 1, 3),
            ProcessingTime("B1", "standard", 20, 20),
        )
    )
    other = replace(
        plan,
        modes=tuple(
            replace(row, actual_ticks=999)
            if row.processing_mode_id in ("slow", "tie")
            else row
            for row in plan.modes
        ),
    )
    left, right = [
        Simulator(factory, workload, processing_times=p) for p in (plan, other)
    ]
    assert left.current_decision == right.current_decision
    chosen = SPTPolicy().select_action(left.current_decision)
    assert chosen == Dispatch("A1", "fast")
    assert left.step(chosen) == right.step(chosen)
    # A remains processing and its future completion is not a snapshot field.
    assert left.current_decision.operations[0].completion_time is None
    assert "actual_ticks" not in json.dumps(primitive(left.current_decision))
    lresult, rresult = [sim.run(SPTPolicy()) for sim in (left, right)]
    assert lresult == rresult
    assert (
        next(
            entry.completion_time
            for entry in lresult.schedule
            if entry.operation_id == "A1"
        )
        == 10
    )


def test_unknown_actual_does_not_affect_observed_prefix_before_completion():
    factory, workload, plan = processing_case()
    changed = replace(
        plan, modes=tuple(replace(row, actual_ticks=30) for row in plan.modes)
    )
    left, right = [
        Simulator(factory, workload, processing_times=p) for p in (plan, changed)
    ]
    for command in (action("A1"), WaitUntil(1)):
        assert left.current_decision == right.current_decision
        assert left.step(command) == right.step(command)
        assert left.trace == right.trace
    assert left.current_decision.simulation_time == 1


def test_unit_multiplier_and_explicit_nominal_plan_preserve_old_trace():
    factory, workload, actions = competition_case()
    plan, draws = generate_processing_times(
        workload, UniformMultiplierProfile(1, 1), 42
    )
    assert all(row.draw is None for row in draws)
    assert replay(factory, workload, actions, processing_times=plan) == replay(
        factory, workload, actions
    )
    with pytest.raises(FrozenInstanceError):
        plan.modes = ()
    with pytest.raises(FrozenInstanceError):
        plan.modes[0].actual_ticks = 1
    sim = Simulator(factory, workload, processing_times=plan)
    with pytest.raises(TypeError):
        sim._processing_times.durations[("A1", "standard")] = 1


@pytest.mark.parametrize("trigger", TRIGGERS)
def test_arrival_combination_waits_and_schedule_replay(trigger):
    factory, workload, plan = processing_case()
    arrivals = ArrivalPlan((JobArrival("A", 3, 1), JobArrival("B", 5, 2)))
    sim = Simulator(
        factory,
        workload,
        processing_times=plan,
        arrivals=arrivals,
        decision_trigger=trigger,
    )
    result = sim.run(SPTPolicy())
    assert min(entry.start_time for entry in result.schedule) >= 3
    assert (
        replay(
            factory,
            workload,
            result.actions,
            processing_times=plan,
            arrivals=arrivals,
            decision_trigger=trigger,
        )
        == result
    )
    assert (
        replay_schedule(
            factory,
            workload,
            result.schedule,
            processing_times=plan,
            arrivals=arrivals,
            decision_trigger=trigger,
        ).schedule
        == result.schedule
    )


def test_simultaneous_actual_completion_is_settled_before_decision():
    factory, workload, plan = processing_case()
    plan = replace(
        plan,
        modes=tuple(
            replace(row, actual_ticks=6) if row.operation_id == "A1" else row
            for row in plan.modes
        ),
    )
    result = Simulator(factory, workload, processing_times=plan).run(SPTPolicy())
    assert [row.kind for row in result.trace if row.simulation_time == 6][:3] == [
        "complete",
        "complete",
        "decision",
    ]


@pytest.mark.parametrize(
    "nominal,actual", [(True, 1), (1, True), (1, 1.0), (0, 1), (1, 0), (1, -1)]
)
def test_invalid_duration_scalars(nominal, actual):
    with pytest.raises(ValueError):
        ProcessingTime("A1", "standard", nominal, actual)


def test_input_coverage_nominal_and_pending_event_invariants():
    factory, workload, plan = processing_case()
    for entries in [
        (),
        plan.modes[:-1],
        plan.modes + (plan.modes[0],),
        (replace(plan.modes[0], operation_id="unknown"), *plan.modes[1:]),
        (replace(plan.modes[0], nominal_ticks=100), *plan.modes[1:]),
    ]:
        with pytest.raises(ValueError):
            Simulator(factory, workload, processing_times=ProcessingTimePlan(entries))
    sim = Simulator(factory, workload, processing_times=plan)
    sim.step(action("A1"))
    pending = sim._calendar.pop()
    sim._calendar.schedule(replace(pending, simulation_time=10))
    with pytest.raises(InvariantViolation, match="pending completion"):
        sim._check_invariants()


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
