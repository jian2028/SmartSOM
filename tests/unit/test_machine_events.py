import json
import random
import runpy
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest
from test_static_engine import action, competition_case, op, problem

from smartsom.algorithms import SPTPolicy
from smartsom.config.codec import primitive
from smartsom.dispatch import Dispatch, WaitUntil
from smartsom.domain import (
    ArrivalPlan,
    Job,
    JobArrival,
    MachineOutage,
    MachineOutagePlan,
    Operation,
    ProcessingMode,
    ProcessingTime,
    ProcessingTimePlan,
    ScheduledOperation,
)
from smartsom.engine import (
    InvalidActionError,
    InvariantViolation,
    ReplayError,
    Simulator,
    replay,
    replay_schedule,
)
from smartsom.engine.calendar import CompletionEvent, EventCalendar
from smartsom.modules.machine_events import MachineEventModule
from smartsom.workloads.machine_events import (
    MachineEventProfile,
    MachineFailureProfile,
    generate_machine_events,
    machine_seed,
)
from smartsom.workloads.static_jsp import IntegerRange


def outage_plan(*windows, machine="M1"):
    return MachineOutagePlan(
        tuple(MachineOutage(machine, start, end) for start, end in windows)
    )


def processing_segments(result):
    """Reconstruct active work solely from actual semantic records."""
    active, segments = {}, {}
    for row in result.trace:
        if row.kind in ("dispatch", "resume"):
            key = row.action.operation_id
            assert key not in active
            active[key] = row.simulation_time
        elif row.kind in ("pause", "complete"):
            key = row.action.operation_id
            segments.setdefault(key, []).append((active.pop(key), row.simulation_time))
    assert not active
    return segments


@pytest.mark.parametrize(
    "windows,segments,end",
    [
        (((2, 4),), [(0, 2), (4, 7)], 7),
        (((1, 3), (4, 6)), [(0, 1), (3, 4), (6, 9)], 9),
        (((5, 8),), [(0, 5)], 5),
        (((0, 3),), [(3, 8)], 8),
        (((20, 30),), [(0, 5)], 5),
    ],
)
def test_hand_segments_completion_and_both_replays(windows, segments, end):
    factory, workload = problem(Job("A", (op("A", "M1", 5),)))
    plan = outage_plan(*windows)
    result = Simulator(factory, workload, machine_events=plan).run(SPTPolicy())
    assert processing_segments(result) == {"A": segments}
    assert result.makespan == end
    assert len(result.actions) == 1 and result.actions == (action("A"),)
    assert result.schedule == (
        ScheduledOperation("A", "standard", "M1", segments[0][0], end),
    )
    assert replay(factory, workload, result.actions, machine_events=plan) == result
    assert (
        replay_schedule(factory, workload, result.schedule, machine_events=plan)
        == result
    )
    assert all(row.simulation_time <= end for row in result.trace)


def test_paused_mode_occupancy_privacy_and_atomic_rejection():
    factory, workload = problem(
        Job(
            "A",
            (
                Operation(
                    "A", (ProcessingMode("x", "M1", 5), ProcessingMode("y", "M1", 2))
                ),
            ),
        ),
        Job("B", (op("B", "M2", 1),)),
        Job("C", (op("C", "M1", 1),)),
    )
    sim = Simulator(factory, workload, machine_events=outage_plan((2, 4)))
    sim.step(Dispatch("A", "x"))
    view = sim.step(WaitUntil(2))
    paused = next(row for row in view.operations if row.operation_id == "A")
    assert (paused.status, paused.start_time, paused.processing_mode_id) == (
        "paused",
        0,
        "x",
    )
    assert paused.actual_processing_ticks is paused.completion_time is None
    assert (
        view.machines[0].availability == "down" and view.machines[0].operation_id == "A"
    )
    before = sim.trace
    for command in (Dispatch("A", "y"), action("C")):
        with pytest.raises(InvalidActionError):
            sim.step(command)
        assert sim.current_decision is view and sim.trace == before
    # Old completion at 5 must not end the wait at 5 after repair at 4.
    assert sim.step(WaitUntil(6)).simulation_time == 4
    assert sim.step(WaitUntil(6)).simulation_time == 6
    view = sim.step(WaitUntil(7))
    assert (
        next(
            row for row in view.operations if row.operation_id == "A"
        ).actual_processing_ticks
        == 5
    )
    result = sim.run(SPTPolicy())
    assert processing_segments(result)["A"] == [(0, 2), (4, 7)]
    assert (
        next(row.start_time for row in result.schedule if row.operation_id == "C") >= 7
    )


def test_future_repair_and_actual_work_do_not_leak():
    factory, workload = problem(
        Job("A", (op("A", "M1", 5),)), Job("B", (op("B", "M2", 1),))
    )
    sims = [
        Simulator(
            factory,
            workload,
            machine_events=outage_plan((2, end)),
            processing_times=ProcessingTimePlan(
                (
                    ProcessingTime("A", "standard", 5, ticks),
                    ProcessingTime("B", "standard", 1, 1),
                )
            ),
        )
        for end, ticks in ((4, 5), (10, 9))
    ]
    for command in (action("A"), WaitUntil(2), WaitUntil(3)):
        assert sims[0].current_decision == sims[1].current_decision
        assert sims[0].step(command) == sims[1].step(command)
        assert sims[0].trace == sims[1].trace
    text = json.dumps(primitive(sims[0].current_decision))
    assert "repair" not in text and "remaining" not in text


@pytest.mark.parametrize("trigger", ["dispatch_available", "arrival_event"])
def test_same_tick_phases_and_composition(trigger):
    factory, workload = problem(
        Job("A", (op("A", "M1", 5),)),
        Job("B", (op("B", "M2", 2), op("B2", "M1", 1, "B"))),
        Job("C", (op("C", "M2", 1),)),
    )
    arrivals = ArrivalPlan(
        (JobArrival("A", 0, 0), JobArrival("B", 0, 0), JobArrival("C", 2, 2))
    )
    durations = ProcessingTimePlan(
        (
            ProcessingTime("A", "standard", 5, 7),
            ProcessingTime("B", "standard", 2, 2),
            ProcessingTime("B2", "standard", 1, 1),
            ProcessingTime("C", "standard", 1, 1),
        )
    )
    plan = MachineOutagePlan((MachineOutage("M1", 2, 4), MachineOutage("M2", 2, 3)))
    kwargs = dict(
        arrivals=arrivals,
        decision_trigger=trigger,
        processing_times=durations,
        machine_events=plan,
    )
    sim = Simulator(factory, workload, **kwargs)
    result = sim.run(SPTPolicy())
    kinds = [row.kind for row in result.trace if row.simulation_time == 2]
    assert kinds[:6] == [
        "complete",
        "breakdown",
        "pause",
        "breakdown",
        "reveal",
        "release",
    ]
    assert processing_segments(result)["A"] == [(0, 2), (4, 9)]
    assert result.makespan == 10
    assert replay(factory, workload, result.actions, **kwargs) == result
    assert (
        replay_schedule(factory, workload, result.schedule, **kwargs).schedule
        == result.schedule
    )


def test_idle_down_machine_and_spt_uses_available_mode():
    factory, workload = problem(
        Job(
            "A",
            (
                Operation(
                    "A",
                    (ProcessingMode("fast", "M1", 1), ProcessingMode("slow", "M2", 3)),
                ),
            ),
        )
    )
    sim = Simulator(factory, workload, machine_events=outage_plan((0, 2)))
    assert sim.current_decision.feasible_actions == (Dispatch("A", "slow"),)
    with pytest.raises(InvalidActionError, match="down"):
        sim.step(Dispatch("A", "fast"))
    assert sim.run(SPTPolicy()).makespan == 3


def test_completion_breakdown_repair_resume_and_arrivals_share_tick():
    factory, workload = problem(
        Job("A", (op("A", "M1", 5),)),
        Job("B", (op("B", "M2", 4),)),
        Job("C", (op("C", "M2", 1),)),
    )
    arrivals = ArrivalPlan(
        (JobArrival("A", 0, 0), JobArrival("B", 0, 0), JobArrival("C", 4, 4))
    )
    plan = MachineOutagePlan((MachineOutage("M1", 2, 4), MachineOutage("M2", 4, 6)))
    result = Simulator(
        factory,
        workload,
        arrivals=arrivals,
        machine_events=plan,
        decision_trigger="arrival_event",
    ).run(SPTPolicy())
    assert [row.kind for row in result.trace if row.simulation_time == 4][:6] == [
        "complete",
        "breakdown",
        "repair",
        "resume",
        "reveal",
        "release",
    ]
    assert result.makespan == 7


def test_resumed_holder_blocks_other_job_for_entire_span():
    factory, workload = problem(
        Job("A", (op("A", "M1", 5),)), Job("B", (op("B", "M1", 1),))
    )
    plan = outage_plan((2, 4))
    schedule = (
        ScheduledOperation("A", "standard", "M1", 0, 7),
        ScheduledOperation("B", "standard", "M1", 4, 5),
    )
    with pytest.raises(ReplayError, match="overlap"):
        replay_schedule(factory, workload, schedule, machine_events=plan)
    schedule = (schedule[0], replace(schedule[1], start_time=7, completion_time=8))
    assert (
        replay_schedule(factory, workload, schedule, machine_events=plan).schedule
        == schedule
    )


def test_calendar_cancelled_root_is_skipped_with_live_later_completion():
    calendar = EventCalendar()
    calendar.schedule(CompletionEvent(5, "A", "x", "M1"))
    calendar.cancel_completion("A")
    new = CompletionEvent(9, "A", "x", "M1")
    calendar.schedule(new)
    assert calendar.pending == (new,) and calendar.next_time == 9
    assert calendar.pop() is new


def test_canonicalization_immutability_and_disabled_equivalence():
    plan = outage_plan((4, 6), (2, 4), (3, 5), (2, 4))
    assert plan == outage_plan((2, 6))
    with pytest.raises(FrozenInstanceError):
        plan.outages = ()
    with pytest.raises(FrozenInstanceError):
        plan.outages[0].end_time = 99
    factory, workload, commands = competition_case()
    assert replay(
        factory, workload, commands, machine_events=MachineOutagePlan()
    ) == replay(factory, workload, commands)
    module = MachineEventModule(factory, plan)
    with pytest.raises(TypeError):
        module.by_machine["M1"] = None
    result = Simulator(factory, workload, machine_events=plan).run(SPTPolicy())
    assert (
        Simulator(
            replace(factory, machines=tuple(reversed(factory.machines))),
            replace(
                workload,
                orders=tuple(
                    replace(order, jobs=tuple(reversed(order.jobs)))
                    for order in workload.orders
                ),
            ),
            machine_events=plan,
        ).run(SPTPolicy())
        == result
    )


@pytest.mark.parametrize(
    "start,end", [(True, 2), (0, False), (0.0, 2), (0, 2.0), (-1, 2), (2, 2), (3, 2)]
)
def test_invalid_outage_scalars(start, end):
    with pytest.raises(ValueError):
        MachineOutage("M1", start, end)


def test_invalid_references_and_progress_invariants():
    factory, workload = problem(
        Job("A", (op("A", "M1", 5),)), Job("B", (op("B", "M2", 1),))
    )
    with pytest.raises(ValueError):
        Simulator(factory, workload, machine_events=outage_plan((1, 2), machine="bad"))
    sim = Simulator(factory, workload, machine_events=outage_plan((2, 4)))
    sim.step(action("A"))
    sim.step(WaitUntil(2))
    sim._state.progress["A"] = replace(sim._state.progress["A"], processed_ticks=3)
    with pytest.raises(InvariantViolation, match="conserved"):
        sim._check_invariants()


def test_cancelled_completion_never_exposed_or_consumed():
    calendar = EventCalendar()
    old = CompletionEvent(5, "A", "standard", "M1")
    calendar.schedule(old)
    calendar.cancel_completion("A")
    assert calendar.pending == () and calendar.next_time is None
    new = replace(old, simulation_time=7)
    calendar.schedule(new)
    assert calendar.pending == (new,) and calendar.next_time == 7
    assert calendar.pop() is new and calendar.next_time is None


def test_fixed_reference_fixtures_without_optional_solvers():
    root = Path(__file__).resolve().parents[2]
    reference = runpy.run_path(str(root / "scripts/validate_machine_events.py"))
    fixture = json.loads(
        (root / "data/reference/machine_events/cases.json").read_text()
    )
    assert [
        reference["core_reference"](case).makespan for case in fixture["cases"]
    ] == [7, 9, 5, 7]


@pytest.mark.parametrize("start,end", [(2, 9), (0, 5), (0, 8), (0, 6), (0, 4)])
def test_schedule_rejects_down_start_wrong_work_and_post_repair_delay(start, end):
    factory, workload = problem(Job("A", (op("A", "M1", 5),)))
    with pytest.raises(ReplayError):
        replay_schedule(
            factory,
            workload,
            (ScheduledOperation("A", "standard", "M1", start, end),),
            machine_events=outage_plan((2, 4)),
        )
    # Five units finish at 5; delaying completion through [5,8) is invalid.
    with pytest.raises(ReplayError):
        replay_schedule(
            factory,
            workload,
            (ScheduledOperation("A", "standard", "M1", 0, 8),),
            machine_events=outage_plan((5, 8)),
        )


def test_small_fjsp_arrival_uncertainty_compositions_against_segment_arithmetic():
    from smartsom.workloads import StaticFJSPProfile, generate_fjsp

    factory, _, _ = competition_case()
    for seed in range(20):
        workload = generate_fjsp(
            factory,
            StaticFJSPProfile(
                1, 2, IntegerRange(3, 3), IntegerRange(1, 2), IntegerRange(1, 5)
            ),
            seed,
        )
        rng = random.Random(seed)
        times = ProcessingTimePlan(
            tuple(
                ProcessingTime(
                    op.operation_id,
                    mode.processing_mode_id,
                    mode.nominal_ticks,
                    rng.randint(1, 8),
                )
                for op in workload.operations
                for mode in op.modes
            )
        )
        arrivals = ArrivalPlan(
            tuple(
                JobArrival(job.job_id, index * 3, index)
                for index, job in enumerate(workload.orders[0].jobs)
            )
        )
        plan = generate_machine_events(factory, profile(), seed)
        result = Simulator(
            factory,
            workload,
            machine_events=plan,
            processing_times=times,
            arrivals=arrivals,
        ).run(SPTPolicy())
        segments = processing_segments(result)
        actual_ticks = {
            (row.operation_id, row.processing_mode_id): row.actual_ticks
            for row in times.modes
        }
        schedule = {row.operation_id: row for row in result.schedule}
        for row in result.schedule:
            parts = segments[row.operation_id]
            assert (
                sum(end - start for start, end in parts)
                == actual_ticks[(row.operation_id, row.processing_mode_id)]
            )
            assert all(start < end for start, end in parts)
            assert all(
                end <= outage.start_time or start >= outage.end_time
                for start, end in parts
                for outage in plan.outages
                if outage.machine_id == row.machine_id
            )
        for operation in workload.operations:
            assert all(
                schedule[pred].completion_time
                <= schedule[operation.operation_id].start_time
                for pred in operation.predecessor_ids
            )
        for machine in factory.machines:
            spans = sorted(
                (row.start_time, row.completion_time)
                for row in result.schedule
                if row.machine_id == machine.machine_id
            )
            assert all(left[1] <= right[0] for left, right in zip(spans, spans[1:]))
        reordered = replace(
            workload,
            orders=tuple(
                replace(
                    order,
                    jobs=tuple(
                        replace(
                            job,
                            operations=tuple(
                                replace(op, modes=tuple(reversed(op.modes)))
                                for op in reversed(job.operations)
                            ),
                        )
                        for job in reversed(order.jobs)
                    ),
                )
                for order in workload.orders
            ),
        )
        assert (
            Simulator(
                factory,
                reordered,
                machine_events=plan,
                processing_times=times,
                arrivals=arrivals,
            ).run(SPTPolicy())
            == result
        )


def profile():
    return MachineEventProfile(
        20,
        (
            MachineFailureProfile("M1", 3, IntegerRange(1, 3)),
            MachineFailureProfile("M2", 5, IntegerRange(2, 4)),
        ),
    )


def test_generator_golden_order_prefix_independence_and_rng_isolation():
    factory, _, _ = competition_case()
    before = random.getstate()
    actual = generate_machine_events(factory, profile(), 42)
    assert random.getstate() == before
    expected = GOLDEN
    assert [
        (r.machine_id, r.start_time, r.end_time) for r in actual.outages
    ] == expected
    assert (
        generate_machine_events(
            replace(factory, machines=tuple(reversed(factory.machines))),
            replace(profile(), machines=tuple(reversed(profile().machines))),
            42,
        )
        == actual
    )
    changed = replace(
        profile(),
        machines=(
            profile().machines[0],
            replace(profile().machines[1], mean_uptime_ticks=2),
        ),
    )
    assert [
        r
        for r in generate_machine_events(factory, changed, 42).outages
        if r.machine_id == "M1"
    ] == [r for r in actual.outages if r.machine_id == "M1"]
    longer = generate_machine_events(
        factory, replace(profile(), generation_until_tick=50), 42
    )
    assert [r for r in longer.outages if r.start_time < 20] == list(actual.outages)
    assert machine_seed(42, "M1") != machine_seed(42, "M2")


@pytest.mark.parametrize("mean", [True, False, 0, -1, float("inf"), float("nan"), "2"])
def test_invalid_mean(mean):
    with pytest.raises(ValueError):
        MachineFailureProfile("M1", mean, IntegerRange(1, 2))


def test_generator_empty_and_window_preserves_full_repair():
    factory, _, _ = competition_case()
    empty = replace(profile(), generation_until_tick=1)
    assert generate_machine_events(factory, empty, 42) == MachineOutagePlan()
    short = MachineEventProfile(
        2, (MachineFailureProfile("M1", 1e-300, IntegerRange(5, 5)),)
    )
    assert generate_machine_events(factory, short, 42) == outage_plan((1, 6))
    with pytest.raises(ValueError):
        generate_machine_events(
            factory,
            replace(
                profile(), machines=(replace(profile().machines[0], machine_id="bad"),)
            ),
            42,
        )
    for seed in (True, -1, 2**64, 1.0):
        with pytest.raises(ValueError):
            generate_machine_events(factory, profile(), seed)


GOLDEN = [
    ("M1", 1, 4),
    ("M1", 8, 10),
    ("M1", 11, 13),
    ("M1", 14, 15),
    ("M1", 16, 19),
    ("M2", 6, 9),
    ("M2", 15, 19),
]
