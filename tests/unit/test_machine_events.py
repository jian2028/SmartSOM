import json
import random
import runpy
from dataclasses import replace
from pathlib import Path

import pytest
from reference_cases import competition_case

from smartsom.domain import (
    MachineOutage,
    MachineOutagePlan,
)
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


@pytest.mark.parametrize(
    "start,end", [(True, 2), (0, False), (0.0, 2), (0, 2.0), (-1, 2), (2, 2), (3, 2)]
)
def test_invalid_outage_scalars(start, end):
    with pytest.raises(ValueError):
        MachineOutage("M1", start, end)


def test_fixed_reference_fixtures_without_optional_solvers():
    root = Path(__file__).resolve().parents[2]
    reference = runpy.run_path(str(root / "scripts/validate_machine_events.py"))
    fixture = json.loads(
        (root / "data/reference/machine_events/cases.json").read_text()
    )
    assert [
        reference["interval_reference"](case).makespan for case in fixture["cases"]
    ] == [7, 9, 5, 7]


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
