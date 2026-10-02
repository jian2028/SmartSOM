"""Task performance keeps one definition and reports unavailable evidence."""

from collections import Counter
from types import SimpleNamespace

import pytest

from smartsom.trace.performance import (
    TaskPerformance,
    lateness,
    qualified_count,
    submitted_count,
    tardiness_totals,
    theoretical_reference,
)


class Recording:
    def __init__(self, rows, demands=()):
        self.rows = rows
        self.last_tick = len(rows) - 1
        self.manifest = {"inputs": {"scenario": {"demands": list(demands)}}}

    def row(self, tick):
        return self.rows[tick]


def frame(tick, completed, submitted=None):
    state = {"completed": list(completed), "metrics": {}}
    if submitted is not None:
        state["metrics"]["submitted"] = submitted
    return {"tick": tick, "state": state, "events": []}


def counted(tick, passed, submitted):
    """A recording that stores counts only, without completion identities."""
    return {
        "tick": tick,
        "state": {"metrics": {"passed": passed, "submitted": submitted}},
        "events": [],
    }


DEMANDS = [
    {"demand_id": "order_01", "due_at": 10},
    {"demand_id": "order_02", "due_at": 10},
    {"demand_id": "order_03", "due_at": 100},
]


def test_hand_computed_cumulative_and_window_values():
    # order_01 on time at 8, order_02 five ticks late at 15, order_03 early at 20.
    rows = [
        frame(0, [], 0),
        *[frame(t, [], 0) for t in range(1, 8)],
        *[frame(t, ["order_01"], 1) for t in range(8, 15)],
        *[frame(t, ["order_01", "order_02"], 3) for t in range(15, 20)],
        frame(20, ["order_01", "order_02", "order_03"], 4),
    ]
    performance = TaskPerformance(Recording(rows, DEMANDS))

    assert performance.completions == [
        (8, "order_01", 0),
        (15, "order_02", 5),
        (20, "order_03", 0),
    ]
    cumulative = performance.cumulative(20)
    assert cumulative["qualified"] == 3
    assert cumulative["submitted"] == 4
    assert cumulative["passing_rate"] == pytest.approx(3 / 4)
    assert cumulative["throughput"] == pytest.approx(3 / 20)
    assert cumulative["total_tardiness"] == 5
    assert cumulative["tardy_jobs"] == 1
    assert cumulative["mean_tardiness"] == pytest.approx(5 / 3)

    # Trailing window of 10 ticks: deliveries at 15 and 20, submissions 1 -> 4.
    recent = performance.window(20, 10)
    assert (recent["span"], recent["deliveries"]) == (10, 2)
    assert recent["throughput"] == pytest.approx(0.2)
    assert recent["passing_rate"] == pytest.approx(2 / 3)
    assert recent["tardiness"] == 5


def test_tick_zero_has_no_rate_and_window_never_precedes_the_run():
    rows = [frame(0, [], 0), frame(1, ["order_01"], 1)]
    performance = TaskPerformance(Recording(rows, DEMANDS))
    assert performance.cumulative(0)["throughput"] is None
    assert performance.window(0, 100)["throughput"] is None
    assert performance.window(1, 100) == {
        "span": 1,
        "deliveries": 1,
        "throughput": 1.0,
        "passing_rate": 1.0,
        "tardiness": 0,
        "tardy_jobs": 0,
    }


def test_counts_without_identities_report_unavailable_tardiness():
    rows = [counted(0, 0, 0), counted(1, 1, 1), counted(2, 2, 3)]
    performance = TaskPerformance(Recording(rows))
    assert performance.identified is False
    cumulative = performance.cumulative(2)
    assert cumulative["qualified"] == 2
    assert cumulative["passing_rate"] == pytest.approx(2 / 3)
    assert cumulative["total_tardiness"] is None
    assert cumulative["tardy_jobs"] is None
    assert performance.window(2, 2)["tardiness"] is None


def test_missing_due_tick_is_unavailable_rather_than_on_time():
    rows = [frame(0, [], 0), frame(1, ["order_99"], 1)]
    performance = TaskPerformance(Recording(rows, DEMANDS))
    assert performance.identified is False
    assert performance.completions == []
    assert performance.cumulative(1)["total_tardiness"] is None


def test_lateness_and_totals_are_shared_by_every_reporting_path():
    assert lateness(10, 12) == 0
    assert lateness(12, 12) == 0
    assert lateness(15, 12) == 3
    assert tardiness_totals([(15, 12), (10, 12), (20, 12)]) == {
        "total_tardiness": 11,
        "tardy_jobs": 2,
        "mean_tardiness": pytest.approx(11 / 3),
    }
    assert tardiness_totals([]) == {
        "total_tardiness": 0,
        "tardy_jobs": 0,
        "mean_tardiness": None,
    }


def test_submission_counts_fall_back_to_recorded_shape():
    assert submitted_count({"metrics": {"submitted": 4}}) == 4
    assert submitted_count({"metrics": {}, "completed": []}) == 0
    assert submitted_count({"metrics": {}}) is None
    assert qualified_count({"metrics": {"fulfilled": 7}}) == 7
    assert qualified_count({"metrics": {"passed": 5}}) == 5
    assert qualified_count({"completed": ["a", "b"], "metrics": {}}) == 2


class Step:
    def __init__(self, operation_type, nominal_ticks):
        self.operation_type = operation_type
        self.nominal_ticks = nominal_ticks


class Demand:
    def __init__(self, steps):
        self.steps = steps


class Machine:
    def __init__(self, operation_types):
        self.operation_types = operation_types


class Factory:
    def __init__(self, machines):
        self.machines = machines


class Scenario:
    def __init__(self, machines, demands, tick_limit=0):
        self.factory = Factory(machines)
        self.demands = demands
        self.tick_limit = tick_limit


def test_theoretical_reference_uses_the_busier_of_route_and_load():
    # Four jobs, each 10 ticks of A then 2 of B; two A machines and one B machine.
    demands = [Demand([Step("a", 10), Step("b", 2)]) for _ in range(4)]
    scenario = Scenario(
        [Machine(("a",)), Machine(("a",)), Machine(("b",))], demands, tick_limit=100
    )
    reference = theoretical_reference(scenario)
    assert reference["available"] is True
    assert reference["longest_route_ticks"] == 12
    # A: 40 ticks over two machines = 20; B: 8 ticks over one machine = 8.
    assert reference["busiest_operation_ticks"] == pytest.approx(20)
    assert reference["makespan_lower_bound"] == pytest.approx(20)
    # A allows 2*4/40 = 0.2 jobs/tick; B allows 1*4/8 = 0.5. The bound is the lower.
    assert reference["max_throughput_jobs_per_tick"] == pytest.approx(0.2)
    assert reference["max_qualified_in_horizon"] == 4
    assert (
        "no transport, buffer blocking, inspection or disposal time"
        in (reference["assumptions"])
    )


def test_theoretical_reference_declines_when_no_machine_is_capable():
    scenario = Scenario([Machine(("a",))], [Demand([Step("z", 5)])])
    reference = theoretical_reference(scenario)
    assert reference["available"] is False
    assert "operation types: z" in reference["reason"]


def test_absent_delivery_evidence_is_unknown_not_zero():
    missing = {"metrics": {}}
    performance = TaskPerformance()
    performance.observe(0, missing)
    performance.observe(1, missing)
    assert performance.cumulative(1)["qualified"] is None
    assert performance.cumulative(1)["throughput"] is None
    assert performance.window(1)["deliveries"] is None
    assert performance.window(1)["throughput"] is None


@pytest.mark.parametrize(
    "capabilities, routes, horizon, expected",
    [
        ([("a",), ("a",)], [[("a", 10)]] * 2, 10, 2),
        ([("a",), ("a",)], [[("a", 10)]] * 2, 9, 0),
        ([("a",)], [[("a", 100)], [("a", 2)], [("a", 2)]], 4, 2),
        ([("a",)], [[("a", 3)], [("a", 7)], [("a", 8)]], 10, 2),
        ([("a", "b")], [[("a", 5), ("b", 5)]] * 2, 10, 1),
        ([("a",), ("b",)], [[("a", 1), ("b", 9)]] * 3, 18, 2),
        ([("a",), ("b",)], [[("a", 1)], [("b", 1)]], 1, 2),
        ([("a",)], [], 10, 0),
        ([("a",)], [[("a", 10)]], 0, None),
    ],
)
def test_horizon_bound_respects_parallelism_and_individual_work(
    capabilities, routes, horizon, expected
):
    scenario = Scenario(
        [Machine(types) for types in capabilities],
        [Demand([Step(op, ticks) for op, ticks in route]) for route in routes],
        tick_limit=horizon,
    )
    assert theoretical_reference(scenario)["max_qualified_in_horizon"] == expected


@pytest.mark.parametrize("status", ["completed", "truncated"])
def test_evaluation_and_replay_count_only_successful_replacement(status):
    from smartsom.experiments.composable import episode_metrics

    scenario = Scenario([Machine(("a",))], [Demand([Step("a", 1)])], 20)
    scenario.transport_matrix = None
    jobs = {
        "failed": {"demand": "d", "location": "out", "since": 12, "quality": "FAIL"},
        "passed": {"demand": "d", "location": "out", "since": 20, "quality": "PASS"},
        "pending": {
            "demand": "other",
            "location": "pre",
            "since": 5,
            "quality": "PASS",
        },
    }
    sim = SimpleNamespace(
        jobs=jobs,
        completed={"d"},
        roles={"out": "system_output", "pre": "machine_pre"},
        demands={"d": SimpleNamespace(release_at=2, due_at=10)},
        metrics=Counter(submitted=2, passed=1, output_rejected=1),
        scenario=scenario,
        status=status,
        tick=20,
        total_reward=0,
    )
    measured = episode_metrics(sim)
    performance = TaskPerformance(due={"d": 10})
    for tick in range(21):
        performance.observe(
            tick,
            frame(tick, ["d"] if tick == 20 else [], int(tick >= 12) + int(tick >= 20))[
                "state"
            ],
        )
    replay = performance.cumulative(20)
    assert measured["delivered"] == replay["qualified"] == 1
    assert measured["flow_time"] == 18
    assert measured["total_tardiness"] == replay["total_tardiness"] == 10
    assert measured["tardy_jobs"] == replay["tardy_jobs"] == 1
    assert measured["mean_tardiness"] == replay["mean_tardiness"] == 10
    assert measured["metrics"]["submitted"] == 2
    assert replay["passing_rate"] == 0.5
