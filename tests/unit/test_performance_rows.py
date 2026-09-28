"""Panel rows and jump markers read recorded evidence without inventing values."""

from smartsom.studio.replay_evidence import (
    MARKER_CATEGORIES,
    ReplayEvidence,
    performance_rows,
    scenario_reference,
)
from smartsom.trace.performance import TaskPerformance


class Recording:
    def __init__(self, rows, demands=(), manifest=None):
        self.rows = rows
        self.last_tick = len(rows) - 1
        self.manifest = (
            manifest
            if manifest is not None
            else {"inputs": {"scenario": {"demands": list(demands)}}}
        )

    def row(self, tick):
        return self.rows[tick]


def frame(tick, completed, submitted, events=(), rejections=None, actions=None):
    return {
        "tick": tick,
        "events": list(events),
        "rejections": rejections or {},
        "actions": actions or {},
        "state": {
            "completed": list(completed),
            "released": [],
            "metrics": {"submitted": submitted},
            "stations": {},
            "jobs": {},
        },
    }


DEMANDS = [{"demand_id": "order_01", "due_at": 3}]


def test_rows_show_run_window_and_bound_values():
    rows = [frame(0, [], 0), *[frame(t, ["order_01"], 2) for t in (1, 2, 3, 4, 5)]]
    performance = TaskPerformance(Recording(rows, DEMANDS))
    reference = {
        "available": True,
        "max_qualified_in_horizon": 9,
        "max_throughput_jobs_per_tick": 0.25,
        "assumptions": ["declared"],
    }
    values = {
        row["label"]: row for row in performance_rows(performance, 5, 4, reference)
    }
    assert values["Qualified jobs"]["total"] == "1"
    assert values["Qualified jobs"]["bound"] == "9"
    assert values["Throughput"]["total"] == "0.200"
    assert values["Throughput"]["bound"] == "0.250"
    assert values["Passing rate"]["total"] == "50.0%"
    assert values["Total tardiness"]["total"] == "0 ticks"
    assert values["Tardy jobs"]["total"] == "0"


def test_rows_print_unavailable_evidence_as_a_dash():
    rows = [
        {"tick": 0, "events": [], "state": {"metrics": {}, "stations": {}, "jobs": {}}}
    ]
    performance = TaskPerformance(Recording(rows))
    values = {row["label"]: row for row in performance_rows(performance, 0, 100)}
    assert values["Throughput"]["total"] == "—"
    assert values["Throughput"]["bound"] == "—"
    assert values["Passing rate"]["total"] == "—"
    assert values["Total tardiness"]["total"] == "—"
    assert values["Tardy jobs"]["total"] == "—"


def test_unavailable_reference_never_prints_a_number():
    performance = TaskPerformance(Recording([frame(0, [], 0)], DEMANDS))
    for reference in (None, {"available": False, "reason": "no capable machine"}):
        values = {
            row["label"]: row for row in performance_rows(performance, 0, 10, reference)
        }
        assert values["Qualified jobs"]["bound"] == "—"
        assert values["Throughput"]["bound"] == "—"


def test_marker_ticks_cover_each_offered_category():
    conflict = frame(
        2,
        [],
        0,
        rejections={"agv:agv_001": "conflict"},
        actions={"movers": {"agv_001": "UP"}},
    )
    rows = [
        frame(0, [], 0),
        frame(1, [], 0, events=[{"kind": "trash", "job_id": "order_02"}]),
        conflict,
        frame(3, ["order_01"], 1, events=[{"kind": "quality_revealed", "job": "j"}]),
    ]
    evidence = ReplayEvidence(Recording(rows, DEMANDS))
    assert evidence.marker_ticks("conflict") == [2]
    assert evidence.marker_ticks("scrap") == [1]
    assert evidence.marker_ticks("inspection") == [3]
    assert evidence.marker_ticks("delivery") == [3]
    assert evidence.marker_ticks("any") == [1, 3]
    assert {key for key, _label in MARKER_CATEGORIES} == {
        "conflict",
        "delivery",
        "scrap",
        "inspection",
        "any",
    }


def test_scenario_reference_is_absent_without_a_frozen_scenario():
    assert scenario_reference(Recording([], manifest={})) is None
    assert scenario_reference(Recording([], manifest={"inputs": {}})) is None
