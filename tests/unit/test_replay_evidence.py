"""Presentation metrics must retain evidence and quality boundaries."""

from copy import deepcopy

import pytest

from smartsom.studio.replay_evidence import (
    ReplayEvidence,
    displayed_quality,
    movement_conflicts,
    resource_conflicts,
    short_job,
    submitted,
)


class Recording:
    def __init__(self, rows):
        self.rows = rows
        self.last_tick = len(rows) - 1

    def row(self, tick):
        return self.rows[tick]


def frame(tick, good, released, events=(), batch=()):
    return {
        "tick": tick,
        "events": events,
        "state": {
            "completed": list(range(good)),
            "shipped": list(range(good)),
            "output_semantics": "blind-shipment/v1",
            "released": list(range(released)),
            "metrics": {},
            "stations": {"q": {"batch": batch}},
            "jobs": {},
        },
    }


def test_rolling_rate_and_recorded_arrival_do_not_count_replacement_attempts():
    rows = [frame(0, 0, 3), frame(1, 1, 3), frame(2, 1, 5), frame(3, 2, 5)]
    rows[1]["events"] = [{"kind": "attempt_queued", "job": "replacement"}]
    original = deepcopy(rows)
    evidence = ReplayEvidence(Recording(rows))
    assert evidence.throughput(0) is None
    assert evidence.throughput(1) == 1
    assert evidence.throughput(3, 2) == 0.5
    assert evidence.throughput(3, 100) == pytest.approx(2 / 3)
    assert evidence.next_arrival(0) == (2, 2)
    assert evidence.next_arrival(2) is None
    assert rows == original


def test_single_tick_inspection_counts_busy_tick_and_completed_jobs():
    start = {"kind": "inspection_started", "station": "q", "jobs": ["j1", "j2"]}
    done = {"kind": "inspection_completed", "station": "q", "jobs": ["j1", "j2"]}
    evidence = ReplayEvidence(
        Recording([frame(0, 0, 1), frame(1, 0, 1, [start, done])])
    )
    assert evidence.station_busy[1]["q"] == 1
    assert evidence.inspected[1]["q"] == 2


def test_quality_override_never_reveals_latent_defect_or_mutates_public_quality():
    state = {
        "jobs": {"job": {"quality": "PASS", "defective": True}},
        "machines": {"machine": {"status": "PROCESSING"}},
    }
    assert displayed_quality(state, "job", "machine") == "UNKNOWN"
    assert state["jobs"]["job"]["quality"] == "PASS"
    assert displayed_quality(state, "job", "buffer") == "PASS"
    state["jobs"]["job"]["quality"] = "FAIL"
    assert displayed_quality(state, "job", "machine") == "FAIL"


def test_conflicts_distinguish_movement_interaction_and_wait_in_current_format():
    current = {
        "actions": {"movers": [("a", "UP"), ("b", "INTERACT"), ("c", "WAIT")]},
        "rejections": {f"agv:{k}": "conflict" for k in "abc"},
    }
    assert movement_conflicts(current) == {"a"}
    assert resource_conflicts(current) == {"b"}


def test_unknown_denominator_is_not_zero_and_job_labels_include_attempt():
    assert submitted({"metrics": {}}) is None
    assert submitted({"metrics": {}, "completed": []}) == 0
    assert short_job("order_01/attempt/2") == "1:2"


def test_per_output_metrics_and_mode_starts_are_independent_of_duration():
    from types import SimpleNamespace

    rows = [frame(0, 0, 3), frame(1, 1, 3), frame(2, 2, 3), frame(3, 2, 3)]
    rows[1]["events"] = [
        {"kind": "drop_off", "owner": "exit_a", "job": "a"},
        {"kind": "processing_started", "machine": "m", "mode": "fast"},
    ]
    rows[1]["state"]["jobs"] = {"a": {"quality": "PASS"}}
    rows[2]["events"] = [
        {"kind": "drop_off", "owner": "exit_a", "job": "b"},
        {"kind": "drop_off", "owner": "exit_b", "job": "c"},
    ]
    rows[2]["state"]["jobs"] = {"b": {"quality": "FAIL"}, "c": {"quality": "PASS"}}
    rows[3]["events"] = [{"kind": "processing_started", "machine": "m", "mode": "slow"}]
    factory = SimpleNamespace(
        buffers=[
            SimpleNamespace(buffer_id=k, role="system_output")
            for k in ("exit_a", "exit_b")
        ]
    )
    evidence = ReplayEvidence(Recording(rows), factory)
    assert evidence.output_metrics("exit_a", 0) == (0, None, None)
    assert evidence.output_metrics("exit_a", 2) == (2, None, 1.0)
    assert evidence.output_metrics("exit_b", 2) == (1, None, 0.5)
    assert evidence.output_metrics("exit_a", 3, window=1) == (2, None, 0.0)
    assert evidence.mode_counts[1] == {"m": {"fast": 1}}
    assert evidence.mode_counts[2] == evidence.mode_counts[1]
    assert evidence.mode_counts[3] == {"m": {"fast": 1, "slow": 1}}


def test_input_integer_countdown_resets_only_on_new_recorded_release():
    from types import SimpleNamespace

    factory = SimpleNamespace(
        buffers=[SimpleNamespace(buffer_id="in", role="system_input")]
    )
    rows = [frame(t, 0, 1 + (t >= 3) + (t >= 5)) for t in range(7)]
    for row in rows:
        row["state"]["queue"] = ["waiting"] if row["tick"] >= 3 else []
    evidence = ReplayEvidence(Recording(rows), factory)
    assert [evidence.input_progress("in", t) for t in range(3)] == [
        (3, 3, 1),
        (3, 2, 1),
        (3, 1, 1),
    ]
    assert evidence.input_progress("in", 3) == (2, 2, 1)
    assert evidence.input_progress("in", 5) is None
    assert evidence.input_progress("in", 1) == (3, 2, 1)  # Seeking backward.
    assert evidence.input_waiting("in", rows[3]["state"]) == 1
    assert evidence.input_waiting("in", rows[0]["state"]) == 0


def test_multiple_inputs_keep_arrivals_and_waiting_separate():
    from types import SimpleNamespace

    factory = SimpleNamespace(
        buffers=[SimpleNamespace(buffer_id=k, role="system_input") for k in ("a", "b")]
    )
    rows = [frame(t, 0, 0) for t in range(4)]
    for row in rows:
        row["state"]["released"] = ["d"] if row["tick"] >= 2 else []
        row["state"]["jobs"] = {"j": {"demand": "d"}}
        row["state"]["queue"] = ["j"] if row["tick"] >= 2 else []
    recording = Recording(rows)
    recording.manifest = {
        "inputs": {"scenario": {"demands": [{"demand_id": "d", "input_id": "b"}]}}
    }
    evidence = ReplayEvidence(recording, factory)
    assert evidence.input_progress("a", 0) is None
    assert evidence.input_progress("b", 0) == (2, 2, 1)
    assert evidence.input_waiting("a", rows[2]["state"]) == 0
    assert evidence.input_waiting("b", rows[2]["state"]) == 1
    assert evidence.input_waiting("a", {"metrics": {"external_backlog": 7}}) is None


def test_quality_history_and_event_index_exclude_latent_fields():
    rows = [frame(0, 0, 1), frame(1, 1, 1)]
    rows[1]["state"]["metrics"]["submitted"] = 2
    rows[1]["events"] = [
        {
            "kind": "processing_started",
            "machine": "m",
            "mode": "normal",
            "job": "j",
            "defective": True,
            "risk": 0.9,
        }
    ]
    evidence = ReplayEvidence(Recording(rows))
    assert evidence.passing_rates == [None, None]
    assert evidence.recorded_events == [(1, "processing_started", "m · j")]


def test_automatic_inspection_counts_elapsed_time_not_arrival_boundary():
    rows = [frame(t, 0, 1, batch=["j"] if t in (1, 2) else []) for t in range(5)]
    for row in rows:
        row["state"]["stations"]["q"]["jobs"] = {}
    rows[1]["events"] = [
        {"kind": "inspection_started", "station": "q", "job": "j", "jobs": ["j"]}
    ]
    rows[3]["events"] = [
        {"kind": "inspection_completed", "station": "q", "job": "j", "jobs": ["j"]}
    ]
    rows[4]["events"] = [
        {"kind": "automatic_disposal", "station": "q", "job": "j", "owner": "bin"}
    ]
    evidence = ReplayEvidence(Recording(rows))
    assert evidence.station_busy[1]["q"] == 0
    assert evidence.station_busy[3]["q"] == 2
    assert evidence.station_busy[4]["q"] == 2
    assert evidence.inspected[4]["q"] == 1
