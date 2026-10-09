"""Composition reports keep population sizes, clocks and validation gaps."""

import json

import pytest

from smartsom.experiments.composition_results import aggregate_cases
from smartsom.experiments.report import build_report, export_figure, load_report_data


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def case(value, status="truncated", **extra):
    return {
        "return": value,
        "delivered": value,
        "makespan": value,
        "status": status,
        "replication": 0,
        **extra,
    }


@pytest.fixture
def saved(tmp_path):
    write(
        tmp_path / "run.json",
        {"schema": "smartsom.experiment/v3", "kind": "training", "updates": 3},
    )
    write(
        tmp_path / "config/prepared.json",
        {
            "config_json": json.dumps(
                {"validation": {"enabled": True, "every_updates": 1}}
            ),
            "validation_json": json.dumps([{}, {}, {}, {}]),
        },
    )
    write(
        tmp_path / "reports/training.json",
        [
            {"update": u, "physical_ticks": t, "optimizations": {"machine": u}}
            for u, t in [(1, 100), (2, 300), (3, 450)]
        ],
    )
    write(
        tmp_path / "logs/validation-000001.json",
        [case(0, "completed"), case(20), case(20), case(20, engineering_failure=True)],
    )
    write(tmp_path / "logs/validation-000003.json", [case(5)])
    return tmp_path


def test_report_preserves_validation_gaps_and_effective_samples(saved):
    row = load_report_data(saved)["runs"][0]
    first, gap, last = row["training"]["validation"]
    assert [r["physical_ticks"] for r in (first, gap, last)] == [100, 300, 450]
    assert first["mean_return"] == pytest.approx(40 / 3)
    assert first["return_samples"] == 3 and first["makespan_samples"] == 1
    assert first["mean_makespan"] == 0
    assert not gap["saved"] and gap["missing"] == 4 and gap["mean_return"] is None
    assert last["missing"] == 3 and last["mean_makespan"] is None
    curve = next(s for s in row["series"] if s["label"] == "Validation return")
    assert curve["points"][1] == [300, None]
    html = build_report(saved, saved / "reports/index.html").read_text()
    assert (
        '"makespan_samples": 1' in html and "if(p[1]===null){line();continue}" in html
    )


def test_old_results_use_recorded_update_and_episode_axes(saved):
    write(saved / "reports/training.json", [{"update": u} for u in (1, 2, 3)])
    write(
        saved / "reports/training-episodes.json",
        [{"case_id": "0", "replication": 8, **case(12)}],
    )
    # Restore the real saved episode number, not a guessed tick count.
    episodes = json.loads((saved / "reports/training-episodes.json").read_text())
    episodes[0]["replication"] = 8
    write(saved / "reports/training-episodes.json", episodes)
    row = load_report_data(saved)["runs"][0]
    assert all(
        "physical ticks" not in s["x_label"].lower() or "unavailable" in s["x_label"]
        for s in row["series"]
    )
    assert next(s for s in row["series"] if s["label"] == "Validation return")[
        "points"
    ][1] == [2, None]
    assert (
        next(s for s in row["series"] if s["label"].startswith("Training env"))[
            "points"
        ][0][0]
        == 8
    )


def test_independent_evaluation_is_available_without_trace_or_weights(saved):
    write(
        saved / "run.json",
        {
            "schema": "smartsom.experiment/v3",
            "kind": "evaluation",
            "results": [case(10, "completed"), case(99), case(2, "exception")],
        },
    )
    row = load_report_data(saved)["runs"][0]
    assert row["summary"]["mean_makespan"] == 10
    assert row["summary"]["makespan_samples"] == 1
    assert row["summary"]["exceptions"] == 1
    assert row["evaluation"][1]["status"] == "truncated"
    assert not row["events"]


def test_figure_keeps_null_gap(saved, tmp_path):
    pytest.importorskip("matplotlib")
    export_figure(saved, tmp_path / "reports/curve.svg")
    assert "<svg" in (tmp_path / "reports/curve.svg").read_text()


def test_aggregate_excludes_error_and_nonfinite_values():
    result = aggregate_cases(
        [case(0, "completed"), case(float("nan")), case(9, "exception")], 4
    )
    assert result["return_samples"] == 1 and result["mean_return"] == 0
    assert result["makespan_samples"] == 1 and result["missing"] == 1


def test_partial_legacy_clock_uses_updates_without_bridging_gap(saved):
    write(
        saved / "reports/training.json",
        [
            {"update": 1, "physical_ticks": 100},
            {"update": 2},
            {"update": 3, "physical_ticks": 450},
        ],
    )
    row = load_report_data(saved)["runs"][0]
    curve = next(s for s in row["series"] if s["label"] == "Validation return")
    assert curve["x_label"] == "Update (recorded; physical ticks unavailable)"
    assert [p[0] for p in curve["points"]] == [1, 2, 3]
    assert curve["points"][1][1] is None
    assert row["training"]["validation"][0]["physical_ticks"] == 100
