import importlib.util
import json
import os
import struct
import sys
import xml.etree.ElementTree as ET
from dataclasses import replace
from pathlib import Path

import pytest

from smartsom.config import resolve_run
from smartsom.experiments.report import (
    build_report,
    export_figure,
    load_report_data,
    timeline,
)
from smartsom.experiments.runner import run_one

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def recorded(tmp_path):
    resolved = resolve_run(
        ROOT / "tests/fixtures/fixed_trace/configs/runs/run_fixed_trace.yaml"
    )
    resolved = replace(
        resolved,
        run=resolved.run.model_copy(update={"output_root": str(tmp_path / "runs")}),
    )
    return run_one(resolved).run_dir


def record(kind, tick, sequence, **extra):
    return {"kind": kind, "simulation_time": tick, "sequence": sequence, **extra}


def test_report_uses_actual_completed_intervals_and_preserves_all_events(recorded):
    data = load_report_data(recorded)
    run = data["runs"][0]
    assert run["summary"]["makespan"] == 6
    assert run["end_time"] == 6
    assert run["jobs"] == ["A", "B"]
    assert len(run["events"]) == len(
        (recorded / "trace.jsonl").read_text().splitlines()
    )
    assert sorted((i["label"], i["start"], i["end"]) for i in run["intervals"]) == [
        ("A1", 1, 4),
        ("A2", 4, 6),
        ("B1", 0, 1),
        ("B2", 1, 4),
    ]


def test_breakdown_pause_segments_do_not_claim_processing_during_downtime():
    action = {"operation_id": "op", "processing_mode_id": "mode"}
    trace = [
        record("dispatch", 0, 0, machine_id="M", action=action),
        record("breakdown", 2, 1, machine_id="M"),
        record("pause", 2, 2, machine_id="M", action=action),
        record("repair", 5, 3, machine_id="M"),
        record("resume", 5, 4, machine_id="M", action=action),
        record("complete", 7, 5, machine_id="M", action=action),
    ]
    data = timeline(trace, {"op": "J"})
    assert [
        (i["start"], i["end"]) for i in data["intervals"] if i["kind"] == "processing"
    ] == [(0, 2), (5, 7)]
    assert [
        (i["start"], i["end"]) for i in data["intervals"] if i["kind"] == "paused"
    ] == [(2, 5)]
    assert all(i["complete"] for i in data["intervals"])


def test_actual_agv_travel_and_unload_wait_are_distinct():
    trip = {"agv_id": "V", "job_id": "J", "transport_sequence": 1}
    rows = [
        record(kind, tick, index, trip=trip)
        for index, (kind, tick) in enumerate(
            [
                ("empty_start", 0),
                ("pickup", 2),
                ("loaded_start", 2),
                ("arrival", 6),
                ("wait_for_unload", 6),
                ("delivery", 9),
            ]
        )
    ]
    data = timeline(rows)
    assert [(i["kind"], i["start"], i["end"]) for i in data["intervals"]] == [
        ("empty travel", 0, 2),
        ("loaded travel", 2, 6),
        ("waiting to unload", 6, 9),
    ]
    assert data["resources"] == ["agv:V"]


def test_failure_prefix_is_shown_as_partial_not_completed():
    data = timeline(
        [
            record("dispatch", 0, 0, machine_id="M", action={"operation_id": "op"}),
            record("decision", 3, 1),
        ]
    )
    assert data["intervals"][0]["end"] == 3
    assert data["intervals"][0]["complete"] is False


def test_out_of_order_trace_is_rejected():
    with pytest.raises(ValueError, match="ordered"):
        timeline([record("decision", 1, 0), record("decision", 0, 1)])


def test_html_is_offline_escapes_input_and_contains_controls(recorded, tmp_path):
    root = tmp_path / "new"
    root.mkdir()
    metadata = {
        "schema": "smartsom.experiment/v2",
        "name": '</script><script>alert("bad")</script>',
        "id": "r",
        "kind": "run",
        "status": "completed",
        "paths": {},
    }
    (root / "run.json").write_text(json.dumps(metadata))
    import shutil

    shutil.copytree(recorded, root / "evidence")
    target = build_report(root, root / "reports/index.html")
    text = target.read_text()
    assert '</script><script>alert("bad")' not in text
    assert "\\u003c/script>" in text
    assert "<script src=" not in text
    assert "fetch(" not in text
    for identifier in (
        "play",
        "first",
        "previous",
        "next",
        "cursor",
        "job",
        "resource",
        "svgExport",
        "pngExport",
        "pdfExport",
    ):
        assert f'id="{identifier}"' in text
    assert "same tick" in text


def test_historical_directory_never_receives_report(recorded, tmp_path):
    before = {p.name: p.read_bytes() for p in recorded.iterdir() if p.is_file()}
    with pytest.raises(ValueError, match="historical"):
        build_report(recorded, recorded / "report.html")
    build_report(recorded, tmp_path / "report.html")
    assert {p.name: p.read_bytes() for p in recorded.iterdir() if p.is_file()} == before


def test_budget_limit_is_explicit_and_does_not_silently_drop_events(recorded, tmp_path):
    with pytest.raises(ValueError, match="row limit"):
        build_report(recorded, tmp_path / "report.html", max_rows=2)
    assert not (tmp_path / "report.html").exists()


def test_training_curves_preserve_raw_return_and_only_completed_makespan(tmp_path):
    (tmp_path / "episodes.jsonl").write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                {"episode": 0, "return": -9, "makespan": 9, "end_reason": "completed"},
                {
                    "episode": 1,
                    "return": -100,
                    "makespan": None,
                    "end_reason": "policy_stalled",
                },
            ]
        )
    )
    (tmp_path / "learner_metrics.jsonl").write_text(
        json.dumps({"update": 1, "metrics": {"vf_loss": 0.4, "timer": 9}})
    )
    data = load_report_data(tmp_path)["runs"][0]
    assert data["events"] == []
    assert data["series"][0]["points"] == [[0, -9], [1, -100]]
    assert data["series"][1]["points"] == [[0, 9]]
    assert data["series"][2]["label"] == "vf_loss"


@pytest.mark.parametrize("suffix", ["png", "svg", "pdf"])
def test_publication_formats_use_real_headless_renderer(recorded, tmp_path, suffix):
    if importlib.util.find_spec("matplotlib") is None:
        if os.environ.get("SMARTSOM_REQUIRE_REPORTS") == "1":
            pytest.fail("required report renderer matplotlib is missing")
        pytest.skip("optional report renderer is missing")
    output = export_figure(recorded, tmp_path / f"figure.{suffix}")
    content = output.read_bytes()
    if suffix == "png":
        assert content.startswith(b"\x89PNG\r\n\x1a\n")
        assert struct.unpack(">II", content[16:24])[0] >= 1000
    elif suffix == "svg":
        assert ET.fromstring(content).tag.endswith("svg")
    else:
        assert content.startswith(b"%PDF-")
    assert len(content) > 1000


def test_missing_optional_renderer_is_actionable_without_output(
    recorded, tmp_path, monkeypatch
):
    monkeypatch.setitem(sys.modules, "matplotlib.backends.backend_agg", None)
    with pytest.raises(RuntimeError, match="extra reports"):
        export_figure(recorded, tmp_path / "figure.png")
    assert not (tmp_path / "figure.png").exists()
