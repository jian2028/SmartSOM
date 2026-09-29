"""Past throughput is a candidate, never a current calibration result."""

import json
from dataclasses import asdict

import pytest

from smartsom.experiments.evidence import write_json
from smartsom.experiments.performance_profiles import select, store
from smartsom.experiments.tuning_resources import ExecutionProfile


def _record(tmp_path, name, at, throughput, *, shape="compatible"):
    report = tmp_path / f"{name}.json"
    profile = asdict(ExecutionProfile(1, 1))
    write_json(
        report,
        {
            "schema": "smartsom.tune-calibration/v1",
            "measurements": [
                {
                    "profile": profile,
                    "throughput": throughput,
                    "valid": True,
                    "termination": None,
                }
            ],
        },
    )
    return {
        "at": at,
        "hardware": "host",
        "shape": shape,
        "profile": profile,
        "throughput": throughput,
        "report": str(report),
        "source": {"git": {"commit": "abc"}},
    }


def test_latest_best_and_explicit_report_require_valid_compatible_evidence(tmp_path):
    output = tmp_path / "runs" / "batch"
    output.mkdir(parents=True)
    fast = _record(tmp_path, "fast", 1, 100)
    recent = _record(tmp_path, "recent", 2, 50)
    wrong = _record(tmp_path, "wrong", 3, 200, shape="other")
    store(output, [fast, recent, wrong])
    assert (
        select(output, hardware="host", shape="compatible", candidate="latest")
        == recent
    )
    assert select(output, hardware="host", shape="compatible", candidate="best") == fast
    report = tmp_path / "selection.json"
    write_json(
        report, {"schema": "smartsom.performance-profiles/v1", "profiles": [fast]}
    )
    assert (
        select(output, hardware="host", shape="compatible", candidate=str(report))
        == fast
    )
    assert (
        select(output, hardware="other", shape="compatible", candidate="best") is None
    )
    write_json(
        report, {"schema": "smartsom.performance-profiles/v1", "profiles": [wrong]}
    )
    with pytest.raises(ValueError, match="shape-compatible"):
        select(output, hardware="host", shape="compatible", candidate=str(report))
    (tmp_path / "recent.json").write_text("broken")
    assert (
        select(output, hardware="host", shape="compatible", candidate="latest") == fast
    )
    (tmp_path / "runs" / ".performance-profiles" / "index.json").write_text("broken")
    assert (
        select(output, hardware="host", shape="compatible", candidate="latest") is None
    )


def test_report_corruption_cannot_replay_index_throughput(tmp_path):
    output = tmp_path / "runs" / "batch"
    output.mkdir(parents=True)
    row = _record(tmp_path, "original", 1, 100)
    store(output, [row])
    report = tmp_path / "original.json"
    body = json.loads(report.read_text())
    body["measurements"][0]["throughput"] = 1000
    write_json(report, body)
    assert (
        select(output, hardware="host", shape="compatible", candidate="latest") is None
    )
