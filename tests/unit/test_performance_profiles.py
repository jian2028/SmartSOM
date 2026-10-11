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


def test_online_cache_is_separate_and_requires_retained_matching_evidence(tmp_path):
    from smartsom.experiments.performance_profiles import (
        ONLINE_SCHEMA,
        select_online,
        store_online,
    )

    output = tmp_path / "runs" / "batch"
    report = tmp_path / "online-performance.json"
    row = {
        "hardware": "host",
        "shape": "task-and-source",
        "concurrency": 4,
        "throughput": 100,
        "at": 1,
        "report": str(report),
    }
    write_json(report, {"schema": ONLINE_SCHEMA, "profiles": [row]})
    store_online(output, [row])
    assert select_online(output, hardware="host", shape="task-and-source") == row
    assert (
        select_online(output, hardware="different-allocation", shape="task-and-source")
        is None
    )
    assert (
        select_online(output, hardware="host", shape="changed-network-or-source")
        is None
    )
    assert (
        select(output, hardware="host", shape="task-and-source", candidate="latest")
        is None
    )
    write_json(
        report, {"schema": ONLINE_SCHEMA, "profiles": [{**row, "throughput": 1000}]}
    )
    assert select_online(output, hardware="host", shape="task-and-source") is None


def test_online_shape_covers_network_parameters_and_frozen_sampler(tmp_path):
    from pathlib import Path

    from smartsom import api
    from smartsom.config.experiment_v3 import prepare_v3
    from smartsom.experiments.performance_profiles import online_shape

    root = Path(__file__).resolve().parents[2]
    prepared = asdict(
        prepare_v3(
            api.load_config(root / "configs/test/runs/train_machine_ppo.yaml"),
            training=True,
        )
    )
    source = {"python": "3.12", "packages": {"torch": "2.14"}}
    first = online_shape({"prepared": prepared}, source, "implementation-a")
    assert first != online_shape({"prepared": prepared}, source, "implementation-b")
    changed = {**prepared, "parameters_json": '{"batch_size": 1024}'}
    assert first != online_shape({"prepared": changed}, source, "implementation-a")
    config = json.loads(prepared["config_json"])
    config["runtime"]["num_envs"] = 4
    changed = {**prepared, "config_json": json.dumps(config)}
    assert first != online_shape({"prepared": changed}, source, "implementation-a")
