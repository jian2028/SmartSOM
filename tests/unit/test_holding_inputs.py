"""Holding is an explicit finite grid store, never an implicit runtime module."""

import json
import os
import subprocess
import sys

import pytest
from test_experiments import bundle as bundle
from test_experiments import edit, json_file, run_path
from test_studies import study_file

from smartsom.config import (
    ConfigurationError,
    load_resolved_run,
    resolve_run,
    resolve_study,
)
from smartsom.config.codec import canonical_json, primitive
from smartsom.experiments import RunFailedError, run_one
from smartsom.trace.production import audit


def test_configuration_snapshot_and_exact_replay(bundle):
    prepared = resolve_run(run_path(bundle, "holding_hand"))
    result = run_one(prepared, verbose=False)
    assert result.simulation_result.makespan == 22
    snapshot = bundle / "frozen.json"
    snapshot.write_text(
        canonical_json(
            {"schema": "smartsom.prepared-grid-experiment/v1", **primitive(prepared)}
        )
    )
    restored = load_resolved_run(snapshot)
    assert restored == prepared
    manifest = json_file(result.run_dir, "run.json")
    holding = next(
        b
        for b in manifest["inputs"]["scenario"]["factory"]["buffers"]
        if b["buffer_id"] == "b-hold"
    )
    assert holding["storage"]["slots"][0]["capacity"] == 1
    assert audit(result.run_dir)["status"] == "passed"
    rows = [
        json.loads(line)
        for line in (result.run_dir / "trace.jsonl").read_text().splitlines()
    ]
    assert any(
        e["kind"] == "drop" and e.get("owner") == "b-hold"
        for r in rows
        for e in r["events"]
    )
    assert (
        run_one(restored, verbose=False).simulation_result == result.simulation_result
    )


@pytest.mark.parametrize("capacity", [True, -1, 1.5])
def test_bad_capacity_rejected_before_allocation(bundle, capacity):
    def change(data):
        buffer = next(
            b for b in data["factory"]["buffers"] if b["buffer_id"] == "b-hold"
        )
        buffer["storage"]["slots"][0]["capacity"] = capacity

    edit(bundle / "configs/factories/holding.yaml", change)
    with pytest.raises(ConfigurationError):
        resolve_run(run_path(bundle, "holding_hand"))
    assert not (bundle / "runs").exists()


@pytest.mark.parametrize(
    "change",
    [
        lambda d: d.update(transport=None),
        lambda d: d.update(holding_buffer={"unknown": 1}),
        lambda d: d.update(holding_buffer={"kind": "unknown"}),
    ],
)
def test_invalid_enablement(bundle, change):
    edit(bundle / "configs/scenarios/holding_hand.yaml", change)
    with pytest.raises(ConfigurationError):
        resolve_run(run_path(bundle, "holding_hand"))
    assert not (bundle / "runs").exists()


def test_cp_rejected_and_explicit_holding_case_preserves_other_inputs(bundle):
    from smartsom.api import load_config, prepare

    path = run_path(bundle, "holding_hand")
    config = load_config(path)
    config.algorithm.source = str(bundle / "configs/algorithms/spt.yaml")
    original = prepare(config, training=False).resolved.scenario
    edit(path, lambda d: d.update(algorithm="../algorithms/cp_sat.yaml"))
    with pytest.raises(ConfigurationError, match="CP-SAT has no grid"):
        resolve_run(path)
    study = study_file(bundle, scenario="holding_hand")
    edit(
        study,
        lambda d: d.update(
            variants=[{"id": "on"}, {"id": "off", "disable": ["holding_buffer"]}]
        ),
    )
    with pytest.raises(ConfigurationError, match="physical facility"):
        resolve_study(study)

    def remove(data):
        factory = data["factory"]
        factory["buffers"] = [
            b for b in factory["buffers"] if b["buffer_id"] != "b-hold"
        ]
        factory["ports"] = [
            p for p in factory["ports"] if p["port_id"] != "port_b-hold"
        ]

    edit(bundle / "configs/factories/holding.yaml", remove)
    changed = prepare(config, training=False).resolved.scenario
    assert changed.demands == original.demands and changed.seed == original.seed
    assert changed.factory.machines == original.factory.machines
    assert changed.factory.agvs == original.factory.agvs
    assert "b-hold" not in {b.buffer_id for b in changed.factory.buffers}


def test_holding_writer_failure_retains_cause_and_partial_trace(bundle, monkeypatch):
    from smartsom.trace.production import Recorder

    original = Recorder.append

    def broken(self, row):
        if any(
            e["kind"] == "drop" and e.get("owner") == "b-hold" for e in row["events"]
        ):
            raise OSError("holding trace write failed")
        return original(self, row)

    monkeypatch.setattr(Recorder, "append", broken)
    with pytest.raises(RunFailedError) as caught:
        run_one(resolve_run(run_path(bundle, "holding_hand")), verbose=False)
    manifest = json_file(caught.value.run_dir, "run.json")
    assert (
        manifest["status"] == "failed"
        and manifest["failure"]["message"] == "holding trace write failed"
    )
    assert manifest["execution_state"]["tick"] == manifest["last_tick"] + 1
    assert (caught.value.run_dir / "trace.jsonl").stat().st_size > 0


def test_reordered_and_other_cwd_hash_seed_inputs_match(bundle):
    path = run_path(bundle, "holding_hand")
    code = "from smartsom.config import resolve_run; from smartsom.config.codec import digest; from smartsom.experiments import run_one; import sys; print(digest(run_one(resolve_run(sys.argv[1]), verbose=False).simulation_result))"
    values = [
        subprocess.check_output(
            [sys.executable, "-c", code, str(path)],
            cwd=cwd,
            env={**os.environ, "PYTHONHASHSEED": seed},
            text=True,
        ).strip()
        for cwd, seed in ((bundle, "1"), (bundle.parent, "123"))
    ]
    assert values[0] == values[1]
