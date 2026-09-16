"""Explicit grid transport, two-file traces and source-bound historical evidence."""

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import FrozenInstanceError

import pytest
from test_experiments import bundle as bundle
from test_experiments import edit, json_file, json_lines, run_path

from smartsom.algorithms.production import GreedyProductionPolicy
from smartsom.config import ConfigurationError, load_resolved_run, resolve_run
from smartsom.config.codec import primitive
from smartsom.engine.production import ProductionSimulator
from smartsom.experiments import RunFailedError, run_one
from smartsom.experiments.cli import main
from smartsom.trace.production import TRACE_SCHEMA, audit, state_hash


def direct_rows(prepared):
    case = prepared.resolved.scenario
    sim = ProductionSimulator(case)
    policy = GreedyProductionPolicy(case.factory, rule="spt")
    rows = []
    while not sim.done and sim.tick < 1024:
        ranking = sim.decision()
        rankings = policy.rank(ranking)
        view = sim.decision(rankings=rankings)
        row = sim.step(policy.act(view))
        row.update(
            rule_decision={
                "ranking": {"sha256": state_hash(ranking)},
                "action": {"sha256": state_hash(view)},
            },
            buffer_scores=policy.scores,
        )
        rows.append(row)
    return sim, rows


def test_configured_hand_matches_code_and_reusable_grid_trace(bundle):
    config = run_path(bundle, "transport_hand")
    prepared = resolve_run(config)
    assert main(["validate", str(config)]) == 0
    assert not (bundle / "runs").exists()
    sim, rows = direct_rows(prepared)
    result = run_one(prepared, verbose=False)
    assert result.simulation_result.final_state == sim.snapshot()
    assert result.simulation_result.makespan == 15
    assert json_lines(result.run_dir, "trace.jsonl") == primitive(
        [
            {"schema": TRACE_SCHEMA, **r, "state_hash": state_hash(r["state"])}
            for r in rows
        ]
    )
    events = [event for row in rows for event in row["events"]]
    assert [
        (e["machine"], e["tick"]) for e in events if e["kind"] == "processing_completed"
    ] == [("M1", 6), ("M2", 12)]
    manifest = json_file(result.run_dir, "run.json")
    assert manifest["result"]["completed"] == ["J"]
    assert manifest["inputs"]["scenario"]["factory"] == primitive(
        prepared.resolved.scenario.factory
    )
    assert audit(result.run_dir)["status"] == "passed"
    assert {p.name for p in result.run_dir.iterdir()} == {"run.json", "trace.jsonl"}
    restored = load_resolved_run(result.run_dir / "run.json")
    assert (
        run_one(restored, verbose=False).simulation_result == result.simulation_result
    )
    with pytest.raises(FrozenInstanceError):
        prepared.resolved.scenario.factory.agvs = ()


@pytest.mark.parametrize(
    "mutation",
    [
        lambda d: d["factory"]["grid"].update(width=True),
        lambda d: d["factory"]["grid"].update(height=1.5),
        lambda d: d["factory"]["ports"][0]["bindings"][0]["target"].update(
            buffer_id="unknown"
        ),
        lambda d: d["factory"]["ports"].append(d["factory"]["ports"][0]),
        lambda d: d["factory"]["agvs"][0].update(initial_cell={"x": 999, "y": 0}),
        lambda d: d["factory"]["agvs"][0].update(capacity=2),
        lambda d: d["factory"]["agvs"][0].update(seed=42),
        lambda d: d["factory"].update(agvs=[]),
    ],
)
def test_bad_transport_inputs_fail_before_simulator_or_directory(
    bundle, monkeypatch, mutation
):
    edit(bundle / "configs/factories/transport_hand.yaml", mutation)
    monkeypatch.setattr(
        "smartsom.engine.production.ProductionSimulator",
        lambda *a, **k: pytest.fail("constructed simulator"),
    )
    with pytest.raises(ConfigurationError):
        resolve_run(run_path(bundle, "transport_hand"))
    assert not (bundle / "runs").exists()


def test_no_implicit_transport_toggle_and_cp_has_no_grid_adapter(bundle):
    config = run_path(bundle, "transport_hand")
    scenario = bundle / "configs/scenarios/transport_hand.yaml"
    edit(scenario, lambda d: d.update(transport=None))
    with pytest.raises(ConfigurationError, match="transport"):
        resolve_run(config)
    edit(scenario, lambda d: d.pop("transport"))
    edit(config, lambda d: d.update(algorithm="../algorithms/cp_sat.yaml"))
    with pytest.raises(ConfigurationError, match="no grid"):
        resolve_run(config)
    assert not (bundle / "runs").exists()


def script(bundle, commands):
    edit(
        bundle / "configs/algorithms/spt_transport.yaml",
        lambda d: d.update(
            algorithm={
                "provider": "builtin.scripted",
                "parameters": {"commands": commands},
            }
        ),
    )


def test_provider_failure_and_write_failure_keep_transport_evidence(
    bundle, monkeypatch
):
    from smartsom.trace.production import Recorder

    config = run_path(bundle, "transport_hand")
    good = resolve_run(config)
    _, rows = direct_rows(good)
    script(bundle, [primitive(r["actions"]) for r in rows[:-1]])
    with pytest.raises(RunFailedError) as err:
        run_one(resolve_run(config), verbose=False)
    failed = err.value.run_dir
    assert json_file(failed, "run.json")["status"] == "failed"
    assert any(
        e["kind"] == "pickup"
        for r in json_lines(failed, "trace.jsonl")
        for e in r["events"]
    )
    original = Recorder.append

    def fail_final(self, row):
        if row["state"]["completed"]:
            raise OSError("injected final trace write failure")
        return original(self, row)

    monkeypatch.setattr(Recorder, "append", fail_final)
    with pytest.raises(RunFailedError, match="final trace write") as err:
        run_one(good, verbose=False)
    record = json_file(err.value.run_dir, "run.json")
    assert record["status"] == "failed" and record["execution_state"]["completed"] == [
        "J"
    ]
    assert not json_lines(err.value.run_dir, "trace.jsonl")[-1]["state"]["completed"]


def test_script_transport_and_cli_failure_codes(bundle):
    config = run_path(bundle, "transport_hand")
    good = resolve_run(config)
    sim, rows = direct_rows(good)
    commands = [primitive(r["actions"]) for r in rows]
    script(bundle, commands)
    assert (
        run_one(resolve_run(config), verbose=False).simulation_result.final_state
        == sim.snapshot()
    )
    script(bundle, commands[:-1])
    assert main(["run", str(config)]) == 1
    commands[0]["agvs"][0][0] = "unknown"
    script(bundle, commands)
    assert main(["validate", str(config)]) == 2


def test_transport_hashseed_cwd_and_input_permutation(bundle, tmp_path):
    source = run_path(bundle, "transport_combined")
    code = """from smartsom.config import resolve_run
from smartsom.config.codec import canonical_json
from smartsom.engine.production import ProductionSimulator
from smartsom.algorithms.production import GreedyProductionPolicy
import sys
r=resolve_run(sys.argv[1]).resolved.scenario
s=ProductionSimulator(r); p=GreedyProductionPolicy(r.factory,rule="spt"); rows=[]
while not s.done and s.tick<1024:
 v=s.decision(rankings=p.rank(s.decision()))
 rows.append((v,s.step(p.act(v))))
print(canonical_json(rows))
"""
    before = subprocess.check_output(
        [sys.executable, "-c", code, str(source)],
        cwd=bundle,
        env={**os.environ, "PYTHONHASHSEED": "1"},
    )

    def reverse_factory(data):
        for field in ("machines", "ports", "buffers", "agvs"):
            data["factory"][field].reverse()

    edit(bundle / "configs/factories/transport_multiple.yaml", reverse_factory)
    edit(
        bundle / "configs/workloads/transport_combined.yaml",
        lambda d: d["demands"].reverse(),
    )
    after = subprocess.check_output(
        [sys.executable, "-c", code, str(source)],
        cwd=tmp_path,
        env={**os.environ, "PYTHONHASHSEED": "987"},
    )
    assert before == after


def test_historical_twelve_matrix_goldens_are_preserved_without_new_core_claim(bundle):
    # New geometry changes physics: the old numerical oracle is historical data.
    path = bundle / "data/reference/transport/disabled_golden.json"
    assert (
        hashlib.sha256(path.read_bytes()).hexdigest()
        == "b42e47754e75bb6b209d41ea3274e651e341ec84c88dcfdc40f1fa77f30a3b0c"
    )
    data = json.loads(path.read_text())
    assert len(data["cases"]) == 12 and data["cases"]["crossing"]["makespan"] == 5
    raw = (bundle / "data/reference/idetc/converted/S00/factory.yaml").read_bytes()
    (bundle / "configs/factories/transport_hand.yaml").write_bytes(raw)
    with pytest.raises(ConfigurationError, match="migrat|grid"):
        resolve_run(run_path(bundle, "transport_hand"))
    assert not (bundle / "runs").exists()


def test_full_replay_verification_failure_preserves_evidence(bundle, monkeypatch):
    from smartsom.trace.production import ExecutionAudit

    prepared = resolve_run(run_path(bundle, "transport_hand"))
    original = ExecutionAudit.append

    def fail(self, row):
        original(self, row)
        if self.sim.done:
            raise ValueError("injected transport result mismatch")

    monkeypatch.setattr(ExecutionAudit, "append", fail)
    with pytest.raises(
        RunFailedError, match="injected transport result mismatch"
    ) as error:
        run_one(prepared, verbose=False)
    record = json_file(error.value.run_dir, "run.json")
    assert record["status"] == "failed" and record["execution_state"]["completed"] == [
        "J"
    ]
    assert "injected transport result mismatch" in record["failure"]["message"]
