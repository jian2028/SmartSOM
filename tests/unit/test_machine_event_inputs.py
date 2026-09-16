"""Frozen grid outages share one state stream with execution and replay."""

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
from smartsom.config.codec import canonical_json, primitive
from smartsom.engine.production import ProductionSimulator
from smartsom.experiments import RunFailedError, run_one
from smartsom.experiments.cli import main
from smartsom.trace.production import TRACE_SCHEMA, audit, state_hash

NAMES = (
    "machine_events_fixed",
    "machine_events_generated",
    "machine_events_arrivals_dispatch",
    "machine_events_arrivals_event",
)


@pytest.mark.parametrize("name", NAMES)
def test_config_code_replay_and_evidence(bundle, name):
    prepared = resolve_run(run_path(bundle, name))
    scenario = prepared.resolved.scenario
    assert not (bundle / "runs").exists()
    sim, policy = (
        ProductionSimulator(scenario),
        GreedyProductionPolicy(scenario.factory, rule="spt"),
    )
    limits = json.loads(prepared.config_json)["training"]
    rows = []
    while sim.status == "running" and sim.tick < min(
        limits["max_ticks"], limits["max_decisions"]
    ):
        ranking = sim.decision()
        rankings = policy.rank(ranking)
        view = sim.decision(rankings=rankings)
        row = sim.step(policy.act(view))
        row["rule_decision"] = {
            "ranking": {"sha256": state_hash(ranking)},
            "action": {"sha256": state_hash(view)},
        }
        row["buffer_scores"] = policy.scores
        rows.append(row)
    actual = run_one(prepared, verbose=False)
    assert actual.simulation_result.final_state == sim.snapshot()
    assert json_lines(actual.run_dir, "trace.jsonl") == primitive(
        [
            {"schema": TRACE_SCHEMA, **r, "state_hash": state_hash(r["state"])}
            for r in rows
        ]
    )
    manifest = json_file(actual.run_dir, "run.json")
    assert manifest["inputs"]["scenario"]["outages"] == primitive(scenario.outages)
    checked = audit(actual.run_dir)
    assert checked["status"] == (
        "passed" if sim.status == "completed" else "partial_verified"
    )
    assert {p.name for p in actual.run_dir.iterdir()} == {"run.json", "trace.jsonl"}
    with pytest.raises(FrozenInstanceError):
        scenario.outages = ()


def test_export_reimport_does_not_regenerate(bundle):
    prepared = resolve_run(run_path(bundle, "machine_events_generated"))
    path = bundle / "snapshot.json"
    path.write_text(
        canonical_json(
            {"schema": "smartsom.prepared-grid-experiment/v1", **primitive(prepared)}
        )
    )
    # A snapshot detaches the complete scientific input from authoring paths.
    (bundle / "configs/scenarios/machine_events_generated.yaml").unlink()
    restored = load_resolved_run(path)
    assert restored == prepared
    assert (
        run_one(restored, verbose=False).simulation_result
        == run_one(prepared, verbose=False).simulation_result
    )


def test_generated_outages_can_be_frozen_as_explicit_scenario_input(bundle):
    name = "machine_events_generated"
    original = resolve_run(run_path(bundle, name)).resolved.scenario

    def freeze(data):
        data.pop("outage_profiles")
        data["outages"] = primitive(original.outages)

    edit(bundle / f"configs/scenarios/{name}.yaml", freeze)
    edit(
        run_path(bundle, name),
        lambda d: d.update(seed=999, algorithm="../algorithms/first_feasible.yaml"),
    )
    restored = resolve_run(run_path(bundle, name)).resolved.scenario
    assert restored.outages == original.outages
    assert restored.seed != original.seed


@pytest.mark.parametrize(
    "mutation",
    [
        lambda d: d.update(extra=True),
        lambda d: d["outages"][0].update(start=True),
        lambda d: d["outages"][0].update(end=4.0),
        lambda d: d["outages"][0].update(machine_id="unknown"),
        lambda d: d["outages"][0].update(start=5),
        lambda d: d["outages"][0].update(extra=1),
        lambda d: d.update(seed=1),
    ],
)
def test_invalid_fixed_input_before_simulator_and_directory(
    bundle, monkeypatch, mutation
):
    edit(bundle / "configs/scenarios/machine_events_fixed.yaml", mutation)
    monkeypatch.setattr(
        "smartsom.engine.production.ProductionSimulator",
        lambda *a, **k: pytest.fail("constructed simulator"),
    )
    with pytest.raises(ConfigurationError):
        resolve_run(run_path(bundle, "machine_events_fixed"))
    assert not (bundle / "runs").exists()


@pytest.mark.parametrize(
    "mutation",
    [
        lambda d: d["outage_profiles"][0].update(until_tick=True),
        lambda d: d["outage_profiles"][0].update(until_tick=0),
        lambda d: d["outage_profiles"][0].update(until_tick=2.0),
        lambda d: d["outage_profiles"].append(d["outage_profiles"][0].copy()),
        lambda d: d["outage_profiles"][0].update(mean_uptime_ticks=True),
        lambda d: d["outage_profiles"][0].update(mean_uptime_ticks=0),
        lambda d: d["outage_profiles"][0].update(mean_uptime_ticks="NaN"),
        lambda d: d["outage_profiles"][0].update(repair_min=0),
        lambda d: d["outage_profiles"][0].update(repair_min=5, repair_max=2),
        lambda d: d["outage_profiles"][0].update(repair_max=True),
        lambda d: d["outage_profiles"][0].update(machine_id="unknown"),
        lambda d: d.update(outages=[{"machine_id": "M1", "start": 1, "end": 2}]),
        lambda d: d.update(seed=1),
    ],
)
def test_invalid_profiles(bundle, mutation):
    edit(bundle / "configs/scenarios/machine_events_generated.yaml", mutation)
    with pytest.raises(ConfigurationError):
        resolve_run(run_path(bundle, "machine_events_generated"))
    assert not (bundle / "runs").exists()


def test_duplicate_keys_and_cli_validation(bundle):
    config = run_path(bundle, "machine_events_fixed")
    assert main(["validate", str(config)]) == 0 and not (bundle / "runs").exists()
    path = bundle / "configs/scenarios/machine_events_fixed.yaml"
    path.write_text(path.read_text() + "\noutages: []\n")
    assert main(["validate", str(config)]) != 0
    assert main(["run", str(config)]) != 0 and not (bundle / "runs").exists()


def test_empty_plan_is_valid_but_cp_remains_unsupported(bundle):
    name = "machine_events_fixed"
    edit(bundle / f"configs/scenarios/{name}.yaml", lambda d: d.update(outages=[]))
    assert resolve_run(run_path(bundle, name)).resolved.scenario.outages == ()
    edit(
        run_path(bundle, name),
        lambda d: d.update(algorithm="../algorithms/cp_sat.yaml"),
    )
    with pytest.raises(ConfigurationError, match="CP-SAT has no grid"):
        resolve_run(run_path(bundle, name))


def test_failure_after_breakdown_retains_actual_records_without_makespan(
    bundle, monkeypatch
):
    from smartsom.trace.production import Recorder

    original = Recorder.append

    def fail(self, row):
        if row["tick"] > 4:
            raise OSError("write failed after breakdown")
        return original(self, row)

    monkeypatch.setattr(Recorder, "append", fail)
    with pytest.raises(RunFailedError) as error:
        run_one(resolve_run(run_path(bundle, "machine_events_fixed")), verbose=False)
    directory = error.value.run_dir
    manifest = json_file(directory, "run.json")
    assert manifest["status"] == "failed" and manifest["result"]["status"] == "running"
    assert manifest["failure"]["message"] == "write failed after breakdown"
    assert manifest["execution_state"]["tick"] == manifest["last_tick"] + 1
    assert any(
        e["kind"] == "breakdown"
        for r in json_lines(directory, "trace.jsonl")
        for e in r["events"]
    )
    assert manifest["inputs"]["scenario"]["outages"]


def test_scripted_provider_with_outages_and_insufficient_commands(bundle):
    # Put a real two-tick outage in the hand-computable one-machine case.
    from test_production_runtime import small_scenario

    from smartsom.config.production import AlgorithmConfig
    from smartsom.domain.production import Outage
    from smartsom.experiments.production import run

    scenario = small_scenario(outages=(Outage("machine", 4, 6),))
    sim, policy, commands = (
        ProductionSimulator(scenario),
        GreedyProductionPolicy(scenario.factory),
        [],
    )
    while sim.status == "running":
        command = policy.act(sim.decision())
        commands.append(command)
        sim.step(command)
    assert sim.status == "completed" and sim.tick == 10
    algorithm = AlgorithmConfig(provider="builtin.scripted", commands=tuple(commands))
    directory = run(scenario, algorithm, output_root=bundle / "runs", verbose=False)
    assert audit(directory)["status"] == "passed"
    with pytest.raises(ValueError, match="exhausted") as error:
        run(
            scenario,
            algorithm.model_copy(update={"commands": tuple(commands[:-1])}),
            output_root=bundle / "runs",
            verbose=False,
        )
    manifest = json_file(error.value.run_dir, "run.json")
    assert manifest["status"] == "failed"
    assert any(
        e["kind"] == "repair"
        for r in json_lines(error.value.run_dir, "trace.jsonl")
        for e in r["events"]
    )


def test_seed_independence_and_hashseed_cwd(bundle):
    path = run_path(bundle, "machine_events_generated")
    original = resolve_run(path).resolved.scenario
    import yaml

    scenario_path = bundle / "configs/scenarios/machine_events_generated.yaml"
    settings = yaml.safe_load(scenario_path.read_text())
    edit(scenario_path, lambda d: d.pop("outage_profiles"))
    off = resolve_run(path).resolved.scenario
    assert (
        off.demands == original.demands
        and off.seed == original.seed
        and not off.outages
    )
    scenario_path.write_text(canonical_json(settings))
    code = "from smartsom.config import resolve_run; from smartsom.config.codec import digest; from smartsom.experiments import run_one; import sys; r=resolve_run(sys.argv[1]); print(digest(r.resolved.scenario)); print(digest(run_one(r, verbose=False).simulation_result))"
    outputs = [
        subprocess.check_output(
            [sys.executable, "-c", code, str(path)],
            cwd=bundle.parent,
            env={**os.environ, "PYTHONHASHSEED": seed},
            text=True,
        )
        for seed in ("1", "87")
    ]
    assert outputs[0] == outputs[1]
