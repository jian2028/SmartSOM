"""Processing duration inputs and evidence for the single grid contract."""

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import replace
from decimal import Decimal
from fractions import Fraction

import pytest
from test_experiments import ROOT, edit, json_file, json_lines, run_path
from test_experiments import bundle as bundle
from test_production_runtime import act, loaded_machine, small_scenario

from smartsom.config import ConfigurationError, load_resolved_run, resolve_run
from smartsom.config.codec import canonical_json, primitive
from smartsom.domain import ScheduledOperation
from smartsom.domain.production import (
    Demand,
    MachineCommand,
    ProcessingSample,
    ProductionStep,
)
from smartsom.engine.production import ProductionSimulator
from smartsom.experiments import RunFailedError, run_one
from smartsom.experiments.cli import main
from smartsom.trace.production import audit


def test_fixed_config_and_hidden_duration_inputs(bundle):
    prepared = resolve_run(run_path(bundle, "processing_fixed"))
    scenario = prepared.resolved.scenario
    assert [(s.operation_id, s.actual_ticks) for s in scenario.processing_samples] == [
        ("A1", 12),
        ("A2", 4),
        ("B1", 6),
        ("B2", 8),
    ]
    assert not (bundle / "runs").exists()
    sim = ProductionSimulator(scenario)
    assert "actual_ticks" not in json.dumps(sim.decision())
    result = run_one(prepared, verbose=False)
    manifest = json_file(result.run_dir, "run.json")
    assert manifest["inputs"]["scenario"]["processing_samples"] == primitive(
        scenario.processing_samples
    )
    started = [
        e
        for r in json_lines(result.run_dir, "trace.jsonl")
        for e in r["events"]
        if e["kind"] == "processing_started"
    ]
    assert {(e["job"], e["actual_ticks"]) for e in started} >= {
        ("A/attempt/1", 12),
        ("B/attempt/1", 6),
    }
    assert audit(result.run_dir)["status"] in ("passed", "partial_verified")


def test_historical_external_schedule_keeps_original_source_identity():
    # This solver schedule excludes physical grid transport. It remains a
    # historical reference, not a makespan expectation for the grid simulator.
    from test_processing_times import REFERENCE

    source_dir = ROOT / "data/reference/processing_times"
    sources = json_file(source_dir, "sources.json")
    external = json_file(source_dir, "direct_solver_result.json")
    assert (
        sources["result_sha256"]
        == hashlib.sha256(
            (source_dir / "direct_solver_result.json").read_bytes()
        ).hexdigest()
    )
    translated = tuple(
        sorted(
            (
                ScheduledOperation(
                    f"{'AB'[r['job']]}{r['position'] + 1}",
                    "standard",
                    f"M{r['machine'] + 1}",
                    r["start"],
                    r["end"],
                )
                for r in external["schedule"]
            ),
            key=lambda r: (r.start_time, r.operation_id),
        )
    )
    fixed = tuple(
        ScheduledOperation(**r) for r in json_file(source_dir, "schedule.json")
    )
    assert translated == fixed == REFERENCE
    assert external["metadata"]["status"] == "optimal" and external["makespan"] == 20


def test_duration_draw_golden_and_two_stage_rounding():
    # Independent keyed hash and integer rational arithmetic. A duration draw is
    # indexed by attempt and operation, so policy order cannot change it.
    scenario = small_scenario(
        seed=42, processing_low=Decimal("0.8"), processing_high=Decimal("1.2")
    )
    sim = ProductionSimulator(scenario)
    key = b'["processing",42,"demand/attempt/1","op"]'
    draw = Fraction(int.from_bytes(hashlib.sha256(key).digest(), "big"), 2**256)
    assert sim._draw("processing", "demand/attempt/1", "op") == draw
    assert (
        hashlib.sha256(key).hexdigest()
        == "14581b89d40f109aac628d53e9961ae56823ef3e7b4c252bdd9290eff66d2c61"
    )
    job = loaded_machine(sim)
    row = act(sim, machine=MachineCommand(job, "normal"))
    start = next(e for e in row["events"] if e["kind"] == "processing_started")
    base = 2 * (Fraction(4, 5) + Fraction(2, 5) * draw)
    assert start["actual_ticks"] == max(
        1, (base.numerator * 2 + base.denominator) // (base.denominator * 2)
    )
    assert "actual_ticks" not in json.dumps(sim.decision())


@pytest.mark.parametrize(
    "name", ["generated", "generated_fjsp_spt", "generated_arrivals_event"]
)
def test_processing_ablation_preserves_workload_arrivals_and_outages(bundle, name):
    path = run_path(bundle, name)
    original = resolve_run(path).resolved.scenario
    # Resolve authoring location from the original run, preserving path semantics.
    import yaml

    scenario_path = (
        path.parent / yaml.safe_load(path.read_text())["scenario"]
    ).resolve()
    edit(scenario_path, lambda d: d.update(processing_low=0.8, processing_high=1.2))
    changed = resolve_run(path).resolved.scenario
    assert (
        changed.demands == original.demands
        and changed.outages == original.outages
        and changed.seed == original.seed
    )
    assert (changed.processing_low, changed.processing_high) == (
        Decimal("0.8"),
        Decimal("1.2"),
    )


@pytest.mark.parametrize("trigger", ["dispatch", "event"])
def test_combination_recording_and_unit_multiplier(bundle, trigger):
    name = "processing_arrivals_" + trigger
    prepared = resolve_run(run_path(bundle, name))
    result = run_one(prepared, verbose=False)
    manifest = json_file(result.run_dir, "run.json")
    assert any(d["release_at"] for d in manifest["inputs"]["scenario"]["demands"])
    assert audit(result.run_dir)["status"] in ("passed", "partial_verified")
    path = bundle / f"configs/scenarios/{name}.yaml"

    def unit(d):
        d.pop("processing_samples", None)
        d.update(processing_low=1, processing_high=1)

    edit(path, unit)
    with_unit = resolve_run(run_path(bundle, name))
    edit(path, lambda d: (d.pop("processing_low"), d.pop("processing_high")))
    implicit = resolve_run(run_path(bundle, name))
    assert with_unit.resolved.scenario == implicit.resolved.scenario
    assert (
        run_one(with_unit, verbose=False).simulation_result
        == run_one(implicit, verbose=False).simulation_result
    )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda d: d["processing_samples"].append(d["processing_samples"][0].copy()),
        lambda d: d["processing_samples"][0].update(operation_id="missing"),
        lambda d: d["processing_samples"][0].update(demand_id="missing"),
        lambda d: d["processing_samples"][0].update(machine_id="M2"),
        lambda d: d["processing_samples"][0].update(extra=99),
        lambda d: d["processing_samples"][0].update(actual_ticks=True),
        lambda d: d["processing_samples"][0].update(actual_ticks=1.0),
        lambda d: d["processing_samples"][0].update(actual_ticks=0),
        lambda d: d["processing_samples"][0].update(attempt=0),
        lambda d: d.update(seed=1),
        lambda d: d.update(schema="unsupported/v1"),
    ],
)
def test_invalid_fixed_inputs_fail_before_directory(bundle, mutation, monkeypatch):
    monkeypatch.setattr(
        "smartsom.engine.production.ProductionSimulator",
        lambda *a, **k: pytest.fail("simulator created"),
    )
    edit(bundle / "configs/scenarios/processing_fixed.yaml", mutation)
    with pytest.raises(ConfigurationError):
        resolve_run(run_path(bundle, "processing_fixed"))
    assert not (bundle / "runs").exists()


@pytest.mark.parametrize(
    "fields",
    [
        {"processing_low": False},
        {"processing_low": 0},
        {"processing_low": 2, "processing_high": 1},
        {"processing_low": "NaN"},
        {"processing_high": float("inf")},
        {"processing_seed": 42},
        {"sample_timing": "dispatch"},
        {"processing_time": {"kind": "fixed", "path": "absent"}},
    ],
)
def test_invalid_profile_and_references(bundle, fields):
    edit(
        bundle / "configs/scenarios/processing_generated.yaml",
        lambda d: d.update(fields),
    )
    with pytest.raises(ConfigurationError):
        resolve_run(run_path(bundle, "processing_generated"))
    assert not (bundle / "runs").exists()


def test_static_solver_is_explicitly_unsupported(bundle):
    path = run_path(bundle, "processing_generated")
    edit(
        bundle / "configs/scenarios/processing_generated.yaml",
        lambda d: d.update(processing_low=1, processing_high=1),
    )
    edit(path, lambda d: d.update(algorithm="../algorithms/cp_sat.yaml"))
    with pytest.raises(ConfigurationError, match="CP-SAT has no grid"):
        resolve_run(path)
    assert not (bundle / "runs").exists()


@pytest.mark.parametrize("tamper", ["identity", "profile", "coverage", "actual"])
def test_frozen_processing_snapshot_verifies_scientific_identity(bundle, tamper):
    prepared = resolve_run(run_path(bundle, "processing_fixed"))
    data = {"schema": "smartsom.prepared-grid-experiment/v1", **primitive(prepared)}
    scenario = json.loads(data["resolved"]["scenario_json"])
    if tamper == "identity":
        data["scientific_sha256"] = "0" * 64
    if tamper == "profile":
        scenario["processing_high"] = "1.5"
    if tamper == "coverage":
        scenario["processing_samples"].pop()
    if tamper == "actual":
        scenario["processing_samples"][0]["actual_ticks"] += 1
    data["resolved"]["scenario_json"] = canonical_json(scenario)
    path = bundle / "snapshot.json"
    path.write_text(canonical_json(data))
    with pytest.raises(ConfigurationError, match="scientific identity"):
        load_resolved_run(path)


def test_duplicate_keys_and_semantic_order(bundle):
    path = bundle / "configs/scenarios/processing_fixed.yaml"
    before = resolve_run(run_path(bundle, "processing_fixed"))
    edit(path, lambda d: d["processing_samples"].reverse())
    after = resolve_run(run_path(bundle, "processing_fixed"))
    assert (
        run_one(before, verbose=False).simulation_result
        == run_one(after, verbose=False).simulation_result
    )
    path.write_text(
        "schema: smartsom.scenario/v2\nprocessing_samples: []\nprocessing_samples: []\n"
    )
    with pytest.raises(ConfigurationError, match="duplicate key"):
        resolve_run(run_path(bundle, "processing_fixed"))


def test_failed_provider_preserves_inputs_and_delivered_states(bundle, monkeypatch):
    from smartsom.algorithms.production import GreedyProductionPolicy

    original = GreedyProductionPolicy.act

    def broken(self, view):
        if view["tick"] >= 18:
            raise RuntimeError("intentional policy failure")
        assert "actual_ticks" not in json.dumps(view)
        return original(self, view)

    monkeypatch.setattr(GreedyProductionPolicy, "act", broken)
    with pytest.raises(RunFailedError) as error:
        run_one(resolve_run(run_path(bundle, "processing_fixed")), verbose=False)
    directory = error.value.run_dir
    manifest = json_file(directory, "run.json")
    assert manifest["status"] == "failed" and manifest["last_tick"] == 18
    assert manifest["failure"]["message"] == "intentional policy failure"
    assert manifest["inputs"]["scenario"]["processing_samples"]
    assert any(
        e["kind"] == "processing_completed"
        for r in json_lines(directory, "trace.jsonl")
        for e in r["events"]
    )


def test_cli_hash_seed_cwd_and_numeric_equivalence(bundle):
    path = run_path(bundle, "processing_generated")
    assert main(["validate", str(path)]) == 0
    assert not (bundle / "runs").exists()
    code = "from smartsom.config import resolve_run; from smartsom.config.codec import canonical_json; import sys; r=resolve_run(sys.argv[1]); print(canonical_json(r.resolved.scenario))"
    outputs = [
        subprocess.check_output(
            [sys.executable, "-c", code, str(path)],
            cwd=cwd,
            env={**os.environ, "PYTHONHASHSEED": seed},
            text=True,
        )
        for cwd, seed in [(ROOT, "1"), (bundle, "99")]
    ]
    assert outputs[0] == outputs[1]
    before = resolve_run(path)
    edit(
        bundle / "configs/scenarios/processing_generated.yaml",
        lambda d: d.update(processing_low=0.800, processing_high=1.200),
    )
    assert before.resolved.scenario == resolve_run(path).resolved.scenario


def test_fixed_samples_override_generated_duration_before_quality_multiplier():
    # Override all encountered processing samples. The selected sample wins over
    # uncertainty, then the machine's quality-mode multiplier applies.
    scenario = small_scenario(
        processing_low=Decimal("0.8"),
        processing_high=Decimal("1.2"),
        processing_samples=(ProcessingSample("demand", "op", "machine", 5),),
    )
    sim = ProductionSimulator(scenario)
    job = loaded_machine(sim)
    row = act(sim, machine=MachineCommand(job, "normal"))
    assert (
        next(
            e["actual_ticks"]
            for e in row["events"]
            if e["kind"] == "processing_started"
        )
        == 5
    )
    assert sim.machine_state["machine"]["remaining"] == 4


def test_duration_machine_override_does_not_add_capability():
    scenario = small_scenario()
    step = ProductionStep("op", "drill", 2, {"missing": 5})
    with pytest.raises(ValueError, match="unknown or incapable"):
        ProductionSimulator(replace(scenario, demands=(Demand("demand", (step,)),)))
