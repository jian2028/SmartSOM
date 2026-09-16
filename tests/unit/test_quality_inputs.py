"""Strict grid quality authoring, seeded attempts and durable run evidence."""

import hashlib
import json
import os
import random
import subprocess
import sys
from dataclasses import FrozenInstanceError, replace
from fractions import Fraction
from pathlib import Path

import pytest
import yaml

from smartsom.algorithms.production import GreedyProductionPolicy
from smartsom.config import ConfigurationError, load_resolved_run, resolve_run
from smartsom.config.codec import canonical_json, primitive
from smartsom.domain.production import Demand, ProductionStep
from smartsom.engine.production import ProductionSimulator
from smartsom.experiments import RunFailedError, run_one
from smartsom.experiments.cli import main
from smartsom.trace.production import audit

ROOT = Path(__file__).resolve().parents[2]


def write(path, data):
    path.write_text(json.dumps(data))


def edit(path, mutate):
    data = yaml.safe_load(path.read_text())
    mutate(data)
    write(path, data)


@pytest.fixture
def files(tmp_path):
    for dest, source in {
        "factory.yaml": "configs/factories/quality.yaml",
        "workload.yaml": "configs/workloads/quality_fixed.yaml",
        "algorithm.yaml": "configs/algorithms/spt_quality_m0.yaml",
    }.items():
        (tmp_path / dest).write_bytes((ROOT / source).read_bytes())
    scenario = yaml.safe_load(
        (ROOT / "configs/scenarios/quality_fixed.yaml").read_text()
    )
    scenario.update(factory="factory.yaml", workload="workload.yaml")
    write(tmp_path / "scenario.yaml", scenario)
    write(
        tmp_path / "run.yaml",
        dict(
            schema="smartsom.run/v1",
            scenario="scenario.yaml",
            algorithm="algorithm.yaml",
            seed=42,
            output_root="runs",
        ),
    )
    return tmp_path


@pytest.mark.parametrize(
    "field,value",
    [
        ("time_scale", True),
        ("time_scale", 0),
        ("time_scale", -1),
        ("time_scale", "nan"),
        ("time_scale", "inf"),
        ("error_rate", False),
        ("error_rate", -0.1),
        ("error_rate", 1.1),
        ("error_rate", "nan"),
        ("error_rate", "-Infinity"),
        ("time_scale", "oops"),
        ("quality_mode_id", ""),
    ],
)
def test_invalid_mode_values_fail_before_simulator_or_directory(
    files, monkeypatch, field, value
):
    edit(
        files / "factory.yaml",
        lambda d: d["factory"]["machines"][0]["quality_modes"][0].update(
            {field: value}
        ),
    )
    monkeypatch.setattr(
        "smartsom.engine.production.ProductionSimulator",
        lambda *a, **k: pytest.fail("constructed Simulator"),
    )
    with pytest.raises(ConfigurationError):
        resolve_run(files / "run.yaml")
    assert main(["validate", str(files / "run.yaml")]) == 2
    assert not (files / "runs").exists()


@pytest.mark.parametrize(
    "case",
    [
        "empty_modes",
        "duplicate_machine",
        "duplicate_mode",
        "unknown_field",
        "unknown_probability_visibility",
        "unknown_kind",
        "unknown_quality_operation",
        "unknown_quality_demand",
        "duplicate_draw",
        "bool_draw",
        "float_draw",
        "negative_draw",
        "overflow_draw",
        "bad_attempt",
        "seed_in_scenario",
        "factory_off",
    ],
)
def test_invalid_structure_and_references(files, case):
    f = yaml.safe_load((files / "factory.yaml").read_text())
    s = yaml.safe_load((files / "scenario.yaml").read_text())
    modes = f["factory"]["machines"][0]["quality_modes"]
    if case == "empty_modes":
        modes.clear()
    if case == "duplicate_machine":
        f["factory"]["machines"].append(f["factory"]["machines"][0])
    if case == "duplicate_mode":
        modes.append(modes[0])
    if case == "unknown_field":
        modes[0]["typo"] = 1
    if case == "unknown_probability_visibility":
        s["quality_probability_visibility"] = "sometimes"
    if case == "unknown_kind":
        s["quality"] = {"kind": "random"}
    if case == "unknown_quality_operation":
        s["quality_samples"][0]["operation_id"] = "unknown"
    if case == "unknown_quality_demand":
        s["quality_samples"][0]["demand_id"] = "unknown"
    if case == "duplicate_draw":
        s["quality_samples"].append(s["quality_samples"][0])
    if case in ("bool_draw", "float_draw", "negative_draw", "overflow_draw"):
        s["quality_samples"][0]["draw"] = {
            "bool_draw": True,
            "float_draw": 1.0,
            "negative_draw": -1,
            "overflow_draw": 2**53,
        }[case]
    if case == "bad_attempt":
        s["quality_samples"][0]["attempt"] = 0
    if case == "seed_in_scenario":
        s["seed"] = 1
    if case == "factory_off":
        f["factory"]["machines"][0].pop("quality_modes")
    write(files / "factory.yaml", f)
    write(files / "scenario.yaml", s)
    with pytest.raises(ConfigurationError):
        resolve_run(files / "run.yaml")
    assert not (files / "runs").exists()


@pytest.mark.parametrize(
    "file,contents",
    [
        (
            "scenario.yaml",
            "schema: smartsom.scenario/v2\nquality_samples: []\nquality_samples: []\n",
        ),
        ("factory.yaml", "schema: smartsom.factory/v2\nfactory: {}\nfactory: {}\n"),
    ],
)
def test_duplicate_keys(files, file, contents):
    (files / file).write_text(contents)
    with pytest.raises(ConfigurationError, match="duplicate key"):
        resolve_run(files / "run.yaml")


def test_fixed_mode_requires_every_machine_to_offer_it(files):
    edit(
        files / "factory.yaml",
        lambda d: d["factory"]["machines"][1].update(
            quality_modes=[
                {"quality_mode_id": "custom", "time_scale": "1", "error_rate": ".01"}
            ]
        ),
    )
    with pytest.raises(ConfigurationError, match="every machine"):
        resolve_run(files / "run.yaml")


def test_unspecified_samples_use_attempt_keyed_draws(files):
    original = resolve_run(files / "run.yaml").resolved.scenario
    edit(files / "scenario.yaml", lambda d: d["quality_samples"].pop())
    partial = resolve_run(files / "run.yaml").resolved.scenario
    assert len(partial.quality_samples) == 1
    sim = ProductionSimulator(partial)
    assert sim._draw("quality", "J/attempt/1", "A1") == Fraction(
        original.quality_samples[0].draw, 2**53
    )
    assert sim._draw("quality", "J/attempt/2", "A1") != sim._draw(
        "quality", "J/attempt/1", "A1"
    )


def test_cp_rejects_quality_even_zero_probability(files):
    edit(
        files / "factory.yaml",
        lambda d: [
            m.update(error_rate="0")
            for machine in d["factory"]["machines"]
            for m in machine["quality_modes"]
        ],
    )
    write(
        files / "algorithm.yaml",
        {
            "schema": "smartsom.algorithm/v1",
            "algorithm": {
                "provider": "pyjobshop.cp_sat",
                "interface_kind": "offline_solver",
                "required_information": "full_static",
            },
        },
    )
    with pytest.raises(ConfigurationError, match="CP-SAT has no grid"):
        resolve_run(files / "run.yaml")


def test_configured_hand_freezing_and_mode_change_do_not_scale_twice(files):
    prepared = resolve_run(files / "run.yaml")
    assert main(["validate", str(files / "run.yaml")]) == 0
    result = run_one(prepared, verbose=False)
    assert (
        result.simulation_result.status == "completed"
        and result.simulation_result.makespan == 40
    )
    rows = [
        e
        for row in (
            json.loads(line)
            for line in (result.run_dir / "trace.jsonl").read_text().splitlines()
        )
        for e in row["events"]
    ]
    starts = [e for e in rows if e["kind"] == "processing_started"]
    assert [(e["tick"], e["actual_ticks"]) for e in starts] == [(4, 12), (19, 12)]
    assert audit(result.run_dir)["status"] == "passed"
    snapshot = files / "frozen.json"
    snapshot.write_text(
        canonical_json(
            {"schema": "smartsom.prepared-grid-experiment/v1", **primitive(prepared)}
        )
    )
    edit(
        files / "algorithm.yaml",
        lambda d: d["algorithm"].update(parameters={"quality_mode": "M2"}),
    )
    fast = resolve_run(files / "run.yaml")
    assert fast.resolved.scenario == prepared.resolved.scenario
    fast_result = run_one(fast, verbose=False)
    assert (
        fast_result.simulation_result.status == "completed"
        and fast_result.simulation_result.makespan == 66
    )
    assert (
        fast_result.simulation_result.final_state["jobs"]["J/attempt/1"]["quality"]
        == "FAIL"
    )
    assert (
        fast_result.simulation_result.final_state["jobs"]["J/attempt/2"]["quality"]
        == "PASS"
    )
    assert audit(fast_result.run_dir)["status"] == "passed"
    with pytest.raises(FrozenInstanceError):
        prepared.resolved.scenario.quality_samples = ()
    (files / "factory.yaml").write_text("broken")
    assert (
        run_one(load_resolved_run(snapshot), verbose=False).simulation_result
        == result.simulation_result
    )


def test_generated_golden_and_seed_isolation(files):
    edit(files / "scenario.yaml", lambda d: d.pop("quality_samples"))
    before = random.getstate()
    prepared = resolve_run(files / "run.yaml")
    assert random.getstate() == before
    sim = ProductionSimulator(prepared.resolved.scenario)
    for op in ("A1", "A2"):
        key = json.dumps(
            ["quality", 42, "J/attempt/1", op], separators=(",", ":")
        ).encode()
        expected = Fraction(int.from_bytes(hashlib.sha256(key).digest(), "big"), 2**256)
        assert sim._draw("quality", "J/attempt/1", op) == expected
    path = files / "frozen.json"
    path.write_text(
        canonical_json(
            {"schema": "smartsom.prepared-grid-experiment/v1", **primitive(prepared)}
        )
    )
    edit(files / "run.yaml", lambda d: d.update(seed=99))
    restored = load_resolved_run(path)
    assert restored == prepared
    assert resolve_run(files / "run.yaml").resolved.scenario.seed == 99


def test_permutation_and_unrelated_operation_do_not_change_draws(files):
    from smartsom.experiments.production import run

    prepared = resolve_run(files / "run.yaml")
    case = prepared.resolved.scenario
    reverse = replace(
        case,
        factory=replace(case.factory, machines=tuple(reversed(case.factory.machines))),
        quality_samples=tuple(reversed(case.quality_samples)),
    )
    a = run_one(prepared, verbose=False)
    b = run(
        reverse, prepared.resolved.algorithm, output_root=files / "runs", verbose=False
    )
    assert (
        json.loads((a.run_dir / "run.json").read_text())["result"]
        == json.loads((b / "run.json").read_text())["result"]
    )
    extra = Demand("unrelated", (ProductionStep("extra", "operation_1", 1),))
    changed = ProductionSimulator(replace(case, demands=(*case.demands, extra)))
    original = ProductionSimulator(case)
    for job in ("J/attempt/1", "J/attempt/2"):
        for operation in ("A1", "A2"):
            assert changed._draw("quality", job, operation) == original._draw(
                "quality", job, operation
            )


def test_hash_seed_and_working_directory(files):
    program = "from smartsom.config import resolve_run; from smartsom.config.codec import digest; from smartsom.experiments import run_one; import sys; r=resolve_run(sys.argv[1]); print(digest(run_one(r,verbose=False).simulation_result))"
    values = [
        subprocess.check_output(
            [sys.executable, "-c", program, str(files / "run.yaml")],
            cwd=cwd,
            env={**os.environ, "PYTHONHASHSEED": seed},
            text=True,
        )
        for cwd, seed in [(ROOT, "1"), (files, "82")]
    ]
    assert values[0] == values[1]


@pytest.mark.parametrize("failure", ["script", "provider", "writer"])
def test_failure_retains_quality_inputs_and_partial_evidence(
    files, monkeypatch, failure
):
    if failure == "script":
        write(
            files / "algorithm.yaml",
            {
                "schema": "smartsom.algorithm/v1",
                "algorithm": {
                    "provider": "builtin.scripted",
                    "parameters": {"commands": [{}]},
                },
            },
        )
    if failure == "provider":
        monkeypatch.setattr(
            GreedyProductionPolicy,
            "act",
            lambda *a: (_ for _ in ()).throw(ValueError("provider exploded")),
        )
    if failure == "writer":
        from smartsom.trace.production import Recorder

        original = Recorder.append

        def broken(self, row):
            if any(e["kind"] == "quality_revealed" for e in row["events"]):
                raise OSError("quality write failed")
            return original(self, row)

        monkeypatch.setattr(Recorder, "append", broken)
    with pytest.raises(RunFailedError) as exc:
        run_one(resolve_run(files / "run.yaml"), verbose=False)
    folder = exc.value.run_dir
    manifest = json.loads((folder / "run.json").read_text())
    assert (
        manifest["inputs"]["scenario"]["quality_samples"]
        and manifest["status"] == "failed"
    )
    assert manifest["failure"]["message"]
    if failure != "provider":
        assert (folder / "trace.jsonl").stat().st_size > 0


def test_cli_success_and_execution_failure(files):
    assert main(["run", str(files / "run.yaml")]) == 0
    write(
        files / "algorithm.yaml",
        {
            "schema": "smartsom.algorithm/v1",
            "algorithm": {
                "provider": "builtin.scripted",
                "parameters": {"commands": []},
            },
        },
    )
    assert main(["run", str(files / "run.yaml")]) == 1


def test_exhausted_run_with_extra_script_action_is_not_quality_success(files):
    case = resolve_run(files / "run.yaml").resolved.scenario
    sim, policy, commands = (
        ProductionSimulator(case),
        GreedyProductionPolicy(case.factory, rule="spt", quality_mode="M0"),
        [],
    )
    while not sim.done:
        ranks = policy.rank(sim.decision())
        command = policy.act(sim.decision(ranks))
        commands.append(primitive(command))
        sim.step(command)
    assert sim.tick == 40 and sim.status == "completed"
    write(
        files / "algorithm.yaml",
        {
            "schema": "smartsom.algorithm/v1",
            "algorithm": {
                "provider": "builtin.scripted",
                "parameters": {"commands": [*commands, {}]},
            },
        },
    )
    with pytest.raises(RunFailedError) as exc:
        run_one(resolve_run(files / "run.yaml"), verbose=False)
    manifest = json.loads((exc.value.run_dir / "run.json").read_text())
    assert manifest["status"] == "failed" and manifest["result"]["tick"] == 40
    assert (
        "unused" in manifest["failure"]["message"]
        or "after" in manifest["failure"]["message"]
    )
