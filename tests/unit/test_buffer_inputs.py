"""Grid buffer configuration, recording failures and semantic identity regressions."""

import hashlib
import json
from dataclasses import FrozenInstanceError

import pytest
from test_experiments import bundle as bundle
from test_experiments import edit, json_file, json_lines, run_path

from smartsom.config import ConfigurationError, resolve_run
from smartsom.config.codec import primitive
from smartsom.engine.production import ProductionSimulator
from smartsom.experiments import RunFailedError, run_one
from smartsom.experiments.cli import main
from smartsom.trace.production import Playback, audit


@pytest.mark.parametrize(
    "name,makespan",
    [
        ("buffers_direct_zero", 39),
        ("buffers_vehicle_zero", 26),
        ("buffers_post_one", 34),
    ],
)
def test_configured_hand_and_exported_v2(bundle, name, makespan):
    path = run_path(bundle, name)
    prepared = resolve_run(path)
    case = prepared.resolved.scenario
    assert main(["validate", str(path)]) == 0
    assert not (bundle / "runs").exists()
    sim = ProductionSimulator(case)
    for command in prepared.resolved.algorithm.commands:
        sim.step(command)
    result = run_one(prepared, verbose=False)
    assert result.simulation_result.final_state == sim.snapshot()
    assert result.simulation_result.makespan == makespan
    assert audit(result.run_dir)["status"] == "passed"
    playback = Playback(result.run_dir)
    assert playback.row(makespan)["state"] == sim.snapshot()
    manifest = json_file(result.run_dir, "run.json")
    assert manifest["inputs"]["scenario"]["factory"] == primitive(case.factory)
    assert len(manifest["result"]["completed"]) == len(case.demands)
    assert manifest["audit"]["status"] == "passed"
    assert {p.name for p in result.run_dir.iterdir()} == {
        "run.json",
        "trace.jsonl",
        "logs",
    }
    with pytest.raises(FrozenInstanceError):
        case.factory.buffers[0].storage.capacity = 10


@pytest.mark.parametrize(
    "mutation",
    [
        lambda d: d["factory"]["buffers"][0]["storage"].update(capacity=True),
        lambda d: d["factory"]["buffers"][0]["storage"].update(capacity=1.5),
        lambda d: d["factory"]["buffers"][0]["storage"].update(capacity=-1),
        lambda d: d["factory"]["buffers"][0]["storage"].update(capacity="1"),
        lambda d: d["factory"]["buffers"][0].update(machine_id="unknown"),
        lambda d: d["factory"]["buffers"][0].update(shared_capacity=2),
        lambda d: d["factory"]["buffers"].append(d["factory"]["buffers"][0]),
    ],
)
def test_bad_capacities_fail_before_execution(bundle, mutation, monkeypatch):
    edit(bundle / "configs/factories/buffers_direct_zero.yaml", mutation)
    monkeypatch.setattr(
        ProductionSimulator,
        "__init__",
        lambda *a, **k: pytest.fail("constructed simulator"),
    )
    with pytest.raises(ConfigurationError):
        resolve_run(run_path(bundle, "buffers_direct_zero"))
    assert not (bundle / "runs").exists()


def test_duplicate_keys_unknown_module_fields_and_cp_rejection(bundle):
    path = run_path(bundle, "buffers_direct_zero")
    factory = bundle / "configs/factories/buffers_direct_zero.yaml"
    old = factory.read_text()
    factory.write_text(
        old.replace("capacity: null", "capacity: null\n      capacity: 1", 1)
    )
    with pytest.raises(ConfigurationError, match="duplicate"):
        resolve_run(path)
    factory.write_text(old)
    scenario = bundle / "configs/scenarios/buffers_direct_zero.yaml"
    edit(scenario, lambda d: d.update(buffers={"seed": 10}))
    with pytest.raises(ConfigurationError):
        resolve_run(path)
    edit(scenario, lambda d: d.pop("buffers"))
    edit(path, lambda d: d.update(algorithm="../algorithms/cp_sat.yaml"))
    with pytest.raises(
        ConfigurationError, match="CP-SAT has no grid production adapter"
    ):
        resolve_run(path)
    assert not (bundle / "runs").exists()


def test_toggle_seed_identity_and_explicit_transfer_compatibility(bundle):
    path = run_path(bundle, "buffers_direct_zero")
    original = resolve_run(path).resolved.scenario
    assert all(
        b.role not in ("machine_pre", "machine_post") for b in original.factory.buffers
    )
    sim = ProductionSimulator(original)
    assert not sim.pre and not sim.post
    assert set(sim.capacity) == {"input", "output"}
    # An obsolete toggle may not synthesize unlimited machine storage.
    edit(
        bundle / "configs/scenarios/buffers_direct_zero.yaml",
        lambda d: d.update(buffers=None),
    )
    with pytest.raises(ConfigurationError, match="buffers"):
        resolve_run(path)


@pytest.mark.parametrize("failure", ["script", "provider", "replay", "write"])
def test_failures_preserve_buffer_records_and_never_success_makespan(
    bundle, monkeypatch, failure
):
    from smartsom.algorithms.production import ScriptedProductionPolicy
    from smartsom.trace.production import ExecutionAudit, Recorder

    path = run_path(bundle, "buffers_vehicle_zero")
    if failure == "script":
        edit(
            bundle / "configs/algorithms/buffers_vehicle_zero.yaml",
            lambda d: d["algorithm"]["parameters"]["commands"].pop(),
        )
    elif failure == "provider":
        original = ScriptedProductionPolicy.act

        def fail(self, view):
            if self.position == 10:
                raise RuntimeError("provider after buffer wait")
            return original(self, view)

        monkeypatch.setattr(ScriptedProductionPolicy, "act", fail)
    elif failure == "replay":
        original = ExecutionAudit.append

        def fail(self, row):
            if row["tick"] == 10:
                raise ValueError("buffer trajectory mismatch")
            return original(self, row)

        monkeypatch.setattr(ExecutionAudit, "append", fail)
    else:
        original = Recorder.append

        def fail(self, row):
            if row["tick"] == 10:
                raise OSError("buffer trajectory writer failed")
            return original(self, row)

        monkeypatch.setattr(Recorder, "append", fail)
    with pytest.raises(RunFailedError) as error:
        run_one(resolve_run(path), verbose=False)
    directory = error.value.run_dir
    manifest = json_file(directory, "run.json")
    assert manifest["status"] == "failed"
    assert manifest["failure"]["message"]
    rows = json_lines(directory, "trace.jsonl")
    assert rows and rows[-1]["tick"] == manifest["last_tick"]
    assert manifest["result"] == rows[-1]["state"]
    assert manifest["inputs"]["scenario"]["factory"]["buffers"]


def test_deadlock_evidence_and_cli_failure(bundle):
    path = run_path(bundle, "buffers_vehicle_zero")
    # A policy that makes no progress reaches the explicit safety horizon;
    # it must not be reported as a completed manufacturing task.
    edit(
        bundle / "configs/scenarios/buffers_vehicle_zero.yaml",
        lambda d: d.update(tick_limit=5),
    )
    edit(
        bundle / "configs/algorithms/buffers_vehicle_zero.yaml",
        lambda d: d["algorithm"]["parameters"].update(commands=[{}] * 5),
    )
    assert main(["run", str(path)]) != 0
    directory = next((bundle / "runs").iterdir())
    manifest = json_file(directory, "run.json")
    assert manifest["status"] == "truncated"
    assert manifest["result"]["tick"] == 5
    assert not manifest["result"]["completed"]
    assert audit(directory)["run_status"] == "truncated"


@pytest.mark.parametrize("machine_buffers", [False, True])
def test_buffer_hashseed_cwd_and_input_permutation(bundle, tmp_path, machine_buffers):
    import os
    import subprocess
    import sys

    source = run_path(bundle, "buffers_combined")
    factory = bundle / "configs/factories/buffers_combined.yaml"
    if not machine_buffers:

        def remove_explicit_buffers(data):
            f = data["factory"]
            removed = {
                b["buffer_id"]
                for b in f["buffers"]
                if b["role"] in ("machine_pre", "machine_post")
            }
            f["buffers"] = [b for b in f["buffers"] if b["buffer_id"] not in removed]
            f["ports"] = [
                p
                for p in f["ports"]
                if not any(
                    b["target"].get("buffer_id") in removed for b in p["bindings"]
                )
            ]

        edit(factory, remove_explicit_buffers)
    code = """from smartsom.config import resolve_run
from smartsom.config.codec import canonical_json
from smartsom.engine.production import ProductionSimulator
from smartsom.algorithms.production import GreedyProductionPolicy
import sys
r=resolve_run(sys.argv[1]); s=ProductionSimulator(r.resolved.scenario)
p=GreedyProductionPolicy(s.factory,seed=r.resolved.scenario.seed,rule='spt'); rows=[]
while not s.done:
 rows.append(s.step(p.act(s.decision(p.rank(s.decision())))))
print(canonical_json(rows))
"""
    before = subprocess.check_output(
        [sys.executable, "-c", code, str(source)],
        cwd=bundle,
        env={**os.environ, "PYTHONHASHSEED": "1"},
    )

    def reverse_factory(data):
        for group in ("machines", "buffers", "agvs", "ports"):
            data["factory"][group].reverse()

    edit(factory, reverse_factory)
    edit(
        bundle / "configs/workloads/buffers_combined.yaml",
        lambda d: d["demands"].reverse(),
    )
    after = subprocess.check_output(
        [sys.executable, "-c", code, str(source)],
        cwd=tmp_path,
        env={**os.environ, "PYTHONHASHSEED": "987"},
    )
    assert before == after


def test_pre_item9_committed_main_goldens_remain_historical(bundle):
    path = bundle / "data/reference/buffers/disabled_golden.json"
    frozen = json.loads(path.read_text())
    assert frozen["source_commit"] == "df3d0cfb542757204069a718ef816efe04657b84"
    assert frozen["cases"]
    # Matrix golden outputs keep their exact bytes and cannot be run through
    # the grid entry by inventing missing facilities or movement actions.
    assert (
        hashlib.sha256(path.read_bytes()).hexdigest()
        == "4548dd9c9ecb1cd28a818c0fd7d06870f31bd716eb6ad24ab7b61e9307b9d8bc"
    )
    with pytest.raises(
        ConfigurationError,
        match="factory/v2|historical scripted actions require migration",
    ):
        resolve_run(run_path(bundle / "historical", "run_fixed_trace"))
