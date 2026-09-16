import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_experiments import edit, json_file, json_lines, run_path

from smartsom.config import resolve_run
from smartsom.experiments import RunFailedError, run_one
from smartsom.experiments.evidence import artifact_digests
from smartsom.experiments.providers import build_provider


@pytest.fixture
def bundle(tmp_path):
    root = Path(__file__).resolve().parents[2]
    for name in ("configs", "data"):
        shutil.copytree(root / name, tmp_path / name)
    return tmp_path


def test_module_switches_and_algorithm_binding_preserve_other_materialized_inputs(
    bundle,
):
    from dataclasses import replace

    from smartsom.engine.production import ProductionSimulator

    scenario = bundle / "configs/scenarios/generated.yaml"
    edit(
        scenario,
        lambda value: value.update(
            mode="dynamic",
            arrivals={"initial_jobs": 1, "release_min": 2, "release_max": 5},
            processing_low=0.8,
            processing_high=1.2,
        ),
    )
    path = run_path(bundle, "generated")
    both = resolve_run(path).resolved.scenario
    edit(scenario, lambda value: value.pop("arrivals"))
    without_arrivals = resolve_run(path).resolved.scenario

    def products(world):
        return tuple(replace(d, release_at=0, reveal_at=0) for d in world.demands)

    assert products(without_arrivals) == products(both)
    assert without_arrivals.outages == both.outages
    a, b = ProductionSimulator(both), ProductionSimulator(without_arrivals)
    for demand in both.demands:
        for operation in demand.steps:
            key = demand.demand_id + "/attempt/1"
            assert a._draw("processing", key, operation.operation_id) == b._draw(
                "processing", key, operation.operation_id
            )
    edit(path, lambda value: value.update(algorithm="../algorithms/spt.yaml"))
    assert resolve_run(path).resolved.scenario == without_arrivals
    edit(
        scenario,
        lambda value: value.update(
            arrivals={"initial_jobs": 1, "release_min": 2, "release_max": 5},
            processing_low=1,
            processing_high=1,
        ),
    )
    without_processing = resolve_run(path).resolved.scenario
    assert without_processing.demands == both.demands
    assert without_processing.seed == both.seed


def test_provider_construction_has_no_unknown_provider_fallback():
    with pytest.raises(ValueError, match="unsupported provider"):
        build_provider(SimpleNamespace(algorithm=SimpleNamespace(provider="unknown")))


@pytest.mark.parametrize("target", ["trace", "manifest", "terminal"])
def test_writer_failure_retains_original_cause_and_actual_progress(
    bundle, monkeypatch, target
):
    from smartsom.engine.production import ProductionSimulator
    from smartsom.experiments.production import TerminalDisplay
    from smartsom.trace import production as trace

    actual = []
    original_step = ProductionSimulator.step
    original_append = trace.Recorder.append
    original_atomic = trace.atomic_json
    original_display = TerminalDisplay.update
    problem = OSError("simulated evidence write failure")

    def step(simulator, action):
        result = original_step(simulator, action)
        actual.append(result)
        return result

    def append(recorder, row):
        if target == "trace" and row["tick"] == 2:
            raise problem
        return original_append(recorder, row)

    def atomic(path, value):
        if target == "manifest" and value.get("status") in ("completed", "truncated"):
            raise problem
        return original_atomic(path, value)

    def display(self, row):
        if target == "terminal" and row["tick"] == 2:
            raise problem
        return original_display(self, row)

    monkeypatch.setattr(ProductionSimulator, "step", step)
    monkeypatch.setattr(trace.Recorder, "append", append)
    monkeypatch.setattr(trace, "atomic_json", atomic)
    monkeypatch.setattr(TerminalDisplay, "update", display)
    resolved = resolve_run(run_path(bundle, "run_test"))
    with pytest.raises(RunFailedError) as caught:
        run_one(resolved)
    assert caught.value.cause is problem
    record = json_file(caught.value.run_dir, "run.json")
    assert record["status"] == "failed"
    assert record["failure"]["message"] == str(problem)
    assert record["execution_state"] == actual[-1]["state"]
    rows = json_lines(caught.value.run_dir, "trace.jsonl")
    assert rows and record["last_tick"] == rows[-1]["tick"]
    assert record["result"] == rows[-1]["state"]
    assert record["inputs"]["scenario"]


def test_artifact_hashes_stream_large_files_and_preserve_exclusions(
    tmp_path, monkeypatch
):
    payload = b"semantic evidence\n" * 100_000
    (tmp_path / "trace.jsonl").write_bytes(payload)
    (tmp_path / "empty.log").touch()
    (tmp_path / "manifest.json").write_text("{}")
    (tmp_path / "summary.json.tmp").write_text("unfinished")

    def forbid_whole_file_reads(*args):
        raise AssertionError("artifact digest must not load an entire file")

    monkeypatch.setattr(Path, "read_bytes", forbid_whole_file_reads)
    assert artifact_digests(tmp_path) == {
        "empty.log": hashlib.sha256(b"").hexdigest(),
        "trace.jsonl": hashlib.sha256(payload).hexdigest(),
    }


@pytest.mark.parametrize("failure", ["source", "origins", "recipe", "attempt"])
def test_training_initialization_failure_retains_allocated_run(
    tmp_path, monkeypatch, failure
):
    from pathlib import Path

    import smartsom.api as api
    from smartsom.experiments import production_training
    from smartsom.experiments.training import TrainingFailedError

    config = api.load_config(
        Path(__file__).resolve().parents[2] / "configs/runs/sb3_production.yaml"
    )
    config.output.root = str(tmp_path)
    prepared = api.prepare(config, require_dependencies=False)
    error = OSError(f"{failure} initialization failed")

    def broken(*args, **kwargs):
        raise error

    if failure == "source":
        monkeypatch.setattr(api, "source_identity", broken)
    elif failure == "attempt":
        monkeypatch.setattr(production_training, "ProductionEvidence", broken)
    elif failure == "recipe":
        original = production_training.write_json

        def write(path, value):
            if path.name == "grid_recipe.json":
                raise error
            return original(path, value)

        monkeypatch.setattr(production_training, "write_json", write)
    else:
        original = Path.write_text

        def write(path, *args, **kwargs):
            if path.name == "origins.json":
                raise error
            return original(path, *args, **kwargs)

        monkeypatch.setattr(Path, "write_text", write)
    with pytest.raises(TrainingFailedError) as caught:
        api.train_prepared(prepared)
    assert caught.value.cause is error
    record = json.loads((caught.value.run_dir / "run.json").read_text())
    assert record["status"] == "failed"
    assert record["failure"]["message"] == str(error)
    assert record["scientific_sha256"] == prepared.scientific_sha256
