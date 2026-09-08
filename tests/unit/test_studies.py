import json
import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest
from test_experiments import bundle as bundle
from test_experiments import edit, json_file, json_lines, run_path

from smartsom.config import (
    ConfigurationError,
    load_resolved_run,
    resolve_run,
    resolve_study,
)
from smartsom.config.codec import digest, primitive
from smartsom.config.models import RecordingSpec
from smartsom.config.snapshots import resolved_from_data
from smartsom.config.study import semantic_run, study_roots
from smartsom.experiments import run_batch, run_one
from smartsom.experiments.batch import exclusive_lock
from smartsom.experiments.cli import main


def study_file(
    bundle, *, scenario="quality_generated", replications=1, algorithms=None
):
    path = bundle / "study.yaml"
    path.write_text(
        json.dumps(
            {
                "schema": "smartsom.study/v1",
                "seed": 101,
                "replications": replications,
                "cases": [
                    {"id": "case", "scenario": f"configs/scenarios/{scenario}.yaml"}
                ],
                "algorithms": algorithms
                or [
                    {"id": "spt", "config": "configs/algorithms/spt.yaml"},
                    {"id": "first", "config": "configs/algorithms/first_feasible.yaml"},
                ],
                "output_root": "studies",
            }
        )
    )
    return path


def test_shared_materialization_pairing_ablation_and_golden(bundle, monkeypatch):
    import smartsom.config.resolver as resolver

    counter = []
    original = resolver.materialize_quality

    def count(*args, **kwargs):
        counter.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(resolver, "materialize_quality", count)
    path = study_file(bundle, replications=2)
    edit(
        path,
        lambda d: d.update(
            variants=[{"id": "all"}, {"id": "no_quality", "disable": ["quality"]}]
        ),
    )
    study = resolve_study(path)
    assert len(study.entries) == 8 and len(counter) == 2
    assert study_roots(101, "case", 0, "spt") == (
        965267409869488611,
        16825626808070815075,
    )
    for rep in range(2):
        rows = [e for e in study.entries if e.replication == rep]
        assert len({e.resolved.workload_sha256 for e in rows}) == 1
        assert all(e.resolved.workload is rows[0].resolved.workload for e in rows)
        enabled = [e for e in rows if not e.disabled]
        assert enabled[0].resolved.quality is enabled[1].resolved.quality
        for domain in ("workload", "demand", "quality"):
            assert (
                len(
                    {
                        next(s.value for s in e.resolved.seeds if s.domain == domain)
                        for e in rows
                    }
                )
                == 1
            )
        assert (
            len(
                {
                    next(s.value for s in e.resolved.seeds if s.domain == "solver")
                    for e in rows
                }
            )
            == 2
        )
    assert not (bundle / "studies").exists()
    with pytest.raises(FrozenInstanceError):
        study.entries = ()
    before = study.plan_sha256
    edit(path, lambda d: d["algorithms"].reverse())
    edit(path, lambda d: d["variants"].reverse())
    assert resolve_study(path).plan_sha256 == before


@pytest.mark.parametrize(
    "name", ["competition", "quality_generated", "quality_combined", "generated"]
)
def test_snapshot_restore_without_authoring_references(bundle, name):
    resolved = resolve_run(run_path(bundle, name))
    first = run_one(resolved)
    shutil.rmtree(bundle / "configs")
    shutil.rmtree(bundle / "data")
    restored = load_resolved_run(first.run_dir / "resolved_run.yaml")
    assert restored == resolved
    assert run_one(restored).simulation_result == first.simulation_result


@pytest.mark.parametrize(
    "change",
    [
        lambda d: d.update(unknown=1),
        lambda d: d["factory"].update(unknown=1),
        lambda d: d.update(workload_sha256="0" * 64),
        lambda d: d["seeds"][0].update(value=0),
        lambda d: d.update(quality=None),
    ],
)
def test_snapshot_rejects_unknown_or_inconsistent_inputs(bundle, change):
    data = {
        "schema": "smartsom.resolved-run/v1",
        **primitive(resolve_run(run_path(bundle, "quality_generated"))),
    }
    change(data)
    with pytest.raises(ConfigurationError):
        resolved_from_data(data)


def test_batch_serial_parallel_equal_resume_and_recording(bundle):
    study = resolve_study(study_file(bundle, replications=2))
    serial = run_batch(study, workers=1)
    parallel = run_batch(study, workers=2)
    assert serial.completed == parallel.completed == 4
    a = json_file(serial.study_dir, "summary.json")["runs"]
    b = json_file(parallel.study_dir, "summary.json")["runs"]
    for x, y in zip(a, b, strict=True):
        for name in ("trace.jsonl", "observation_hashes.jsonl", "summary.json"):
            assert (Path(x["run_dir"]) / name).read_bytes() == (
                Path(y["run_dir"]) / name
            ).read_bytes()
        assert not (Path(x["run_dir"]) / "observations.jsonl").exists()
        resolved = load_resolved_run(Path(x["run_dir"]) / "resolved_run.yaml")
        assert semantic_run(resolved) == semantic_run(
            next(e.resolved for e in study.entries if e.entry_id == x["entry_id"])
        )
    before = list(serial.study_dir.rglob("attempt-*"))
    shutil.rmtree(bundle / "configs")
    shutil.rmtree(bundle / "data")
    assert run_batch(resume=serial.study_dir).completed == 4
    assert list(serial.study_dir.rglob("attempt-*")) == before
    restored = load_resolved_run(Path(a[0]["run_dir"]) / "resolved_run.yaml")
    assert run_one(restored).simulation_result.quality is not None
    with exclusive_lock(serial.study_dir / "coordinator.lock"):
        with pytest.raises(RuntimeError, match="active"):
            run_batch(resume=serial.study_dir)
    with exclusive_lock(serial.study_dir / "locks" / f"{a[0]['entry_id']}.lock"):
        with pytest.raises(RuntimeError, match="active"):
            run_batch(resume=serial.study_dir)
    (Path(a[0]["run_dir"]) / "trace.jsonl").write_text("changed")
    with pytest.raises(ConfigurationError, match="evidence"):
        run_batch(resume=serial.study_dir)


def test_failed_script_continues_and_retry_is_explicit(bundle):
    (bundle / "bad.yaml").write_text(
        "schema: smartsom.algorithm/v1\nalgorithm:\n  provider: builtin.scripted\n  parameters: {actions: []}\n"
    )
    path = study_file(
        bundle,
        algorithms=[
            {"id": "bad", "config": "bad.yaml"},
            {"id": "good", "config": "configs/algorithms/spt.yaml"},
        ],
    )
    result = run_batch(resolve_study(path))
    assert (result.completed, result.failed, result.pending) == (1, 1, 0)
    initial = list(result.study_dir.rglob("attempt-*"))
    assert run_batch(resume=result.study_dir).failed == 1
    assert list(result.study_dir.rglob("attempt-*")) == initial
    assert run_batch(resume=result.study_dir, retry_failed=True).failed == 1
    assert len(list(result.study_dir.rglob("attempt-*"))) == len(initial) + 1
    failed = next(
        r
        for r in json_file(result.study_dir, "summary.json")["runs"]
        if r["status"] == "failed"
    )
    assert failed.get("makespan") is None


def test_incomplete_restarts_and_changed_source_rejected(bundle, monkeypatch):
    import smartsom.experiments.batch as batch

    study = resolve_study(
        study_file(
            bundle, algorithms=[{"id": "spt", "config": "configs/algorithms/spt.yaml"}]
        )
    )
    result = run_batch(study)
    entry = study.entries[0]
    partial = (
        result.study_dir / "children" / entry.entry_id / "attempt-00000002-partial"
    )
    partial.mkdir()
    (partial / "partial.log").write_text("preserve")
    assert run_batch(resume=result.study_dir).completed == 1
    assert (partial / "partial.log").read_text() == "preserve"
    original = batch.execution_identity()
    monkeypatch.setattr(
        batch, "execution_identity", lambda: original | {"commit": "different"}
    )
    with pytest.raises(ConfigurationError, match="identity changed"):
        run_batch(resume=result.study_dir)


def test_full_observation_and_hash_are_same_context(bundle):
    base = resolve_run(run_path(bundle, "quality_generated"))
    full = run_one(
        replace(
            base,
            run=base.run.model_copy(
                update={"recording": RecordingSpec(observations="full", debug=True)}
            ),
        )
    )
    hashed = run_one(
        replace(
            base,
            run=base.run.model_copy(
                update={"recording": RecordingSpec(observations="hash")}
            ),
        )
    )
    rows = json_lines(full.run_dir, "observations.jsonl")
    hashes = json_lines(hashed.run_dir, "observation_hashes.jsonl")
    assert hashes == [
        {"decision_index": i, "sha256": digest(row)} for i, row in enumerate(rows)
    ]
    assert full.simulation_result == hashed.simulation_result
    assert (full.run_dir / "debug.log").exists()
    assert not (hashed.run_dir / "debug.log").exists()


def test_invalid_study_and_cli_preview(bundle, capsys):
    path = study_file(bundle)
    assert main(["plan", str(path)]) == 0
    assert json.loads(capsys.readouterr().out)["runs"] == 2
    assert not (bundle / "studies").exists()
    edit(path, lambda d: d.update(seed=True))
    assert main(["plan", str(path)]) == 2
    assert not (bundle / "studies").exists()


@pytest.mark.parametrize("twice", [False, True])
def test_ctrl_c_drains_then_interrupts_only_owned_workers(bundle, twice):
    path = study_file(
        bundle,
        replications=3,
        algorithms=[{"id": "spt", "config": "configs/algorithms/spt.yaml"}],
    )
    launcher = bundle / "interrupt_probe.py"
    launcher.write_text("""
import sys, time
from pathlib import Path
import smartsom.experiments.batch as batch
from smartsom.config import resolve_study
original_worker = batch._worker

def slow_worker(*args):
    original_run = batch.run_one
    def slow_run(resolved, *, on_progress):
        (Path(args[2]) / "started").touch()
        time.sleep(2)
        return original_run(resolved, on_progress=on_progress)
    batch.run_one = slow_run
    return original_worker(*args)

if __name__ == "__main__":
    batch._worker = slow_worker
    result = batch.run_batch(resolve_study(sys.argv[1]), workers=1)
    print(result.study_dir, flush=True)
""")
    log = (bundle / "probe.log").open("w")
    process = subprocess.Popen(
        [sys.executable, str(launcher), str(path)],
        stdout=log,
        stderr=log,
        start_new_session=True,
    )
    try:
        deadline = time.monotonic() + 20
        while not list((bundle / "studies").rglob("started")):
            assert process.poll() is None
            assert time.monotonic() < deadline
            time.sleep(0.05)
        os.killpg(process.pid, signal.SIGINT)
        if twice:
            time.sleep(0.2)
            os.killpg(process.pid, signal.SIGINT)
        assert process.wait(timeout=20) == 0
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        log.close()
    directory = next((bundle / "studies").iterdir())
    summary = json_file(directory, "summary.json")
    assert summary["interrupted"] is True
    assert summary["completed"] == (0 if twice else 1)
    assert len(list(directory.rglob("attempt-*"))) == 1
    resumed = run_batch(resume=directory, workers=2)
    assert resumed.completed == 3 and resumed.failed == 0


def test_incompatible_ablation_fails_before_allocation(bundle):
    path = study_file(
        bundle, algorithms=[{"id": "fixed", "config": "configs/algorithms/spt_m0.yaml"}]
    )
    # Fixed-quality provider cannot be silently weakened when quality is disabled.
    (bundle / "configs/algorithms/spt_m0.yaml").write_text(
        "schema: smartsom.algorithm/v1\nalgorithm:\n  provider: builtin.spt\n  parameters: {quality_mode: M0}\n"
    )
    edit(path, lambda d: d.update(variants=[{"id": "off", "disable": ["quality"]}]))
    with pytest.raises(ConfigurationError, match="requires enabled quality"):
        resolve_study(path)
    assert not (bundle / "studies").exists()


def test_hash_writer_failure_retains_failed_evidence(bundle, monkeypatch):
    import smartsom.experiments.evidence as evidence
    from smartsom.experiments import RunFailedError

    original = evidence.append_json

    def broken(stream, value):
        if Path(stream.name).name == "observation_hashes.jsonl":
            raise OSError("hash writer failed")
        original(stream, value)

    monkeypatch.setattr(evidence, "append_json", broken)
    entry = resolve_study(study_file(bundle)).entries[0]
    with pytest.raises(RunFailedError) as error:
        run_one(entry.resolved)
    assert json_file(error.value.run_dir, "summary.json")["makespan"] is None
    assert (
        json_file(error.value.run_dir, "failure.json")["message"]
        == "hash writer failed"
    )


def test_plan_identity_cwd_hash_seed_and_reordering(bundle):
    path = study_file(bundle)
    code = "from smartsom.config import resolve_study; import sys; print(resolve_study(sys.argv[1]).plan_sha256)"
    results = [
        subprocess.check_output(
            [sys.executable, "-c", code, str(path)],
            cwd=cwd,
            env={**os.environ, "PYTHONHASHSEED": seed},
            text=True,
        ).strip()
        for cwd, seed in ((bundle, "1"), (bundle.parent, "99"))
    ]
    assert results[0] == results[1] == resolve_study(path).plan_sha256
