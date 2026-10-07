"""Disposable native updates stop with complete recovery and resume on CPU."""

import json
import os
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import pytest
from test_tuning_probe_native import (
    frozen_engineering_input,
    require_optional_cpu_runtime,
)

from smartsom import api
from smartsom.config.codec import canonical_json
from smartsom.config.experiment_v3 import prepare_v3
from smartsom.experiments.control import stop


@pytest.mark.parametrize("algorithm", ["ppo", "dqn"])
def test_native_safe_stop_then_resume(tmp_path, algorithm, capsys):
    require_optional_cpu_runtime()
    prepared, _ = frozen_engineering_input(tmp_path, algorithm)
    config = prepared.config
    total_ticks = 2 * config.training.ticks_per_update
    config.training.total_ticks = total_ticks
    prepared = prepare_v3(config)
    snapshot = tmp_path / "prepared.json"
    snapshot.write_text(json.dumps(asdict(prepared)))
    script = """
import json, sys, time
from pathlib import Path
from smartsom import api
from smartsom.config.experiment_v3 import PreparedComposition
p = PreparedComposition(**json.loads(Path(sys.argv[1]).read_text()))
def updated(record):
    (Path(sys.argv[1]).parent / "ready").write_text("committed")
    time.sleep(.5)
api.train_prepared(p, on_progress=updated)
"""
    process = subprocess.Popen(
        [sys.executable, "-c", script, str(snapshot)],
        env={**os.environ, "PYTHONPATH": "src"},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    try:
        deadline = time.monotonic() + 45
        while not (tmp_path / "ready").exists():
            if process.poll() is not None or time.monotonic() > deadline:
                raise AssertionError("native update did not reach a saved boundary")
            time.sleep(0.1)
        root = next(Path(config.output.root).glob("*"))
        assert stop(root, timeout=20)["status"] == "stopped"
        record = json.loads((root / "run.json").read_text())
        assert record["status"] == "interrupted"
        assert 0 < record["physical_ticks"] < total_ticks
        assert (root / "checkpoints/recovery.json").is_file()
        result = api.resume(root)
        assert result.status == "completed" and result.environment_steps == total_ticks
        from smartsom.experiments.cli import main

        original = {
            str(path): path.read_bytes() for path in root.rglob("*") if path.is_file()
        }
        assert (
            main(
                [
                    "check",
                    "--task",
                    "evaluate",
                    "--source",
                    str(root),
                    "--checkpoint",
                    "last",
                ]
            )
            == 0
        )
        assert original == {
            str(path): path.read_bytes() for path in root.rglob("*") if path.is_file()
        }
        capsys.readouterr()
        assert (
            main(
                [
                    "run",
                    "--task",
                    "evaluate",
                    "--source",
                    str(root),
                    "--checkpoint",
                    "last",
                    "--preview",
                    "--output-root",
                    str(tmp_path / "preview"),
                ]
            )
            == 0
        )
        preview = json.loads(capsys.readouterr().out)
        assert len(preview["results"]) == 1
        assert (
            json.loads((Path(preview["run_dir"]) / "run.json").read_text())["purpose"]
            == "preview"
        )
    finally:
        if process.poll() is None:
            process.kill()
        process.communicate(timeout=10)


def _driver(arguments, stderr_path):
    # An undrained PIPE can block the driver while it logs worker shutdown.
    # The child owns its inherited file handle after the parent's handle closes.
    with stderr_path.open("wb") as stream:
        process = subprocess.Popen(
            [sys.executable, "-m", "smartsom.experiments.cli", *arguments],
            env={
                **os.environ,
                "PYTHONPATH": str(Path(__file__).resolve().parents[2] / "src"),
            },
            stdout=subprocess.DEVNULL,
            stderr=stream,
        )
    process.stderr_log_path = stderr_path
    return process


def _wait(process, predicate, timeout=45):
    deadline = time.monotonic() + timeout
    while not predicate():
        if process.poll() is not None or time.monotonic() > deadline:
            raise AssertionError(
                "driver exited or did not reach the requested engineering phase; "
                f"stderr in {process.stderr_log_path}:\n"
                + process.stderr_log_path.read_text(errors="replace")[-16000:]
            )
        time.sleep(0.1)


def test_parallel_study_stops_dispatch_and_drains_workers(tmp_path):
    require_optional_cpu_runtime()
    from test_composable_study_parallel import _freeze_study

    _freeze_study(tmp_path, 2, count=4, total_ticks=128)
    process = _driver(
        ["run", "--task", "train-evaluate", "--study", str(tmp_path)],
        tmp_path / "driver-stderr.log",
    )
    try:
        _wait(
            process,
            lambda: (
                len(
                    list((tmp_path / "experiments").glob("*/checkpoints/recovery.json"))
                )
                >= 2
            ),
        )
        assert stop(tmp_path, timeout=30)["status"] == "stopped"
        state = json.loads((tmp_path / "study.json").read_text())
        assert state["status"] == "interrupted"
        assert any(row["status"] == "queued" for row in state["entries"].values())
        assert not any(row["status"] == "running" for row in state["entries"].values())
    finally:
        if process.poll() is None:
            process.kill()
        process.communicate(timeout=10)


def test_tune_calibration_cancels_owned_probe(tmp_path):
    require_optional_cpu_runtime()
    prepared, _ = frozen_engineering_input(tmp_path, "ppo")
    recipe = tmp_path / "experiment.json"
    recipe.write_text(prepared.config_json)
    batch = tmp_path / "batch.json"
    output = tmp_path / "tune-runs"
    batch.write_text(
        canonical_json(
            {
                "schema": "smartsom.tune-batch/v1",
                "entries": [{"id": "one", "config": str(recipe)}],
                "mode": "performance",
                "active_limit": 45,
                "output_root": str(output),
            }
        )
    )
    process = _driver(
        ["run", "--task", "train-evaluate", "--config", str(batch)],
        tmp_path / "driver-stderr.log",
    )
    try:
        _wait(
            process,
            lambda: bool(
                list(output.glob("*/calibration/probes/*/worker-*/phase.json"))
            ),
        )
        root = next(output.iterdir())
        assert stop(root, timeout=20)["status"] == "stopped"
        state = json.loads((root / "batch.json").read_text())
        assert state["status"] == "interrupted"
        assert not any(row.get("attempts") for row in state["entries"].values())
        assert (root / "calibration.json").is_file()
    finally:
        if process.poll() is None:
            process.kill()
        process.communicate(timeout=10)


def test_tune_native_trial_stops_and_resumes_committed_progress(tmp_path):
    require_optional_cpu_runtime()
    prepared, _ = frozen_engineering_input(tmp_path, "ppo")
    config = prepared.config
    config.training.total_ticks = 64
    recipe = tmp_path / "experiment.json"
    recipe.write_text(canonical_json(config))
    batch = tmp_path / "batch.json"
    output = tmp_path / "tune-runs"
    batch.write_text(
        canonical_json(
            {
                "schema": "smartsom.tune-batch/v1",
                "entries": [{"id": "one", "config": str(recipe)}],
                "mode": "performance",
                "execution": "fixed",
                "active_limit": 45,
                "output_root": str(output),
            }
        )
    )
    process = _driver(
        ["run", "--task", "train-evaluate", "--config", str(batch)],
        tmp_path / "driver-stderr.log",
    )
    root = None
    try:
        _wait(
            process,
            lambda: bool(
                list(
                    output.glob(
                        "*/experiments/*/attempt-*/checkpoints/adaptive-recovery.json"
                    )
                )
            ),
            timeout=120,
        )
        root = next(output.iterdir())
        assert stop(root, timeout=30)["status"] == "stopped"
        saved = json.loads((root / "batch.json").read_text())
        assert saved["status"] == "interrupted"
        entry = saved["entries"]["one"]
        assert entry["status"] == "interrupted"
        assert 0 < entry["physical_ticks"] < 64 and entry["checkpoint"]
        result = api.resume_tune_batch(root)
        assert result["status"] == "completed" and result["completed"] == 1
    finally:
        if process.poll() is None:
            if root is not None:
                stop(root, timeout=2, force=True)
            if process.poll() is None:
                process.kill()
        process.communicate(timeout=10)
