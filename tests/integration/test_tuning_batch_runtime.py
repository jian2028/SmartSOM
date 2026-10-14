"""End-to-end disposable batch with real observation, probes and native learning."""

import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import pytest
from test_tuning_probe_native import (
    frozen_engineering_input,
    require_optional_cpu_runtime,
)

from smartsom.experiments import tuning_batch as batch
from smartsom.experiments.tuning_session import verify_identity

pytestmark = pytest.mark.learning


def test_cli_check_import_is_json_and_does_not_allocate_or_mutate(tmp_path):
    require_optional_cpu_runtime()
    from smartsom.config.codec import digest
    from smartsom.experiments.composable import archive_inputs
    from smartsom.experiments.evidence import write_json

    prepared, _ = frozen_engineering_input(tmp_path, "ppo")
    root = tmp_path / "prepared-study"
    (root / "snapshots/a/config").mkdir(parents=True)
    frozen = archive_inputs(root / "snapshots/a", prepared)
    snapshot = root / "snapshots/a/config/prepared.json"
    plan = {
        "schema": "smartsom.composable-study-plan/v1",
        "source": {"old": True},
        "recipe": {"controls": ["initial", "rule", "random"]},
        "entries": [
            {
                "id": "a",
                "snapshot": "snapshots/a/config/prepared.json",
                "snapshot_sha256": digest(json.loads(snapshot.read_text())),
                "scientific_sha256": frozen.scientific_sha256,
            }
        ],
    }
    write_json(root / "plan.json", plan)
    write_json(root / "study.json", {"plan_sha256": digest(plan)})
    before = {
        str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()
    }
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "smartsom.experiments.cli",
            "tune",
            "check",
            "--study",
            str(root),
            "--log-format",
            "json",
            "--progress",
            "off",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["status"] == "feasible"
    assert not (tmp_path / "tune-runs").exists()
    assert before == {
        str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()
    }


@pytest.mark.parametrize(
    ("algorithm", "execution"), [("ppo", "fixed"), ("dqn", "adaptive")]
)
def test_real_batch_calibrates_executes_and_skips_completed_resume(
    tmp_path, algorithm, execution
):
    require_optional_cpu_runtime()
    import ray

    prepared, _ = frozen_engineering_input(tmp_path, algorithm)
    inputs = batch.BatchInputs(
        (
            {
                "experiment_id": algorithm,
                "prepared": asdict(prepared),
                "control_spec": {},
            },
        ),
        mode="performance",
        execution=execution,
        active_limit=45,
        output_root=str(tmp_path / "engineering-only"),
    )
    checked = batch.preflight(inputs)
    assert checked["status"] == "feasible"
    assert not Path(inputs.output_root).exists() and not ray.is_initialized()
    root, plan, state = batch.allocate_batch(inputs)
    result = batch.execute_batch(root, plan, state)
    assert result["status"] == "completed", json.loads(
        (root / "batch.json").read_text()
    )
    assert result["completed"] == 1 and result["failed"] == 0
    assert not ray.is_initialized()
    calibration = json.loads((root / "calibration.json").read_text())
    assert 0 < calibration["active_seconds"] <= 45
    assert calibration["recommendations"] and not calibration["missing_groups"]
    assert any(
        m["valid"] and m["stages"]["updates"] >= 1 for m in calibration["measurements"]
    )
    saved = json.loads((root / "batch.json").read_text())
    row = saved["entries"][algorithm]
    assert row["physical_ticks"] == prepared.config.training.total_ticks
    assert row["updates"] == (
        prepared.config.training.total_ticks
        // prepared.config.training.ticks_per_update
    )
    record = json.loads((Path(row["checkpoint"]) / "record.json").read_text())
    batch.load_run(root)
    frozen = row.get("selected_prepared", plan["entries"][0]["prepared"])
    from smartsom.config.experiment_v3 import PreparedComposition

    marker = verify_identity(PreparedComposition(**frozen), record, row["checkpoint"])
    assert marker["phase"] == "experiment_complete"
    before = (root / "batch.json").read_bytes()
    assert batch.resume_batch(root)["completed"] == 1
    assert (root / "batch.json").read_bytes() == before
    assert not ray.is_initialized()


@pytest.mark.parametrize("algorithm", ["ppo", "dqn"])
def test_real_subprocess_tune_runtime_pause_hold_and_resume(tmp_path, algorithm):
    """Public intents pause genuine actors after native commits, then continue.

    This establishes checkpoint selection, resource release and remaining-budget
    continuation. Exact optimizer/RNG equality is covered by the separate native
    continuation test; observing counters here does not establish that stronger
    property or cross-platform numerical reproducibility.
    """
    import hashlib
    import time

    from smartsom.config.experiment_v3 import prepare_v3
    from smartsom.experiments import control, queue_admission
    from smartsom.experiments import runtime_control as runtime

    require_optional_cpu_runtime()
    original, _ = frozen_engineering_input(tmp_path, algorithm)
    config = original.config
    # Author the disposable acceptance budget before freezing. There are enough
    # real update boundaries to issue controls without artificial learner sleeps.
    config.training.total_ticks = 32 * config.training.ticks_per_update
    prepared = prepare_v3(config)
    root, plan, _ = batch.allocate_batch(
        batch.BatchInputs(
            (
                {
                    "experiment_id": algorithm,
                    "prepared": asdict(prepared),
                    "control_spec": {},
                },
            ),
            mode="performance",
            execution="fixed",
            calibration_level="off",
            active_limit=0,
            output_root=str(tmp_path / "runtime-proof"),
        )
    )
    script = """
import json, sys
from pathlib import Path
from smartsom.experiments import control, tuning_batch
root = Path(sys.argv[1])
scope = control.Scope()
scope.bind(root)
token = control.CURRENT.set(scope)
status = "failed"
try:
    root, plan, state = tuning_batch.load_run(root)
    result = tuning_batch.execute_batch(root, plan, state)
    status = result["status"]
    (root / "subprocess-result.json").write_text(json.dumps(result))
finally:
    scope.finish(status)
    control.CURRENT.reset(token)
"""
    log = root / "runtime-subprocess.log"
    with log.open("w") as stream:
        process = subprocess.Popen(
            [sys.executable, "-c", script, str(root)],
            stdout=stream,
            stderr=subprocess.STDOUT,
        )

    def read_row():
        return json.loads((root / "batch.json").read_text())["entries"][algorithm]

    def until(predicate, label, timeout=180):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return
            assert process.poll() is None, (
                label,
                process.returncode,
                log.read_text()[-16000:],
            )
            time.sleep(0.02)
        pytest.fail(f"{label}: {runtime.snapshot(root)!r}\n{log.read_text()[-16000:]}")

    try:
        until(lambda: read_row().get("updates", 0) >= 1, "first native commit")
        first = read_row()
        native_root = Path(first["attempts"][-1]["run_dir"])
        worker = json.loads((native_root / "tuning-runtime.json").read_text())
        original_pid = worker["pid"]
        runtime.submit(root, "pause", task_id=algorithm)
        until(
            lambda: (
                runtime.snapshot(root).get("tasks", {}).get(algorithm, {}).get("status")
                == "paused"
                and not queue_admission.snapshot(root).get("slots")
            ),
            "checkpoint pause and observed resource release",
        )
        paused = read_row()
        assert 1 <= paused["updates"] < 32
        checkpoint = Path(paused["checkpoint"])
        record = json.loads((checkpoint / "record.json").read_text())
        from smartsom.config.experiment_v3 import PreparedComposition

        selected = paused.get("selected_prepared", plan["entries"][0]["prepared"])
        marker = verify_identity(PreparedComposition(**selected), record, checkpoint)
        assert marker["updates"] == paused["updates"]
        assert marker["physical_ticks"] == paused["physical_ticks"]
        continuation_hash = hashlib.sha256(
            (checkpoint / "continuation.pkl").read_bytes()
        ).hexdigest()
        import psutil

        assert (
            not psutil.pid_exists(original_pid)
            or psutil.Process(original_pid).create_time() != worker["pid_create_time"]
            or psutil.Process(original_pid).status() == psutil.STATUS_ZOMBIE
        )
        runtime.submit(root, "pause")
        until(
            lambda: runtime.snapshot(root)["status"] == "paused",
            "global paused service",
        )
        runtime.submit(root, "resume")
        until(
            lambda: not runtime.snapshot(root)["paused"], "global resume acknowledged"
        )
        held = runtime.snapshot(root)
        assert algorithm in held["paused_tasks"]
        assert held["tasks"][algorithm]["status"] == "paused"
        assert not queue_admission.snapshot(root).get("slots")
        assert read_row()["updates"] == paused["updates"]
        runtime.submit(root, "resume", task_id=algorithm)
        until(
            lambda: (
                (native_root / "tuning-runtime.json").exists()
                and json.loads((native_root / "tuning-runtime.json").read_text())["pid"]
                != original_pid
                and json.loads((native_root / "tuning-runtime.json").read_text())[
                    "updates"
                ]
                >= paused["updates"]
            ),
            "new actor resumes saved task",
        )
        resumed = json.loads((native_root / "tuning-runtime.json").read_text())
        assert resumed["physical_ticks"] >= paused["physical_ticks"]
        assert resumed["updates"] >= paused["updates"]
        process.wait(timeout=240)
        assert process.returncode == 0, log.read_text()[-16000:]
        result = json.loads((root / "subprocess-result.json").read_text())
        assert result["status"] == "completed"
        final = read_row()
        assert final["physical_ticks"] == prepared.config.training.total_ticks
        assert final["updates"] == 32
        assert (
            hashlib.sha256((checkpoint / "continuation.pkl").read_bytes()).hexdigest()
            == continuation_hash
        )
        assert runtime.snapshot(root)["status"] == "completed"
        assert not queue_admission.snapshot(root)["slots"]
    finally:
        if process.poll() is None:
            # Ordinary test cleanup is cooperative; timeout never upgrades to kill.
            control.stop(root, timeout=60)
            process.wait(timeout=60)
