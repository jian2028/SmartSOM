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
from smartsom.telemetry.runtime import CURRENT, DisplayOptions, RuntimeDisplay
root = Path(sys.argv[1])
scope = control.Scope()
scope.bind(root)
token = control.CURRENT.set(scope)
display = RuntimeDisplay(DisplayOptions(progress="off", verbose=False), kind="tune")
display_token = CURRENT.set(display)
status = "failed"
try:
    display.start()
    root, plan, state = tuning_batch.load_run(root)
    result = tuning_batch.execute_batch(root, plan, state, display=display)
    status = result["status"]
    (root / "subprocess-result.json").write_text(json.dumps(result))
finally:
    display.finish(status)
    display.close()
    CURRENT.reset(display_token)
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

    native_root = None
    timing = {
        "schema": "smartsom.pause-latency-acceptance/v1",
        "algorithm": algorithm,
        "prepared": asdict(prepared),
        "poll_seconds": 0.02,
        "trials": [],
        "phase_observations": [],
        "interpretation": "Observed intervals, not phase durations or latency guarantees; brief phases may be missed.",
    }
    timing_path = root / "pause-latency.json"

    def observe_phase():
        if native_root is None:
            return None
        path = native_root / "control/stage-resources.json"
        if not path.is_file():
            return None
        evidence = json.loads(path.read_text())
        sample = {
            "observed_at": time.time(),
            "stage": evidence.get("current_stage"),
            "published_at": evidence.get("updated_at"),
            "generation": evidence.get("id"),
        }
        previous = timing["phase_observations"][-1:]
        if not previous or any(
            previous[0].get(key) != sample[key]
            for key in ("stage", "published_at", "generation")
        ):
            timing["phase_observations"].append(sample)
        return sample

    def submit_pause(label, *, task_id=None, active_worker, phase_sample=None):
        trial = {
            "label": label,
            "active_worker": active_worker,
            "phase_at_request": phase_sample or observe_phase(),
            "submitted_at": time.time(),
            "submitted_monotonic": time.monotonic(),
        }
        receipt = runtime.submit(root, "pause", task_id=task_id)
        trial["receipt"] = receipt
        timing["trials"].append(trial)
        timing_path.write_text(json.dumps(timing, indent=2))
        return trial

    def pause_completed(trial):
        receipt = runtime.snapshot(root)["commands"][trial["receipt"]["id"]]
        return receipt.get("progress", {}).get("state") == "completed"

    def record_paused(trial):
        trial["observed_paused_at"] = time.time()
        trial["observed_pause_seconds"] = time.monotonic() - trial.pop(
            "submitted_monotonic"
        )
        trial["receipt"] = runtime.snapshot(root)["commands"][trial["receipt"]["id"]]
        trial["received_to_observed_paused_seconds"] = (
            trial["observed_paused_at"] - trial["receipt"]["received_at"]
        )
        trial["checkpoint"] = read_row().get("checkpoint")
        trial["updates"] = read_row().get("updates")
        trial["phase_at_observed_pause"] = observe_phase()
        timing_path.write_text(json.dumps(timing, indent=2))
        receipt = trial["receipt"]
        assert receipt["status"] == "applied"
        assert trial["submitted_at"] <= receipt["received_at"] <= receipt["applied_at"]
        assert receipt["progress"]["state"] == "completed"
        assert receipt["progress"]["targets"] == [algorithm]
        assert receipt["progress"]["total_count"] == 1
        assert receipt["progress"]["paused_count"] == 1
        assert receipt["progress"]["failed_count"] == 0
        assert receipt["progress"]["naturally_completed_count"] == 0
        assert receipt["progress"]["remaining_seconds"] is None
        assert receipt["progress"]["finished_at"] <= trial["observed_paused_at"]

    def until(predicate, label, timeout=180):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            observe_phase()
            if predicate():
                return
            assert process.poll() is None, (
                label,
                process.returncode,
                log.read_text()[-16000:],
            )
            time.sleep(0.02)
        pytest.fail(f"{label}: {runtime.snapshot(root)!r}\n{log.read_text()[-16000:]}")

    def await_phase(stages, worker):
        matched = None

        def observed():
            nonlocal matched
            sample = observe_phase()
            if (
                sample
                and sample["stage"] in stages
                and sample["published_at"] >= worker["pid_create_time"]
            ):
                matched = sample
                return True
            return False

        until(observed, f"actual phase in {sorted(stages)} for current worker")
        return matched

    try:
        until(lambda: read_row().get("updates", 0) >= 1, "first native commit")
        first = read_row()
        native_root = Path(first["attempts"][-1]["run_dir"])
        worker = json.loads((native_root / "tuning-runtime.json").read_text())
        original_pid = worker["pid"]
        training_sample = await_phase({"sampling", "optimizing"}, worker)
        selected_trial = submit_pause(
            "selected_active",
            task_id=algorithm,
            active_worker=True,
            phase_sample=training_sample,
        )
        until(
            lambda: (
                runtime.snapshot(root).get("tasks", {}).get(algorithm, {}).get("status")
                == "paused"
                and not queue_admission.snapshot(root).get("slots")
                and pause_completed(selected_trial)
            ),
            "checkpoint pause and observed resource release",
        )
        record_paused(selected_trial)
        paused = read_row()
        from smartsom.experiments import runtime_resources as resource_model

        paused_estimate = runtime.snapshot(root)["tasks"][algorithm][
            "resource_estimate"
        ]
        resource_group = paused_estimate["resource_group"]
        paused_resources = resource_model.snapshot(root)["groups"][resource_group][
            "stages"
        ]
        assert paused_resources
        assert paused_estimate["memory"] >= max(
            row["memory_peak"] for row in paused_resources.values()
        )
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
        held_trial = submit_pause("global_already_held", active_worker=False)
        until(
            lambda: (
                runtime.snapshot(root)["status"] == "paused"
                and pause_completed(held_trial)
            ),
            "global paused service",
        )
        record_paused(held_trial)
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
                and json.loads((native_root / "tuning-runtime.json").read_text())[
                    "phase"
                ]
                == "training"
            ),
            "new actor resumes saved task",
        )
        resumed = json.loads((native_root / "tuning-runtime.json").read_text())
        assert resumed["physical_ticks"] >= paused["physical_ticks"]
        assert resumed["updates"] >= paused["updates"]
        validation_sample = await_phase({"validation"}, resumed)
        global_trial = submit_pause(
            "global_active", active_worker=True, phase_sample=validation_sample
        )
        until(
            lambda: (
                runtime.snapshot(root)["status"] == "paused"
                and runtime.snapshot(root)["tasks"][algorithm]["status"] == "paused"
                and not queue_admission.snapshot(root).get("slots")
                and pause_completed(global_trial)
            ),
            "global safe pause of resumed actor",
        )
        record_paused(global_trial)
        assert read_row()["updates"] >= paused["updates"]
        assert read_row()["updates"] < 32
        runtime.submit(root, "resume")
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
        from smartsom.telemetry.monitor import read_snapshot

        public = read_snapshot(root)
        assert (root / "logs/progress.json").is_file()
        assert any(row["id"] == algorithm for row in public["tasks"])
        final_resources = resource_model.snapshot(root)["groups"][resource_group][
            "stages"
        ]
        for phase in ("sampling", "optimizing", "validation"):
            observation = final_resources[phase]
            assert observation["memory_peak"] > 0
            assert observation["cpu_samples"] > 0
            assert observation["cpu_peak"] is not None
        assert sum(row["cpu_peak"] or 0 for row in final_resources.values()) > 0
        for phase, before in paused_resources.items():
            assert final_resources[phase]["memory_peak"] >= before["memory_peak"]
            if before["cpu_peak"] is not None:
                assert final_resources[phase]["cpu_peak"] >= before["cpu_peak"]
        final_estimate = runtime.snapshot(root)["tasks"][algorithm]["resource_estimate"]
        assert final_estimate["resource_group"] == resource_group
        assert final_estimate["memory"] >= max(
            row["memory_peak"] for row in final_resources.values()
        )
        assert not queue_admission.snapshot(root)["slots"]
    finally:
        timing["sample_count"] = len(timing["trials"])
        timing["active_worker_sample_count"] = sum(
            row["active_worker"] for row in timing["trials"]
        )
        timing["observed_stages"] = sorted(
            {row["stage"] for row in timing["phase_observations"] if row["stage"]}
        )
        timing_path.write_text(json.dumps(timing, indent=2))
        if process.poll() is None:
            # Ordinary test cleanup is cooperative; timeout never upgrades to kill.
            control.stop(root, timeout=60)
            process.wait(timeout=60)
