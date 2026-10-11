"""Real v4 -> calibration -> local Ray -> native CPU PPO, in disposable runs.

The engineering driver restricts candidate enumeration to one numerical thread
and the owned Ray runtime to two CPUs. Monitoring, hard probe deadlines, native
training/validation/checkpoints, Tune scheduling and final evaluation stay real.
The limits keep this integration check bounded; they are not performance claims.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml
from test_author_learning import ROOT, _inputs, _require_cpu

pytestmark = pytest.mark.learning


DRIVER = """
import json, sys
from pathlib import Path
import psutil
import ray
from smartsom.config.experiment_v4 import compile_experiment
from smartsom.experiments.author_driver import allocate, execute_saved
from smartsom.experiments import tuning_calibration
from smartsom.experiments.tuning_resources import ExecutionProfile

source, evidence = map(Path, sys.argv[1:3])
# Test-only bounded enumeration: no fake throughput or resource observations.
tuning_calibration.generate_candidates = lambda snapshot, mode="balanced": (
    ExecutionProfile(1, 1, "cpu"),
)
original_init, original_shutdown = ray.init, ray.shutdown
cluster_started = False

def owned_processes():
    identities = []
    node = ray._private.worker._global_node
    if node is not None:
        for infos in node.all_processes.values():
            for info in infos:
                try:
                    process = psutil.Process(info.process.pid)
                    for owned in [process, *process.children(recursive=True)]:
                        identities.append({"pid": owned.pid, "create_time": owned.create_time()})
                except psutil.Error:
                    pass
    (evidence / "owned-ray-processes.json").write_text(json.dumps(identities))


def init(**kwargs):
    global cluster_started
    # This wrapper only configures the fresh local runtime owned by this driver.
    assert kwargs.get("address") == "local"
    assert not ray.is_initialized()
    kwargs.update(num_cpus=2, object_store_memory=80 * 1024**2,
                  _temp_dir=sys.argv[3])
    result = original_init(**kwargs)
    cluster_started = True
    owned_processes()
    return result


def shutdown():
    if ray.is_initialized():
        owned_processes()
    return original_shutdown()

ray.init, ray.shutdown = init, shutdown
root = allocate(compile_experiment(source, require_dependencies=True))
(evidence / "allocated.json").write_text(json.dumps(str(root)))
try:
    result = execute_saved(root)
    resumed = execute_saved(root) if result["status"] == "completed" else None
    source_check = None
    if resumed:
        import io
        from contextlib import redirect_stdout
        from smartsom.experiments.cli import main
        captured = io.StringIO()
        with redirect_stdout(captured):
            assert main(["check", "--task", "evaluate", "--source", str(root)]) == 0
        source_check = json.loads(captured.getvalue())
    (evidence / "result.json").write_text(json.dumps({
        "result": result, "resumed": resumed,
        "cluster_started": cluster_started,
        "ray_initialized_after": ray.is_initialized(),
        "source_check": source_check,
    }))
except Exception as exc:
    (evidence / "failure.json").write_text(json.dumps({
        "exception": type(exc).__name__, "message": str(exc),
        "cluster_started": cluster_started,
        "ray_initialized_after": ray.is_initialized(),
    }))
    raise
finally:
    shutdown()
"""


def _json(path):
    return json.loads(Path(path).read_text())


def _live(identity):
    import psutil

    try:
        process = psutil.Process(identity["pid"])
        return (
            process.create_time() == identity["create_time"]
            and process.status() != psutil.STATUS_ZOMBIE
        )
    except psutil.NoSuchProcess:
        return False


def _cleanup_owned(evidence):
    """Emergency test cleanup uses this driver's exact recorded PID identities."""
    import psutil

    path = evidence / "owned-ray-processes.json"
    if not path.exists():
        return
    identities = _json(path)
    processes = [psutil.Process(row["pid"]) for row in identities if _live(row)]
    for process in processes:
        try:
            process.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(processes, timeout=3)
    for process in alive:
        if any(row["pid"] == process.pid and _live(row) for row in identities):
            process.kill()
    psutil.wait_procs(alive, timeout=3)


def _driver(tmp_path, performance, seconds, *, skip_calibration=False):
    _require_cpu()
    source = _inputs(tmp_path / "inputs", "sb3")
    document = yaml.safe_load(source.read_text())
    document["execution"] = {
        "tuning": performance,
        "mode": "performance",
        "scheduling": "fixed" if skip_calibration else "adaptive",
        **(
            {"calibration_level": "off"}
            if skip_calibration
            else {"calibration_level": "quick", "calibration_seconds": seconds}
        ),
    }
    source.write_text(yaml.safe_dump(document, sort_keys=False))
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    script = evidence / "engineering_tune_driver.py"
    script.write_text(DRIVER)
    # Ray requires a short UNIX socket path on macOS; the test owns this directory.
    import tempfile

    with tempfile.TemporaryDirectory(
        prefix="som-v4-", dir=None if os.name == "nt" else "/tmp"
    ) as ray_root:
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(ROOT / "src")
        with (
            (evidence / "stdout.log").open("wb") as out,
            (evidence / "stderr.log").open("wb") as err,
        ):
            process = subprocess.Popen(
                [sys.executable, str(script), str(source), str(evidence), ray_root],
                cwd=ROOT,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=out,
                stderr=err,
                start_new_session=True,
            )
            try:
                process.wait(timeout=120)
            except subprocess.TimeoutExpired:
                # The run knows its driver and descendants. This affects only the
                # disposable run, never a shared Ray cluster or external process.
                allocated = evidence / "allocated.json"
                if allocated.exists():
                    from smartsom.experiments.control import stop

                    stop(_json(allocated), timeout=5, force=True)
                if process.poll() is None:
                    process.kill()
                process.wait(timeout=10)
                pytest.fail("owned v4 Tune driver exceeded its 120 s wall deadline")
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait(timeout=10)
                identities_path = evidence / "owned-ray-processes.json"
                if identities_path.exists():
                    deadline = time.monotonic() + 5
                    while (
                        any(_live(row) for row in _json(identities_path))
                        and time.monotonic() < deadline
                    ):
                        time.sleep(0.05)
                    (evidence / "ray-left-before-cleanup.json").write_text(
                        json.dumps(
                            [row for row in _json(identities_path) if _live(row)]
                        )
                    )
                _cleanup_owned(evidence)
    return _json(evidence / "allocated.json"), evidence, process.returncode


def _no_owned_ray_left(evidence):
    path = evidence / "owned-ray-processes.json"
    if not path.exists():
        return
    deadline = time.monotonic() + 5
    while any(_live(row) for row in _json(path)) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not [row for row in _json(path) if _live(row)]
    assert _json(evidence / "ray-left-before-cleanup.json") == []


@pytest.mark.parametrize("performance", ["recommend", "auto"])
def test_real_v4_tune_calibration_and_pipeline(tmp_path, performance):
    root, evidence, exit_code = _driver(tmp_path, performance, 20.0)
    assert exit_code == 0, (evidence / "stderr.log").read_text()
    root = Path(root)
    observed = _json(evidence / "result.json")
    result = observed["result"]
    assert Path(result["run_directory"]) == root
    state = _json(root / "batch.json")
    tune_root = Path(state["tune_directory"])
    tune_state = _json(tune_root / "batch.json")
    calibration = _json(tune_root / "calibration.json")
    progress = _json(root / "logs/progress.json")
    assert progress["kind"] == "tune" and progress["tuning"]
    assert progress["tuning"]["stage"] == result["status"]
    assert progress["tuning"]["entries"][0]["status"] == result["status"]
    assert _json(root / "preflight.json")["status"] == "passed"
    workflow = next(row for row in progress["tasks"] if row["id"] == "entry-0001")[
        "values"
    ]["workflow"]
    assert workflow["training_total"] == 16
    assert workflow["training_unit"] == "physical ticks"
    assert workflow["round_size"] == 8 and workflow["round_total"] == 2
    assert workflow["validation_cases"] == 1 and workflow["validation_rounds"] == 2
    assert calibration["ready"] and not calibration["missing_groups"]
    assert 0 < calibration["active_seconds"] <= 20.05
    assert calibration["recommendations"]
    assert any(
        row["valid"]
        and row["stages"]["updates"] >= 1
        and row["stages"]["validation_cases"] == 1
        for row in calibration["measurements"]
    )
    assert all(
        row
        == {
            "threads": 1,
            "concurrency": 1,
            "device": "cpu",
            "num_envs": 1,
            "sampling_processes": 0,
        }
        for row in calibration["recommendations"].values()
    )
    assert not observed["ray_initialized_after"]
    row = tune_state["entries"]["entry-0001"]
    if performance == "recommend":
        assert result["status"] == state["status"] == "recommended"
        assert not observed["cluster_started"]
        assert not tune_state["segments"] and not row.get("attempts")
        assert not (tune_root / "experiments").exists()
        assert not list((root / "entries").glob("*/runs/*"))
    else:
        assert result["status"] == state["status"] == "completed"
        assert result["completed"] == 1 and result["failed"] == 0
        assert observed["cluster_started"]
        assert observed["resumed"]["completed"] == 1
        assert observed["source_check"]["task"] == "evaluate"
        assert len(tune_state["segments"]) == len(row["attempts"]) == 1
        assert row["status"] == "completed"
        assert row["physical_ticks"] == 16 and row["updates"] == 2
        selected = row["selected_option"]["profile"]
        actual = json.loads(row["selected_prepared"]["config_json"])["runtime"]
        assert actual["num_envs"] == selected["num_envs"]
        assert actual["sampling_processes"] == selected["sampling_processes"]
        assert actual["numerical_threads"] == selected["threads"]
        assert f"num_envs={selected['num_envs']}" in workflow["runtime_mode"]
        values = next(item for item in progress["tasks"] if item["id"] == "entry-0001")[
            "values"
        ]
        assert values["validation_finished"] == values["validation_requested"] == 1
        assert values["validation_batches_finished"] == 2
        assert values["evaluation_finished"] == values["evaluation_requested"] == 1
        checkpoint = Path(row["checkpoint"])
        record = _json(checkpoint / "record.json")
        from smartsom.config.experiment_v3 import PreparedComposition
        from smartsom.experiments.tuning_session import verify_identity

        frozen = row.get(
            "selected_prepared",
            _json(tune_root / "plan.json")["entries"][0]["prepared"],
        )
        marker = verify_identity(PreparedComposition(**frozen), record, checkpoint)
        assert marker["phase"] == "experiment_complete"
        attempt = Path(row["attempts"][0]["run_dir"])
        assert len(list((attempt / "logs").glob("validation-*.json"))) == 2
        evaluated = _json(attempt / "evaluation/tuning-final.json")
        assert len(evaluated) == 1 and not evaluated[0].get("engineering_failure")
        assert not _json(attempt / "evaluation/summary.json")["exceptions"]
    _no_owned_ray_left(evidence)


def test_real_v4_fixed_layout_runs_without_performance_probes(tmp_path):
    root, evidence, exit_code = _driver(tmp_path, "auto", None, skip_calibration=True)
    assert exit_code == 0, (evidence / "stderr.log").read_text()
    root = Path(root)
    observed = _json(evidence / "result.json")
    assert observed["result"]["status"] == "completed"
    tune_root = Path(_json(root / "batch.json")["tune_directory"])
    report = _json(tune_root / "calibration.json")
    assert report["status"] == "skipped"
    assert report["calibration_level"] == "off"
    assert report["active_seconds"] == 0
    assert report["measurements"] == []
    assert report["uncalibrated_groups"]
    assert (
        _json(tune_root / "batch.json")["entries"]["entry-0001"]["status"]
        == "completed"
    )
    assert not list((tune_root / "calibration").glob("probes/*"))
    _no_owned_ray_left(evidence)


def test_short_real_calibration_never_launches_a_trial_without_valid_baseline(tmp_path):
    root, evidence, exit_code = _driver(tmp_path, "recommend", 2.0)
    root = Path(root)
    state = _json(root / "batch.json")
    tune_root = Path(state["tune_directory"])
    calibration = _json(tune_root / "calibration.json")
    assert 0 <= calibration["active_seconds"] <= 2.05
    if calibration["active_seconds"] == 0:
        # Host sampling/startup may exhaust the budget before any probe starts.
        assert calibration["status"] == "deadline"
        assert calibration["measurements"] == []
        assert calibration["uncalibrated_groups"]
    assert not _json(tune_root / "batch.json")["segments"]
    assert not (tune_root / "experiments").exists()
    if exit_code == 0:
        # Recommendation can retain an explicitly uncalibrated starting layout;
        # neither this case nor a completed warm probe may launch a trial.
        assert calibration["ready"]
        assert _json(evidence / "result.json")["result"]["status"] == "recommended"
    else:
        failure = _json(evidence / "failure.json")
        assert not calibration["ready"] and calibration["missing_groups"]
        assert "no valid constrained calibration baseline" in failure["message"]
        assert not failure["cluster_started"]
        assert not failure["ray_initialized_after"]
    _no_owned_ray_left(evidence)
