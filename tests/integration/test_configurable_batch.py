"""Disposable public directory execution of declarative PPO/DQN contracts."""

import json
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml
from test_author_learning import ROOT, _inputs, _require_cpu, _write

pytestmark = pytest.mark.learning


def inputs(directory, algorithm):
    path = _inputs(directory, "rllib")
    factory = yaml.safe_load(
        (ROOT / "src/smartsom/studio/templates/template_010.yaml").read_text(
            encoding="utf-8"
        )
    )
    for buffer in factory["factory"]["buffers"]:
        if buffer["role"] == "system_input":
            buffer["storage"]["capacity"] = 10
    _write(directory / "factory.yaml", factory)
    method = yaml.safe_load((directory / "algorithm.yaml").read_text(encoding="utf-8"))
    method["learner"].update(
        algorithm=algorithm,
        parameters={
            "batch_size": 2,
            **(
                {"n_epochs": 1}
                if algorithm == "ppo"
                else {"warmup_ticks": 0, "replay_capacity": 16, "train_every_ticks": 1}
            ),
        },
    )
    for role in ("machine", "buffer", "dispatcher"):
        extensions = method["agents"][role]["default"]["extensions"]
        extensions.update(
            observation={
                "name": "builtin.physical_job",
                "version": "1",
                "parameters": {"include_inspection": True},
            },
            network_implementation={
                "name": "builtin.physical_job_candidate",
                "version": "1",
            },
            network={
                branch: {"hidden_sizes": [8]}
                for branch in (("actor", "critic") if algorithm == "ppo" else ("q",))
            },
        )
    method["agents"]["mover"]["default"]["name"] = "automatic_travel"
    _write(directory / "algorithm.yaml", method)
    experiment = yaml.safe_load(path.read_text(encoding="utf-8"))
    experiment["training"].update(total_ticks=8, ticks_per_update=4, max_ticks=4)
    experiment["runtime"]["environment"].update(tick_limit=4)
    _write(
        directory / "travel.yaml",
        {"schema": "smartsom.travel-time-matrix/v1", "source": "auto", "overrides": {}},
    )
    experiment["runtime"]["environment"]["transport"] = {
        "mode": "travel_time_matrix",
        "matrix": "../travel.yaml",
    }
    experiment["validation"].update(
        every_updates=None, updates=[1, 2], best_mode="all_complete"
    )
    experiment["evaluation"].update(
        checkpoint="best",
        no_eligible_best={
            "outcome": "record_unselected",
            "last_diagnostic": {"enabled": True},
        },
    )
    experiment["checkpointing"] = {"retention": {"mode": "latest_full_and_best"}}
    experiment["execution"] = {
        "calibration_level": "off",
        "scheduling": "fixed",
        "max_concurrent": 2,
        "background": False,
    }
    _write(path, experiment)
    # Only Experiment YAMLs belong in the scanned directory.
    experiments = directory / "experiments"
    experiments.mkdir()
    experiment.update(
        factory="../factory.yaml",
        workload="../workload.yaml",
        algorithm="../algorithm.yaml",
    )
    _write(experiments / "experiment.yaml", experiment)
    return experiments


@pytest.mark.parametrize("algorithm", ["ppo", "dqn"])
def test_public_batch_run_without_calibration_and_unselected_final(tmp_path, algorithm):
    _require_cpu()
    directory = inputs(tmp_path / "inputs", algorithm)
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "smartsom.experiments.cli",
            "batch-run",
            str(directory),
            "--no-background",
            "--calibration-level",
            "off",
            "--log-format",
            "json",
            "--progress",
            "off",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    outputs = tmp_path / "outputs"
    selections = list(outputs.rglob("selection-outcome.json"))
    assert selections
    outcome = json.loads(selections[0].read_text(encoding="utf-8"))
    assert all(
        json.loads(path.read_text(encoding="utf-8")) == outcome for path in selections
    )
    assert outcome["status"] == "no_eligible_best"
    assert outcome["final"] is None and not outcome["ranking_eligible"]
    assert not list(outputs.rglob("tuning-final.json"))
    diagnostic = json.loads(
        next(outputs.rglob("tuning-final-last-diagnostic.json")).read_text(
            encoding="utf-8"
        )
    )
    assert diagnostic and all(
        row["evaluation_label"] == "FINAL-LAST-DIAGNOSTIC"
        and not row["ranking_eligible"]
        for row in diagnostic
    )
    calibrations = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in outputs.rglob("calibration.json")
    ]
    assert calibrations and all(
        not record["measurements"] and record["calibration_level"] == "off"
        for record in calibrations
    )
    for path in outputs.rglob("retention.json"):
        if (
            path.parent.name == "checkpoints"
            and (path.parent / "retention-owner.json").exists()
            and path.parent.parent.name != "support"
        ):
            from smartsom.experiments.checkpoint_retention import manifest

            value = manifest(path.parent.parent)
            assert value["updates"] == 2
            assert value["selected_best"] is None
            assert len(list(path.parent.glob("update-*"))) == 1


@pytest.mark.parametrize("algorithm", ["ppo", "dqn"])
def test_declared_policy_spawn_worker_count_preserves_logical_streams(
    tmp_path, algorithm
):
    _require_cpu()
    from smartsom.config.experiment_v4 import compile_experiment
    from smartsom.experiments.composable import TrainingSession, allocate

    path = inputs(tmp_path / "inputs", algorithm) / "experiment.yaml"
    experiment = yaml.safe_load(path.read_text(encoding="utf-8"))
    experiment["runtime"]["num_envs"] = 2
    experiment["validation"]["enabled"] = False
    experiment["validation"].pop("updates")
    experiment["validation"]["every_updates"] = 1
    experiment["evaluation"].pop("no_eligible_best")
    experiment["evaluation"]["checkpoint"] = "last"
    observed = []
    for processes in (1, 2):
        experiment["runtime"]["sampling_processes"] = processes
        experiment["output"]["name"] = f"declared-{algorithm}-{processes}"
        _write(path, experiment)
        prepared = (
            compile_experiment(path, require_dependencies=True).entries[0].prepared
        )
        root, record, prepared = allocate(prepared, "training")
        session = TrainingSession(prepared, root, record)
        try:
            while not session.training_done:
                session.step_update()
            observed.append(
                (
                    session.actions,
                    {
                        key: value.fingerprint()
                        for key, value in session.policies.items()
                        if hasattr(value, "fingerprint")
                    },
                    session.optimizations,
                )
            )
        finally:
            session.close()
    assert observed[0] == observed[1]


@pytest.mark.parametrize("algorithm", ["ppo", "dqn"])
def test_public_committed_stop_and_fresh_process_resume(tmp_path, algorithm):
    _require_cpu()
    from smartsom.experiments.tuning_session import verify_commit

    directory = inputs(tmp_path / "inputs", algorithm)
    path = directory / "experiment.yaml"
    experiment = yaml.safe_load(path.read_text(encoding="utf-8"))
    experiment["training"].update(total_ticks=512, ticks_per_update=64, max_ticks=512)
    experiment["runtime"]["environment"]["tick_limit"] = 512
    experiment["validation"]["updates"] = [2, 5, 8]
    experiment["execution"]["max_concurrent"] = 1
    _write(path, experiment)
    env = dict(
        os.environ,
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        PYTHONUTF8="1",
    )
    command = [sys.executable, "-m", "smartsom.experiments.cli"]
    display = ["--log-format", "json", "--progress", "off"]
    log = tmp_path / "public-stop-run.log"
    child = None
    with log.open("w", encoding="utf-8") as stream:
        child = subprocess.Popen(
            [
                *command,
                "batch-run",
                str(directory),
                "--no-background",
                "--calibration-level",
                "off",
                *display,
            ],
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
        try:
            deadline = time.monotonic() + 180
            root = None
            while time.monotonic() < deadline and child.poll() is None:
                for manifest in (tmp_path / "outputs").rglob("batch.json"):
                    state = json.loads(manifest.read_text(encoding="utf-8"))
                    if any(
                        row.get("checkpoint") and 0 < row.get("updates", 0) < 8
                        for row in state.get("entries", {}).values()
                    ):
                        root = manifest.parent
                        break
                if root is not None:
                    break
                time.sleep(0.1)
            assert root is not None, log.read_text(encoding="utf-8")
            owner = json.loads(
                (root / "control/owner.json").read_text(encoding="utf-8")
            )
            driver_root = Path(owner["driver_root"])
            stop = subprocess.run(
                [*command, "stop", str(driver_root), "--timeout", "60"],
                env=env,
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=90,
            )
            assert stop.returncode == 0, stop.stdout + stop.stderr
            child.wait(timeout=90)
            state = json.loads((root / "batch.json").read_text(encoding="utf-8"))
            assert state["status"] == "interrupted", log.read_text(encoding="utf-8")
            assert (
                json.loads((driver_root / "batch.json").read_text(encoding="utf-8"))[
                    "status"
                ]
                == "stopped"
            )
            before = next(iter(state["entries"].values()))
            boundary = verify_commit(before["checkpoint"])
            assert 0 < boundary["updates"] < 8
            assert boundary["physical_ticks"] == boundary["updates"] * 64
        finally:
            if child.poll() is None:
                # Cooperative cancellation belongs only to this disposable test run.
                if root is not None:
                    subprocess.run(
                        [
                            *command,
                            "stop",
                            str(
                                json.loads(
                                    (root / "control/owner.json").read_text(
                                        encoding="utf-8"
                                    )
                                )["driver_root"]
                            ),
                            "--timeout",
                            "30",
                        ],
                        env=env,
                        capture_output=True,
                        timeout=60,
                    )
                child.wait(timeout=60)
    resumed = subprocess.run(
        [*command, "tune", "resume", str(root), *display],
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=300,
    )
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    state = json.loads((root / "batch.json").read_text(encoding="utf-8"))
    assert state["status"] == "completed" and all(
        row["status"] == "completed" for row in state["entries"].values()
    )
    entry = next(iter(state["entries"].values()))
    checkpoint = Path(entry["checkpoint"])
    marker = verify_commit(checkpoint)
    assert (
        marker["phase"] == "experiment_complete"
        and marker["updates"] == 8
        and marker["physical_ticks"] == 512
    )
    continuation = pickle.loads((checkpoint / "continuation.pkl").read_bytes())
    assert continuation["validation_phase"] == {
        "pending_update": None,
        "completed_updates": [2, 5, 8],
    }
    assert all(
        continuation["optimizations"][role] > 0
        for role in ("machine", "buffer", "dispatcher")
    )
    assert len(continuation["history"]) == 8 and len(continuation["actions"]) == 512
    original = json.loads(
        (checkpoint / "original-prepared.json").read_text(encoding="utf-8")
    )
    for declaration in json.loads(original["policies_json"]).values():
        if declaration["role"] != "mover":
            extensions = declaration["implementation"]["extensions"]
            assert extensions["observation"]["name"] == "builtin.physical_job"
            assert (
                extensions["network_implementation"]["name"]
                == "builtin.physical_job_candidate"
            )
            assert (
                extensions["observation"]["code_sha256"]
                == extensions["network_implementation"]["code_sha256"]
            )
    calibrations = [
        json.loads(p.read_text(encoding="utf-8"))
        for p in root.rglob("calibration.json")
    ]
    assert calibrations and all(
        not record["measurements"] and record["calibration_level"] == "off"
        for record in calibrations
    )
