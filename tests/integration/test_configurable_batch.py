"""Disposable public directory execution of declarative PPO/DQN contracts."""

import json
import os
import subprocess
import sys

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
