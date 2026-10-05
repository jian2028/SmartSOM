"""Real, disposable CPU calibration with original v3 PPO/DQN input closures."""

import copy
import json
import time
from dataclasses import asdict
from pathlib import Path

import pytest

from smartsom import api
from smartsom.config.experiment_v3 import prepare_v3
from smartsom.experiments.tuning_calibration import ExecutionProfile
from smartsom.experiments.tuning_probe import ProbeSupervisor, run_training_probe

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.learning


def frozen_engineering_input(tmp_path, algorithm):
    # Optional early inspection delays the second closed DQN transition. Author
    # a sufficient engineering budget before freezing the calibration input.
    probe_ticks = 16 if algorithm == "dqn" else 4
    config = api.load_config(ROOT / f"configs/test/runs/train_all_{algorithm}.yaml")
    config.composition = str(
        ROOT / f"configs/test/compositions/small_train_{algorithm}.yaml"
    )
    config.scenario = str(ROOT / "configs/test/scenarios/small_matrix_zero.yaml")
    config.scenario_overrides["tick_limit"] = 2 * probe_ticks
    config.training.groups = ("machine", "buffer", "dispatcher")
    config.training.total_ticks = 2 * probe_ticks
    config.training.ticks_per_update = probe_ticks
    config.training.record_initial = True
    config.runtime.num_envs = 1
    config.runtime.sampling_processes = 0
    config.runtime.numerical_threads = 1
    config.runtime.device = "cpu"
    config.validation.enabled = True
    config.validation.every_updates = 1
    config.validation.replications = 1
    config.evaluation.replications = 1
    config.evaluation.record = False
    config.evaluation.full_replay = True
    config.output.root = str(tmp_path / "unused-formal-output")
    config.output.name = f"engineering-only-{algorithm}"
    parameters = {"batch_size": 2}
    if algorithm == "ppo":
        parameters["n_epochs"] = 1
    else:
        parameters.update(warmup_ticks=0, target_update_ticks=4)
    parameter_file = tmp_path / f"engineering-{algorithm}-parameters.json"
    parameter_file.write_text(
        json.dumps(
            {"schema": "smartsom.algorithm-parameters/v1", "parameters": parameters}
        )
    )
    config.training.parameters = str(parameter_file)
    # Freeze every scientific input after fixture authoring. Calibration itself
    # gets no permission to shorten a budget, change a network or replace seeds.
    prepared = prepare_v3(config)
    return prepared, {"prepared": asdict(prepared)}


def require_optional_cpu_runtime():
    pytest.importorskip("torch")
    pytest.importorskip("ray")
    pytest.importorskip("psutil")
    pytest.importorskip("threadpoolctl")


@pytest.mark.parametrize(("algorithm", "concurrency"), [("ppo", 1), ("dqn", 2)])
def test_real_probe_complete_update_validation_checkpoint_and_frozen_inputs(
    tmp_path, algorithm, concurrency
):
    require_optional_cpu_runtime()
    prepared, group = frozen_engineering_input(tmp_path, algorithm)
    probe_ticks = prepared.config.training.ticks_per_update
    original = copy.deepcopy(group)
    polls = []
    runner = ProbeSupervisor(
        work_root=tmp_path / "probes", keep_artifacts=True, on_poll=polls.append
    )
    measurement = runner.run(
        run_training_probe, group, ExecutionProfile(1, concurrency, "cpu"), 90
    )
    assert measurement.valid, measurement.reason
    assert measurement.profile == ExecutionProfile(1, concurrency, "cpu")
    assert measurement.stages["physical_ticks"] == probe_ticks * concurrency
    assert measurement.stages["updates"] == concurrency
    assert measurement.stages["cpu_request"] == concurrency
    assert measurement.stages["sampling_cpu_overhead"] == 0
    assert measurement.peak_memory > 0
    assert measurement.stages["per_trial_peak_memory"] > 0
    assert measurement.stages["cold_start"] > 0
    assert measurement.stages["saving"] > 0
    assert measurement.stages["validation"] > 0
    assert measurement.stages["validation_cases"] == 1
    assert measurement.throughput == pytest.approx(
        (probe_ticks * concurrency) / measurement.elapsed_seconds
    )
    assert polls and polls[-1]["elapsed_seconds"] > 0
    if concurrency > 1:
        assert measurement.stages["overlap_seconds"] > 0
        assert measurement.peak_memory > measurement.stages["per_trial_peak_memory"]
    assert group == original
    assert not Path(prepared.config.output.root).exists()
    candidates = list((tmp_path / "probes").glob("smartsom-probe-*/worker-*/native"))
    assert len(candidates) == concurrency
    for directory in candidates:
        original_saved = json.loads(
            (directory / "config/original-prepared.json").read_text()
        )
        assert original_saved == original["prepared"]
        saved = json.loads((directory / "config/prepared.json").read_text())
        assert saved["scientific_sha256"] == prepared.scientific_sha256
        assert json.loads(saved["config_json"]) == json.loads(prepared.config_json)
        for key in (
            "scenario_json",
            "composition_json",
            "parameters_json",
            "validation_json",
            "evaluation_json",
            "policies_json",
        ):
            assert saved[key] == original["prepared"][key]
        record = json.loads((directory / "run.json").read_text())
        assert record["kind"] == "calibration"
        assert record["formal_evidence"] is False
        assert record["physical_ticks"] == probe_ticks and record["updates"] == 1
        assert sum(record["optimizations"].values()) > 0
        assert (directory / "checkpoints/update-000000/continuation.pkl").is_file()
        assert (directory / "checkpoints/update-000001/continuation.pkl").is_file()
        snapshot = json.loads(
            (directory / "checkpoints/update-000001/snapshot.json").read_text()
        )
        assert snapshot["scientific_sha256"] == prepared.scientific_sha256
        rows = json.loads((directory / "logs/validation-000001.json").read_text())
        assert len(rows) == 1
        assert not any(row.get("engineering_failure") for row in rows)
        assert list((directory / "evaluation").iterdir()) == []
    import psutil

    assert all(not psutil.pid_exists(pid) for pid in runner.last_pids)


def test_real_native_probe_cold_start_timeout_joins_owned_process(tmp_path):
    require_optional_cpu_runtime()
    _, group = frozen_engineering_input(tmp_path, "ppo")
    runner = ProbeSupervisor(work_root=tmp_path / "probes", keep_artifacts=True)
    started = time.monotonic()
    measurement = runner.run(
        run_training_probe, group, ExecutionProfile(1, 1, "cpu"), 1.0
    )
    assert not measurement.valid
    assert "deadline" in measurement.reason
    assert "cleanup failed" not in measurement.reason
    assert measurement.elapsed_seconds <= 1.05
    assert time.monotonic() - started < 1.2
    import psutil

    assert runner.last_pids and all(
        not psutil.pid_exists(pid) for pid in runner.last_pids
    )
