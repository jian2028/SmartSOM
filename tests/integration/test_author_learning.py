"""Disposable v4 CPU learning, validation and exact stage-recovery behavior."""

import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml

import smartsom
from smartsom.config.experiment_v4 import compile_experiment
from smartsom.experiments.author_driver import allocate, execute_saved
from smartsom.experiments.control import stop
from smartsom.experiments.tuning_batch import _short_probe_prepared

ROOT = Path(smartsom.__file__).resolve().parents[2]
pytestmark = pytest.mark.learning


def _require_cpu():
    missing = [
        name
        for name in ("torch", "ray", "gymnasium", "sb3_contrib")
        if importlib.util.find_spec(name) is None
    ]
    if missing:
        if os.environ.get("SMARTSOM_REQUIRE_LEARNING") == "1":
            pytest.fail(f"required optional CPU environment is missing {missing}")
        pytest.skip("optional learning extras are not installed")


def _write(path, value):
    path.write_text(yaml.safe_dump(value, sort_keys=False))


def _inputs(directory, backend):
    directory.mkdir()
    factory = yaml.safe_load(
        (ROOT / "configs/test/factories/factory_hand.yaml").read_text()
    )
    factory["factory"]["grid"].update(width=5, height=4)
    factory["factory"]["inspection_stations"] = [
        {
            "inspection_station_id": "inspect",
            "name": "Inspection",
            "footprint": {"x": 1, "y": 3, "width": 1, "height": 1},
            "slots": [
                {"slot_id": "slot", "local_cell": {"x": 0, "y": 0}, "capacity": 1}
            ],
            "inspection_ticks": 2,
        }
    ]
    factory["factory"]["ports"].append(
        {
            "port_id": "port_inspect",
            "name": "Inspection",
            "cell": {"x": 2, "y": 3},
            "bindings": [
                {
                    "target": {
                        "kind": "inspection_slot",
                        "inspection_station_id": "inspect",
                        "slot_id": "slot",
                    },
                    "operations": ["pickup", "drop_off"],
                }
            ],
        }
    )
    _write(directory / "factory.yaml", factory)
    workload = {
        "schema": "smartsom.workload/v3",
        "demands": [
            {
                "demand_id": f"job-{i}",
                "steps": [
                    {
                        "operation_id": "op-1",
                        "operation_type": "operation_1",
                        "nominal_ticks": 1,
                    }
                ],
                "release_at": 0,
                "due_at": 30,
            }
            for i in range(1, 4)
        ],
    }
    policy = {
        "kind": "new_model",
        "projection": {"max_jobs": 4},
        "extensions": {
            "network": {branch: {"hidden_sizes": [8]} for branch in ("actor", "critic")}
        },
    }
    algorithm = {
        "schema": "smartsom.algorithm/v2",
        "mode": "central" if backend == "sb3" else "resource",
        "learner": {
            "backend": backend,
            "algorithm": "ppo",
            "parameters": {"batch_size": 8, "n_epochs": 1},
        },
    }
    if backend == "sb3":
        algorithm["controller"] = policy
    else:
        algorithm["agents"] = {
            role: {"default": policy} for role in ("machine", "buffer", "dispatcher")
        }
        algorithm["agents"]["mover"] = {
            "default": {"kind": "rule", "name": "shortest_path"}
        }
    experiment = {
        "schema": "smartsom.experiment-config/v4",
        "task": "train-evaluate",
        "factory": "factory.yaml",
        "workload": "workload.yaml",
        "algorithm": "algorithm.yaml",
        "seed": 101,
        "data_seed": 709,
        "execution": {"executor": "native", "tuning": "off"},
        "training": {
            "total_ticks": 16,
            "ticks_per_update": 8,
            "max_ticks": 8,
        },
        "runtime": {
            "device": "cpu",
            "num_envs": 1,
            "numerical_threads": 1,
            "sampling_processes": 0,
            "environment": {"mode": "finite", "tick_limit": 8},
        },
        "validation": {
            "enabled": True,
            "every_updates": 1,
            "replications": 1,
            "full_replay": False,
        },
        "evaluation": {
            "checkpoint": "last",
            "replications": 1,
            "record": False,
            "full_replay": False,
        },
        "logging": {
            "verbose": False,
            "progress": "off",
            "tensorboard": False,
            "wandb": False,
        },
        "output": {
            "root": str(directory.parent / "outputs"),
            "name": f"engineering-only-{backend}",
        },
    }
    for name, document in (
        ("workload", workload),
        ("algorithm", algorithm),
        ("experiment", experiment),
    ):
        _write(directory / f"{name}.yaml", document)
    return directory / "experiment.yaml"


def _ledger(root):
    return json.loads((root / "entries/entry-0001/stages.json").read_text())


def _record(path):
    return json.loads((Path(path) / "run.json").read_text())


def _child_files(root):
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in (root / "entries").glob("*/runs/*/run.json")
    }


def test_directory_probe_is_short_and_derived_from_frozen_v4_case(tmp_path):
    path = _inputs(tmp_path / "inputs", "sb3")
    original = compile_experiment(path).entries[0].prepared
    from dataclasses import asdict

    shortened = _short_probe_prepared(asdict(original), tick_limit=4)
    config = json.loads(shortened["config_json"])
    cases = json.loads(shortened["validation_json"])
    assert config["training"]["total_ticks"] == 4
    assert len(cases) == 1 and cases[0]["scenario"]["tick_limit"] == 4
    assert (
        cases[0]["scenario"]["demands"]
        == json.loads(original.validation_json)[0]["scenario"]["demands"]
    )
    assert shortened["scientific_sha256"] != original.scientific_sha256


def test_train_only_v4_checkpoint_evaluates_frozen_held_out_case(tmp_path):
    from smartsom.experiments.cli import main

    _require_cpu()
    source = _inputs(tmp_path / "inputs", "sb3")
    experiment = yaml.safe_load(source.read_text())
    experiment["task"] = "train"
    _write(source, experiment)
    plan = compile_experiment(source, require_dependencies=True)
    assert len(json.loads(plan.entries[0].prepared.evaluation_json)) == 1
    assert plan.summary()["entries"][0]["evaluation_cases"] == 0
    root = allocate(plan)
    result = execute_saved(root)
    assert result["status"] == "completed", result
    stages = _ledger(root)["stages"]
    assert set(stages) == {"training"}
    training_dir = Path(stages["training"]["run_dir"])
    later = tmp_path / "later"
    exit_code = main(
        [
            "evaluate",
            str(training_dir),
            "--checkpoint",
            "last",
            "--output-root",
            str(later),
        ]
    )
    runs = list(later.glob("*/run.json"))
    assert len(runs) == 1
    record = _record(runs[0].parent)
    assert record["status"] == "completed"
    assert len(record["results"]) == 1
    assert record["summary"]["requested"] == 1
    assert not record["summary"]["exceptions"]
    assert exit_code == (0 if record["results"][0]["status"] == "completed" else 1)


@pytest.mark.parametrize("backend", ["sb3", "rllib", "central-rllib"])
def test_true_short_learning_validation_evaluation_and_completed_stage_skip(
    tmp_path, backend
):
    _require_cpu()
    source = _inputs(
        tmp_path / "inputs", "rllib" if backend == "central-rllib" else backend
    )
    if backend == "central-rllib":
        algorithm_path = source.parent / "algorithm.yaml"
        algorithm = yaml.safe_load(algorithm_path.read_text())
        algorithm["mode"] = "central"
        algorithm["controller"] = algorithm.pop("agents")["machine"]["default"]
        _write(algorithm_path, algorithm)
    plan = compile_experiment(source, require_dependencies=True)
    other_seed = compile_experiment(source, seed=999)
    # Policy randomness does not change externally frozen demands or case roots.
    for original, other in zip(plan.entries, other_seed.entries, strict=True):
        assert original.prepared.scenario_json == other.prepared.scenario_json
        assert original.prepared.validation_json == other.prepared.validation_json
        assert original.prepared.evaluation_json == other.prepared.evaluation_json
    root = allocate(plan)
    result = execute_saved(root)
    assert result["status"] == "completed", result
    assert result["completed"] == 1 and result["failed"] == 0
    ledger = _ledger(root)
    assert ledger["status"] == "completed"
    assert set(ledger["stages"]) == {"training", "evaluation"}
    training = ledger["stages"]["training"]
    evaluation = ledger["stages"]["evaluation"]
    assert training["status"] == evaluation["status"] == "completed"
    trained = _record(training["run_dir"])
    assert trained["physical_ticks"] == 16 and trained["updates"] == 2
    assert sum(trained["actual_optimization_steps"].values()) > 0
    assert any(trained["changed_weights"].values())
    assert trained["frozen_partners_unchanged"] is True
    train_dir = Path(training["run_dir"])
    history = json.loads((train_dir / "reports/training.json").read_text())
    assert [row["physical_ticks"] for row in history] == [8, 16]
    validation = sorted((train_dir / "logs").glob("validation-*.json"))
    assert len(validation) == 2
    for path in validation:
        rows = json.loads(path.read_text())
        assert len(rows) == 1 and not rows[0].get("engineering_failure")
        assert rows[0]["physical_ticks"] <= 8
    evaluated = _record(evaluation["run_dir"])
    assert evaluated["kind"] == "evaluation"
    assert len(evaluated["results"]) == 1
    assert not evaluated["summary"]["exceptions"]
    assert len(evaluation["attempts"]) == 1
    before = _child_files(root)
    before_ledger = (root / "entries/entry-0001/stages.json").read_bytes()
    resumed = execute_saved(root)
    assert resumed["status"] == "completed"
    assert _child_files(root) == before
    assert (root / "entries/entry-0001/stages.json").read_bytes() == before_ledger


@pytest.mark.parametrize("executor", ["native", "online"])
def test_algorithm_seed_matrix_saves_independent_best_and_evaluates(tmp_path, executor):
    _require_cpu()
    source = _inputs(tmp_path / "inputs", "sb3")
    algorithm_path = source.parent / "algorithm.yaml"
    second_path = source.parent / "algorithm-second.yaml"
    second = yaml.safe_load(algorithm_path.read_text())
    second["learner"]["gamma"] = 0.95
    _write(second_path, second)
    experiment = yaml.safe_load(source.read_text())
    del experiment["factory"], experiment["workload"], experiment["algorithm"]
    experiment["matrix"] = {
        "factories": ["factory.yaml"],
        "workloads": ["workload.yaml"],
        "algorithms": ["algorithm.yaml", "algorithm-second.yaml"],
        "seeds": [101, 202],
    }
    experiment["validation"].update(
        best_mode="completion_delivery_return", replications=2
    )
    experiment["evaluation"].update(checkpoint="best", replications=2)
    experiment["execution"] = {"max_concurrent": 2}
    if executor == "native":
        experiment["execution"]["executor"] = "native"
    _write(source, experiment)
    plan = compile_experiment(source, require_dependencies=True)
    assert len(plan.entries) == 4
    assert all(
        entry.prepared.validation_json == plan.entries[0].prepared.validation_json
        for entry in plan.entries
    )
    result = execute_saved(allocate(plan))
    assert result["status"] == "completed" and result["completed"] == 4, result
    root = Path(result["run_directory"])
    state = json.loads((root / "batch.json").read_text())
    for entry in plan.entries:
        if executor == "native":
            ledger = json.loads(
                (root / "entries" / entry.id / "stages.json").read_text()
            )
            train_dir = Path(ledger["stages"]["training"]["run_dir"])
            evaluation = _record(ledger["stages"]["evaluation"]["run_dir"])["summary"]
        else:
            row = state["entries"][entry.id]
            assert len(row["attempts"]) == 1 and row["updates"] == 2
            train_dir = Path(row["attempts"][0]["run_dir"])
            evaluation = json.loads((train_dir / "evaluation/summary.json").read_text())
        assert (train_dir / "checkpoints/best.json").is_file()
        selections = sorted((train_dir / "logs").glob("selection-*.json"))
        assert selections and any(
            json.loads(path.read_text())["selected"] for path in selections
        )
        assert evaluation["requested"] == 2 and not evaluation["exceptions"]
    assert execute_saved(root)["completed"] == 4


def _barrier_process(root, directory, phase):
    script = directory / "engineering_driver.py"
    marker = directory / "phase-ready"
    script.write_text(
        """
import sys, time
from pathlib import Path
from smartsom import api
from smartsom.experiments.author_driver import execute_saved
from smartsom.experiments import composable
ROOT = Path(sys.argv[1])
MARKER = Path(sys.argv[2])
PHASE = sys.argv[3]
original_train = api.train_prepared
original_evaluate = composable.evaluate

def train_barrier(prepared, **kwargs):
    def updated(record):
        if record['physical_ticks'] == 8:
            MARKER.write_text('complete update')
            time.sleep(1.5)
    return original_train(prepared, on_progress=updated, **kwargs)

def evaluate_barrier(*args, **kwargs):
    MARKER.write_text('training committed; evaluation entered')
    time.sleep(1.5)
    return original_evaluate(*args, **kwargs)

if PHASE == 'training':
    api.train_prepared = train_barrier
else:
    composable.evaluate = evaluate_barrier

if __name__ == '__main__':
    result = execute_saved(ROOT)
    print(result, flush=True)
"""
    )
    return (
        subprocess.Popen(
            [sys.executable, str(script), str(root), str(marker), phase],
            cwd=ROOT,
            env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        ),
        marker,
    )


def _wait_marker(process, marker):
    deadline = time.monotonic() + 90
    while not marker.is_file():
        if process.poll() is not None:
            stdout, stderr = process.communicate(timeout=5)
            raise AssertionError(f"driver exited before safe phase: {stdout}\n{stderr}")
        if time.monotonic() >= deadline:
            raise AssertionError("driver did not reach its bounded engineering phase")
        time.sleep(0.1)


@pytest.mark.parametrize("phase", ["training", "evaluation"])
def test_stop_resume_restores_update_or_restarts_eval_without_retraining(
    tmp_path, phase
):
    _require_cpu()
    source = _inputs(tmp_path / "inputs", "sb3")
    root = allocate(compile_experiment(source, require_dependencies=True))
    process, marker = _barrier_process(root, tmp_path, phase)
    try:
        _wait_marker(process, marker)
        stopped = stop(root, timeout=30)
        assert stopped["status"] == "stopped", stopped
        stdout, stderr = process.communicate(timeout=10)
        assert process.returncode == 0, stdout + stderr
        assert json.loads((root / "batch.json").read_text())["status"] == "stopped"
        before = _ledger(root)
        training = before["stages"]["training"]
        training_dir = Path(training["run_dir"])
        record = _record(training_dir)
        if phase == "training":
            assert record["physical_ticks"] == 8
            assert (training_dir / "checkpoints/recovery.json").is_file()
        else:
            assert training["status"] == "completed"
            assert record["physical_ticks"] == 16
            assert before["stages"]["evaluation"]["status"] == "stopped"
        training_bytes = (training_dir / "run.json").read_bytes()
        resumed = execute_saved(root)
        assert resumed["status"] == "completed", resumed
        after = _ledger(root)
        assert after["stages"]["training"]["run_dir"] == str(training_dir)
        assert _record(training_dir)["physical_ticks"] == 16
        assert _record(training_dir)["updates"] == 2
        if phase == "evaluation":
            assert (training_dir / "run.json").read_bytes() == training_bytes
            assert len(after["stages"]["evaluation"]["attempts"]) > len(
                before["stages"]["evaluation"]["attempts"]
            )
        assert after["stages"]["evaluation"]["status"] == "completed"
        assert execute_saved(root)["status"] == "completed"
        assert _ledger(root) == after
    finally:
        if process.poll() is None:
            try:
                stop(root, timeout=10, force=True)
            finally:
                if process.poll() is None:
                    # Exact disposable Popen child only; verified worker cleanup uses stop.
                    process.kill()
        process.communicate(timeout=10)


def test_v4_tensorboard_uses_logging_option_and_resumed_cursor(tmp_path):
    _require_cpu()
    pytest.importorskip("tensorboard")
    import pickle

    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    from smartsom.experiments.composable import TrainingSession, prepared_from_run

    source = _inputs(tmp_path / "tensorboard-inputs", "sb3")
    experiment = yaml.safe_load(source.read_text())
    experiment["task"] = "train"
    experiment["logging"]["tensorboard"] = True
    _write(source, experiment)
    root = allocate(compile_experiment(source, require_dependencies=True))
    result = execute_saved(root)
    assert result["status"] == "completed"
    training = Path(_ledger(root)["stages"]["training"]["run_dir"])
    directory = training / "logs/tensorboard"
    accumulator = EventAccumulator(str(directory))
    accumulator.Reload()
    assert [row.step for row in accumulator.Scalars("training/updates")] == [8, 16]
    assert [row.step for row in accumulator.Scalars("validation/observed")] == [8, 16]
    assert [
        r["training_physical_ticks"]
        for r in json.loads((training / "reports/training-episodes.json").read_text())
    ] == [8, 16]
    with (training / "checkpoints/update-000001/continuation.pkl").open("rb") as stream:
        saved = pickle.load(stream)
    prepared = prepared_from_run(training)
    session = TrainingSession(prepared, training, _record(training))
    session.restore(saved)
    session.execute()
    accumulator.Reload()
    assert [row.step for row in accumulator.Scalars("training/updates")] == [8, 16]
    assert [row.step for row in accumulator.Scalars("validation/observed")] == [8, 16]
    assert session.telemetry.writer is None


def test_distributed_fixed_demands_public_batch_initial_dev_final_rule_and_frozen_resume(
    tmp_path, capsys, monkeypatch
):
    from smartsom.config.codec import digest
    from smartsom.experiments.author_batch import compile_directory
    from smartsom.experiments.author_batch import load as load_directory
    from smartsom.experiments.author_driver import load
    from smartsom.experiments.cli import main
    from smartsom.experiments.composable import prepared_from_run

    _require_cpu()
    source = _inputs(tmp_path / "inputs", "sb3")
    workload_path = source.parent / "workload.yaml"
    workload = yaml.safe_load(workload_path.read_text())
    workload.update(
        arrivals={
            "initial_jobs": 1,
            "interval_seconds": {"distribution": "exponential", "mean": 30},
        },
        due={"nominal_work_factor": {"distribution": "uniform", "low": 1, "high": 3}},
    )
    _write(workload_path, workload)
    experiment = yaml.safe_load(source.read_text())
    experiment["evaluation"]["initial_control"] = True
    experiment["execution"]["background"] = False
    experiment.update(
        factory="../factory.yaml",
        workload="../workload.yaml",
        algorithm="../algorithm.yaml",
    )
    folder = source.parent / "batch"
    folder.mkdir()
    _write(folder / "learn.yaml", experiment)
    rules = {
        "schema": "smartsom.algorithm/v2",
        "mode": "rules",
        "agents": {
            role: {"default": {"kind": "rule", "name": name}}
            for role, name in {
                "machine": "spt",
                "buffer": "edd",
                "dispatcher": "nearest",
                "mover": "shortest_path",
            }.items()
        },
    }
    _write(source.parent / "rules.yaml", rules)
    rule_experiment = dict(experiment, task="evaluate", algorithm="../rules.yaml")
    rule_experiment.pop("training")
    rule_experiment["validation"] = {"enabled": False}
    rule_experiment["evaluation"] = {
        "replications": 1,
        "full_replay": False,
        "record": False,
    }
    _write(folder / "rule.yaml", rule_experiment)
    checked = compile_directory(folder, calibration_level="off")
    plans = {row["id"]: row["compiled"] for row in checked.files}
    prepared = plans["learn"].entries[0].prepared
    other_policy = (
        compile_experiment(folder / "learn.yaml", seed=999).entries[0].prepared
    )
    assert prepared.scenario_json == other_policy.scenario_json
    assert prepared.validation_json == other_policy.validation_json
    assert prepared.evaluation_json == other_policy.evaluation_json
    final_cases = json.loads(prepared.evaluation_json)
    rule_cases = json.loads(plans["rule"].entries[0].prepared.evaluation_json)
    assert [c["scenario"]["demands"] for c in final_cases] == [
        c["scenario"]["demands"] for c in rule_cases
    ]
    assert any(d["release_at"] > 8 for d in final_cases[0]["scenario"]["demands"])
    assert main(["check", str(folder)]) == 0
    capsys.readouterr()
    assert (
        main(
            ["batch-run", str(folder), "--no-background", "--calibration-level", "off"]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    root = Path(result["directory"])
    assert result["status"] == "completed"
    learn_root = root / "experiments/learn"
    state = json.loads((root / "batch.json").read_text())
    tune_root = root / state["tune_directories"]["0"]
    tuned = json.loads((tune_root / "batch.json").read_text())["entries"][
        "learn__entry-0001"
    ]
    assert tuned["status"] == "completed" and len(tuned["attempts"]) == 1
    train_dir = Path(tuned["attempts"][0]["run_dir"])
    train = _record(train_dir)
    assert train["physical_ticks"] == 16 and train["updates"] == 2
    assert sum(train["actual_optimization_steps"].values()) > 0
    dev_logs = sorted((train_dir / "logs").glob("validation-*.json"))
    assert len(dev_logs) == 2
    for path in dev_logs:
        assert not json.loads(path.read_text())[0].get("engineering_failure")
    initial = json.loads((train_dir / "evaluation/controls/initial.json").read_text())
    final = json.loads((train_dir / "evaluation/tuning-final.json").read_text())
    rule_ledger = _ledger(root / "experiments/rule")["stages"]
    rule = _record(rule_ledger["evaluation"]["run_dir"])
    assert rule["status"] == "completed" and not rule["summary"]["exceptions"]
    for results in (initial, final, rule["results"]):
        assert len(results) == 1 and not results[0]["engineering_failure"]
        assert results[0]["world_sha256"] == initial[0]["world_sha256"]
    snapshot = learn_root / load(learn_root)[1]["entries"][0]["snapshot"]
    restored_before = prepared_from_run(snapshot)
    assert json.loads(restored_before.evaluation_json) == final_cases
    # Saved continuation reads detached tables, even if the author source changes.
    workload_path.write_text("invalid edited source\n")

    def reject_redraw(*_args, **_kwargs):
        raise AssertionError("timing redrawn on restore")

    monkeypatch.setattr(
        "smartsom.workloads.workload_v3.apply_workload_timing", reject_redraw
    )
    load_directory(root)
    assert main(["resume", str(root), "--no-background"]) == 0
    resumed = json.loads(capsys.readouterr().out)
    assert resumed["status"] == "completed"
    restored = prepared_from_run(snapshot)
    assert digest(json.loads(restored.evaluation_json)) == digest(final_cases)
    assert restored.training_inputs_json == restored_before.training_inputs_json
    from smartsom.experiments.composable import evaluate

    reevaluated = evaluate(
        source=train_dir, selection="last", output_root=tmp_path / "restored-eval"
    )
    record = _record(reevaluated.run_dir)
    assert record["status"] == "completed" and not record["summary"]["exceptions"]
    assert record["results"][0]["world_sha256"] == initial[0]["world_sha256"]
