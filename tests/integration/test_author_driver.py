"""Disposable real native rule plans and their stage ledger boundaries."""

import copy
import json
from pathlib import Path

import pytest
import yaml

from smartsom.config.codec import canonical_json
from smartsom.config.experiment_v4 import compile_experiment
from smartsom.experiments.author_driver import allocate, execute_saved, load, run

DISPLAY = {"progress": "off", "verbose": False, "format": "json"}


@pytest.fixture
def author_input(tmp_path):
    directory = tmp_path / "author"
    directory.mkdir()
    factory = directory / "factory.yaml"
    source = (
        Path(__file__).resolve().parents[2] / "configs/test/factories/factory_hand.yaml"
    )
    # The older hand fixture has no inspection station. The composable physical
    # protocol requires an explicit inspection destination even with zero defects.
    factory_data = yaml.safe_load(source.read_text())
    factory_data["factory"]["grid"].update(width=5, height=4)
    factory_data["factory"]["inspection_stations"] = [
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
    factory_data["factory"]["ports"].append(
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
    factory.write_text(yaml.safe_dump(factory_data))
    documents = {
        "workload": {
            "schema": "smartsom.workload/v3",
            "demands": [
                {
                    "demand_id": "job-1",
                    "steps": [
                        {
                            "operation_id": "op-1",
                            "operation_type": "operation_1",
                            "nominal_ticks": 2,
                        }
                    ],
                    "release_at": 0,
                    "due_at": 30,
                }
            ],
        },
        "algorithm": {
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
        },
        "experiment": {
            "schema": "smartsom.experiment-config/v4",
            "task": "evaluate",
            "factory": "factory.yaml",
            "workload": "workload.yaml",
            "algorithm": "algorithm.yaml",
            "data_seed": 9001,
            "runtime": {"environment": {"mode": "finite", "tick_limit": 64}},
            "evaluation": {
                "replications": 1,
                "full_replay": False,
                "verbose": False,
                "record": False,
            },
            "logging": {"progress": "off", "verbose": False, "format": "json"},
            "output": {"root": str(tmp_path / "results"), "name": "disposable-rules"},
        },
    }
    for name, data in documents.items():
        (directory / f"{name}.yaml").write_text(yaml.safe_dump(data, sort_keys=False))
    return directory / "experiment.yaml", documents


def read(path):
    return json.loads(path.read_text())


def children(root):
    return tuple(sorted((root / "entries").glob("*/runs/*/run.json")))


def test_real_single_rule_plan_records_only_evaluation_and_completed_resume_is_idempotent(
    author_input,
):
    path, _ = author_input
    result = run(compile_experiment(path), display_options=DISPLAY)
    root = Path(result["run_directory"])
    assert result["status"] == "completed" and result["completed"] == 1
    assert result["failed"] == result["pending"] == 0
    ledger_path = root / "entries/entry-0001/stages.json"
    ledger = read(ledger_path)
    assert ledger["status"] == "completed"
    assert set(ledger["stages"]) == {"evaluation"}
    stage = ledger["stages"]["evaluation"]
    assert stage["status"] == "completed" and len(stage["attempts"]) == 1
    evaluation = Path(stage["run_dir"])
    record = read(evaluation / "run.json")
    assert record["kind"] == "evaluation" and record["status"] == "completed"
    assert record["summary"]["completed"] == 1
    assert len(children(root)) == 1
    before = {p: p.read_bytes() for p in (ledger_path, *children(root))}
    resumed = execute_saved(root, display_options=DISPLAY)
    assert resumed["status"] == "completed" and resumed["completed"] == 1
    assert children(root) == tuple(p for p in before if p.name == "run.json")
    assert {p: p.read_bytes() for p in before} == before
    assert read(root / "run.json")["status"] == "completed"
    from smartsom.experiments.batch import exclusive_lock

    with exclusive_lock(root / "driver.lock"):
        pass


def test_real_matrix_runs_true_cartesian_entries_with_two_workers(author_input):
    path, documents = author_input
    directory = path.parent
    factory = yaml.safe_load((directory / "factory.yaml").read_text())
    factory["factory"]["factory_id"] = "second-factory"
    (directory / "factory-second.yaml").write_text(yaml.safe_dump(factory))
    workload = copy.deepcopy(documents["workload"])
    workload["demands"][0]["due_at"] = 40
    (directory / "workload-second.yaml").write_text(yaml.safe_dump(workload))
    experiment = copy.deepcopy(documents["experiment"])
    del experiment["factory"], experiment["workload"]
    experiment["matrix"] = {
        "factories": ["factory.yaml", "factory-second.yaml"],
        "workloads": ["workload.yaml", "workload-second.yaml"],
    }
    experiment["execution"] = {"max_concurrent": 2}
    path.write_text(yaml.safe_dump(experiment))
    result = run(compile_experiment(path), display_options=DISPLAY)
    root = Path(result["run_directory"])
    assert result["status"] == "completed" and result["completed"] == 4
    assert result["failed"] == result["pending"] == 0
    assert len(children(root)) == 4
    saved_plan = read(root / "plan.json")
    assert len(saved_plan["entries"]) == 4
    assert all(entry["task"] == "evaluate" for entry in saved_plan["entries"])
    for entry in saved_plan["entries"]:
        ledger = read(root / "entries" / entry["id"] / "stages.json")
        assert set(ledger["stages"]) == {"evaluation"}
        assert len(ledger["stages"]["evaluation"]["attempts"]) == 1
        assert (
            read(Path(ledger["stages"]["evaluation"]["run_dir"]) / "run.json")[
                "summary"
            ]["completed"]
            == 1
        )
    progress = read(root / "logs/progress.json")
    assert progress["status"] == "completed"
    assert len(progress["tasks"]) == 4
    assert all(row["status"] == "completed" for row in progress["tasks"])
    assert all(row["unit"] == "evaluation episodes" for row in progress["tasks"])
    assert all(row["total"] == 1 for row in progress["tasks"])


def test_frozen_prepared_input_drift_is_rejected_before_any_execution(author_input):
    path, _ = author_input
    root = allocate(compile_experiment(path))
    prepared_path = root / "inputs/entry-0001/config/prepared.json"
    prepared = read(prepared_path)
    config = json.loads(prepared["config_json"])
    config["seed"] += 1
    prepared["config_json"] = canonical_json(config)
    prepared_path.write_text(json.dumps(prepared))
    with pytest.raises(ValueError, match="frozen author input changed"):
        load(root)
    assert not (root / "entries").exists()
    assert read(root / "batch.json")["status"] == "prepared"


def test_frozen_plan_drift_is_rejected_before_any_execution(author_input):
    path, _ = author_input
    root = allocate(compile_experiment(path))
    plan_path = root / "plan.json"
    saved = read(plan_path)
    saved["entries"][0]["task"] = "train"
    plan_path.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="frozen author plan changed"):
        load(root)
    assert not (root / "entries").exists()


def test_execution_uses_frozen_inputs_after_original_author_files_are_removed(
    author_input,
):
    path, _ = author_input
    root = allocate(compile_experiment(path))
    for original in path.parent.glob("*.yaml"):
        original.unlink()
    result = execute_saved(root, display_options=DISPLAY)
    assert result["status"] == "completed" and result["completed"] == 1
    assert len(children(root)) == 1
    assert read(children(root)[0])["summary"]["completed"] == 1


def test_completed_evaluation_stage_is_not_repeated_when_outer_entry_was_interrupted(
    author_input,
):
    path, _ = author_input
    result = run(compile_experiment(path), display_options=DISPLAY)
    root = Path(result["run_directory"])
    ledger_path = root / "entries/entry-0001/stages.json"
    ledger = read(ledger_path)
    completed_stage = copy.deepcopy(ledger["stages"]["evaluation"])
    ledger["status"] = "stopped"
    ledger_path.write_text(json.dumps(ledger))
    before = children(root)
    resumed = execute_saved(root, display_options=DISPLAY)
    assert resumed["status"] == "completed" and children(root) == before
    assert read(ledger_path)["stages"]["evaluation"] == completed_stage
