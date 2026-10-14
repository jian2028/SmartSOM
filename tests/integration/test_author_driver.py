"""Disposable real native rule plans and their stage ledger boundaries."""

import copy
import json
from pathlib import Path

import pytest
import yaml

from smartsom.config.codec import canonical_json
from smartsom.config.experiment_v4 import compile_experiment
from smartsom.experiments.author_batch import (
    allocate as allocate_directory,
)
from smartsom.experiments.author_batch import (
    compile_directory,
)
from smartsom.experiments.author_batch import (
    execute_saved as execute_directory,
)
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


def test_directory_batch_checks_freezes_runs_and_resumes_two_v4_files(
    author_input, monkeypatch
):
    path, documents = author_input
    folder = path.parent / "experiments"
    folder.mkdir()
    original = copy.deepcopy(documents["experiment"])
    for name, seed, stage in (("a", 9001, 10), ("b", 9002, 20)):
        document = copy.deepcopy(original)
        document.update(
            factory="../factory.yaml",
            workload="../workload.yaml",
            algorithm="../algorithm.yaml",
            data_seed=seed,
            batch={
                "stage": stage,
                "parallel_files": 1,
                **({"gate": {"min_cases": 1}} if name == "a" else {}),
            },
        )
        (folder / f"{name}.yaml").write_text(yaml.safe_dump(document))
    checked = compile_directory(folder)
    assert checked.calibration_level == "quick"
    assert checked.calibration_seconds == 300
    assert (
        compile_directory(folder, calibration_level="full").calibration_seconds == 1800
    )
    assert (
        compile_directory(
            folder, calibration_level="full", calibration_seconds=17
        ).calibration_seconds
        == 17
    )
    assert [row["id"] for row in checked.files] == ["a", "b"]
    assert not Path(original["output"]["root"]).exists()
    root = allocate_directory(checked)
    saved = read(root / "plan.json")
    assert len(saved["files"]) == 2
    assert (
        saved["files"][0]["scientific_sha256"] != saved["files"][1]["scientific_sha256"]
    )
    result = execute_directory(root, display_options=DISPLAY)
    assert result["status"] == "completed"
    smoke = read(root / "smoke.json")
    assert smoke["status"] == "completed"
    assert len(smoke["entries"]) == 2
    assert all(row["status"] == "passed" for row in smoke["entries"].values())
    assert all(row["status"] == "completed" for row in result["files"].values())
    assert execute_directory(root, display_options=DISPLAY)["status"] == "completed"
    from smartsom.telemetry.monitor import read_snapshot

    snapshot = read_snapshot(root)
    assert snapshot["status"] == "completed"
    assert {row["id"] for row in snapshot["tasks"]} == {"a", "b"}
    from smartsom.experiments import control

    for name in ("a", "b"):
        child = root / "experiments" / name
        assert control.read(child)["driver_root"] == str(root)
        with pytest.raises(ValueError, match="child of a shared driver"):
            control.stop(child)
    from smartsom.experiments import author_batch

    def reject_gate(*_args):
        raise RuntimeError("gate rechecked on completed resume")

    monkeypatch.setattr(author_batch, "_gate", reject_gate)
    assert execute_directory(root, display_options=DISPLAY)["status"] == "failed"


def test_directory_batch_rejects_duplicate_science_and_gate_stops_next_stage(
    author_input,
):
    path, documents = author_input
    folder = path.parent / "experiments"
    folder.mkdir()
    first = copy.deepcopy(documents["experiment"])
    first.update(
        factory="../factory.yaml",
        workload="../workload.yaml",
        algorithm="../algorithm.yaml",
        batch={"stage": 10, "gate": {"min_cases": 1, "min_deliveries_each": 9999}},
    )
    second = copy.deepcopy(first)
    (folder / "a.yaml").write_text(yaml.safe_dump(first))
    (folder / "b.yaml").write_text(yaml.safe_dump(second))
    with pytest.raises(ValueError, match="duplicate scientific entry"):
        compile_directory(folder)
    second["data_seed"] = 9002
    second["batch"] = {"stage": 20}
    (folder / "b.yaml").write_text(yaml.safe_dump(second))
    root = allocate_directory(compile_directory(folder))
    result = execute_directory(root, display_options=DISPLAY)
    assert result["status"] == "failed"
    assert result["files"]["a"]["status"] == "completed"
    assert result["files"]["b"]["status"] == "queued"
    assert not (root / "experiments/b/entries").exists()


def test_directory_batch_joins_started_child_if_next_launch_fails(
    tmp_path, monkeypatch
):
    import multiprocessing
    import time

    from test_runtime_scheduler import _counter_worker

    from smartsom.experiments import author_batch, control, queue_admission

    rows = [
        {"id": name, "child": f"experiments/{name}", "task": "evaluate"}
        for name in ("a", "b")
    ]
    for row in rows:
        (tmp_path / row["child"]).mkdir(parents=True)
    scope = control.Scope()
    scope.bind(tmp_path)
    queue_admission.initialize(tmp_path, supported=True)
    real_context = multiprocessing.get_context("spawn")
    started = []
    joined = []
    attempts = []

    class FailedLaunch:
        def start(self):
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                if (tmp_path / "experiments/a/counter.json").exists():
                    raise OSError("cannot start second child")
                time.sleep(0.02)
            pytest.fail("first real worker did not persist progress")

    class StartedProcess:
        def __init__(self, process):
            self.process = process

        @property
        def pid(self):
            return self.process.pid

        @property
        def exitcode(self):
            return self.process.exitcode

        def start(self):
            self.process.start()

        def is_alive(self):
            return self.process.is_alive()

        def join(self, *args, **kwargs):
            self.process.join(*args, **kwargs)
            joined.append(self.pid)

    class Context:
        def Process(self, **kwargs):
            attempts.append(kwargs["args"][0])
            if started:
                return FailedLaunch()
            process = StartedProcess(real_context.Process(**kwargs))
            started.append(process)
            return process

    monkeypatch.setattr(
        author_batch.multiprocessing, "get_context", lambda _: Context()
    )
    monkeypatch.setattr(author_batch, "_quiet_worker", _counter_worker)
    monkeypatch.setattr(author_batch, "_save", lambda *args: None)
    monkeypatch.setattr(author_batch, "_publish", lambda *args: None)
    state = {"files": {name: {"status": "queued"} for name in ("a", "b")}}
    try:
        with pytest.raises(OSError, match="second child"):
            author_batch._run_children(
                tmp_path, {"files": rows}, state, rows, parallel_files=2
            )
        assert len(started) == 1
        assert len(attempts) == 2  # cleanup must never launch another task
        assert not started[0].is_alive()
        assert started[0].exitcode == 0
        assert started[0].pid in joined
        assert read(tmp_path / "experiments/a/batch.json")["status"] == "stopped"
        assert read(tmp_path / "experiments/a/counter.json")["counter"] > 0
        assert queue_admission.snapshot(tmp_path)["slots"] == {}
    finally:
        author_batch._request_owned_stop(tmp_path)
        for process in started:
            if process.pid is not None:
                process.join(timeout=10)
        scope.finish("stopped")


def test_directory_batch_parent_hash_uses_compiled_input(author_input):
    path, documents = author_input
    folder = path.parent / "experiments"
    folder.mkdir()
    experiment = copy.deepcopy(documents["experiment"])
    experiment.update(
        factory="../factory.yaml",
        workload="../workload.yaml",
        algorithm="../algorithm.yaml",
    )
    source = folder / "one.yaml"
    source.write_text(yaml.safe_dump(experiment))
    checked = compile_directory(folder)
    original = json.loads(
        checked.files[0]["compiled"].entries[0].prepared.training_inputs_json
    )["authoring"]["input_sha256"][str(source)]
    source.write_text(source.read_text() + "\n# edited after compilation\n")
    root = allocate_directory(checked)
    assert read(root / "plan.json")["files"][0]["source_sha256"] == original


def test_directory_check_rejects_per_experiment_overrides(tmp_path):
    from smartsom.experiments.cli import main

    assert main(["check", str(tmp_path), "--algorithm", "other.yaml"]) == 2
    assert main(["check", str(tmp_path), "--set", "algorithm.x=1"]) == 2


def test_real_matrix_runs_true_cartesian_entries_with_two_workers(author_input):
    path, documents = author_input
    directory = path.parent
    factory = yaml.safe_load((directory / "factory.yaml").read_text())
    factory["factory"]["factory_id"] = "second-factory"
    (directory / "factory-second.yaml").write_text(yaml.safe_dump(factory))
    workload = copy.deepcopy(documents["workload"])
    workload["demands"][0]["due_at"] = 40
    (directory / "workload-second.yaml").write_text(yaml.safe_dump(workload))
    algorithm = copy.deepcopy(documents["algorithm"])
    algorithm["pickup_matching"] = "priority_greedy"
    (directory / "algorithm-second.yaml").write_text(yaml.safe_dump(algorithm))
    experiment = copy.deepcopy(documents["experiment"])
    del experiment["factory"], experiment["workload"], experiment["algorithm"]
    experiment["matrix"] = {
        "factories": ["factory.yaml", "factory-second.yaml"],
        "workloads": ["workload.yaml", "workload-second.yaml"],
        "algorithms": ["algorithm.yaml", "algorithm-second.yaml"],
    }
    experiment["execution"] = {"max_concurrent": 2}
    path.write_text(yaml.safe_dump(experiment))
    result = run(compile_experiment(path), display_options=DISPLAY)
    root = Path(result["run_directory"])
    assert result["status"] == "completed" and result["completed"] == 8
    assert result["failed"] == result["pending"] == 0
    assert len(children(root)) == 8
    saved_plan = read(root / "plan.json")
    assert len(saved_plan["entries"]) == 8
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
    assert len(progress["tasks"]) == 8
    assert all(row["status"] == "completed" for row in progress["tasks"])
    assert all(row["unit"] == "evaluation episodes" for row in progress["tasks"])
    assert all(row["total"] == 1 for row in progress["tasks"])
    assert {"algorithm", "algorithm-second"} == {
        row["name"].split(" · ")[0] for row in progress["tasks"]
    }
    assert execute_saved(root, display_options=DISPLAY)["completed"] == 8


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


def test_public_batch_run_resume_and_readonly_monitor_reuse_three_of_five(
    author_input, tmp_path
):
    """Fresh CLI processes, real native workers; the test hook faults after commit."""
    import os
    import subprocess
    import sys

    path, documents = author_input
    document = copy.deepcopy(documents["experiment"])
    document["evaluation"]["replications"] = 5
    document.update(
        factory="../factory.yaml",
        workload="../workload.yaml",
        algorithm="../algorithm.yaml",
    )
    inputs = path.parent / "experiments"
    inputs.mkdir()
    (inputs / "a.yaml").write_text(yaml.safe_dump(document))
    hooks = tmp_path / "fault-hook"
    hooks.mkdir()
    (hooks / "sitecustomize.py").write_text(
        "import os\n"
        "if os.environ.get('SMARTSOM_TEST_CASE_INTERRUPT') == 'yes':\n"
        " from smartsom.experiments.evaluation_resume import CaseStore\n"
        " original = CaseStore.commit\n"
        " def commit(self, index, row):\n"
        "  original(self, index, row)\n"
        "  if len(self.cases) == 5 and index == 2:\n"
        "   raise KeyboardInterrupt('test fault after third durable receipt')\n"
        " CaseStore.commit = commit\n"
    )
    source = Path(__file__).resolve().parents[2]
    env = dict(
        os.environ,
        PYTHONPATH=os.pathsep.join([str(hooks), str(source / "src")]),
        PYTHONDONTWRITEBYTECODE="1",
        SMARTSOM_TEST_CASE_INTERRUPT="yes",
    )
    cli = [sys.executable, "-m", "smartsom.experiments.cli"]
    display = ["--progress", "off", "--no-verbose", "--log-format", "json"]
    first = subprocess.run(
        cli
        + [
            "batch-run",
            str(inputs),
            "--calibration-level",
            "off",
            "--no-background",
            *display,
        ],
        cwd=source,
        env=env,
        capture_output=True,
        text=True,
        timeout=45,
    )
    parent = next(Path(document["output"]["root"]).glob("*/plan.json")).parent
    ledger = next(parent.glob("experiments/*/entries/*/stages.json"))
    stage = read(ledger)["stages"]["evaluation"]
    assert stage["status"] == "stopped", first.stderr + first.stdout
    evaluation = ledger.parent / "runs/final-evaluation"
    receipts = evaluation / "evidence/.case-resume"
    before = {p.name: p.read_bytes() for p in receipts.glob("case-*.json")}
    assert len(before) == 3
    env.pop("SMARTSOM_TEST_CASE_INTERRUPT")
    resumed = subprocess.run(
        cli + ["resume", str(parent), "--retry-failed", *display],
        cwd=source,
        env=env,
        capture_output=True,
        text=True,
        timeout=45,
    )
    assert resumed.returncode == 0, resumed.stderr + resumed.stdout
    assert read(parent / "batch.json")["status"] == "completed"
    assert all(
        (receipts / name).read_bytes() == content for name, content in before.items()
    )
    assert len(list(receipts.glob("case-*.json"))) == 5
    saved = read(ledger.parent / "logs/progress.json")
    values = next(r for r in saved["tasks"] if r["id"] == "evaluation")["values"]
    assert (
        values["evaluation_historical_completed"],
        values["evaluation_current_completed"],
        values["evaluation_remaining"],
        values["evaluation_target"],
    ) == (3, 2, 0, 5)
    snapshot_before = {p: p.read_bytes() for p in parent.rglob("*") if p.is_file()}
    monitor = subprocess.run(
        cli + ["monitor", str(parent), "--once", "--log-format", "json"],
        cwd=source,
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert monitor.returncode == 0, monitor.stderr
    assert '"evaluation_historical_completed": 3' in monitor.stdout + monitor.stderr
    assert '"evaluation_current_completed": 2' in monitor.stdout + monitor.stderr
    assert all(p.read_bytes() == content for p, content in snapshot_before.items())
    assert set(p for p in parent.rglob("*") if p.is_file()) == set(snapshot_before)


@pytest.mark.parametrize("appended", (False, True))
def test_runtime_default_native_worker_pause_resume_uses_one_slot(
    author_input, tmp_path, appended
):
    """Exercise real _quiet_worker -> author driver -> native entry, at limit one."""
    import time

    from smartsom.experiments import control, queue_admission, runtime_control
    from smartsom.experiments.runtime_workers import NativeScheduler

    path, _ = author_input
    parent = tmp_path / "runtime-parent"
    parent.mkdir()
    child = allocate(compile_experiment(path), root=parent / "child")
    (parent / "plan.json").write_text(
        json.dumps(
            {
                "schema": "smartsom.author-batch-plan/v1",
                "files": [] if appended else [{"id": "one", "child": "child"}],
            }
        )
    )
    scope = control.Scope()
    scope.bind(parent)
    runtime_control.initialize(parent, 1)
    queue_admission.initialize(parent, supported=True)
    jobs = {"one": {"root": str(child), "status": "queued"}}
    if appended:
        with runtime_control.transaction(parent) as state:
            state["additions"].update(jobs)
    scheduler = NativeScheduler(parent, {} if appended else jobs)

    def wait_for(predicate):
        deadline = time.monotonic() + 45
        while time.monotonic() < deadline:
            observed = scheduler.tick()
            if predicate(observed):
                return observed
            time.sleep(0.01)
        raise AssertionError(json.dumps(runtime_control.snapshot(parent)))

    try:
        wait_for(lambda _: control.read(child) is not None)
        runtime_control.submit(parent, "pause", task_id="one")
        wait_for(lambda _: scheduler.jobs["one"]["status"] == "paused")
        assert not runtime_control.snapshot(parent)["stopped_tasks"]
        assert not queue_admission.snapshot(parent)["slots"]
        runtime_control.submit(parent, "resume", task_id="one")
        wait_for(lambda _: scheduler.jobs["one"]["status"] in {"completed", "failed"})
        assert scheduler.jobs["one"]["status"] == "completed"
        assert read(child / "batch.json")["status"] == "completed"
        assert children(child)
    finally:
        runtime_control.submit(parent, "stop")
        wait_for(lambda _: not scheduler.processes)
        scope.finish("stopped")


def test_runtime_native_matrix_holds_and_removes_individual_entries(
    author_input, tmp_path
):
    import time
    from dataclasses import replace

    from smartsom.experiments import control, queue_admission, runtime_control
    from smartsom.experiments.runtime_workers import NativeScheduler

    path, _ = author_input
    parent = tmp_path / "matrix-parent"
    parent.mkdir()
    compiled = compile_experiment(path)
    first = replace(compiled.entries[0], id="first")
    second = replace(compiled.entries[0], id="second")
    child = allocate(replace(compiled, entries=(first, second)), root=parent / "child")
    (parent / "plan.json").write_text(
        json.dumps(
            {
                "schema": "smartsom.author-batch-plan/v1",
                "files": [{"id": "matrix", "child": "child"}],
            }
        )
    )
    scope = control.Scope()
    scope.bind(parent)
    runtime_control.initialize(parent, 1)
    queue_admission.initialize(parent, supported=True)
    with runtime_control.transaction(parent) as state:
        state["tasks"].update(
            {
                "matrix__first": {"status": "queued"},
                "matrix__second": {"status": "queued"},
            }
        )
    runtime_control.submit(parent, "pause", task_id="matrix__first")
    runtime_control.submit(parent, "pause", task_id="matrix__second")
    scheduler = NativeScheduler(
        parent, {"matrix": {"root": str(child), "status": "queued", "own_slots": True}}
    )

    def wait_for(predicate):
        deadline = time.monotonic() + 45
        while time.monotonic() < deadline:
            scheduler.tick()
            if predicate():
                return
            time.sleep(0.02)
        raise AssertionError(json.dumps(runtime_control.snapshot(parent)))

    try:
        wait_for(
            lambda: (
                (child / "batch.json").exists()
                and read(child / "batch.json")["status"] == "running"
            )
        )
        runtime_control.submit(parent, "resume", task_id="matrix__second")
        wait_for(
            lambda: (
                read(child / "batch.json")["entries"]["second"]["status"] == "completed"
            )
        )
        assert scheduler.processes["matrix"].is_alive()
        assert read(child / "batch.json")["entries"]["first"]["status"] != "completed"
        target = runtime_control.removal_target(parent, "matrix__first")
        runtime_control.submit(
            parent,
            "remove",
            task_id="matrix__first",
            confirmation=target["confirmation"],
            expected_revision=target["revision"],
        )
        wait_for(lambda: not scheduler.processes)
        assert scheduler.jobs["matrix"]["status"] == "partial_removed"
        assert read(child / "batch.json")["status"] == "partial_removed"
        assert read(child / "batch.json")["entries"]["first"]["status"] == "removed"
    finally:
        runtime_control.submit(parent, "stop")
        wait_for(lambda: not scheduler.processes)
        scope.finish("stopped")


@pytest.mark.parametrize(
    "second,collision", [("foo__entry-0001", True), ("foo__bar", False)]
)
def test_directory_authored_ids_do_not_alias_generated_entries(
    author_input, second, collision
):
    from smartsom.config.codec import ConfigurationError

    path, documents = author_input
    inputs = path.parent / "namespace"
    inputs.mkdir()
    for index, name in enumerate(("foo", second)):
        document = copy.deepcopy(documents["experiment"])
        document.update(
            factory="../factory.yaml",
            workload="../workload.yaml",
            algorithm="../algorithm.yaml",
            data_seed=100 + index,
        )
        (inputs / f"{name}.yaml").write_text(yaml.safe_dump(document))
    if collision:
        with pytest.raises(ConfigurationError, match="collides"):
            compile_directory(inputs)
    else:
        assert len(compile_directory(inputs).files) == 2


@pytest.mark.parametrize("standalone_tune", [False, True])
def test_append_rejects_frozen_child_id_before_first_publish(
    author_input, tmp_path, standalone_tune
):
    from smartsom.experiments import runtime_control as runtime
    from smartsom.experiments.evidence import write_json

    path, documents = author_input
    inputs = path.parent / "appended"
    inputs.mkdir()
    document = copy.deepcopy(documents["experiment"])
    document.update(
        factory="../factory.yaml",
        workload="../workload.yaml",
        algorithm="../algorithm.yaml",
    )
    (inputs / "foo__entry-0001.yaml").write_text(yaml.safe_dump(document))
    root = tmp_path / "parent"
    runtime.initialize(root, 2)
    write_json(
        root / "plan.json",
        {"entries": [{"experiment_id": "foo__entry-0001"}]}
        if standalone_tune
        else {"files": [{"id": "foo", "entry_ids": ["entry-0001"]}]},
    )
    with pytest.raises(ValueError, match="duplicate task ID"):
        runtime.append(root, inputs)
    assert not (root / "added").exists()
