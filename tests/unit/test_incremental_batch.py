"""Short inventory/selection lifecycle fixtures; no scientific training or ML imports."""

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from smartsom.experiments import author_batch, background, reservation, runtime_control
from smartsom.experiments.evidence import write_json
from smartsom.experiments.launch import compile_launch, run_launch
from tests.unit.test_algorithm_components import author, recipe


def batch(tmp_path):
    folder = tmp_path / "batch"
    folder.mkdir()
    template = author(folder, recipe("ppo"))
    document = yaml.safe_load(template.read_text())
    template.unlink()
    for index, name in enumerate(
        ("small-base", "small-sl", "medium-base", "medium-sl")
    ):
        value = {
            **document,
            "train_seeds": [4101, 4102, 4103],
            "train_ticks": 10 + index,
            "drain_frozen": True,
            "final_evaluation": True,
            "run": {"task": "train", "output": str(tmp_path / "outputs" / name)},
        }
        (folder / f"{name}.yaml").write_text(yaml.safe_dump(value))
    (folder / "scenario.yaml").write_text("schema: smartsom.scenario/v1\n")
    return folder


def compile_batch(folder, **kwargs):
    kwargs.setdefault("operations", {"output_root": str(folder.parent / "results")})
    kwargs["operations"].setdefault("output_root", str(folder.parent / "results"))
    return compile_launch(folder, ids=["medium-base", "medium-sl"], **kwargs)


def hashes(root):
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.rglob("*")
        if p.is_file()
    }


@pytest.fixture
def lightweight(monkeypatch):
    # Materialization and parent source/frozen-plan checking remain real; only the
    # per-trial source copy and worker are replaced by bounded fixture evidence.
    def allocate(compiled, *, root, tracking_group):
        (root / "inputs").mkdir(parents=True)
        compiled.materialize(root / "inputs")
        write_json(root / "plan.json", {"science": compiled.scientific_sha256})
        write_json(root / "batch.json", {"status": "allocated"})
        return root

    monkeypatch.setattr(reservation, "allocate", allocate)
    monkeypatch.setattr(
        reservation, "load", lambda root: json.loads((root / "plan.json").read_text())
    )
    monkeypatch.setattr(background, "_check_owner", lambda root: None)
    monkeypatch.setattr(author_batch, "bind", lambda root: None)
    monkeypatch.setattr(author_batch, "requested", lambda root: False)
    monkeypatch.setattr(
        "smartsom.experiments.queue_admission.initialize", lambda *a, **k: None
    )
    calls = []

    def run_children(root, plan, state, rows, **kwargs):
        for row in rows:
            calls.append(row["id"])
            child = root / row["child"]
            (child / "last.pt").write_bytes(row["id"].encode())
            (child / "continuation").mkdir(exist_ok=True)
            generation = child / "continuation/generation-0"
            generation.mkdir(exist_ok=True)
            write_json(
                generation / "state.json", {"rows": [], "ticks": 0, "condition_rng": []}
            )
            (generation / "model.pt").write_bytes(b"fixture-model")
            write_json(
                child / "continuation/current.json",
                {
                    "generation": "generation-0",
                    "files": {
                        name: hashlib.sha256(
                            (generation / name).read_bytes()
                        ).hexdigest()
                        for name in ("state.json", "model.pt")
                    },
                },
            )
            (child / "evaluation").mkdir(exist_ok=True)
            write_json(child / "evaluation/completed.json", {"fixture": True})
            state["files"][row["id"]]["status"] = "completed"
        return True

    monkeypatch.setattr(author_batch, "_run_children", run_children)

    class Tail:
        def __init__(self, root):
            self.jobs, self.unfinished = {}, False

        def tick(self):
            pass

    monkeypatch.setattr("smartsom.experiments.runtime_workers.NativeScheduler", Tail)
    return calls


def test_initial_two_supplement_four_repeat_zero_and_unchanged_artifacts(
    tmp_path, lightweight
):
    folder = batch(tmp_path)
    initial = compile_batch(folder, first_train_seeds=1)
    assert len(initial.files) == 12
    assert len(initial.selection["tasks"]) == 2
    assert {r["compiled"].experiment.train_seeds for r in initial.files} == {
        (4101, 4102, 4103)
    }
    first = run_launch(initial)
    root = Path(first["directory"])
    assert first["status"] == "partial"
    assert first["completed_tasks"] == 2 and first["deferred_tasks"] == 10
    old = {
        name: hashes(root / "trials" / name / "seed-4101")
        for name in ("medium-base", "medium-sl")
    }
    assert not runtime_control.snapshot(root)["accepting"]
    second = run_launch(compile_batch(folder, resume_missing=True))
    assert second["directory"] == first["directory"]
    assert second["submitted_tasks"] == 4 and second["completed_tasks"] == 6
    assert second["deferred_tasks"] == 6
    assert len(lightweight) == len(set(lightweight)) == 6
    for name, before in old.items():
        assert hashes(root / "trials" / name / "seed-4101") == before
    third = run_launch(compile_batch(folder, resume_missing=True))
    assert third["submitted_tasks"] == 0 and len(lightweight) == 6
    assert len(list(root.rglob("last.pt"))) == 6
    assert all("medium" in task for task in lightweight)


def test_full_inventory_and_path_binding_reject_changes(tmp_path, lightweight):
    folder = batch(tmp_path)
    initial = compile_batch(folder, first_train_seeds=1)
    run_launch(initial)
    # Even an unselected small experiment cannot change under this batch.
    path = folder / "small-base.yaml"
    old = path.read_text()
    path.write_text(old.replace("train_ticks: 10", "train_ticks: 99"))
    with pytest.raises(ValueError, match="inventory changed"):
        run_launch(compile_batch(folder, resume_missing=True))
    path.write_text(old)
    with pytest.raises(ValueError, match="inventory changed"):
        run_launch(
            compile_batch(
                folder, resume_missing=True, sets=["experiment.train_seeds=1"]
            )
        )
    import shutil

    moved = tmp_path / "other" / "batch"
    shutil.copytree(folder, moved)
    with pytest.raises(ValueError, match="directory or scientific"):
        run_launch(
            compile_batch(
                moved,
                resume_missing=True,
                operations={"output_root": str(initial.output_root)},
            )
        )


def test_edited_seed_list_keeps_old_results_and_selects_new(tmp_path, lightweight):
    folder = batch(tmp_path)
    initial = compile_batch(folder)
    run_launch(initial)
    root = initial.output_root
    old = hashes(root / "trials/medium-base/seed-4101")
    path = folder / "medium-base.yaml"
    config = yaml.safe_load(path.read_text())
    config["train_seeds"] = [4103, 4101, 101, 102]
    path.write_text(yaml.safe_dump(config))
    result = run_launch(
        compile_launch(
            folder,
            ids=["medium-base"],
            selectors=[{"train_seeds": [101]}],
            operations={"output_root": str(root)},
        )
    )
    assert result["submitted_tasks"] == 1
    assert hashes(root / "trials/medium-base/seed-4101") == old
    assert (root / "trials/medium-base/seed-101/last.pt").is_file()
    assert not (root / "trials/medium-base/seed-102/last.pt").exists()
    result = run_launch(
        compile_launch(
            folder,
            ids=["medium-base"],
            resume_missing=True,
            operations={"output_root": str(root)},
        )
    )
    assert result["submitted_tasks"] == 1
    assert (root / "trials/medium-base/seed-102/last.pt").is_file()
    assert hashes(root / "trials/medium-base/seed-4101") == old
    result = run_launch(
        compile_launch(
            folder,
            ids=["medium-base"],
            resume_missing=True,
            operations={"output_root": str(root)},
        )
    )
    assert result["submitted_tasks"] == 0

    # Reintroducing an already completed seed must not run it again.
    config["train_seeds"] = [4102, 101, 102]
    path.write_text(yaml.safe_dump(config))
    result = run_launch(
        compile_launch(
            folder,
            ids=["medium-base"],
            resume_missing=True,
            operations={"output_root": str(root)},
        )
    )
    assert result["submitted_tasks"] == 0
    assert hashes(root / "trials/medium-base/seed-4101") == old

    # The authored run.seed is not the per-child seed and must not be hidden.
    original_seed = config["run"].get("seed")
    config["run"]["seed"] = 999
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="scientific inventory changed"):
        run_launch(
            compile_launch(
                folder,
                ids=["medium-base"],
                resume_missing=True,
                operations={"output_root": str(root)},
            )
        )
    if original_seed is None:
        del config["run"]["seed"]
    else:
        config["run"]["seed"] = original_seed

    # An unrelated scientific change is still rejected before any work.
    config["train_ticks"] += 1
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="scientific inventory changed"):
        run_launch(
            compile_launch(
                folder,
                ids=["medium-base"],
                resume_missing=True,
                operations={"output_root": str(root)},
            )
        )


def test_single_file_run_select_reuses_owner_batch(tmp_path, lightweight):
    from smartsom.experiments.cli import main

    folder = batch(tmp_path)
    path = folder / "medium-base.yaml"
    root = tmp_path / "results"
    assert (
        main(
            [
                "run",
                str(path),
                "--output-root",
                str(root),
                "--select",
                "train_seeds: [4101]",
            ]
        )
        == 0
    )
    old = hashes(root / "trials/medium-base/seed-4101")
    config = yaml.safe_load(path.read_text())
    config["train_seeds"].append(101)
    path.write_text(yaml.safe_dump(config))
    assert (
        main(
            [
                "run",
                str(path),
                "--output-root",
                str(root),
                "--select",
                "train_seeds: [101]",
            ]
        )
        == 0
    )
    assert (root / "trials/medium-base/seed-101/last.pt").is_file()
    assert hashes(root / "trials/medium-base/seed-4101") == old
    assert lightweight == ["medium-base--seed-4101", "medium-base--seed-101"]
    assert main(["run", str(path), "--output-root", str(root), "--resume-missing"]) == 0
    assert lightweight == [
        "medium-base--seed-4101",
        "medium-base--seed-101",
        "medium-base--seed-4102",
        "medium-base--seed-4103",
    ]


def test_source_change_and_live_driver_rejected(tmp_path, lightweight, monkeypatch):
    folder = batch(tmp_path)
    initial = compile_batch(folder, first_train_seeds=1)
    run_launch(initial)
    from smartsom.experiments.locking import exclusive_lock

    with exclusive_lock(initial.output_root / "driver.lock"):
        with pytest.raises(RuntimeError, match="still active"):
            run_launch(compile_batch(folder, resume_missing=True))
    monkeypatch.setattr(
        author_batch, "implementation_identity", lambda: "different-source"
    )
    with pytest.raises(ValueError, match="source/dependency identity changed"):
        run_launch(compile_batch(folder, resume_missing=True))


def test_completed_training_pending_evaluation_and_failed_tasks(tmp_path, lightweight):
    folder = batch(tmp_path)
    initial = compile_batch(folder, first_train_seeds=1)
    run_launch(initial)
    root = initial.output_root
    first_child = root / "trials/medium-base/seed-4101"
    (first_child / "evaluation/completed.json").unlink()
    # A completed parent must not cause the missing final panel to be skipped.
    state = json.loads((root / "batch.json").read_text())
    state["status"] = "completed"
    write_json(root / "batch.json", state)
    result = run_launch(compile_batch(folder, first_train_seeds=1, resume_missing=True))
    assert result["submitted_tasks"] == 1
    assert lightweight.count("medium-base--seed-4101") == 2
    state = json.loads((root / "batch.json").read_text())
    state["files"]["medium-base--seed-4102"]["status"] = "failed"
    write_json(root / "batch.json", state)
    before = hashes(root)
    with pytest.raises(ValueError, match="no complete continuation"):
        run_launch(compile_batch(folder, resume_missing=True))
    assert hashes(root) == before
    assert "medium-base--seed-4102" not in lightweight
    # Failed submission did not change the prior selected task set.
    assert (
        "medium-base--seed-4102" not in author_batch.load(root)[2]["selection"]["tasks"]
    )


def test_interrupted_without_continuation_is_preserved(tmp_path, lightweight):
    folder = batch(tmp_path)
    initial = compile_batch(folder, first_train_seeds=1)
    run_launch(initial)
    root = initial.output_root
    state = json.loads((root / "batch.json").read_text())
    state["files"]["medium-base--seed-4102"]["status"] = "running"
    write_json(root / "batch.json", state)
    before = hashes(root)
    with pytest.raises(ValueError, match="no complete continuation"):
        run_launch(compile_batch(folder, resume_missing=True))
    assert hashes(root) == before


def test_submission_claim_covers_background_handoff(tmp_path, lightweight, monkeypatch):
    folder = batch(tmp_path)
    initial = compile_batch(
        folder, first_train_seeds=1, operations={"launch": {"mode": "background"}}
    )
    from smartsom.experiments.incremental_batch import submission

    seen = []

    def launch(root, **kwargs):
        with pytest.raises(RuntimeError, match="still active"):
            with submission(root):
                pass
        seen.append(root)
        return {"directory": str(root), "startup_status": "ready", "background": True}

    monkeypatch.setattr(background, "launch", launch)
    result = run_launch(initial)
    assert result["startup_status"] == "ready" and seen == [initial.output_root]


def test_stage_barrier_and_holds_survive(tmp_path, lightweight):
    folder = batch(tmp_path)
    for name in ("medium-base", "medium-sl"):
        path = folder / (name + ".yaml")
        data = yaml.safe_load(path.read_text())
        data["batch"] = {"stage": 1}
        path.write_text(yaml.safe_dump(data))
    initial = compile_batch(folder, first_train_seeds=1)
    result = run_launch(initial)
    assert result["status"] == "partial" and result["completed_tasks"] == 2
    assert len(lightweight) == 2
    root = initial.output_root
    with runtime_control.transaction(root) as saved:
        saved["paused_tasks"] = ["medium-base--seed-4101"]
    from smartsom.experiments.incremental_batch import activate

    activate(compile_batch(folder, resume_missing=True))
    assert runtime_control.snapshot(root)["paused_tasks"] == ["medium-base--seed-4101"]


def test_count_override_remains_scientific_and_parser_accepts_selection(tmp_path):
    folder = batch(tmp_path)
    with pytest.raises(ValueError, match="conflicts"):
        compile_batch(folder, first_train_seeds=1, sets=["experiment.train_seeds=1"])
    plan = compile_batch(folder, sets=["experiment.train_seeds=1"])
    assert {row["compiled"].experiment.run.seed for row in plan.files} == {101}
    from smartsom.experiments.cli import _parser

    args = _parser().parse_args(
        ["batch-run", str(folder), "--first-train-seeds", "1", "--background"]
    )
    assert args.first_train_seeds == 1 and args.background
    with pytest.raises(ValueError, match="positive integer"):
        compile_batch(folder, first_train_seeds=0)


def test_named_default_destination_is_under_runs_not_inputs(tmp_path, monkeypatch):
    folder = batch(tmp_path)
    monkeypatch.chdir(tmp_path)
    plan = compile_launch(folder, first_train_seeds=1)
    assert plan.output_root.parent == tmp_path / "runs/batches"
    assert not plan.output_root.is_relative_to(folder)


def test_report_deduplicates_case_identity_and_rejects_conflicts():
    from smartsom.experiments.research_report import unique_cases

    row = {
        "evaluation_identity": {"world_sha256": "world", "world_index": 0},
        "method": "ppo",
        "seed": 1,
        "action_seed": 2,
        "score": 3,
    }
    assert unique_cases([row, dict(row)]) == [row]
    with pytest.raises(ValueError, match="conflicting evaluation receipts"):
        unique_cases([row, {**row, "score": 4}])


def test_background_real_owner_handshake_rejects_second_submission(
    tmp_path, monkeypatch
):
    """Use the real detached launcher/owner handshake with a bounded idle driver."""
    import subprocess
    import sys

    from smartsom.experiments import control
    from smartsom.experiments.incremental_batch import submission

    folder = batch(tmp_path)
    compiled = compile_batch(
        folder, first_train_seeds=1, operations={"launch": {"mode": "background"}}
    )
    original = subprocess.Popen
    processes = []
    # Only the runtime body is a short fixture. The launcher, request nonce,
    # control registration, detached process and ready verification are real.
    program = """
import sys, time
from pathlib import Path
from smartsom.experiments import background

def fixture(root, **kwargs):
    deadline = time.monotonic() + 10
    while not (root / 'release-fixture').exists():
        if time.monotonic() > deadline:
            raise RuntimeError('fixture release timeout')
        time.sleep(.025)
    return {'directory': str(root), 'status': 'partial'}
background._execute = fixture
background.child(Path(sys.argv[1]), nonce=sys.argv[2])
"""

    def spawn(command, **kwargs):
        if command[1:3] != ["-m", "smartsom.experiments.background"]:
            return original(command, **kwargs)
        process = original(
            [sys.executable, "-c", program, command[3], command[5]], **kwargs
        )
        processes.append(process)
        return process

    monkeypatch.setattr(background.subprocess, "Popen", spawn)
    root = compiled.output_root
    try:
        result = run_launch(compiled)
        assert result["startup_status"] == "ready" and result["background"]
        assert control.read(root)["owner"] == result["owner"]
        assert processes[0].poll() is None
        with pytest.raises(ValueError, match="live control owner"):
            with submission(root):
                pass
    finally:
        if root.exists():
            (root / "release-fixture").touch()
        for process in processes:
            assert process.wait(timeout=15) == 0


def test_stopped_and_explicit_failed_recovery_keep_existing_continuation(
    tmp_path, lightweight
):
    folder = batch(tmp_path)
    initial = compile_batch(folder, first_train_seeds=1)
    run_launch(initial)
    root = initial.output_root
    state = json.loads((root / "batch.json").read_text())
    state["files"]["medium-base--seed-4101"]["status"] = "stopped"
    state["files"]["medium-sl--seed-4101"]["status"] = "failed"
    state["status"] = "stopped"
    write_json(root / "batch.json", state)
    marker = root / "trials/medium-base/seed-4101/continuation/current.json"
    prior = marker.read_bytes()
    result = run_launch(compile_batch(folder, first_train_seeds=1, resume_missing=True))
    assert result["submitted_tasks"] == 2 and result["failed_tasks"] == 0
    assert marker.read_bytes() == prior
    result = author_batch.execute_saved(root, retry_failed=True)
    assert result["failed_tasks"] == 0 and result["completed_tasks"] == 2
    assert lightweight.count("medium-base--seed-4101") == 2
    assert lightweight.count("medium-sl--seed-4101") == 2


def test_official_runtime_routes_pending_final_evaluation_without_overwriting_model(
    tmp_path, lightweight, monkeypatch
):
    folder = batch(tmp_path)
    initial = compile_batch(folder, first_train_seeds=1)
    run_launch(initial)
    child = initial.output_root / "trials/medium-base/seed-4101"
    (child / "evaluation/completed.json").unlink()
    write_json(child / "progress.json", {"status": "completed"})
    model = (child / "last.pt").read_bytes()
    called = []

    def complete_panel(args):
        assert (args.output / "last.pt").read_bytes() == model
        assert (args.output / "continuation/current.json").is_file()
        called.append(args.output)
        write_json(args.output / "evaluation/completed.json", {"fixture": True})
        write_json(args.output / "progress.json", {"status": "completed"})

    monkeypatch.setattr(reservation.runtime, "train", complete_panel)
    result = reservation.execute_saved(child)
    assert result["status"] == "completed" and called == [child]
    assert (child / "last.pt").read_bytes() == model
    reservation.execute_saved(child)
    assert called == [child]


def test_deferred_tasks_are_not_displayed_as_running_or_finished():
    from smartsom.telemetry.runtime import DisplayOptions, RuntimeDisplay

    display = RuntimeDisplay(
        DisplayOptions(progress="off"), kind="batch-directory", quiet=True
    )
    try:
        display.update("later", {"status": "deferred", "stage": "deferred"})
        assert display.tasks["later"]["status"] == "deferred"
        display.finish("partial")
        assert display.tasks["later"]["status"] == "deferred"
        assert display.task_counts() == (0, 0)
    finally:
        display.close()
