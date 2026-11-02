"""Cooperative resource thresholds are operational, bounded and fail closed."""

import json

import pytest

from smartsom.config.launch import ResourceGuard, Resources
from smartsom.experiments import resource_guard as guard

LIMITS = ResourceGuard(
    max_worker_rss_gib=8, min_host_available_gib=8, min_disk_free_gib=2
)
SAFE = dict(worker_rss=1, host_available=16 * guard.GIB, disk_free=4 * guard.GIB)


def test_disabled_guard_does_not_observe_or_read_clock(monkeypatch, tmp_path):
    def forbidden(*args):
        raise AssertionError("disabled guard did work")

    monkeypatch.setattr(guard, "observe", forbidden)
    assert Resources().guard is None
    with guard.scope(tmp_path, None):
        guard.check("rollout", force=True)
    assert not list(tmp_path.iterdir())


def test_throttle_forced_boundaries_and_nested_scope(monkeypatch, tmp_path):
    clock = [10.0]
    observed = []
    monkeypatch.setattr(guard.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(guard, "observe", lambda root: observed.append(root) or SAFE)
    with guard.scope(tmp_path, LIMITS):
        current = guard.CURRENT.get()
        for _ in range(10000):
            guard.check("train_rollout")
        assert len(observed) == 1
        clock[0] += 1
        guard.check()
        assert len(observed) == 2
        with guard.scope(tmp_path / "nested", LIMITS):
            assert guard.CURRENT.get() is current
            guard.check("case_commit", force=True)
        assert len(observed) == 3
    assert guard.CURRENT.get() is None


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("worker_rss", 8 * guard.GIB),
        ("host_available", 8 * guard.GIB - 1),
        ("disk_free", 2 * guard.GIB - 1),
    ],
)
def test_threshold_evidence_and_latched_failure(tmp_path, monkeypatch, key, value):
    monkeypatch.setattr(guard, "observe", lambda root: SAFE | {key: value})
    worker = guard.WorkerGuard(tmp_path, LIMITS)
    with pytest.raises(guard.ResourceGuardExceeded):
        worker.check("checkpoint_publish", force=True)
    evidence = json.loads((tmp_path / "resource-guard-failure.json").read_text())
    assert evidence["reason"] == [key]
    assert evidence["stage"] == "checkpoint_publish"
    assert evidence["observed_bytes"][key] == value
    monkeypatch.setattr(guard, "observe", lambda root: SAFE)
    with pytest.raises(guard.ResourceGuardExceeded):
        worker.check(force=True)


def test_minimum_available_and_disk_are_inclusive(tmp_path, monkeypatch):
    monkeypatch.setattr(
        guard,
        "observe",
        lambda root: dict(
            worker_rss=8 * guard.GIB - 1,
            host_available=8 * guard.GIB,
            disk_free=2 * guard.GIB,
        ),
    )
    guard.WorkerGuard(tmp_path, LIMITS).check(force=True)


def test_failed_observation_is_not_silently_accepted(tmp_path, monkeypatch):
    def unavailable(root):
        raise OSError("cannot observe")

    monkeypatch.setattr(guard, "observe", unavailable)
    with pytest.raises(guard.ResourceGuardExceeded, match="observation_unavailable"):
        guard.WorkerGuard(tmp_path, LIMITS).check(force=True)


def test_guard_does_not_change_scientific_identity(tmp_path):
    from smartsom.config.reservation import compile_reservation
    from tests.unit.test_algorithm_components import author, recipe

    path = author(tmp_path, recipe())
    original = compile_reservation(
        path,
        task="evaluate",
        sets=["experiment.drain_frozen=true", "experiment.run.methods=[rule]"],
    )
    protected = compile_reservation(
        path,
        task="evaluate",
        sets=["experiment.drain_frozen=true", "experiment.run.methods=[rule]"],
        operations={"resources": {"guard": LIMITS.model_dump()}},
    )
    assert protected.scientific_sha256 == original.scientific_sha256
    assert protected.experiment.resources.guard == LIMITS


def test_real_resource_sample_is_read_only(tmp_path):
    observed = guard.observe(tmp_path)
    assert observed["worker_rss"] > 0
    assert observed["host_available"] > 0
    assert observed["disk_free"] > 0


@pytest.mark.parametrize("task,tune", [("smoke", "off"), ("train", "auto")])
def test_uncovered_execution_modes_reject_opt_in_guard(task, tune):
    from smartsom.config.base_diagnostic import BaseExperiment

    with pytest.raises(ValueError, match="resources.guard requires native"):
        BaseExperiment.model_validate(
            dict(
                schema="smartsom.base-experiment/v1",
                factory="f",
                workload="w",
                algorithm="a",
                resources={"guard": LIMITS.model_dump()},
                run={"task": task, "tune": tune},
            )
        )


def test_evidence_write_failure_retains_nonretryable_guard_error(tmp_path, monkeypatch):
    from smartsom.experiments import evidence

    monkeypatch.setattr(guard, "observe", lambda root: SAFE | {"disk_free": 0})

    def full(*args):
        raise OSError("disk full")

    monkeypatch.setattr(evidence, "write_json", full)
    with pytest.raises(guard.ResourceGuardExceeded) as raised:
        guard.WorkerGuard(tmp_path, LIMITS).check(force=True)
    assert "disk full" in raised.value.__notes__[0]


def execution_entry(folder, *, task="smoke", tune="off"):
    import yaml

    from tests.unit.test_algorithm_components import author, recipe

    folder.mkdir(parents=True)
    path = author(folder, recipe("dqn" if task == "train" else "rule"))
    document = yaml.safe_load(path.read_text())
    document["drain_frozen"] = True
    document["run"] = dict(
        task=task, tune=tune, methods=["rule"], smoke_ticks=2, seconds=30
    )
    path.write_text(yaml.safe_dump(document))
    return path


def file_bytes(root):
    return {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}


@pytest.mark.parametrize("mode", ["smoke", "tune"])
@pytest.mark.parametrize("through_profile", [False, True])
def test_execution_update_validates_frozen_task_before_writing(
    tmp_path, mode, through_profile
):
    import yaml

    from smartsom.config.reservation import compile_reservation
    from smartsom.experiments import reservation
    from smartsom.experiments.launch import update_execution

    path = execution_entry(
        tmp_path / "author",
        task="train" if mode == "tune" else "smoke",
        tune="auto" if mode == "tune" else "off",
    )
    root = reservation.allocate(compile_reservation(path), root=tmp_path / "run")
    before = file_bytes(root)
    operations = {"resources": {"guard": LIMITS.model_dump()}}
    if through_profile:
        profile = tmp_path / "profile.yaml"
        profile.write_text(yaml.safe_dump(operations))
        arguments = {"profile": profile}
    else:
        arguments = {"operations": operations}
    with pytest.raises(ValueError, match="resources.guard requires native"):
        update_execution(root, **arguments)
    assert file_bytes(root) == before


def test_closed_smoke_resume_rejects_guard_without_touching_results(tmp_path):
    from smartsom import api
    from smartsom.config.reservation import compile_reservation
    from smartsom.experiments import reservation

    path = execution_entry(tmp_path / "author")
    root = reservation.allocate(compile_reservation(path), root=tmp_path / "run")
    assert reservation.execute_saved(root)["status"] == "completed"
    before = file_bytes(root)
    with pytest.raises(ValueError, match="resources.guard requires native"):
        api.resume(str(root), operations={"resources": {"guard": LIMITS.model_dump()}})
    assert file_bytes(root) == before


@pytest.mark.parametrize("mode", ["smoke", "tune"])
def test_invalid_execution_sidecar_cannot_bypass_saved_entrypoint(
    tmp_path, monkeypatch, mode
):
    from smartsom.config.reservation import compile_reservation
    from smartsom.experiments import reservation

    path = execution_entry(
        tmp_path / "author",
        task="train" if mode == "tune" else "smoke",
        tune="auto" if mode == "tune" else "off",
    )
    root = reservation.allocate(compile_reservation(path), root=tmp_path / "run")
    sidecar = root / "execution.json"
    saved = json.loads(sidecar.read_text())
    saved["settings"]["resources"]["guard"] = LIMITS.model_dump()
    sidecar.write_text(json.dumps(saved))
    # A completed marker must not bypass validation either.
    (root / "progress.json").write_text(json.dumps({"status": "completed"}))
    before = file_bytes(root)

    def forbidden(*args, **kwargs):
        raise AssertionError("unsupported execution reached a runtime")

    monkeypatch.setattr(reservation, "smoke", forbidden)
    monkeypatch.setattr(guard, "observe", forbidden)
    with pytest.raises(ValueError, match="resources.guard requires native"):
        reservation.execute_saved(root)
    assert file_bytes(root) == before


@pytest.mark.parametrize("external", [False, True])
def test_invalid_late_child_leaves_batch_children_and_runtime_unchanged(
    tmp_path, external
):

    from smartsom.config.reservation import compile_reservation
    from smartsom.experiments import author_batch, reservation, runtime_control
    from smartsom.experiments.launch import compile_launch, update_execution

    folder = tmp_path / "batch"
    execution_entry(folder / "a-valid", task="calibrate")
    if not external:
        execution_entry(folder / "z-invalid")
    root = author_batch.allocate(
        compile_launch(folder, operations={"output_root": str(tmp_path / "run")})
    )
    runtime_control.initialize(root, 2, keep_open=False)
    if external:
        bad = reservation.allocate(
            compile_reservation(execution_entry(tmp_path / "z-author")),
            root=tmp_path / "z-external",
        )
        with runtime_control.transaction(root) as state:
            state["additions"]["z-external"] = {"root": str(bad), "status": "queued"}
    # Exercise the same update used to reopen a closed queue, without launching it.
    with runtime_control.transaction(root) as state:
        state["accepting"] = False
        state["status"] = "completed"
    before = file_bytes(tmp_path)
    with pytest.raises(ValueError, match="resources.guard requires native"):
        update_execution(
            root,
            operations={
                "resources": {"guard": LIMITS.model_dump()},
                "concurrency": {"max_concurrent": 7},
            },
        )
    assert file_bytes(tmp_path) == before


def test_valid_closed_child_update_keeps_science_and_condition_inputs(tmp_path):
    from pathlib import Path

    from smartsom.config.base_diagnostic import load_inputs
    from smartsom.config.reservation import compile_reservation
    from smartsom.experiments import reservation, runtime_control
    from smartsom.experiments.launch import update_execution

    source = (
        Path(__file__).resolve().parents[2]
        / "experiments/rule-dqn/narrow-layout/medium-base.yaml"
    )
    root = reservation.allocate(
        compile_reservation(source, operations={"resources": {"guard": None}}),
        root=tmp_path / "run",
    )
    immutable = {
        path: contents
        for path, contents in file_bytes(root).items()
        if path.name != "execution.json"
    }
    runtime_control.initialize(root, 1, keep_open=False)
    with runtime_control.transaction(root) as state:
        state["accepting"] = False
        state["status"] = "completed"
    update_execution(root, operations={"resources": {"guard": LIMITS.model_dump()}})
    ex, factory, workload, *_ = load_inputs(root / "inputs/experiment.yaml")
    assert ex.resources.guard == LIMITS
    assert set(ex._condition_inputs) == {"normal", "low", "medium", "high"}
    assert ex.inputs_for("low", factory, workload) == ex._condition_inputs["low"]
    assert all(path.read_bytes() == contents for path, contents in immutable.items())


def test_saved_entrypoint_checks_prepared_execution_before_bind(tmp_path, monkeypatch):
    from smartsom.config.reservation import compile_reservation
    from smartsom.experiments import reservation

    root = reservation.allocate(
        compile_reservation(execution_entry(tmp_path / "author")), root=tmp_path / "run"
    )
    args, settings = reservation.arguments(root)
    inputs, identity = args._prepared_inputs
    # Simulate an adapter handing in a model_copy that bypassed validation.
    invalid = inputs[0].model_copy(update={"resources": Resources(guard=LIMITS)})
    args._prepared_inputs = ((invalid, *inputs[1:]), identity)
    monkeypatch.setattr(reservation, "arguments", lambda root: (args, settings))

    def forbidden(*args):
        raise AssertionError("invalid prepared settings registered an owner")

    monkeypatch.setattr("smartsom.telemetry.runtime.bind", forbidden)
    with pytest.raises(ValueError, match="resources.guard requires native"):
        reservation.execute_saved(root)


def test_candidate_update_can_explicitly_remove_old_invalid_guard(tmp_path):
    from smartsom.config.base_diagnostic import load_inputs
    from smartsom.config.reservation import compile_reservation
    from smartsom.experiments import reservation
    from smartsom.experiments.launch import update_execution

    root = reservation.allocate(
        compile_reservation(execution_entry(tmp_path / "author")), root=tmp_path / "run"
    )
    path = root / "execution.json"
    document = json.loads(path.read_text())
    document["settings"]["resources"]["guard"] = LIMITS.model_dump()
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="resources.guard requires native"):
        load_inputs(root / "inputs/experiment.yaml")
    frozen = {p: content for p, content in file_bytes(root).items() if p != path}
    update_execution(root, operations={"resources": {"guard": None}})
    assert load_inputs(root / "inputs/experiment.yaml")[0].resources.guard is None
    assert all(p.read_bytes() == content for p, content in frozen.items())
