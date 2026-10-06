"""Real compiler, public recovery and retained-input lifetime regressions."""

import json
import pickle
import shutil
import sys
from dataclasses import replace
from types import ModuleType

import pytest
import yaml
from test_author_learning import _require_cpu, _write
from test_configurable_batch import inputs

from smartsom import api
from smartsom.config.codec import canonical_json
from smartsom.config.experiment_v4 import compile_experiment
from smartsom.config.scientific_contract import CONTRACT, identity
from smartsom.experiments.composable import allocate, archive_inputs
from smartsom.experiments.tuning_session import scientific_identity

pytestmark = pytest.mark.learning


def compiled(path):
    return compile_experiment(path).entries[0].prepared


def test_real_v4_compiler_separates_science_from_execution(tmp_path):
    path = inputs(tmp_path / "inputs", "ppo") / "experiment.yaml"
    experiment = yaml.safe_load(path.read_text(encoding="utf-8"))
    experiment["interface_contract"] = CONTRACT
    _write(path, experiment)
    baseline = compiled(path)
    experiment["checkpointing"].pop("retention")
    experiment["runtime"]["numerical_threads"] = 2
    experiment["execution"]["max_concurrent"] = 4
    _write(path, experiment)
    execution = compiled(path)
    assert baseline.scientific_sha256 == execution.scientific_sha256
    assert (
        baseline.scientific_sha256
        == identity(baseline)
        == scientific_identity(baseline)
    )
    experiment["validation"]["updates"] = [2]
    _write(path, experiment)
    assert compiled(path).scientific_sha256 != baseline.scientific_sha256
    experiment["validation"]["updates"] = [1, 2]
    experiment["seed"] += 1
    _write(path, experiment)
    assert compiled(path).scientific_sha256 != baseline.scientific_sha256


@pytest.mark.parametrize("algorithm", ["ppo", "dqn"])
@pytest.mark.parametrize("failed_update", [1, 2])
def test_public_resume_completes_pending_dev_without_retraining(
    tmp_path, monkeypatch, algorithm, failed_update
):
    _require_cpu()
    import torch
    from test_tuning_native_continuation import assert_state_equal

    from smartsom.experiments import composable

    path = inputs(tmp_path / "inputs", algorithm) / "experiment.yaml"
    experiment = yaml.safe_load(path.read_text(encoding="utf-8"))
    experiment["checkpointing"].pop("retention")
    _write(path, experiment)
    prepared = compiled(path)
    original = composable.evaluate_cases

    def fault(frozen, cases, **kwargs):
        if kwargs.get("validation"):
            source = json.loads(frozen.policies_json)["machine"]["resolved_model"][
                "source"
            ]
            from pathlib import Path

            update = int(Path(source).parents[1].name[7:])
            if update == failed_update:
                original(frozen, cases, **kwargs)
                raise RuntimeError("injected unfinished dev")
        return original(frozen, cases, **kwargs)

    with monkeypatch.context() as context:
        context.setattr(composable, "evaluate_cases", fault)
        with pytest.raises(RuntimeError, match="unfinished dev") as error:
            api.train_prepared(prepared)
    root = error.value.run_dir
    pointer = json.loads((root / "checkpoints/recovery.json").read_text())
    state = pickle.loads(
        (root / "checkpoints" / pointer["checkpoint"] / "continuation.pkl").read_bytes()
    )
    assert state["validation_phase"] == {
        "pending_update": failed_update,
        "completed_updates": list(range(1, failed_update)),
    }
    assert state["ticks"] == failed_update * 4
    resumed = api.resume(root)
    continuous = api.train_prepared(prepared)
    assert resumed.environment_steps == continuous.environment_steps == 8
    assert resumed.learner_updates == continuous.learner_updates == 2
    left = pickle.loads((resumed.last_checkpoint / "continuation.pkl").read_bytes())
    right = pickle.loads((continuous.last_checkpoint / "continuation.pkl").read_bytes())
    assert left["validation_phase"] == {
        "pending_update": None,
        "completed_updates": [1, 2],
    }
    assert left["optimizations"] == right["optimizations"]
    assert sum(left["optimizations"].values()) > 0
    assert left["actions"] == right["actions"]
    for role in ("machine", "buffer", "dispatcher"):
        assert_state_equal(
            torch.load(
                resumed.last_checkpoint / "groups" / role / "weights.pt",
                weights_only=True,
            ),
            torch.load(
                continuous.last_checkpoint / "groups" / role / "weights.pt",
                weights_only=True,
            ),
            exact=True,
        )
    record = json.loads((root / "run.json").read_text())
    assert record["validation_completed_updates"] == [1, 2]
    assert record["selection_outcome"]["validation_rounds"] == 2
    assert [
        row["update"]
        for row in json.loads((root / "reports/training.json").read_text())
    ] == [1, 2]


def test_retained_lookup_is_pinned_before_later_publication(tmp_path):
    _require_cpu()
    from smartsom.config.experiment_v3 import model_location
    from smartsom.config.policies import ModelSelector
    from smartsom.experiments.checkpoint_retention import resolve
    from smartsom.experiments.composable import TrainingSession

    prepared = compiled(inputs(tmp_path / "inputs", "ppo") / "experiment.yaml")
    root, record, prepared = allocate(prepared, "training")
    session = TrainingSession(prepared, root, record)
    try:
        session.step_update()
        old = resolve(root, "last")
        pinned = model_location(
            ModelSelector(source=str(root), checkpoint="last", group="machine")
        )
        session.step_update()
        assert not old.exists()
        declarations = json.loads(prepared.policies_json)
        declarations["machine"].update(
            implementation={
                "kind": "model",
                "model": {
                    "source": str(root),
                    "checkpoint": "last",
                    "group": "machine",
                },
            },
            resolved_model=pinned,
        )
        consumer = replace(prepared, policies_json=canonical_json(declarations))
        output = tmp_path / "consumer"
        (output / "config").mkdir(parents=True)
        archived = archive_inputs(output, consumer)
        assert (
            json.loads(archived.policies_json)["machine"]["resolved_model"][
                "weights_sha256"
            ]
            == pinned["weights_sha256"]
        )
    finally:
        session.close()


def test_relocated_initial_model_and_archive_do_not_change_identity(tmp_path):
    _require_cpu()
    path = inputs(tmp_path / "inputs", "ppo") / "experiment.yaml"
    source = api.train_prepared(compiled(path)).run_dir
    relocated = tmp_path / "relocated"
    shutil.copytree(source, relocated)
    method_path = path.parent.parent / "algorithm.yaml"
    method = yaml.safe_load(method_path.read_text(encoding="utf-8"))
    records = []
    for directory in (source, relocated):
        for role in ("machine", "buffer", "dispatcher"):
            method["agents"][role]["default"] = {
                "kind": "model",
                "model": {
                    "source": str(directory),
                    "checkpoint": "last",
                    "group": role,
                },
            }
        _write(method_path, method)
        prepared = compiled(path)
        _, _, archived = allocate(prepared, "training")
        assert identity(prepared) == identity(archived)
        records.append(prepared.scientific_sha256)
    assert records[0] == records[1]


def test_actual_human_worker_reads_archived_models_without_source_lock(
    tmp_path, monkeypatch
):
    _require_cpu()
    prepared = compiled(inputs(tmp_path / "inputs", "ppo") / "experiment.yaml")
    source = api.train_prepared(prepared).run_dir
    playback = ModuleType("smartsom.studio.playback")
    threads = []

    def window(factory, controls, thread):
        threads.append(thread)
        thread.start()
        thread.join(timeout=10)
        if thread.is_alive():
            controls.stop()
            raise AssertionError("human worker blocked by retained source lock")

    playback.live_window = window
    monkeypatch.setitem(sys.modules, "smartsom.studio.playback", playback)
    options = prepared.config.evaluation.model_copy(
        update={"checkpoint": "last", "render_mode": "human", "no_eligible_best": None}
    )
    result = api.evaluate(source, options, output_root=source / "evaluation")
    assert result.status == "completed" and threads and not threads[0].is_alive()


def test_adaptive_gc_waits_for_actual_reader_lease(tmp_path):
    _require_cpu()
    import threading

    from test_tuning_native_continuation import wrapper

    from smartsom.experiments.checkpoint_retention import lease
    from smartsom.experiments.tuning_session import verify_commit

    prepared = compiled(inputs(tmp_path / "inputs", "ppo") / "experiment.yaml")
    session = wrapper(prepared)
    try:
        session.step()
        old = session.last_commit
        session.session.step_update()
        attempting, finished = threading.Event(), threading.Event()
        errors = []

        def publish():
            attempting.set()
            try:
                session._commit()
            except BaseException as exc:
                errors.append(exc)
            finally:
                finished.set()

        with lease(session.root):
            thread = threading.Thread(target=publish)
            thread.start()
            assert attempting.wait(5)
            assert not finished.wait(0.1)
            assert verify_commit(old)["updates"] == 1
        thread.join(timeout=15)
        assert not thread.is_alive() and not errors
        assert not old.exists()
        assert verify_commit(session.last_commit)["updates"] == 2
    finally:
        session.close()


@pytest.mark.parametrize("boundary", ["before_pointer", "after_replace_before_sync"])
def test_adaptive_pointer_fault_preserves_previous_generation(
    tmp_path, monkeypatch, boundary
):
    _require_cpu()
    from test_tuning_native_continuation import wrapper

    from smartsom.experiments import checkpoint_retention as retention
    from smartsom.experiments import tuning_session
    from smartsom.experiments.tuning_session import verify_commit

    prepared = compiled(inputs(tmp_path / "inputs", "ppo") / "experiment.yaml")
    session = wrapper(prepared)
    try:
        session.step()
        old = session.last_commit
        previous = verify_commit(old)
        session.session.step_update()
        pointer = session.root / "checkpoints/adaptive-recovery.json"
        saved_pointer = pointer.read_bytes()
        original = tuning_session.durable_write
        sync = retention._sync_directory

        def publish(path, value):
            if boundary == "before_pointer" and path == pointer:
                raise OSError("injected adaptive pointer failure")
            return original(path, value)

        def directory_sync(path):
            if boundary == "after_replace_before_sync" and path == pointer.parent:
                raise OSError("injected adaptive pointer failure")
            return sync(path)

        with monkeypatch.context() as context:
            context.setattr(tuning_session, "durable_write", publish)
            context.setattr(retention, "_sync_directory", directory_sync)
            with pytest.raises(OSError, match="adaptive pointer failure"):
                session._commit()
        assert old.exists() and verify_commit(old) == previous
        if boundary == "before_pointer":
            assert pointer.read_bytes() == saved_pointer
        else:
            new = json.loads(pointer.read_text())
            assert verify_commit(pointer.parent / new["checkpoint"])["updates"] == 2
        session._commit()
        assert not old.exists()
    finally:
        session.close()
