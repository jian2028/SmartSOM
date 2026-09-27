"""Bounded real backend checks for adaptive native continuation and evidence.

These 4/8-tick probes verify engineering state, never policy performance.
"""

import copy
import json
import pickle
import shutil
from dataclasses import replace
from pathlib import Path

import pytest

from smartsom import api
from smartsom.config.codec import canonical_json
from smartsom.config.experiment import prepare
from smartsom.experiments.composable import (
    TrainingSession,
    allocate,
    checkpoint_path,
    export,
)
from smartsom.experiments.tuning_session import (
    AdaptiveSession,
    control_recipe,
    verify_commit,
)

torch = pytest.importorskip("torch")
pytest.importorskip("ray")

ROOT = Path(__file__).resolve().parents[2]


def tiny(algorithm, output, *, validation=True, empty_validation=False, patience=None):
    config = api.load_config(
        ROOT / "configs/test/runs" / ("train_all_" + algorithm + ".yaml")
    )
    config.training.total_ticks = 8
    config.training.ticks_per_update = 4
    config.training.record_initial = False
    config.validation.enabled = validation
    config.validation.every_updates = 1
    config.validation.replications = 1
    config.validation.patience = patience
    config.evaluation.replications = 1
    config.evaluation.checkpoint = "last"
    config.output.root = str(output)
    config.scenario_overrides["tick_limit"] = 8
    config.runtime.numerical_threads = 1
    config.runtime.sampling_processes = 0
    frozen = prepare(config)
    parameters = json.loads(frozen.parameters_json)
    if algorithm == "dqn":
        parameters.update(batch_size=2, warmup_ticks=0, target_update_ticks=4)
    else:
        parameters.update(batch_size=2, n_epochs=1)
    # An empty verification case deliberately exercises real best/patience
    # transitions without a learned policy needing to finish manufacturing.
    if empty_validation:
        cases = json.loads(frozen.validation_json)
        for case in cases:
            case["scenario"]["demands"] = []
        frozen = replace(frozen, validation_json=canonical_json(cases))
    return replace(frozen, parameters_json=canonical_json(parameters))


def wrapper(frozen, *, root=None, rec=None, controls=None, threads=1, finalizer=None):
    if rec is None:
        root, rec, frozen = allocate(frozen, "training")
    rec = copy.deepcopy(rec)
    rec.setdefault("tuning", {}).update(
        experiment_id="native-acceptance", control_spec=controls
    )
    return AdaptiveSession(frozen, root, rec, threads=threads, finalizer=finalizer)


def run_complete(session):
    while True:
        result = session.step()
        if result["done"]:
            return result


def saved_state(session):
    return pickle.loads((session.last_commit / "continuation.pkl").read_bytes())


def assert_state_equal(actual, expected):
    import numpy as np

    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
    elif isinstance(expected, np.ndarray):
        assert actual.dtype == expected.dtype and actual.shape == expected.shape
        if np.issubdtype(expected.dtype, np.floating):
            # CPU reduction order changes with numerical threads. Serialized
            # float32 weights need the same numerical comparison as tensors;
            # integer RNG state and discrete simulator state remain exact.
            np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=2e-6)
        else:
            np.testing.assert_array_equal(actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_state_equal(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected, strict=True):
            assert_state_equal(a, b)
    elif hasattr(expected, "snapshot"):
        assert actual.snapshot() == expected.snapshot()
    else:
        assert actual == expected


@pytest.mark.parametrize("algorithm", ("ppo", "dqn"))
def test_real_paused_native_state_matches_continuous_with_new_threads(
    algorithm, tmp_path
):
    frozen = tiny(algorithm, tmp_path / "runs")
    continuous = wrapper(frozen)
    try:
        run_complete(continuous)
        expected = saved_state(continuous)
        assert sum(continuous.session.optimizations.values()) > 0
    finally:
        continuous.close()
    partial = wrapper(frozen)
    result = partial.step()
    assert not result["done"] and result["physical_ticks"] == 4
    state = saved_state(partial)
    assert state["no_improvement"] == 1
    assert (partial.last_commit / "support/logs/validation-000001.json").is_file()
    if algorithm == "dqn":
        assert state["replays"]
    original = (partial.root / "config/original-prepared.json").read_bytes()
    saved = partial.save_checkpoint(tmp_path / "ray")
    record = copy.deepcopy(partial.record)
    record["tuning"]["continuation"] = saved
    partial.close()
    resumed = wrapper(frozen, root=tmp_path / "resumed", rec=record, threads=2)
    try:
        assert torch.get_num_threads() == 2
        assert resumed.session.no_improvement == 1
        assert (resumed.root / "config/original-prepared.json").read_bytes() == original
        run_complete(resumed)
        assert_state_equal(saved_state(resumed), expected)
        marker = verify_commit(resumed.last_commit)
        assert marker["phase"] == "experiment_complete" and marker["threads"] == 2
    finally:
        resumed.close()


def test_real_validation_best_patience_and_failed_final_phase_restore(tmp_path):
    frozen = tiny("ppo", tmp_path / "runs", empty_validation=True, patience=1)

    def failing_finalizer(session):
        raise RuntimeError("temporary final-evaluation failure")

    before = wrapper(frozen, finalizer=failing_finalizer)
    first = before.step()
    assert not first["done"] and before.session.best_update == 1
    assert before.session.best_score["completed"] == 1
    with pytest.raises(RuntimeError, match="temporary final-evaluation"):
        before.step()
    assert before.session.record["status"] == "early_stopped"
    assert before.session.no_improvement == 1
    marker = verify_commit(before.last_commit)
    assert marker["phase"] == "training_complete"
    assert (
        before.last_commit / "support/checkpoints/update-000001/snapshot.json"
    ).is_file()
    state = saved_state(before)
    assert state["best_update"] == 1 and state["no_improvement"] == 1
    saved = before.save_checkpoint(tmp_path / "ray")
    record = copy.deepcopy(before.record)
    record["tuning"]["continuation"] = saved
    before.close()
    after = wrapper(frozen, root=tmp_path / "resumed", rec=record, threads=2)
    try:
        updates, ticks = after.session.updates, after.session.ticks
        assert run_complete(after)["done"]
        assert (after.session.updates, after.session.ticks) == (updates, ticks)
        assert after.session.best_update == 1 and after.session.no_improvement == 1
        terminal = after.save_checkpoint(tmp_path / "terminal")
        record = copy.deepcopy(after.record)
        record["tuning"]["continuation"] = terminal
    finally:
        after.close()
    terminal_session = wrapper(frozen, root=tmp_path / "terminal-run", rec=record)
    try:
        assert terminal_session.step()["done"]
        assert terminal_session.session.ticks == ticks
    finally:
        terminal_session.close()


def test_real_final_evaluation_and_all_paired_controls_are_portable(tmp_path):
    frozen = tiny("ppo", tmp_path / "runs", validation=False)
    controls = {
        "names": ["initial", "rule", "random"],
        "pair_key": "paired-case-1",
        "directory": str(tmp_path / "shared"),
    }
    before = wrapper(frozen, controls=controls)
    try:
        assert (before.root / "checkpoints/update-000000/snapshot.json").is_file()
        result = run_complete(before)
        assert result["done"]
        assert (before.root / "evaluation/tuning-final.json").is_file()
        assert set(before.record["tuning"]["controls"]) == {"initial", "rule", "random"}
        for name in controls["names"]:
            path = before.root / before.record["tuning"]["controls"][name]["path"]
            rows = json.loads(path.read_text())
            assert len(rows) == 1 and not rows[0]["engineering_failure"]
            assert rows[0]["seed"] == json.loads(frozen.evaluation_json)[0]["seed"]
        saved = before.save_checkpoint(tmp_path / "ray")
        record = copy.deepcopy(before.record)
        record["tuning"]["continuation"] = saved
    finally:
        before.close()
    shutil.rmtree(before.root)
    shutil.rmtree(tmp_path / "shared")
    after = wrapper(frozen, root=tmp_path / "resumed", rec=record, controls=controls)
    try:
        assert after.step()["done"]
        assert set(after.record["tuning"]["controls"]) == {"initial", "rule", "random"}
        assert (after.root / "evaluation/evidence/case-0000/trace.jsonl").is_file()
        for name in controls["names"]:
            assert (after.root / "evaluation/controls" / (name + ".json")).is_file()
    finally:
        after.close()


def test_real_frozen_partner_zip_survives_removed_original_and_run(tmp_path):
    frozen = tiny("ppo", tmp_path / "source", validation=False)
    source = wrapper(frozen)
    try:
        run_complete(source)
        package = export(source.root, tmp_path / "dispatcher.zip", group="dispatcher")
    finally:
        source.close()
    declarations = json.loads(frozen.policies_json)
    from smartsom.config.experiment_v3 import model_location
    from smartsom.config.policies import ModelSelector

    declarations["dispatcher"]["implementation"] = {
        "kind": "model",
        "model": {"source": str(package)},
    }
    declarations["dispatcher"]["resolved_model"] = model_location(
        ModelSelector(source=str(package))
    )
    config = json.loads(frozen.config_json)
    config["training"]["groups"].remove("dispatcher")
    frozen = replace(
        frozen,
        config_json=canonical_json(config),
        policies_json=canonical_json(declarations),
    )
    mixed = wrapper(frozen)
    frozen = mixed.original
    try:
        expected = mixed.session.policies["dispatcher"].fingerprint()
        mixed.step()
        saved = mixed.save_checkpoint(tmp_path / "ray")
        record = copy.deepcopy(mixed.record)
        record["tuning"]["continuation"] = saved
    finally:
        mixed.close()
    shutil.rmtree(source.root)
    shutil.rmtree(mixed.root)
    Path(package).unlink()
    after = wrapper(frozen, root=tmp_path / "resumed", rec=record, threads=2)
    try:
        assert after.session.policies["dispatcher"].fingerprint() == expected
        run_complete(after)
        assert after.session.policies["dispatcher"].fingerprint() == expected
        assert after.record["frozen_partners_unchanged"]
        assert (
            checkpoint_path(after.root, "last") / "groups/dispatcher/model.json"
        ).is_file()
    finally:
        after.close()


def test_native_partial_finish_then_continue_refreshes_result(tmp_path):
    frozen = tiny("ppo", tmp_path / "runs", validation=False)
    root, record, frozen = allocate(frozen, "training")
    native = TrainingSession(frozen, root, record)
    try:
        partial = native.execute(stop_after_updates=1)
        assert partial.environment_steps == 4 and partial.status == "interrupted"
        finished = native.execute()
        assert finished.environment_steps == 8 and finished.status == "completed"
        assert len(json.loads((root / "reports/training.json").read_text())) == 2
    finally:
        native.close()


@pytest.mark.parametrize(
    "name",
    ("train_all_ppo", "central_rllib_ppo", "central_sb3_ppo", "small_rules_auto"),
)
@pytest.mark.parametrize("random", (False, True))
def test_real_role_controls_cover_grid_matrix_and_central_worlds(name, random):
    from smartsom.experiments.composable import evaluate_cases

    config = api.load_config(ROOT / "configs/test/runs" / (name + ".yaml"))
    if config.training:
        config.training.total_ticks = 4
        config.training.ticks_per_update = 4
    config.validation.enabled = False
    config.evaluation.replications = 1
    config.scenario_overrides["tick_limit"] = 4
    original = prepare(config)
    recipe = control_recipe(original, random=random)
    if name.startswith("central"):
        assert original.composition_json != recipe.composition_json
    else:
        assert original.composition_json == recipe.composition_json
    rows = evaluate_cases(recipe, json.loads(recipe.evaluation_json))
    assert len(rows) == 1 and not rows[0]["engineering_failure"]
    assert rows[0]["physical_ticks"] <= 4


def _worker_numerical_threads():
    return torch.get_num_threads()


def test_real_parallel_sampler_threads_stay_frozen_through_restore(tmp_path):
    frozen = tiny("ppo", tmp_path / "runs", validation=False)
    config = json.loads(frozen.config_json)
    config["runtime"].update(num_envs=2, sampling_processes=2)
    frozen = replace(frozen, config_json=canonical_json(config))
    continuous = wrapper(frozen)
    try:
        run_complete(continuous)
        expected = saved_state(continuous)
    finally:
        continuous.close()
    partial = wrapper(frozen)
    try:
        partial.step()
        saved = partial.save_checkpoint(tmp_path / "ray")
        record = copy.deepcopy(partial.record)
        record["tuning"]["continuation"] = saved
    finally:
        partial.close()
    after = wrapper(frozen, root=tmp_path / "resumed", rec=record, threads=2)
    try:
        assert torch.get_num_threads() == 2
        assert (
            after.session.executor.submit(_worker_numerical_threads).result(timeout=30)
            == 1
        )
        run_complete(after)
        assert_state_equal(saved_state(after), expected)
    finally:
        after.close()
