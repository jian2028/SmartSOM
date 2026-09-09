"""Real locked-framework recovery: optimizer, RNG, physics, and validation isolation."""

import importlib.util
import json
import os
from dataclasses import replace
from pathlib import Path

import pytest

from smartsom.config import resolve_training_run
from smartsom.experiments.training import train_one
from smartsom.experiments.training_audit import audit_training
from smartsom.experiments.training_controls import TrainingControls, ValidationControls
from smartsom.experiments.training_lifecycle import inspect_resume_checkpoint

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.learning


def assert_same_state(left, right):
    import numpy as np
    import torch

    if isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_same_state(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            assert_same_state(a, b)
    elif isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    else:
        assert left == right


def small_training(name, output):
    resolved = resolve_training_run(ROOT / f"configs/runs/learning_{name}.yaml")
    spec = resolved.algorithm.algorithm
    parameters = spec.parameters.model_copy(
        update={
            "n_steps": 32,
            "batch_size": 16,
            "n_epochs": 2,
            "hidden_sizes": (32, 32),
        }
    )
    return replace(
        resolved,
        algorithm=resolved.algorithm.model_copy(
            update={"algorithm": spec.model_copy(update={"parameters": parameters})}
        ),
        run=resolved.run.model_copy(
            update={
                "budget": resolved.run.budget.model_copy(
                    update={"environment_steps": 128}
                ),
                "output_root": str(output),
            }
        ),
    )


@pytest.fixture(scope="module", params=["sb3", "rllib", "marl"])
def recovered(request, tmp_path_factory):
    packages = [
        "torch",
        "gymnasium",
        "sb3_contrib" if request.param == "sb3" else "ray",
    ]
    if request.param == "marl":
        packages.append("pettingzoo")
    missing = [name for name in packages if importlib.util.find_spec(name) is None]
    if missing:
        required = (
            "SMARTSOM_REQUIRE_MARL"
            if request.param == "marl"
            else "SMARTSOM_REQUIRE_LEARNING"
        )
        if os.environ.get(required) == "1":
            pytest.fail(f"required learning extras missing: {missing}")
        pytest.skip(f"optional learning extras missing: {missing}")
    name = request.param
    resolved = small_training(name, tmp_path_factory.mktemp(f"recovery-{name}"))
    controls = TrainingControls(
        checkpoint_every_updates=1,
        validation=ValidationControls(
            every_updates=2, replications=2, full_replay=True
        ),
    )
    continuous = train_one(resolved, controls=controls)
    partial = train_one(resolved, controls=replace(controls, stop_after_updates=2))
    resumed = train_one(
        resolved, controls=replace(controls, resume_from=partial.last_checkpoint)
    )
    return name, resolved, controls, continuous, partial, resumed


def test_real_resume_matches_weights_optimizer_rng_and_raw_episode_ledger(recovered):
    from smartsom.learning.training_state import load_state

    name, _, _, continuous, partial, resumed = recovered
    assert partial.status == "interrupted" and partial.environment_steps == 64
    assert resumed.status == "completed" and resumed.environment_steps == 128
    a = json.loads((continuous.checkpoint_dir / "checkpoint.json").read_text())
    b = json.loads((resumed.checkpoint_dir / "checkpoint.json").read_text())
    assert a["final_weights_sha256"] == b["final_weights_sha256"]
    assert (continuous.run_dir / "episodes.jsonl").read_bytes() == (
        resumed.run_dir / "episodes.jsonl"
    ).read_bytes()
    assert_same_state(
        load_state(continuous.last_checkpoint / "rng.pkl"),
        load_state(resumed.last_checkpoint / "rng.pkl"),
    )
    if name == "sb3":
        from sb3_contrib import MaskablePPO

        optimizers = [
            MaskablePPO.load(
                r.last_checkpoint / "training/model.zip"
            ).policy.optimizer.state_dict()
            for r in (continuous, resumed)
        ]
        assert_same_state(*optimizers)
        assert resumed.learner_updates == 8
    else:
        states = [
            load_state(r.last_checkpoint / "training/algorithm.pkl")
            for r in (continuous, resumed)
        ]
        assert_same_state(
            *(state["learner_group"]["learner"]["optimizer"] for state in states)
        )
        assert_same_state(
            *(
                load_state(r.last_checkpoint / "training/ppo_dynamic.pkl")
                for r in (continuous, resumed)
            )
        )
        assert resumed.learner_updates == 4
    assert inspect_resume_checkpoint(resumed.last_checkpoint)["ppo_updates"] == 4
    assert audit_training(resumed.run_dir)["environment_steps"] == 128


def test_periodic_validation_is_rng_isolated_and_replays_each_fixed_input(recovered):
    _, resolved, controls, continuous, _, resumed = recovered
    without_validation = train_one(
        resolved, controls=replace(controls, validation=None)
    )
    a = json.loads((continuous.checkpoint_dir / "checkpoint.json").read_text())
    b = json.loads((without_validation.checkpoint_dir / "checkpoint.json").read_text())
    assert a["final_weights_sha256"] == b["final_weights_sha256"]
    assert (continuous.run_dir / "episodes.jsonl").read_bytes() == (
        without_validation.run_dir / "episodes.jsonl"
    ).read_bytes()
    report = json.loads((continuous.run_dir / "validation-000004.json").read_text())
    assert all(row["replay"] == "passed" for row in report["results"])
    assert report == json.loads(
        (resumed.run_dir / "validation-000004.json").read_text()
    )


def test_weights_initialization_starts_an_independent_budget_and_episode_sequence(
    recovered,
):
    _, resolved, controls, continuous, _, _ = recovered
    budget = resolved.run.budget.model_copy(update={"environment_steps": 64})
    warm = train_one(
        replace(resolved, run=resolved.run.model_copy(update={"budget": budget})),
        controls=replace(
            controls, validation=None, initialize_from=continuous.last_checkpoint
        ),
    )
    source = json.loads((continuous.checkpoint_dir / "checkpoint.json").read_text())
    target = json.loads((warm.checkpoint_dir / "checkpoint.json").read_text())
    assert target["initial_weights_sha256"] == source["final_weights_sha256"]
    assert target["final_weights_sha256"] != target["initial_weights_sha256"]
    assert warm.environment_steps == 64
    rows = [
        json.loads(line)
        for line in (warm.run_dir / "episodes.jsonl").read_text().splitlines()
    ]
    assert rows[0]["episode"] == 0
    assert audit_training(warm.run_dir)["environment_steps"] == 64


def test_resume_rejects_exhausted_budget_before_creating_attempt(recovered):
    _, resolved, controls, continuous, _, _ = recovered
    before = set(Path(resolved.run.output_root).iterdir())
    with pytest.raises(ValueError, match="exhausted"):
        train_one(
            resolved, controls=replace(controls, resume_from=continuous.last_checkpoint)
        )
    assert set(Path(resolved.run.output_root).iterdir()) == before


def test_explicit_save_disable_leaves_no_persistent_model(recovered):
    _, resolved, controls, _, _, _ = recovered
    stopped = train_one(
        resolved,
        controls=replace(
            controls,
            validation=None,
            save_last=False,
            save_best=False,
            stop_after_updates=1,
        ),
    )
    assert stopped.status == "interrupted"
    assert stopped.environment_steps == 32
    assert stopped.last_checkpoint is None and stopped.checkpoint_dir is None
    assert stopped.best_checkpoint is None
    assert not (stopped.run_dir / "checkpoint").exists()
    assert not list((stopped.run_dir / "checkpoints").glob("update-*"))


def test_stochastic_inference_has_private_repeatable_rng_and_legal_masks(recovered):
    from smartsom.experiments.training_validation import predictor
    from smartsom.learning.training_state import rng_state

    name, resolved, _, continuous, _, _ = recovered
    if name == "marl":
        from smartsom.learning.pettingzoo import SmartSOMParallelEnv

        env = SmartSOMParallelEnv(
            resolved.episode(0).input, resolved.algorithm.algorithm.projection
        )
    else:
        from smartsom.learning.gymnasium import SchedulingEnv

        env = SchedulingEnv(
            resolved.episode(0).input, resolved.algorithm.algorithm.projection
        )
    try:
        env.reset()
        state = rng_state()

        def sequence():
            predict = predictor(
                resolved.algorithm.algorithm.provider,
                continuous.checkpoint_dir,
                {},
                deterministic=False,
                seed=515,
            )
            return [predict(env.projected) for _ in range(30)]

        actions = sequence()
        assert actions == sequence()
        assert_same_state(state, rng_state())
        if name == "marl":
            masks = {view.agent_id: view.action_mask for view in env.projected.views}
            assert all(
                masks[agent][index]
                for joint in actions
                for agent, index in joint.items()
            )
        else:
            assert all(env.projected.action_mask[index] for index in actions)
    finally:
        env.close()


def test_disabling_periodic_saves_still_saves_last_on_normal_budget_end(tmp_path):
    if importlib.util.find_spec("sb3_contrib") is None:
        if os.environ.get("SMARTSOM_REQUIRE_LEARNING") == "1":
            pytest.fail("required sb3_contrib dependency missing")
        pytest.skip("optional sb3_contrib dependency missing")
    resolved = small_training("sb3", tmp_path)
    result = train_one(
        resolved,
        controls=TrainingControls(checkpoint_every_updates=None, validation=None),
    )
    assert result.last_checkpoint is not None
    assert result.last_checkpoint.name == "update-000004"
    assert result.checkpoint_dir == result.last_checkpoint / "inference"
    assert not (result.run_dir / "checkpoint").exists()
    manifest = json.loads((result.run_dir / "manifest.json").read_text())
    assert manifest["checkpoint_dir"] == str(result.checkpoint_dir)
    assert manifest["checkpoint_files"]
    assert [p.name for p in (result.run_dir / "checkpoints").glob("update-*")] == [
        "update-000004"
    ]


def test_validation_callback_prunes_at_saved_update_without_engineering_failure(
    tmp_path,
):
    if importlib.util.find_spec("sb3_contrib") is None:
        if os.environ.get("SMARTSOM_REQUIRE_LEARNING") == "1":
            pytest.fail("required sb3_contrib dependency missing")
        pytest.skip("optional sb3_contrib dependency missing")
    observed = []

    def progress(event):
        if event["stage"] == "validation":
            observed.append(event)
            return {"stop": "pruned"}
        return None

    result = train_one(
        small_training("sb3", tmp_path),
        on_progress=progress,
        controls=TrainingControls(
            checkpoint_every_updates=None,
            validation=ValidationControls(every_updates=1, replications=1),
        ),
    )
    assert result.status == "pruned" and result.environment_steps == 32
    assert result.last_checkpoint is not None
    assert inspect_resume_checkpoint(result.last_checkpoint)["ppo_updates"] == 1
    assert len(observed) == 1 and observed[0]["report"]["episodes"] == 1
    assert (result.run_dir / "validation-000001.json").exists()
    assert not (result.run_dir / "failure.json").exists()
