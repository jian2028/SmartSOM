"""Real ordered sampling, exact quotas, and complete recovery across three PPOs."""

import importlib.util
import json
import os
from dataclasses import replace
from pathlib import Path

import pytest

from smartsom.config.experiment import ExperimentConfig, load_config, prepare
from smartsom.experiments.production_training import verify_checkpoint
from smartsom.experiments.training import train_one
from smartsom.experiments.training_audit import audit_training
from smartsom.experiments.training_controls import TrainingControls

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.learning


def small_training(name, output):
    config = load_config(ROOT / f"configs/runs/learning_{name}.yaml")
    config.training.total_steps = 128
    config.training.steps_per_update = 32
    config.training.max_decisions = 32
    config.algorithm.batch_size = 16
    config.algorithm.n_epochs = 2
    config.algorithm.hidden_sizes = (32, 32)
    config.output.root = str(output)
    config.logging.progress = "off"
    config.logging.tensorboard = False
    config.logging.verbose = False
    config.validation.enabled = False
    return prepare(config)


def checkpoint_rows(result):
    checkpoint = result.last_checkpoint
    rows = [
        json.loads(line)
        for line in (checkpoint / "episodes.jsonl").read_text().splitlines()
    ]
    return rows + json.loads((checkpoint / "active_episodes.json").read_text())


def optimizer_and_rng(result, name):
    import cloudpickle

    if name == "sb3":
        from sb3_contrib import MaskablePPO

        model = MaskablePPO.load(result.checkpoint_dir / "model.zip", device="cpu")
        optimizer = model.policy.optimizer.state_dict()
        with (result.last_checkpoint / "sampler.pkl").open("rb") as stream:
            state = cloudpickle.load(stream)
        rng = {key: state[key] for key in ("python_rng", "numpy_rng", "torch_rng")}
    else:
        from smartsom.learning.training_state import load_state

        optimizer = load_state(result.last_checkpoint / "training/algorithm.pkl")[
            "learner_group"
        ]["learner"]["optimizer"]
        rng = load_state(result.last_checkpoint / "rng.pkl")
    return optimizer, rng


def assert_state_equal(left, right):
    import numpy as np
    import torch

    if isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_state_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            assert_state_equal(a, b)
    elif isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    else:
        assert left == right


@pytest.fixture(scope="module", params=["sb3", "rllib", "marl"])
def sampled(request, tmp_path_factory):
    packages = [
        "torch",
        "gymnasium",
        "sb3_contrib" if request.param == "sb3" else "ray",
    ]
    if request.param == "marl":
        pass  # The grid resource wrapper uses RLlib directly.
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
    resolved = small_training(
        request.param, tmp_path_factory.mktemp(f"streams-{request.param}")
    )
    results = []
    for processes in (0, 2):
        controls = TrainingControls(
            checkpoint_every_updates=1,
            keep_last=4,
            validation=None,
            num_envs=2,
            sampling_processes=processes,
        )
        continuous = train_one(resolved, controls=controls)
        partial = train_one(resolved, controls=replace(controls, stop_after_updates=2))
        resumed = train_one(
            resolved, controls=replace(controls, resume_from=partial.last_checkpoint)
        )
        results.append((controls, continuous, partial, resumed))
    return request.param, resolved, results


def test_two_stream_resume_matches_weights_optimizer_rng_and_episode_prefix(sampled):
    name, _, results = sampled
    for _, continuous, partial, resumed in results:
        assert partial.status == "interrupted" and partial.environment_steps == 64
        assert resumed.status == "completed" and resumed.environment_steps == 128
        assert partial.run_dir == resumed.run_dir
        manifests = [
            json.loads((r.checkpoint_dir / "checkpoint.json").read_text())
            for r in (continuous, resumed)
        ]
        assert (
            manifests[0]["final_weights_sha256"] == manifests[1]["final_weights_sha256"]
        )
        assert checkpoint_rows(continuous) == checkpoint_rows(resumed)
        assert_state_equal(
            optimizer_and_rng(continuous, name), optimizer_and_rng(resumed, name)
        )
        update = verify_checkpoint(resumed.last_checkpoint)
        assert update["steps"] == 128 and update["updates"] == (
            8 if name == "sb3" else 4
        )
        report = audit_training(resumed.run_dir)
        assert report["environment_steps"] == 128 and report["num_envs"] == 2
        assert report["ppo_updates"] == 4 and report["budget_completed"]
        partial_report = audit_training(partial.last_checkpoint)
        assert (
            partial_report["environment_steps"] == 64
            and not partial_report["budget_completed"]
        )
        rows = checkpoint_rows(resumed)
        for stream in range(2):
            members = [row for row in rows if row["stream_id"] == stream]
            assert sorted(row["local_episode"] for row in members) == list(
                range(len(members))
            )
            assert sum(len(row["steps"]) for row in members) == 64
        if name == "marl":
            assert (
                sum(len(step["indices"]) for row in rows for step in row["steps"])
                == 128
            )


def test_process_completion_order_does_not_change_training(sampled):
    name, _, results = sampled
    left, right = results[0][1], results[1][1]
    assert checkpoint_rows(left) == checkpoint_rows(right)
    a, b = [
        json.loads((r.checkpoint_dir / "checkpoint.json").read_text())
        for r in (left, right)
    ]
    assert a["final_weights_sha256"] == b["final_weights_sha256"]
    assert_state_equal(optimizer_and_rng(left, name), optimizer_and_rng(right, name))


def test_full_resume_rejects_changed_stream_execution_identity(sampled):
    _, resolved, results = sampled
    controls, _, partial, _ = results[0]
    output = Path(
        ExperimentConfig.model_validate_json(resolved.config_json).output.root
    )
    before = set(output.iterdir())
    with pytest.raises(ValueError, match="identical|identity"):
        train_one(
            resolved,
            controls=replace(
                controls, resume_from=partial.last_checkpoint, sampling_processes=2
            ),
        )
    assert set(output.iterdir()) == before
