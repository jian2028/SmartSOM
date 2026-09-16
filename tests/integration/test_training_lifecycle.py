"""Real locked-framework recovery: optimizer, RNG, physics, and validation isolation."""

import importlib.util
import json
import os
from dataclasses import replace
from pathlib import Path

import pytest
from test_training_sampling import checkpoint_rows, optimizer_and_rng, small_training

from smartsom.config.experiment import ExperimentConfig, prepare
from smartsom.experiments.production_training import verify_checkpoint
from smartsom.experiments.training import train_one
from smartsom.experiments.training_audit import audit_training
from smartsom.experiments.training_controls import TrainingControls, ValidationControls

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


@pytest.fixture(scope="module", params=["sb3", "rllib", "marl"])
def recovered(request, tmp_path_factory):
    packages = [
        "torch",
        "gymnasium",
        "sb3_contrib" if request.param == "sb3" else "ray",
    ]
    if request.param == "marl":
        pass  # No PettingZoo dependency for the grid RLlib wrapper.
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
        keep_last=4,
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
    name, _, _, continuous, partial, resumed = recovered
    assert partial.status == "interrupted" and partial.environment_steps == 64
    assert resumed.status == "completed" and resumed.environment_steps == 128
    a = json.loads((continuous.checkpoint_dir / "checkpoint.json").read_text())
    b = json.loads((resumed.checkpoint_dir / "checkpoint.json").read_text())
    assert a["final_weights_sha256"] == b["final_weights_sha256"]
    assert checkpoint_rows(continuous) == checkpoint_rows(resumed)
    assert_same_state(
        optimizer_and_rng(continuous, name), optimizer_and_rng(resumed, name)
    )
    assert resumed.learner_updates == (8 if name == "sb3" else 4)
    assert verify_checkpoint(resumed.last_checkpoint)["steps"] == 128
    assert audit_training(resumed.run_dir)["environment_steps"] == 128


def attempt(result):
    record = json.loads((result.run_dir / "run.json").read_text())
    return result.run_dir / record["paths"]["training"]


def test_periodic_validation_is_rng_isolated_and_replays_each_fixed_input(recovered):
    _, resolved, controls, continuous, _, resumed = recovered
    without_validation = train_one(
        resolved, controls=replace(controls, validation=None)
    )
    a = json.loads((continuous.checkpoint_dir / "checkpoint.json").read_text())
    b = json.loads((without_validation.checkpoint_dir / "checkpoint.json").read_text())
    assert a["final_weights_sha256"] == b["final_weights_sha256"]
    assert checkpoint_rows(continuous) == checkpoint_rows(without_validation)
    report = json.loads((attempt(continuous) / "validation-000004.json").read_text())
    assert all(row["replay"] == "passed" for row in report["results"])
    assert report == json.loads(
        (attempt(resumed) / "validation-000004.json").read_text()
    )


def test_weights_initialization_starts_an_independent_budget_and_episode_sequence(
    recovered,
):
    _, resolved, controls, continuous, _, _ = recovered
    config = ExperimentConfig.model_validate_json(resolved.config_json)
    config.training.total_steps = 64
    warm = train_one(
        prepare(config),
        controls=replace(
            controls, validation=None, initialize_from=continuous.last_checkpoint
        ),
    )
    source = json.loads((continuous.checkpoint_dir / "checkpoint.json").read_text())
    target = json.loads((warm.checkpoint_dir / "checkpoint.json").read_text())
    assert target["initial_weights_sha256"] == source["final_weights_sha256"]
    assert target["final_weights_sha256"] != target["initial_weights_sha256"]
    assert warm.environment_steps == 64
    rows = checkpoint_rows(warm)
    assert rows[0]["episode"] == 0
    assert audit_training(warm.run_dir)["environment_steps"] == 64


def test_resume_rejects_exhausted_budget_before_creating_attempt(recovered):
    _, resolved, controls, continuous, _, _ = recovered
    output = Path(
        ExperimentConfig.model_validate_json(resolved.config_json).output.root
    )
    before = set(output.iterdir())
    with pytest.raises(ValueError, match="exhausted"):
        train_one(
            resolved, controls=replace(controls, resume_from=continuous.last_checkpoint)
        )
    assert set(output.iterdir()) == before


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
    from smartsom.config.training import episode_root
    from smartsom.learning.production import LearnedProductionDriver
    from smartsom.learning.training_state import rng_state

    _, resolved, _, continuous, _, _ = recovered
    case = resolved.resolved.episode(episode_root(101, 0))

    def sequence():
        driver = LearnedProductionDriver(
            continuous.checkpoint_dir, case, deterministic=False, seed=515
        )
        # Loading initializes a backend. Action sampling itself must leave global RNG alone.
        state = rng_state()
        rows = []
        try:
            while not driver.env.finished:
                row = driver.next_tick()
                if row is not None:
                    for decision in row["decisions"]:
                        assert decision["mask"][decision["selected_index"]]
                    rows.append(row)
            assert_same_state(state, rng_state())
            return rows
        finally:
            driver.env.close()

    assert sequence() == sequence()
