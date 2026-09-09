"""Real public workflows: train, exact resume, evaluate and portable initialization."""

import importlib.util
import json
import os

import pytest

from smartsom import api
from smartsom.config.codec import primitive
from smartsom.config.experiment import apply_overrides
from smartsom.experiments.packaging import export_model, model_locator, verify_bundle
from smartsom.experiments.training_audit import audit_training


def require_backend(name):
    packages = {
        "sb3_micro": ("sb3_contrib", "torch"),
        "rllib_micro": ("ray", "torch"),
        "marl_micro": ("ray", "torch", "pettingzoo"),
    }[name]
    if not all(importlib.util.find_spec(p) for p in packages):
        required = (
            "SMARTSOM_REQUIRE_MARL"
            if name == "marl_micro"
            else "SMARTSOM_REQUIRE_LEARNING"
        )
        if os.environ.get(required) == "1":
            pytest.fail(f"required backend missing: {name}")
        pytest.skip(f"optional backend missing: {name}")


def recipe(name, root):
    return apply_overrides(
        api.load_preset(name),
        [
            ("training.total_steps", 128),
            ("training.steps_per_update", 32),
            ("algorithm.batch_size", 16),
            ("algorithm.n_epochs", 2),
            ("validation.every_updates", 2),
            ("validation.replications", 1),
            ("validation.full_replay", True),
            ("checkpointing.every_updates", 2),
            ("logging.tensorboard", False),
            ("logging.progress", "off"),
            ("logging.verbose", 0),
            ("output.root", str(root)),
        ],
    )


@pytest.mark.parametrize("name", ["sb3_micro", "rllib_micro", "marl_micro"])
def test_public_train_resume_evaluate_and_export(name, tmp_path):
    require_backend(name)
    uninterrupted = api.train(recipe(name, tmp_path / "continuous"))
    config = recipe(name, tmp_path / "resumed")

    def stop(event):
        if event["stage"] == "validation" and event["ppo_updates"] == 2:
            return {"stop": "pruned"}

    stopped = api.train(config, on_progress=stop)
    assert (stopped.status, stopped.environment_steps, stopped.ppo_updates) == (
        "pruned",
        64,
        2,
    )
    assert not audit_training(stopped.training_dir)["budget_completed"]
    resumed = api.resume(stopped.run_dir)
    assert resumed.run_dir == stopped.run_dir
    assert (resumed.status, resumed.environment_steps, resumed.ppo_updates) == (
        "completed",
        128,
        4,
    )
    assert audit_training(resumed.training_dir)["budget_completed"]
    assert (
        json.loads((model_locator(resumed.run_dir) / "checkpoint.json").read_text())[
            "final_weights_sha256"
        ]
        == json.loads(
            (model_locator(uninterrupted.run_dir) / "checkpoint.json").read_text()
        )["final_weights_sha256"]
    )
    assert (resumed.training_dir / "episodes.jsonl").read_bytes() == (
        uninterrupted.training_dir / "episodes.jsonl"
    ).read_bytes()
    assert len(json.loads((resumed.run_dir / "run.json").read_text())["attempts"]) == 1
    options = api.EvaluationOptions(replications=1, baselines=("spt",))
    result = api.evaluate(resumed.run_dir, options, output_root=tmp_path / "evaluation")
    manifest = json.loads((result.run_dir / "run.json").read_text())
    assert len(manifest["results"]) == 2
    assert all(
        row["replay"]["status"] in {"passed", "partial_verified"}
        for row in manifest["results"]
    )
    package = export_model(resumed.run_dir, tmp_path / "model.zip")
    assert verify_bundle(package)["kind"] == "model"
    initialized = api.train(
        recipe(name, tmp_path / "initialized"), initialize_from=package
    )
    assert initialized.environment_steps == 128
    initial_metadata = json.loads(
        (model_locator(initialized.run_dir) / "checkpoint.json").read_text()
    )
    assert (
        initial_metadata["initial_weights_sha256"]
        == json.loads((model_locator(resumed.run_dir) / "checkpoint.json").read_text())[
            "final_weights_sha256"
        ]
    )
    assert primitive(config)["training"]["total_steps"] == 128
