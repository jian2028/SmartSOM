"""Real public workflows: train, exact resume, evaluate and portable initialization."""

import importlib.util
import json
import os
import shutil

import pytest

from smartsom import api
from smartsom.config.codec import primitive
from smartsom.config.experiment import (
    PRESET_ROOT,
    apply_overrides,
    load_config,
    prepare,
)
from smartsom.experiments.packaging import (
    export_experiment,
    export_model,
    import_bundle,
    model_locator,
    verify_bundle,
)
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


def test_frozen_training_does_not_require_original_authoring_files(tmp_path):
    require_backend("sb3_micro")
    inputs = tmp_path / "inputs"
    shutil.copytree(PRESET_ROOT, inputs)
    config = apply_overrides(
        load_config(inputs / "configs/runs/learning_sb3.yaml"),
        [
            ("training.total_steps", 64),
            ("training.steps_per_update", 32),
            ("algorithm.batch_size", 16),
            ("algorithm.n_epochs", 2),
            ("output.root", str(tmp_path / "runs")),
            ("logging.verbose", 0),
        ],
    )
    frozen = prepare(config)
    shutil.rmtree(inputs)
    result = api.train_prepared(frozen)
    assert result.environment_steps == 64
    assert audit_training(result.training_dir)["status"] == "passed"


def test_public_parallel_configuration_survives_resume(tmp_path):
    require_backend("sb3_micro")
    config = recipe("sb3_micro", tmp_path / "parallel")
    config.runtime.num_envs = 2
    config.runtime.sampling_processes = 2
    config.runtime.numerical_threads = 2

    def stop(event):
        if event["stage"] == "validation" and event["ppo_updates"] == 2:
            return {"stop": "pruned"}

    partial = api.train(config, on_progress=stop)
    assert partial.environment_steps == 64
    resumed = api.resume(partial.run_dir)
    audit = audit_training(resumed.training_dir)
    assert audit["num_envs"] == 2 and audit["environment_steps"] == 128
    assert audit["ppo_updates"] == 4 and audit["budget_completed"]
    controls = json.loads((resumed.training_dir / "training_controls.json").read_text())
    assert controls["sampling_processes"] == controls["numerical_threads"] == 2


def test_external_validation_survives_missing_authoring_files_on_resume(tmp_path):
    require_backend("sb3_micro")
    inputs = tmp_path / "inputs"
    shutil.copytree(PRESET_ROOT, inputs)
    config = recipe("sb3_micro", tmp_path / "validation-runs")
    config.validation.scenarios = (
        str(inputs / "configs/scenarios/learning_micro.yaml"),
    )
    frozen = prepare(config)
    shutil.rmtree(inputs)

    def stop(event):
        if event["stage"] == "validation" and event["ppo_updates"] == 2:
            return {"stop": "pruned"}

    stopped = api.train_prepared(frozen, on_progress=stop)
    archive = export_experiment(stopped.run_dir, tmp_path / "experiment.zip")
    imported = import_bundle(archive, tmp_path / "moved-experiment")
    original = stopped.run_dir
    original.rename(tmp_path / "retained-original")
    resumed = api.resume(imported)
    assert resumed.run_dir == imported
    assert resumed.training_dir.is_relative_to(imported / "evidence/training")
    assert not original.exists()
    assert resumed.environment_steps == 128 and resumed.ppo_updates == 4
    original = json.loads(frozen.validation_json)
    for result in (resumed,):
        assert (
            json.loads((result.training_dir / "validation_inputs.json").read_text())
            == original
        )
        assert (
            json.loads((result.last_checkpoint / "validation_inputs.json").read_text())
            == original
        )
    assert audit_training(resumed.training_dir)["status"] == "passed"
