"""Reward-only extensions cannot silently change the network experiment factor."""

import importlib.util
import json
import os

import pytest
from test_training_sampling import (
    assert_state_equal,
    checkpoint_rows,
    optimizer_and_rng,
    small_training,
)

from smartsom.config.experiment import ExperimentConfig, prepare
from smartsom.config.extensions import ExtensionRef, ExtensionSpec, RewardSpec
from smartsom.experiments.training import train_one
from smartsom.experiments.training_audit import audit_training
from smartsom.experiments.training_controls import TrainingControls

pytestmark = pytest.mark.learning


@pytest.fixture(scope="module", params=["sb3", "rllib", "marl"])
def baseline(request, tmp_path_factory):
    name = request.param
    packages = ["torch", "gymnasium", "sb3_contrib" if name == "sb3" else "ray"]
    missing = [
        package for package in packages if importlib.util.find_spec(package) is None
    ]
    if missing:
        required = (
            "SMARTSOM_REQUIRE_MARL" if name == "marl" else "SMARTSOM_REQUIRE_LEARNING"
        )
        if os.environ.get(required) == "1":
            pytest.fail(f"required learning extras missing: {missing}")
        pytest.skip(f"optional learning extras missing: {missing}")
    resolved = small_training(name, tmp_path_factory.mktemp(f"reward-isolation-{name}"))
    controls = TrainingControls(validation=None, checkpoint_every_updates=None)
    plain = train_one(resolved, controls=controls)
    return name, resolved, controls, plain


def with_extensions(resolved, extensions):
    config = ExperimentConfig.model_validate_json(resolved.config_json)
    config.algorithm.extensions = extensions
    return prepare(config)


def test_identity_reward_preserves_initial_weights_optimizer_and_physical_trajectory(
    baseline,
):
    name, resolved, controls, plain = baseline
    result = train_one(
        with_extensions(resolved, ExtensionSpec(reward=RewardSpec())), controls=controls
    )
    a, b = [
        json.loads((run.checkpoint_dir / "checkpoint.json").read_text())
        for run in (plain, result)
    ]
    assert a["initial_weights_sha256"] == b["initial_weights_sha256"]
    assert a["final_weights_sha256"] == b["final_weights_sha256"]
    assert_state_equal(optimizer_and_rng(plain, name), optimizer_and_rng(result, name))
    original, extended = checkpoint_rows(plain), checkpoint_rows(result)
    for rows in (original, extended):
        for row in rows:
            for step in row["steps"]:
                values = step["reward_values"]
                # Explicit identity hooks additionally report each role's values.
                for role in values.pop("roles").values():
                    assert role == {
                        key: values[key] for key in ("raw", "research", "learner")
                    }
    assert [
        {key: value for key, value in row.items() if key != "extension_state"}
        for row in extended
    ] == [
        {key: value for key, value in row.items() if key != "extension_state"}
        for row in original
    ]
    assert audit_training(result.run_dir)["environment_steps"] == 128


def test_dict_fallback_network_is_frozen_in_framework_construction(baseline):
    name, resolved, controls, _ = baseline
    extended = with_extensions(
        resolved,
        ExtensionSpec(observation=ExtensionRef(name="builtin.dict", version="1")),
    )
    result = train_one(extended, controls=controls)
    if name == "sb3":
        from sb3_contrib import MaskablePPO

        model = MaskablePPO.load(result.checkpoint_dir / "model.zip", device="cpu")
        configured = [model.policy_kwargs["extensions"]]
    else:
        from smartsom.learning.production import LearnedProductionDriver

        driver = LearnedProductionDriver(
            result.checkpoint_dir, extended.resolved.scenario
        )
        try:
            configured = [
                module.model_config["extensions"] for module in driver.modules.values()
            ]
            assert set(driver.modules) == (
                {"default_policy"}
                if name == "rllib"
                else {"machine_policy", "agv_policy", "buffer_policy", "quality_policy"}
            )
        finally:
            driver.env.close()
    for spec in configured:
        assert spec["network"]["actor"]["encoder"]["code_sha256"]
        assert spec["network"]["critic"]["encoder"]["code_sha256"]
        assert spec["network"]["actor"]["hidden_sizes"] == [32, 32]
    assert audit_training(result.run_dir)["environment_steps"] == 128
