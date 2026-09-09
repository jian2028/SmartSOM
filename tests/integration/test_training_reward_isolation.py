"""Reward-only extensions cannot silently change the network experiment factor."""

import importlib.util
import json
import os
from dataclasses import replace

import pytest
from test_training_sampling import small_training

from smartsom.config.extensions import ExtensionRef, ExtensionSpec, RewardSpec
from smartsom.experiments.training import train_one
from smartsom.experiments.training_audit import audit_training
from smartsom.experiments.training_controls import TrainingControls

pytestmark = pytest.mark.learning


@pytest.fixture(scope="module", params=["sb3", "rllib", "marl"])
def baseline(request, tmp_path_factory):
    name = request.param
    packages = ["torch", "gymnasium", "sb3_contrib" if name == "sb3" else "ray"]
    if name == "marl":
        packages.append("pettingzoo")
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
    return replace(
        resolved,
        algorithm=resolved.algorithm.model_copy(
            update={
                "algorithm": resolved.algorithm.algorithm.model_copy(
                    update={"extensions": extensions}
                )
            }
        ),
    )


def test_identity_reward_preserves_initial_weights_optimizer_and_physical_trajectory(
    baseline,
):
    _, resolved, controls, plain = baseline
    result = train_one(
        with_extensions(resolved, ExtensionSpec(reward=RewardSpec())), controls=controls
    )
    a, b = [
        json.loads((run.checkpoint_dir / "checkpoint.json").read_text())
        for run in (plain, result)
    ]
    assert a["initial_weights_sha256"] == b["initial_weights_sha256"]
    assert a["final_weights_sha256"] == b["final_weights_sha256"]
    assert (plain.last_checkpoint / "state_summary.json").read_bytes() == (
        result.last_checkpoint / "state_summary.json"
    ).read_bytes()
    original = [
        json.loads(line)
        for line in (plain.run_dir / "episodes.jsonl").read_text().splitlines()
    ]
    extended = [
        json.loads(line)
        for line in (result.run_dir / "episodes.jsonl").read_text().splitlines()
    ]
    assert [
        {
            key: value
            for key, value in row.items()
            if key not in ("extension_state", "extension_steps")
        }
        for row in extended
    ] == original
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
        from ray.rllib.core.rl_module.rl_module import RLModule

        names = ["module"] if name == "rllib" else ["machine_policy", "agv_policy"]
        configured = [
            RLModule.from_checkpoint(result.checkpoint_dir / role).model_config[
                "extensions"
            ]
            for role in names
        ]
    for spec in configured:
        assert spec["network"]["actor"]["encoder"]["code_sha256"]
        assert spec["network"]["critic"]["encoder"]["code_sha256"]
        assert spec["network"]["actor"]["hidden_sizes"] == [32, 32]
    assert audit_training(result.run_dir)["environment_steps"] == 128
