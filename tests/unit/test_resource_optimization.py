"""Optimization units stay outside simulator rewards and checkpoint semantics."""

import importlib.util
import os

import pytest
from pydantic import ValidationError

from smartsom.config.models import PPOParameters, ResourcePPOParameters


@pytest.mark.parametrize("value", [0, -1, True, float("inf"), float("nan")])
def test_resource_reward_scale_rejects_invalid_values(value):
    with pytest.raises(ValidationError):
        ResourcePPOParameters(learner_reward_scale=value)


def test_default_resource_parameters_preserve_old_serialized_identity():
    assert ResourcePPOParameters().model_dump() == PPOParameters().model_dump()
    parameters = ResourcePPOParameters(learner_reward_scale=0.0001)
    assert ResourcePPOParameters.model_validate_json(parameters.model_dump_json()) == (
        parameters
    )
    with pytest.raises(ValidationError):
        PPOParameters(learner_reward_scale=0.0001)


def test_scaled_training_targets_keep_raw_rewards_and_value_gradients():
    if not all(importlib.util.find_spec(p) for p in ("ray", "torch", "pettingzoo")):
        if os.environ.get("SMARTSOM_REQUIRE_MARL") == "1":
            pytest.fail("required MARL backend is absent")
        pytest.skip("optional MARL backend is absent")
    from types import SimpleNamespace

    import torch
    from ray.rllib.core.columns import Columns
    from ray.rllib.evaluation.postprocessing import Postprocessing

    from smartsom.learning.production_ray import PhysicalGAE

    for raw_rewards in ((-2.0, -5.0), (-86.0, -9915.0)):
        rewards = torch.tensor(raw_rewards)
        actions = torch.tensor([1, 0])
        data = {
            Columns.REWARDS: rewards,
            Columns.ACTIONS: actions,
            Columns.OBS: {"physical_tick": torch.tensor([0.0, 1.0])},
            Columns.TERMINATEDS: torch.tensor([False, True]),
            Columns.TRUNCATEDS: torch.tensor([False, False]),
        }
        module = SimpleNamespace(compute_values=lambda batch: torch.zeros(2))
        result = PhysicalGAE(1.0, 1.0, reward_scale=0.0001)(
            rl_module={"machine_policy": module}, batch={"machine_policy": data}
        )["machine_policy"]
        assert rewards.tolist() == list(raw_rewards)
        assert torch.equal(result[Columns.ACTIONS], actions)
        targets = result[Postprocessing.VALUE_TARGETS]
        assert targets[0] == pytest.approx(sum(raw_rewards) * 0.0001)
        assert targets[1] == pytest.approx(raw_rewards[1] * 0.0001)
        prediction = torch.tensor(0.0, requires_grad=True)
        loss = (prediction - targets[0]).square().clamp(0, 10)
        loss.backward()
        assert 0 < loss.item() < 10 and prediction.grad.item() > 0
