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
    import numpy as np
    import torch
    from ray.rllib.core.columns import Columns
    from ray.rllib.utils.postprocessing.value_predictions import compute_value_targets

    from smartsom.learning.rllib_resource import ResourcePPOConfig, ScaleLearnerRewards

    # A successful episode and a failed episode retain their distinct tick costs.
    for raw_rewards in ((-2.0, -5.0), (-86.0, -9915.0)):
        rewards = torch.tensor(raw_rewards)
        actions = torch.tensor([1, 0])
        raw = {"machine_policy": {Columns.REWARDS: rewards, Columns.ACTIONS: actions}}
        scaled = ScaleLearnerRewards(0.0001)(batch=raw)
        assert rewards.tolist() == list(raw_rewards)
        assert scaled["machine_policy"][Columns.ACTIONS] is actions
        assert (
            scaled is not raw and scaled["machine_policy"] is not raw["machine_policy"]
        )
        targets = compute_value_targets(
            values=np.zeros(2, dtype=np.float32),
            rewards=scaled["machine_policy"][Columns.REWARDS].numpy(),
            terminateds=np.array([False, True]),
            truncateds=np.array([False, False]),
            gamma=1.0,
            lambda_=1.0,
        )
        assert targets[0] == pytest.approx(sum(raw_rewards) * 0.0001)
        prediction = torch.tensor(0.0, requires_grad=True)
        loss = (prediction - targets[0]).square().clamp(0, 10)
        loss.backward()
        assert 0 < loss.item() < 10 and prediction.grad.item() > 0

    config = ResourcePPOConfig(learner_reward_scale=0.0001)
    restored = ResourcePPOConfig.from_dict(config.to_dict())
    assert restored.learner_reward_scale == 0.0001
