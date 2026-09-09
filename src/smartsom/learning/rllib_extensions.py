"""RLlib PPO module with separate registered actor and critic feed-forward paths."""

import gymnasium as gym
import torch
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.apis import ValueFunctionAPI
from ray.rllib.core.rl_module.torch.torch_rl_module import TorchRLModule

from smartsom.learning.extension_tensors import network_configuration, public_space
from smartsom.learning.torch_extensions import build_actor_critic


class ExtensionPPOTorchRLModule(TorchRLModule, ValueFunctionAPI):
    """All three forward modes retain RLlib's ordinary PPO loss/distribution API."""

    def setup(self):
        if (
            not isinstance(self.observation_space, gym.spaces.Dict)
            or set(self.observation_space.spaces) != {"observations", "action_mask"}
            or not isinstance(self.action_space, gym.spaces.Discrete)
        ):
            raise ValueError(
                "extension RLModule requires masked fixed Discrete observations"
            )
        config = self.model_config
        provider, role = config["provider"], config.get("role")
        if (
            provider not in ("rllib.ppo", "rllib.resource_ppo")
            or (provider == "rllib.ppo" and role is not None)
            or (
                provider == "rllib.resource_ppo"
                and role not in ("machine_policy", "agv_policy")
            )
        ):
            raise ValueError("extension RLModule provider/role is incompatible")
        spec = network_configuration(
            config["extensions"], provider, config["fallback_hidden_sizes"]
        )
        self.network = build_actor_critic(
            public_space(self.observation_space["observations"]),
            int(self.action_space.n),
            spec,
            provider,
            role=role,
        )

    @staticmethod
    def _unpack(batch):
        observation = batch[Columns.OBS]
        if isinstance(observation, dict) and set(observation) == {
            "observations",
            "action_mask",
        }:
            return observation["observations"], observation["action_mask"]
        # Some connector paths split the mask into its own batch column.
        return observation, batch["action_mask"]

    def _forward(self, batch, **kwargs):
        observations, mask = self._unpack(batch)
        logits = self.network.actor(observations)
        if not torch.isfinite(logits).all() or logits.shape != mask.shape:
            raise ValueError("nonfinite logits or mismatched extension action mask")
        if not torch.isfinite(mask).all() or not ((mask == 0) | (mask == 1)).all():
            raise ValueError("extension action mask must contain only zero or one")
        # Learner bootstrap rows may legitimately contain a terminal all-zero mask.
        return {
            Columns.ACTION_DIST_INPUTS: logits.masked_fill(
                mask == 0, torch.finfo(logits.dtype).min
            )
        }

    def compute_values(self, batch, embeddings=None):
        observations, _ = self._unpack(batch)
        return self.network.critic(observations).squeeze(-1)
