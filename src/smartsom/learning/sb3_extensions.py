"""MaskablePPO policy using registered independent actor and critic encoders."""

import gymnasium as gym
import torch
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

from smartsom.learning.extension_tensors import network_configuration, public_space
from smartsom.learning.torch_extensions import build_actor_critic


class ExtensionMaskableActorCriticPolicy(MaskableActorCriticPolicy):
    """Replace policy networks only; SB3 still owns distributions, PPO and optimizer."""

    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule,
        *,
        extensions,
        provider="sb3.maskable_ppo",
        role=None,
        fallback_hidden_sizes=(64, 64),
        **kwargs,
    ):
        if provider != "sb3.maskable_ppo" or role is not None:
            raise ValueError("SB3 extension policy requires the central SB3 provider")
        if not isinstance(action_space, gym.spaces.Discrete):
            raise ValueError(
                "SmartSOM extension policy requires fixed Discrete actions"
            )
        self.extension_config = extensions
        self.extension_provider = provider
        self.extension_role = role
        self.fallback_hidden_sizes = tuple(fallback_hidden_sizes)
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

    def _build(self, lr_schedule):
        spec = network_configuration(
            self.extension_config, self.extension_provider, self.fallback_hidden_sizes
        )
        self.network = build_actor_critic(
            public_space(self.observation_space),
            int(self.action_space.n),
            spec,
            self.extension_provider,
        )
        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )

    def get_distribution(self, obs, action_masks=None):
        logits = self.network.actor(obs)
        if not torch.isfinite(logits).all():
            raise ValueError("nonfinite extension actor logits")
        distribution = self.action_dist.proba_distribution(action_logits=logits)
        if action_masks is not None:
            distribution.apply_masking(action_masks)
        return distribution

    def predict_values(self, obs):
        return self.network.critic(obs)

    def forward(self, obs, deterministic=False, action_masks=None):
        distribution = self.get_distribution(obs, action_masks)
        actions = distribution.get_actions(deterministic=deterministic)
        return (
            actions.reshape((-1, *self.action_space.shape)),
            self.predict_values(obs),
            distribution.log_prob(actions),
        )

    def evaluate_actions(self, obs, actions, action_masks=None):
        distribution = self.get_distribution(obs, action_masks)
        return (
            self.predict_values(obs),
            distribution.log_prob(actions),
            distribution.entropy(),
        )

    def _get_constructor_parameters(self):
        return {
            **super()._get_constructor_parameters(),
            "extensions": self.extension_config,
            "provider": self.extension_provider,
            "role": self.extension_role,
            "fallback_hidden_sizes": self.fallback_hidden_sizes,
        }
