"""Public RLlib catalogs for research candidate networks and learned waits.

RLlib owns PPO/DQN modules, exploration, losses, targets and optimizers. These
catalogs only construct observation encoders and neural prediction heads.
"""

from dataclasses import asdict

import torch
from ray.rllib.algorithms.dqn.dqn_catalog import DQNCatalog
from ray.rllib.algorithms.ppo.ppo_catalog import PPOCatalog
from ray.rllib.core.columns import Columns
from ray.rllib.core.models.base import ACTOR, CRITIC, ENCODER_OUT
from ray.rllib.core.models.configs import (
    ActorCriticEncoderConfig,
    MLPEncoderConfig,
    MLPHeadConfig,
)
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from ray.rllib.models.torch.torch_distributions import TorchCategorical
from torch import nn

from smartsom.learning.rllib_observation import SUPPLY_FIELDS
from smartsom.representations.v2.network import Network
from smartsom.representations.v2.social_config import social_config


def mlp_head(width, config, output=1):
    return MLPHeadConfig(
        input_dims=(width,),
        hidden_layer_dims=config["head_fcnet_hiddens"],
        hidden_layer_activation=config["head_fcnet_activation"],
        output_layer_dim=output,
    ).build(framework="torch")


class ActorFeatures(nn.Module):
    def forward(self, batch):
        return {
            ENCODER_OUT: {k: v for k, v in batch[Columns.OBS].items() if k != "critic"}
        }


class CriticFeatures(nn.Module):
    def forward(self, batch):
        return {ENCODER_OUT: batch[Columns.OBS]["critic"]}


class ActorCriticFeatures(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor_encoder = ActorFeatures()
        self.critic_encoder = CriticFeatures()

    def forward(self, batch):
        features = {ACTOR: self.actor_encoder(batch)[ENCODER_OUT]}
        if hasattr(self, "critic_encoder"):
            features[CRITIC] = self.critic_encoder(batch)[ENCODER_OUT]
        return {ENCODER_OUT: features}


class CandidateHead(nn.Module):
    def __init__(self, space, config):
        super().__init__()
        settings = social_config(config["research_social"])
        self.model = Network(
            config["research_group"],
            social_enabled=settings.kind == "implicit",
            concat_slots=settings.slots,
            value_head=False,
        )
        self.stages = tuple(config["research_stages"])
        # Attention embeds owner supply once; concat retains its candidate slots.
        self.expand_supply = settings.fusion == "concat"
        width = space["stages"][self.stages[0]]["context"].shape[0]
        # A true trainable waiting value, used in ordinary transitions too.
        # The constant waiting indicator also permits an empty local context.
        self.wait_head = mlp_head(width + 1, config)

    def forward(self, obs):
        mask = obs["action_mask"].bool()
        scores = torch.full(mask.shape, -torch.inf, device=mask.device)
        waiting = obs["waiting"].bool().reshape(-1)
        stage_index = obs["stage"].reshape(-1)
        for index, stage in enumerate(self.stages):
            indices = torch.where((stage_index == index) & ~waiting)[0]
            if len(indices):
                data = {k: v[indices] for k, v in obs["stages"][stage].items()}
                data = {
                    k: v.unsqueeze(1).expand(-1, mask.shape[1], *v.shape[1:])
                    if k in SUPPLY_FIELDS and self.expand_supply
                    else v
                    for k, v in data.items()
                }
                data = {
                    k: v.bool() if k.endswith("mask") else v.float()
                    for k, v in data.items()
                }
                data["indices"] = torch.arange(len(indices), device=mask.device)
                values = self.model.actor(
                    {
                        "size": len(indices),
                        "width": mask.shape[1],
                        "stages": {stage: data},
                    }
                )
                scores = scores.index_copy(0, indices, values)
            indices = torch.where((stage_index == index) & waiting)[0]
            if len(indices):
                context = obs["stages"][stage]["context"][indices].float()
                values = self.wait_head(
                    torch.cat((context, context.new_ones((len(indices), 1))), -1)
                ).squeeze(-1)
                scores[indices, 0] = values
        return scores.masked_fill(~mask, -torch.inf)


class LocalDuelingValue(nn.Module):
    """DQN value stream uses only permitted local/candidate observations."""

    def __init__(self, space, config):
        super().__init__()
        self.stages = tuple(config["research_stages"])
        self.heads = nn.ModuleDict(
            {
                stage: mlp_head(
                    space["stages"][stage]["context"].shape[-1]
                    + space["stages"][stage]["candidates"].shape[-1]
                    + 1,
                    config,
                )
                for stage in self.stages
            }
        )

    def forward(self, obs):
        mask = obs["action_mask"].bool()
        result = torch.zeros((mask.shape[0], 1), device=mask.device)
        for index, stage in enumerate(self.stages):
            indices = torch.where(obs["stage"].reshape(-1) == index)[0]
            if len(indices):
                data = obs["stages"][stage]
                valid = mask[indices].unsqueeze(-1)
                pooled = data["candidates"][indices].float().masked_fill(~valid, 0).sum(
                    1
                ) / valid.sum(1).clamp_min(1)
                values = self.heads[stage](
                    torch.cat(
                        (
                            data["context"][indices].float(),
                            pooled,
                            obs["waiting"][indices].float(),
                        ),
                        -1,
                    )
                )
                result = result.index_copy(0, indices, values)
        return result


class ResearchDQNCatalog(DQNCatalog):
    def __init__(self, observation_space, action_space, model_config_dict):
        if "critic" in observation_space.spaces:
            raise ValueError("DQN observation must not contain global critic inputs")
        self.observation_space = observation_space
        self.action_space = action_space
        self.config = asdict(DefaultModelConfig()) | model_config_dict

    def build_encoder(self, framework):
        return ActorFeatures()

    def build_af_head(self, framework):
        return CandidateHead(self.observation_space, self.config)

    def build_vf_head(self, framework):
        return LocalDuelingValue(self.observation_space, self.config)


class ResearchPPOCatalog(PPOCatalog):
    def __init__(self, observation_space, action_space, model_config_dict):
        self.observation_space = observation_space
        self.action_space = action_space
        self.config = asdict(DefaultModelConfig()) | model_config_dict
        width = observation_space["critic"].shape[-1]
        self.actor_critic_encoder_config = ActorCriticEncoderConfig(
            base_encoder_config=MLPEncoderConfig(input_dims=(width,)),
            shared=False,
        )

    def build_actor_critic_encoder(self, framework):
        return ActorCriticFeatures()

    def build_pi_head(self, framework):
        return CandidateHead(self.observation_space, self.config)

    def build_vf_head(self, framework):
        return mlp_head(self.observation_space["critic"].shape[-1], self.config)

    def get_action_dist_cls(self, framework):
        if framework != "torch":
            raise ValueError("research modules require the Torch framework")
        return TorchCategorical
