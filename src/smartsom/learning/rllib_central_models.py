"""PR42 E1a heads within the official PPO module and loss."""

import numpy as np
import torch
from ray.rllib.connectors.common import AddObservationsFromEpisodesToBatch
from ray.rllib.connectors.connector_v2 import ConnectorV2
from ray.rllib.connectors.learner import AddColumnsFromEpisodesToTrainBatch
from ray.rllib.core.columns import Columns
from torch import nn

from smartsom.controllers.e1a.network import Network, mlp
from smartsom.learning.rllib_models import ResearchPPOCatalog


class CentralCandidateHead(nn.Module):
    def __init__(self, space, config):
        super().__init__()
        self.model = Network(False, config["advance_only_when_idle"])
        self.owner_count = config["owner_count"]
        width = space["global"].shape[0] + 15 + 3 * self.owner_count
        self.model.scorer[0].initialize_parameters(torch.empty(1, width))

    def forward(self, obs):
        rows, columns = torch.where(obs["valid"].bool())
        numeric = obs["candidates"][rows, columns].float()
        return self.model.actor(
            {
                "global": obs["global"].float(),
                "candidates": numeric,
                "rows": rows,
                "columns": columns,
                "width": obs["valid"].shape[-1],
                "advance": numeric[:, 0].bool(),
                "associations": obs["associations"][rows, columns].long(),
                "owner_count": self.owner_count,
                # Layout is validated at the semantic environment boundary.
                "layouts": (),
            }
        )


class CentralPPOCatalog(ResearchPPOCatalog):
    def build_actor_critic_encoder(self, framework):
        encoder = super().build_actor_critic_encoder(framework)
        if self.actor_critic_encoder_config.inference_only:
            del encoder.critic_encoder
        return encoder

    def build_pi_head(self, framework):
        return CentralCandidateHead(self.observation_space, self.config)

    def build_vf_head(self, framework):
        head = mlp()
        head[0].initialize_parameters(
            torch.empty(1, self.observation_space["critic"].shape[0])
        )
        return head


class CentralCandidates(ConnectorV2):
    """Pad only the current batch, retaining compact episode observations.

    This is a data-format adapter. Official connectors still perform bootstrap,
    GAE, device conversion and batching; PPO retains its distribution and loss.
    The scorer gathers real rows before neural work, as in PR42.
    """

    def __init__(self, *, learner=False):
        super().__init__()
        self.learner = learner
        self.observations = AddObservationsFromEpisodesToBatch(
            as_learner_connector=learner
        )
        self.columns = AddColumnsFromEpisodesToTrainBatch() if learner else None

    def __call__(self, *, rl_module, batch, episodes, **kwargs):
        args = dict(rl_module=rl_module, batch=batch, episodes=episodes, **kwargs)
        self.observations(**args)
        if self.columns is not None:
            self.columns(**args)
        items = batch[Columns.OBS]
        width = max(len(obs["valid"]) for rows in items.values() for obs in rows)
        old_logits = batch.get(Columns.ACTION_DIST_INPUTS, {})
        if old_logits:
            width = max(
                width, max(len(row) for rows in old_logits.values() for row in rows)
            )
        for key, rows in items.items():
            padded = []
            for obs in rows:
                count = len(obs["valid"])
                row = dict(obs)
                for name in ("candidates", "associations", "valid"):
                    value = obs[name]
                    padding = [(0, width - count)] + [(0, 0)] * (value.ndim - 1)
                    row[name] = np.pad(
                        value,
                        padding,
                        constant_values=-1 if name == "associations" else 0,
                    )
                padded.append(row)
            items[key] = padded
        for key, rows in old_logits.items():
            old_logits[key] = [
                np.pad(row, (0, width - len(row)), constant_values=-np.inf)
                for row in rows
            ]
        return batch


def central_env_connector(env, spaces, device):
    return CentralCandidates()


def central_learner_connector(observation_space, action_space):
    return CentralCandidates(learner=True)
