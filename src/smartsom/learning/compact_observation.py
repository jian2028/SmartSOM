"""Version-two public fields with independent actor/Q and value inputs."""

import copy
import json

import numpy as np
import torch
from torch import nn

from smartsom.config.codec import digest, primitive
from smartsom.config.extensions import ActorCriticSpec
from smartsom.config.policies import QNetwork
from smartsom.learning.extensions import ObservationSpace
from smartsom.learning.production_models import PREFIX_WIDTH, CandidateNetwork
from smartsom.learning.registered_observation import (
    CONTEXT_ENCODERS,
    PublicFieldsEncoder,
)
from smartsom.learning.torch_extensions import _Branch


class CompactPublicFieldsEncoder(PublicFieldsEncoder):
    DIAGNOSTIC_CONTRACT = {
        **PublicFieldsEncoder.DIAGNOSTIC_CONTRACT,
        "schema": "smartsom.public-fields/v2",
        "context_planes": ["compact actor/Q", "compact critic (PPO only)"],
    }

    def __init__(self, *args, algorithm="ppo", **kwargs):
        super().__init__(*args, **kwargs)
        if algorithm not in ("ppo", "dqn"):
            raise ValueError("unsupported compact observation algorithm")
        self.algorithm = algorithm
        self.layout = []
        offset = 0
        sizes = {}
        for branch in ("actor", "critic"):
            start = offset
            if branch == "actor" or algorithm == "ppo":
                for selection in self.settings.context:
                    if selection.visibility not in (branch, "both"):
                        continue
                    width = self._width(selection)
                    self.layout.append(
                        dict(
                            component=primitive(selection),
                            branch=branch,
                            offset=offset,
                            width=width,
                        )
                    )
                    offset += width
            sizes[branch] = offset - start
        self.context_layout = sizes
        if algorithm == "ppo" and sizes["critic"] == 0:
            raise ValueError("compact PPO requires critic-visible context")
        self.context_size = offset
        self.observation_identity = (
            "smartsom.public-fields/v2/"
            + digest(primitive(self.settings))
            + "/"
            + algorithm
        )

    def state_dict(self):
        return {
            **super().state_dict(),
            "algorithm": self.algorithm,
            "context_layout": copy.deepcopy(self.context_layout),
        }

    def _context(self, request, view):
        # Compute each selected component once. No equal-width zero planes are
        # constructed, copied into packets or retained in DQN replay.
        values_by_name = {}
        for selection in self.settings.context:
            if self.algorithm == "dqn" and selection.visibility == "critic":
                continue
            rows = np.asarray(
                CONTEXT_ENCODERS[selection.name](self, request, view, selection),
                dtype=np.float32,
            ).reshape(-1, self.widths[selection.name])
            if selection.aggregation == "per_entity":
                values = rows.reshape(-1)
            elif len(rows):
                values = getattr(rows, selection.aggregation)(axis=0)
            else:
                values = np.zeros(self.widths[selection.name], np.float32)
            if (
                selection.applies_to is not None
                and request.role not in selection.applies_to
            ):
                values = np.zeros_like(values)
            values_by_name[selection.name] = values
        return (
            np.concatenate(
                [values_by_name[row["component"]["name"]] for row in self.layout]
            ).astype(np.float32, copy=False)
            if self.layout
            else np.empty(0, np.float32)
        )


class CompactCandidateNetwork(CandidateNetwork):
    DIAGNOSTIC_CONTRACT = {
        **CandidateNetwork.DIAGNOSTIC_CONTRACT,
        "schema": "smartsom.scoped-candidate/v2",
        "context_planes": ["compact actor/Q only", "compact critic only"],
        "critic_prefix": "absent",
        "critic_candidate_pool": "absent",
    }

    def __init__(
        self,
        context_size,
        network,
        provider,
        algorithm="ppo",
        *,
        context_layout,
        central=False,
        central_private_end=0,
        candidate_width=16,
    ):
        nn.Module.__init__(self)
        if (
            set(context_layout) != {"actor", "critic"}
            or any(
                type(value) is not int or value < 0 for value in context_layout.values()
            )
            or sum(context_layout.values()) != context_size
        ):
            raise ValueError("compact context layout mismatch")
        if algorithm not in ("ppo", "dqn") or (
            algorithm == "dqn" and context_layout["critic"]
        ):
            raise ValueError("compact DQN must not include critic context")
        self.context_size, self.algorithm, self.central = (
            context_size,
            algorithm,
            central,
        )
        self.central_private_end = central_private_end
        self.candidate_width = candidate_width
        self.actor_context_size = context_layout["actor"]
        self.critic_context_size = context_layout["critic"]
        self.actor_prefix = nn.GRU(candidate_width, PREFIX_WIDTH, batch_first=True)
        actor_space = ObservationSpace(
            vector=(self.actor_context_size + PREFIX_WIDTH + candidate_width,)
        )
        if algorithm == "dqn":
            selected = QNetwork.model_validate_json(json.dumps(network))
            self.actor = _Branch(actor_space, 1, selected.q, provider)
            self.critic = None
        else:
            selected = ActorCriticSpec.model_validate_json(json.dumps(network))
            self.actor = _Branch(actor_space, 1, selected.actor, provider)
            self.critic = _Branch(
                ObservationSpace(vector=(self.critic_context_size,)),
                1,
                selected.critic,
                provider,
            )

    def forward(self, observations):
        context = observations["context"].float()
        if context.shape[-1] != self.context_size:
            raise ValueError("compact context shape mismatch")
        actor = context[:, : self.actor_context_size]
        prefix = self.prefix_embedding(
            observations["prefix"].float(),
            observations["prefix_length"].long(),
            self.actor_prefix,
        )
        candidates = observations["candidates"].float()
        expanded = (
            torch.cat((actor, prefix), -1)
            .unsqueeze(1)
            .expand(
                -1,
                candidates.shape[1],
                -1,
            )
        )
        scores = self.actor(torch.cat((expanded, candidates), -1)).squeeze(-1)
        scores = scores.masked_fill(
            ~observations["mask"].bool(),
            torch.finfo(scores.dtype).min,
        )
        if self.critic is None:
            return scores, scores.max(-1).values
        return scores, self.critic(context[:, self.actor_context_size :]).squeeze(-1)
