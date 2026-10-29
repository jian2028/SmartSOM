"""Shared semantic candidate scoring with an optional independent critic."""

import torch
from torch import nn

from smartsom.domain.base_decisions import FEATURES
from smartsom.representations.base.features import mlp


class CandidateHead(nn.Module):
    INPUT_SCHEMA = "smartsom.candidate-features/v1"

    def __init__(self, width, *, empty_wait=False, value_head=True):
        super().__init__()
        self.heads = nn.ModuleList(mlp(2 * width, width, 1) for _ in range(2))
        self.stop = mlp(width, width, 1)
        self.critic = mlp(FEATURES, width * 2, 1) if value_head else None
        self.empty_wait = mlp(width, width, 1) if empty_wait else None
        self.social = None

    def forward(self, batch, features):
        candidate, context = features
        combined = torch.cat(
            (candidate, context.unsqueeze(1).expand(-1, candidate.shape[1], -1)), dim=2
        )
        heads = torch.stack([head(combined).squeeze(-1) for head in self.heads], dim=-1)
        logits = heads.gather(
            2, batch["stage"][:, None, None].expand(-1, candidate.shape[1], 1)
        ).squeeze(-1)
        logits = torch.where(batch["stop"], self.stop(context), logits)
        if self.empty_wait is not None:
            logits = torch.where(batch["empty_wait"], self.empty_wait(context), logits)
        if self.social is not None:
            logits = self.social(batch, candidate, context, logits)
        logits = logits.masked_fill(~batch["mask"], -torch.inf)
        return logits
