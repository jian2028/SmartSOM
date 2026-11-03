"""One central candidate scorer and an independent native 48-field critic."""

import torch
from torch import nn

from smartsom.domain.base_decisions import FEATURES


def mlp(size, output):
    return nn.Sequential(
        nn.Linear(size, 64),
        nn.Tanh(),
        nn.Linear(64, 64),
        nn.Tanh(),
        nn.Linear(64, output),
    )


class Network(nn.Module):
    def __init__(self, context_size, token_size):
        super().__init__()
        self.prefix_encoder = nn.GRU(token_size + FEATURES, 32, batch_first=True)
        self.scorer = mlp(context_size + 32 + FEATURES, 1)
        self.critic = mlp(FEATURES, 1)

    def actor(self, batch):
        prefix, lengths = batch["prefix"], batch["prefix_lengths"]
        if prefix.shape[1]:
            output, _ = self.prefix_encoder(prefix)
            encoded = output[torch.arange(len(lengths)), (lengths - 1).clamp(min=0)]
            encoded = torch.where(
                (lengths > 0)[:, None], encoded, torch.zeros_like(encoded)
            )
        else:
            encoded = batch["context"].new_zeros((len(lengths), 32))
        context = torch.cat((batch["context"], encoded), -1)
        context = context[:, None, :].expand(-1, batch["candidates"].shape[1], -1)
        logits = self.scorer(torch.cat((context, batch["candidates"]), -1)).squeeze(-1)
        return logits.masked_fill(~batch["mask"], -torch.inf)

    def forward(self, batch):
        return self.actor(batch), self.critic(batch["critic"]).squeeze(-1)


def pack(requests, *, critics, options):
    count = max(len(r.candidates) for r in requests)
    prefix_count = max(len(r.inputs["prefix"]) for r in requests)
    token_width = len(requests[0].inputs["token"]) + FEATURES
    candidates = torch.zeros((len(requests), count, FEATURES))
    mask = torch.zeros((len(requests), count), dtype=torch.bool)
    prefix = torch.zeros((len(requests), prefix_count, token_width))
    for i, request in enumerate(requests):
        size = len(request.candidates)
        candidates[i, :size] = torch.tensor([c.features for c in request.candidates])
        mask[i, :size] = True
        if request.inputs["prefix"]:
            prefix[i, : len(request.inputs["prefix"])] = torch.tensor(
                request.inputs["prefix"]
            )
    return {
        "context": torch.tensor(
            [r.inputs["context"] for r in requests], dtype=torch.float32
        ),
        "candidates": candidates,
        "mask": mask,
        "prefix": prefix,
        "prefix_lengths": torch.tensor([len(r.inputs["prefix"]) for r in requests]),
        "critic": torch.tensor(critics, dtype=torch.float32)
        if critics is not None
        else None,
    }


def create_network(*, group, options, value_head, context_size=None, token_size=None):
    if group != "central" or not value_head:
        raise ValueError("this network requires the central PPO learner")
    return Network(context_size, token_size)


create_network.INPUT_SCHEMA = "smartsom.central-rule-observation/v1"
create_network.pack = pack
