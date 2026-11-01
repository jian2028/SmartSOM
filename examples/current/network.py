"""Example explicit PPO/DQN network; observation shape need not be 48."""

import torch
from torch import nn

INPUT_SCHEMA = "example.public-counts/v1"


def pack(requests, *, critics, options):
    count = max(len(r.candidates) for r in requests)
    candidates = torch.zeros(len(requests), count, 2)
    mask = torch.zeros(len(requests), count, dtype=torch.bool)
    for i, request in enumerate(requests):
        size = len(request.candidates)
        candidates[i, :size] = torch.tensor(request.inputs["candidates"])
        mask[i, :size] = True
    return {
        "context": torch.tensor(
            [r.inputs["context"] for r in requests], dtype=torch.float32
        ),
        "candidates": candidates,
        "mask": mask,
        "critic": torch.tensor(critics, dtype=torch.float32)
        if critics is not None
        else None,
    }


class Network(nn.Module):
    def __init__(self, width, value_head):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(5, width), nn.Tanh(), nn.Linear(width, 1))
        # Native PPO's privileged value channel is separate from actor inputs.
        self.critic = (
            nn.Sequential(nn.Linear(48, width), nn.Tanh(), nn.Linear(width, 1))
            if value_head
            else None
        )

    def actor(self, batch):
        context = (
            batch["context"].unsqueeze(1).expand(-1, batch["candidates"].shape[1], -1)
        )
        scores = self.score(torch.cat((context, batch["candidates"]), dim=-1)).squeeze(
            -1
        )
        return scores.masked_fill(~batch["mask"], -torch.inf)

    def forward(self, batch):
        return self.actor(batch), self.critic(batch["critic"]).squeeze(-1)


def create_network(*, group, options, value_head):
    return Network(options.width, value_head)


create_network.INPUT_SCHEMA = INPUT_SCHEMA
create_network.pack = pack
