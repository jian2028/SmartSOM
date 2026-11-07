"""Shared 64x64 candidate scorer; separate 64x64 value network."""

import torch
from torch import nn


def mlp():
    return nn.Sequential(
        nn.LazyLinear(64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1)
    )


class Network(nn.Module):
    def __init__(self, value_head):
        super().__init__()
        self.layout = None
        self.scorer = mlp()
        if value_head:
            self.critic = mlp()

    def get_extra_state(self):
        return self.layout

    def set_extra_state(self, state):
        self.layout = state

    def check_layouts(self, layouts):
        for layout in layouts:
            if self.layout is not None and self.layout != layout:
                raise ValueError("network observation layout mismatch")
            self.layout = layout

    def actor(self, batch):
        self.check_layouts(batch["layouts"])
        context = (
            batch["global"].unsqueeze(1).expand(-1, batch["candidates"].shape[1], -1)
        )
        scores = self.scorer(torch.cat((context, batch["candidates"]), -1)).squeeze(-1)
        return scores.masked_fill(~batch["mask"], -torch.inf)

    def value(self, inputs):
        self.check_layouts([v["layout"] for v in inputs])
        return self.critic(
            torch.tensor([v["global"] for v in inputs], dtype=torch.float32)
        ).squeeze(-1)

    def forward(self, batch):
        return self.actor(batch), self.value(batch["critics"])


def pack(requests, *, critics, options):
    count = max(len(r.candidates) for r in requests)
    candidates = torch.zeros(
        len(requests), count, len(requests[0].inputs["candidates"][0])
    )
    mask = torch.zeros(len(requests), count, dtype=torch.bool)
    for i, r in enumerate(requests):
        candidates[i, : len(r.candidates)] = torch.tensor(r.inputs["candidates"])
        mask[i, : len(r.candidates)] = True
    return {
        "global": torch.tensor(
            [r.inputs["global"] for r in requests], dtype=torch.float32
        ),
        "candidates": candidates,
        "mask": mask,
        "critics": critics,
        "layouts": [r.inputs["layout"] for r in requests],
    }


def create_network(*, group, options, value_head):
    return Network(value_head)


create_network.INPUT_SCHEMA = "smartsom.e1a-current/actor-v1"
create_network.CRITIC_SCHEMA = "smartsom.e1a-current/critic-v1"
create_network.pack = pack
