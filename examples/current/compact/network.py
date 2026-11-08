"""Candidate scoring for the named compact layout; reused by PPO and DQN."""

import torch
from torch import nn

from .features import SCHEMA, layout


def selected_layout(options):
    return layout(options.observation.params.get("social", False))


def pack(requests, *, critics, options):
    fields = selected_layout(options)
    width = sum(map(len, fields.values()))
    count = max(len(r.candidates) for r in requests)
    inputs = torch.zeros(len(requests), count, width)
    mask = torch.zeros(len(requests), count, dtype=torch.bool)
    for i, request in enumerate(requests):
        data = request.inputs
        size = len(request.candidates)
        if len(data["base"]) != len(fields["base"]) or any(
            len(data[key]) != size for key in ("candidates", "social")
        ):
            raise ValueError("compact observation shape does not match decision")
        for j in range(size):
            if any(
                len(data[key][j]) != len(fields[key])
                for key in ("candidates", "social")
            ):
                raise ValueError("compact observation feature width mismatch")
            inputs[i, j] = torch.tensor(
                (*data["base"], *data["candidates"][j], *data["social"][j])
            )
        mask[i, :size] = True
    return {
        "inputs": inputs,
        "mask": mask,
        "critic": torch.tensor(critics, dtype=torch.float32)
        if critics is not None
        else None,
    }


class Network(nn.Module):
    def __init__(self, input_width, width, value_head):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(input_width, width), nn.ReLU(), nn.Linear(width, 1)
        )
        self.critic = (
            nn.Sequential(nn.Linear(48, width), nn.ReLU(), nn.Linear(width, 1))
            if value_head
            else None
        )

    def actor(self, batch):
        return (
            self.score(batch["inputs"])
            .squeeze(-1)
            .masked_fill(~batch["mask"], -torch.inf)
        )

    def forward(self, batch):
        return self.actor(batch), self.critic(batch["critic"]).squeeze(-1)


def create_network(*, group, options, value_head):
    return Network(
        sum(map(len, selected_layout(options).values())), options.width, value_head
    )


create_network.INPUT_SCHEMA = SCHEMA
create_network.pack = pack
