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
        # This is the same first affine map over [global, numeric, three one-hots].
        # Compute the global term once and gather one-hot columns, avoiding a
        # repeated dense global vector and mostly-zero owner features per action.
        context, numeric = batch["global"], batch["candidates"]
        first = self.scorer[0]
        width = context.shape[-1] + numeric.shape[-1] + 3 * batch["owner_count"]
        if isinstance(first, nn.LazyLinear) and first.has_uninitialized_params():
            first.initialize_parameters(context.new_empty((1, width)))
        boundary = context.shape[-1]
        numeric_end = boundary + numeric.shape[-1]
        hidden = torch.nn.functional.linear(
            context, first.weight[:, :boundary], first.bias
        ).unsqueeze(1)
        hidden = hidden + torch.nn.functional.linear(
            numeric, first.weight[:, boundary:numeric_end]
        )
        for slot in range(3):
            identities = batch["associations"][:, :, slot]
            columns = first.weight[
                :,
                numeric_end + slot * batch["owner_count"] : numeric_end
                + (slot + 1) * batch["owner_count"],
            ].T
            selected = columns[identities.clamp(min=0)]
            hidden = hidden + selected * (identities >= 0).unsqueeze(-1)
        scores = self.scorer[1:](hidden).squeeze(-1)
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
    associations = torch.full((len(requests), count, 3), -1, dtype=torch.long)
    owners = requests[0].inputs["owners"]
    positions = {owner: index for index, owner in enumerate(owners)}
    for i, r in enumerate(requests):
        candidates[i, : len(r.candidates)] = torch.tensor(r.inputs["candidates"])
        mask[i, : len(r.candidates)] = True
        if r.inputs["owners"] != owners:
            raise ValueError("network observation owner layout mismatch")
        associations[i, : len(r.candidates)] = torch.tensor(
            [
                [positions.get(owner, -1) for owner in row]
                for row in r.inputs["associations"]
            ]
        )
    return {
        "global": torch.tensor(
            [r.inputs["global"] for r in requests], dtype=torch.float32
        ),
        "candidates": candidates,
        "mask": mask,
        "associations": associations,
        "owner_count": len(owners),
        "critics": critics,
        "layouts": [r.inputs["layout"] for r in requests],
    }


def create_network(*, group, options, value_head):
    return Network(value_head)


create_network.INPUT_SCHEMA = "smartsom.e1a-current/actor-v2"
create_network.CRITIC_SCHEMA = "smartsom.e1a-current/critic-v1"
create_network.pack = pack
# Each row is scored alone and padding is masked, so learners may pad
# minibatch rows in separate width buckets.
create_network.ROW_INDEPENDENT = True
