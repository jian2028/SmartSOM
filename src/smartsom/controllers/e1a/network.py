"""Shared 64x64 candidate scorer; separate 64x64 value network."""

import numpy
import torch
from torch import nn


def mlp():
    return nn.Sequential(
        nn.LazyLinear(64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1)
    )


class Network(nn.Module):
    def __init__(self, value_head, advance_only_when_idle=False):
        super().__init__()
        self.layout = None
        # Offer ADVANCE only while no work task is legal (a mask, so the engine
        # candidate set and every other score are unchanged).
        self.advance_only_when_idle = advance_only_when_idle
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
        # Candidates are flat (no padding); scores are scattered into a padded
        # logit matrix whose empty cells are -inf.
        context, numeric, rows = batch["global"], batch["candidates"], batch["rows"]
        first = self.scorer[0]
        width = context.shape[-1] + numeric.shape[-1] + 3 * batch["owner_count"]
        if isinstance(first, nn.LazyLinear) and first.has_uninitialized_params():
            first.initialize_parameters(context.new_empty((1, width)))
        boundary = context.shape[-1]
        numeric_end = boundary + numeric.shape[-1]
        hidden = torch.nn.functional.linear(
            context, first.weight[:, :boundary], first.bias
        )[rows]
        hidden = hidden + torch.nn.functional.linear(
            numeric, first.weight[:, boundary:numeric_end]
        )
        for slot in range(3):
            identities = batch["associations"][:, slot]
            columns = first.weight[
                :,
                numeric_end + slot * batch["owner_count"] : numeric_end
                + (slot + 1) * batch["owner_count"],
            ].T
            selected = columns[identities.clamp(min=0)]
            hidden = hidden + selected * (identities >= 0).unsqueeze(-1)
        scores = self.scorer[1:](hidden).squeeze(-1)
        if self.advance_only_when_idle:
            advance = batch["advance"]
            busy = torch.zeros(len(context), dtype=torch.bool, device=rows.device)
            busy[rows[~advance]] = True
            scores = scores.masked_fill(advance & busy[rows], -torch.inf)
        logits = scores.new_full((len(context), batch["width"]), -torch.inf)
        return logits.index_put((rows, batch["columns"]), scores)

    def value(self, inputs):
        self.check_layouts([v["layout"] for v in inputs])
        return self.critic(
            torch.from_numpy(
                numpy.array([v["global"] for v in inputs], dtype=numpy.float32)
            )
        ).squeeze(-1)

    def forward(self, batch):
        return self.actor(batch), self.value(batch["critics"])


def pack(requests, *, critics, options):
    owners = requests[0].inputs["owners"]
    positions = {owner: index for index, owner in enumerate(owners)}
    counts, numeric, associations, advance = [], [], [], []
    for r in requests:
        if r.inputs["owners"] != owners:
            raise ValueError("network observation owner layout mismatch")
        counts.append(len(r.candidates))
        numeric.extend(r.inputs["candidates"])
        associations.extend(r.inputs["associations"])
        advance.extend(c.kind == "advance" for c in r.candidates)
    # numpy converts nested Python floats several times faster than torch.
    counts = torch.tensor(counts)
    rows = torch.repeat_interleave(torch.arange(len(requests)), counts)
    starts = torch.cumsum(counts, 0) - counts
    return {
        "global": torch.from_numpy(
            numpy.array([r.inputs["global"] for r in requests], dtype=numpy.float32)
        ),
        "candidates": torch.from_numpy(numpy.array(numeric, dtype=numpy.float32)),
        "rows": rows,
        "columns": torch.arange(len(rows)) - starts[rows],
        "width": int(counts.max()),
        "advance": torch.tensor(advance, dtype=torch.bool),
        "associations": torch.tensor(
            [[positions.get(owner, -1) for owner in row] for row in associations],
            dtype=torch.long,
        ).view(len(rows), 3),
        "owner_count": len(owners),
        "critics": critics,
        "layouts": [r.inputs["layout"] for r in requests],
    }


def create_network(*, group, options, value_head, advance_only_when_idle=False):
    if not isinstance(advance_only_when_idle, bool):
        raise ValueError("advance_only_when_idle must be true or false")
    if advance_only_when_idle and not value_head:
        # Value-based learners explore uniformly over the request candidates,
        # which would bypass a network-side mask.
        raise ValueError("advance_only_when_idle requires a policy-gradient learner")
    return Network(value_head, advance_only_when_idle)


create_network.INPUT_SCHEMA = "smartsom.e1a-current/actor-v2"
create_network.CRITIC_SCHEMA = "smartsom.e1a-current/critic-v1"
create_network.pack = pack
