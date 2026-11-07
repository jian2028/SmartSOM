"""Network-owned value packing, including bootstrap without an Actor call."""

import torch


def values(model, inputs):
    if hasattr(model, "value"):
        return model.value(inputs)
    # The existing built-in and external vector critic contract.
    return model.critic(torch.tensor(inputs, dtype=torch.float32)).squeeze(-1)
