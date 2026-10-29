"""Base feature encoder. Input and output contracts are independent of learners."""

import torch
from torch import nn

from smartsom.domain.base_decisions import FEATURES, SET_NAMES

INPUT_SCHEMA = "smartsom.base-observation/native-v2"
OUTPUT_SCHEMA = "smartsom.candidate-features/v1"


def mlp(input_size, width, output):
    return nn.Sequential(
        nn.Linear(input_size, width), nn.Tanh(), nn.Linear(width, output)
    )


class BaseEncoder(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.width = width
        self.encoders = nn.ModuleList(mlp(FEATURES, width, width) for _ in SET_NAMES)
        self.context = mlp(
            FEATURES + len(SET_NAMES) * (2 * width + 1), width * 2, width
        )
        self.candidate = mlp(FEATURES, width, width)

    def forward(self, batch):
        summaries = []
        for encoder, (values, mask) in zip(self.encoders, batch["sets"], strict=True):
            encoded = encoder(values)
            count = mask.sum(1, keepdim=True)
            mean = (encoded * mask.unsqueeze(-1)).sum(1) / count.clamp(min=1)
            maximum = encoded.masked_fill(~mask.unsqueeze(-1), -torch.inf).max(1).values
            maximum = torch.where(count > 0, maximum, torch.zeros_like(maximum))
            summaries.extend((mean, maximum, count / batch["count_scale"]))
        context = self.context(torch.cat((batch["context"], *summaries), dim=1))
        candidate = self.candidate(batch["candidates"])
        return candidate, context
