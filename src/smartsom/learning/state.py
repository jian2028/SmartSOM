"""Atomic complete-state files with component/scientific identity checks."""

from pathlib import Path

import torch


def save(path, options, state):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    torch.save(
        {
            "contract": "smartsom.component-checkpoint/v2",
            "identity": options.science_identity(),
            "state": state,
        },
        temporary,
    )
    temporary.replace(path)


def load(path, options):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (
        payload.get("contract") != "smartsom.component-checkpoint/v2"
        or payload.get("identity") != options.science_identity()
    ):
        raise ValueError(
            "checkpoint contract mismatch; use the original archived source"
        )
    return payload["state"]
