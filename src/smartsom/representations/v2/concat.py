"""Parameter-free, lossless padded concatenation of candidate-linked relations.

Each valid element carries a mask bit. Slots are ranks in semantic-ID order,
not entity identifiers or simulator state. Overflow is an error, never truncation.
"""

import torch
from torch.nn import functional as F


def width(group, stage, slots):
    station = slots.stations * 10  # nine station features plus a validity bit
    if group == "dispatcher":
        return slots.vehicles * 4
    if group in ("input", "post") and stage == "accept":
        count = slots.post_jobs if group == "post" else slots.input_jobs
        condition = 4 if group == "post" else 3
        return count * (condition + station + 2 + 1) + (group == "post")
    return station + 2 + (station + 1 if group == "pre" and stage == "accept" else 0)


def padded(values, mask, capacity, name):
    if values.shape[-2] > capacity:
        raise ValueError(f"concat {name} exceeds configured slots ({capacity})")
    # Invalid payloads cannot influence a score, including nested job padding.
    valid = mask.unsqueeze(-1)
    rows = torch.cat([values.masked_fill(~valid, 0), valid.to(values.dtype)], -1)
    rows = F.pad(rows, (0, 0, 0, capacity - rows.shape[-2]))
    return rows.flatten(-2)


def features(batch, group, stage, slots):
    if group == "dispatcher":
        return padded(batch["peers"], batch["peer_mask"], slots.vehicles, "vehicles")
    if group in ("input", "post") and stage == "accept":
        peers = padded(
            batch["job_peers"], batch["job_peer_mask"], slots.stations, "stations"
        )
        jobs = torch.cat([batch["jobs"], peers, batch["job_meta"]], -1)
        count = slots.post_jobs if group == "post" else slots.input_jobs
        result = padded(jobs, batch["job_mask"], count, group + " jobs")
        if group == "post":
            result = torch.cat([result, batch["job_count"]], -1)
        return result
    parts = [
        padded(
            batch["downstream"], batch["downstream_mask"], slots.stations, "stations"
        ),
        batch["downstream_meta"],
    ]
    if group == "pre" and stage == "accept":
        parts.extend(
            [
                padded(
                    batch["alternatives"],
                    batch["alternatives_mask"],
                    slots.stations,
                    "stations",
                ),
                batch["alternatives_count"],
            ]
        )
    return torch.cat(parts, -1)
