"""Immutable Base request batching, shared by PPO and DQN."""

import torch

from smartsom.domain.base_decisions import FEATURES, SET_NAMES

STAGES = {"process": 0, "pickup": 0, "delivery": 1, "accept": 0, "service": 1}


def encode_requests(requests):
    """Cache immutable input tensors for one update, never learned embeddings."""
    return {
        id(r): (
            torch.tensor([c.features for c in r.candidates]),
            tuple(torch.tensor(values) for values in r.sets),
        )
        for r in requests
    }


def pack(requests, critics, count_scale, encoded=None, *, allow_empty_wait=False):
    n = len(requests)
    max_candidates = max(len(r.candidates) for r in requests)
    candidate = torch.zeros(n, max_candidates, FEATURES)
    mask = torch.zeros(n, max_candidates, dtype=torch.bool)
    stop = mask.clone()
    empty_wait = mask.clone()
    sets = []
    for index in range(len(SET_NAMES)):
        size = max(1, max(len(r.sets[index]) for r in requests))
        values = torch.zeros(n, size, FEATURES)
        valid = torch.zeros(n, size, dtype=torch.bool)
        for i, r in enumerate(requests):
            if r.sets[index]:
                values[i, : len(r.sets[index])] = (
                    encoded[id(r)][1][index]
                    if encoded is not None
                    else torch.tensor(r.sets[index])
                )
                valid[i, : len(r.sets[index])] = True
        sets.append((values, valid))
    for i, r in enumerate(requests):
        count = len(r.candidates)
        candidate[i, :count] = (
            encoded[id(r)][0]
            if encoded is not None
            else torch.tensor([c.features for c in r.candidates])
        )
        mask[i, :count] = True
        stop[i, :count] = torch.tensor([c.action == ("STOP",) for c in r.candidates])
        if empty_wait is not None:
            empty_wait[i, :count] = torch.tensor(
                [c.action == ("EMPTY_WAIT",) for c in r.candidates]
            )
    from smartsom.learning.social_actor import pack_social

    return {
        "social_active": torch.tensor([r.social is not None for r in requests]),
        "social": pack_social(requests, max_candidates, count_scale)
        if any(r.social is not None for r in requests)
        else None,
        "context": torch.tensor([r.context for r in requests]),
        "sets": sets,
        "candidates": candidate,
        "mask": mask,
        "stop": stop,
        "empty_wait": empty_wait,
        "stage": torch.tensor([STAGES[r.stage] for r in requests]),
        "critic": torch.tensor(critics, dtype=torch.float32)
        if critics is not None
        else None,
        "count_scale": count_scale,
    }
