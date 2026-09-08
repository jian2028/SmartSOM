"""Optional tensor evidence helpers, imported only by actual learner backends."""

import hashlib

import numpy as np
import torch


def weights_digest(state) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        if isinstance(value, dict):
            digest.update(key.encode() + weights_digest(value).encode())
            continue
        array = (
            value.detach().cpu().numpy()
            if isinstance(value, torch.Tensor)
            else np.asarray(value)
        )
        if not np.isfinite(array).all():
            raise ValueError(f"nonfinite learner parameters: {key}")
        digest.update(
            key.encode()
            + str(array.dtype).encode()
            + repr(array.shape).encode()
            + array.tobytes()
        )
    return digest.hexdigest()
