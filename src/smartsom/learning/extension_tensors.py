"""Optional tensor boundary shared by the three extension-enabled backends."""

import json

import gymnasium as gym
import numpy as np
import torch

from smartsom.config.extensions import ExtensionSpec, NetworkBranch, NetworkSpec
from smartsom.learning.extensions import ObservationSpace, bind_extensions


def public_space(space):
    """Read the actual declared Gym space, without consulting an authoring file."""
    if isinstance(space, gym.spaces.Box):
        return ObservationSpace(vector=tuple(space.shape))
    if isinstance(space, gym.spaces.Dict) and all(
        isinstance(value, gym.spaces.Box) for value in space.spaces.values()
    ):
        return ObservationSpace(
            fields=tuple(
                (key, tuple(value.shape)) for key, value in space.spaces.items()
            )
        )
    raise ValueError(
        "extensions require a fixed Box or a flat fixed-shape Dict of Boxes"
    )


def observation_tensor(value, *, device=None, add_batch=False):
    """Preserve Dict structure while converting the public float32 wire value."""
    if isinstance(value, dict):
        return {
            key: observation_tensor(item, device=device, add_batch=add_batch)
            for key, item in value.items()
        }
    tensor = torch.as_tensor(np.asarray(value, dtype=np.float32), device=device)
    if not torch.isfinite(tensor).all():
        raise ValueError("nonfinite encoded observations")
    return tensor.unsqueeze(0) if add_batch else tensor


def network_configuration(extensions, provider, fallback_hidden_sizes):
    if not isinstance(extensions, dict):
        raise ValueError("backend extensions must be a JSON object")
    spec = ExtensionSpec.model_validate_json(json.dumps(extensions, allow_nan=False))
    bound = bind_extensions(spec, provider)
    if bound != spec:
        raise ValueError(
            "backend extensions must include pinned implementation digests"
        )
    if spec.network is not None:
        return spec.network
    branch = NetworkBranch(hidden_sizes=tuple(fallback_hidden_sizes), activation="tanh")
    # The fallback is explicit in each framework's serialized construction args.
    return bind_extensions(
        ExtensionSpec(network=NetworkSpec(actor=branch, critic=branch)), provider
    ).network
