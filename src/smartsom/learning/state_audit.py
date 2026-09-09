"""Recompute counters and optimizer fingerprints from verified local state members."""

import hashlib

from smartsom.config.codec import digest


def _numeric_state(value):
    import numpy as np
    import torch

    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        if not np.isfinite(value).all():
            raise ValueError("nonfinite optimizer state")
        return {
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "sha256": hashlib.sha256(value.tobytes()).hexdigest(),
        }
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _numeric_state(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_numeric_state(child) for child in value]
    return value


def checkpoint_state_summary(provider, directory):
    if provider == "sb3.maskable_ppo":
        from stable_baselines3.common.save_util import load_from_zip_file

        data, parameters, _ = load_from_zip_file(directory / "model.zip", device="cpu")
        return {
            "environment_steps": data["num_timesteps"],
            "learner_updates": data["_n_updates"],
            "optimizer_sha256": digest(_numeric_state(parameters["policy.optimizer"])),
        }
    from smartsom.learning.training_state import load_state

    state = load_state(directory / "algorithm.pkl")
    optimizer = state["learner_group"]["learner"]["optimizer"]
    return {
        "environment_steps": state["env_runner"]["num_env_steps_sampled_lifetime"],
        "learner_updates": state["training_iteration"],
        "optimizer_sha256": digest(_numeric_state(optimizer)),
        "ppo_dynamic_sha256": digest(
            _numeric_state(load_state(directory / "ppo_dynamic.pkl"))
        ),
    }
