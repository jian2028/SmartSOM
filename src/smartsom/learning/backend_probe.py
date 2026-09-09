"""Observe constructed framework state without collecting data or updating PPO."""

import torch


def sb3_probe(model, controls):
    if model.num_timesteps != 0 or model._n_updates != 0:
        raise ValueError("backend probe unexpectedly sampled or updated PPO")
    return {
        "sampled_steps": model.num_timesteps,
        "learner_updates": model._n_updates,
        "device": str(model.device),
        "numerical_threads": torch.get_num_threads(),
        "num_envs": model.n_envs,
        "sampling_processes": controls.sampling_processes,
        "policy_parameters": {
            "policy": sum(parameter.numel() for parameter in model.policy.parameters())
        },
    }


def rllib_probe(algorithm, controls):
    learner = algorithm.learner_group._learner
    modules = {role: learner.module[role] for role in learner.module.keys()}
    state = algorithm.get_state()
    if (
        state["training_iteration"] != 0
        or state["env_runner"]["num_env_steps_sampled_lifetime"] != 0
    ):
        raise ValueError("backend probe unexpectedly sampled or updated PPO")
    devices = {
        str(parameter.device)
        for module in modules.values()
        for parameter in module.parameters()
    }
    if len(devices) != 1:
        raise ValueError("backend probe found inconsistent learner devices")
    return {
        "sampled_steps": 0,
        "learner_updates": 0,
        "device": devices.pop(),
        "numerical_threads": torch.get_num_threads(),
        "num_envs": controls.num_envs,
        "sampling_processes": controls.sampling_processes,
        "policy_parameters": {
            role: sum(parameter.numel() for parameter in module.parameters())
            for role, module in modules.items()
        },
    }
