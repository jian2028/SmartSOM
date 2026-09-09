"""Training-side wiring for public research hooks and model-only state exports."""

from dataclasses import replace

from smartsom.config.codec import digest, primitive
from smartsom.learning.extensions import (
    EncodedDecision,
    EncodedResourceDecision,
    bind_extensions,
)


def bind_training_extensions(resolved):
    spec = resolved.algorithm.algorithm
    pinned = bind_extensions(spec.extensions, spec.provider)
    if pinned == spec.extensions:
        return resolved
    return replace(
        resolved,
        algorithm=resolved.algorithm.model_copy(
            update={"algorithm": spec.model_copy(update={"extensions": pinned})}
        ),
    )


def environment_arguments(resolved):
    spec = resolved.algorithm.algorithm
    if spec.extensions is None:
        return {}
    resource = spec.provider == "rllib.resource_ppo"
    return {
        "extensions": spec.extensions,
        "learner_scale": spec.parameters.learner_reward_scale if resource else 1.0,
        **({} if resource else {"provider": spec.provider}),
    }


def learner_scale(resolved):
    spec = resolved.algorithm.algorithm
    scale = (
        spec.parameters.learner_reward_scale
        if spec.provider == "rllib.resource_ppo"
        else 1.0
    )
    return scale * (
        spec.extensions.reward.learner_scale
        if spec.extensions and spec.extensions.reward
        else 1.0
    )


def environment_state(env):
    if callable(getattr(env, "extension_state_dict", None)):
        return env.extension_state_dict()
    return getattr(env, "extension_state", None)


def restore_initial_state(env, state):
    if state is not None:
        env.extensions.load_state_dict(state["episode_initial_state"])


def verify_restored_state(env, state):
    if digest(environment_state(env)) != digest(state):
        raise ValueError(
            "replayed extension state/observation cache differs from checkpoint"
        )
    if state is not None:
        env.load_extension_state_dict(state)


def inference_view(env):
    """Use the already encoded decision; never advance stateful encoders twice."""
    if not env.extensions:
        return env.projected
    if hasattr(env, "possible_agents"):
        return EncodedResourceDecision(
            env.projected,
            tuple(
                EncodedDecision(view, env.encoded_observations[view.agent_id])
                for view in env.projected.views
            ),
        )
    return EncodedDecision(env.projected, env.encoded_observation)


def export_extension_metadata(directory, env, spec):
    if spec.extensions is None:
        return {}
    from smartsom.experiments.evidence import write_json
    from smartsom.learning.checkpoint import CheckpointFile, file_hash

    state = environment_state(env)
    if state is None:
        raise ValueError("extended inference export lacks its selected stream state")
    path = directory / "extension_state.json"
    write_json(path, state["runtime"])
    write_json(
        directory / "extension_selection.json",
        {
            "scope": "model-only",
            "stream_id": getattr(env, "stream_id", 0),
            "episode": env.episode_index,
            "state": "current stream runtime; evaluation begins an independent episode",
        },
    )
    return {
        "extensions": spec.extensions,
        "extension_state": CheckpointFile(path=path.name, sha256=file_hash(path)),
    }


def extension_step_record(step):
    if step.encoded_observations_sha256 is None:
        return {}
    return {
        "encoded_observations_sha256": step.encoded_observations_sha256,
        **(
            {"reward_values": primitive(step.reward_values)}
            if hasattr(step, "reward_values")
            else {
                "research_reward": step.research_reward,
                "learner_reward": step.learner_reward,
            }
        ),
    }


def inference_runtime_state(directory):
    import json

    from smartsom.learning.checkpoint import file_hash

    if not (directory / "checkpoint.json").exists():
        return None
    manifest = json.loads((directory / "checkpoint.json").read_text())
    member = manifest.get("extension_state")
    if member is None:
        return None
    path = (directory / member["path"]).resolve()
    if (
        not path.is_relative_to(directory.resolve())
        or file_hash(path) != member["sha256"]
    ):
        raise ValueError("inference extension state member digest mismatch")
    return json.loads(path.read_text())


def torch_observation_batch(value):
    import numpy as np
    import torch

    if isinstance(value, dict):
        return {key: torch_observation_batch(child) for key, child in value.items()}
    return torch.as_tensor(np.asarray(value, dtype=np.float32)).unsqueeze(0)


def stack_observations(values):
    import numpy as np

    if isinstance(values[0], dict):
        return {
            key: stack_observations([value[key] for value in values])
            for key in values[0]
        }
    return np.stack(values)


def uses_extension_network(extensions, observation_space):
    """Changing only rewards never changes a backend's existing vector network."""
    import gymnasium as gym

    return bool(extensions) and (
        extensions.network is not None
        or isinstance(observation_space, gym.spaces.Dict)
        or len(observation_space.shape) != 1
    )


def effective_network_extensions(extensions, provider, hidden_sizes):
    """Pin the actual fallback architecture in the framework construction snapshot."""
    from smartsom.config.extensions import NetworkBranch, NetworkSpec

    if extensions.network is None:
        branch = NetworkBranch(hidden_sizes=tuple(hidden_sizes), activation="tanh")
        extensions = extensions.model_copy(
            update={"network": NetworkSpec(actor=branch, critic=branch)}
        )
    return primitive(bind_extensions(extensions, provider))
