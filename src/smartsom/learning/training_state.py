"""Locked-framework state adapters for complete single-environment update points.

These adapters extend framework checkpoints, not the PPO loss or simulator. Pickle
members are local trusted training artifacts and are verified before loading.
"""

import copy
import pickle
import random
from contextlib import contextmanager
from pathlib import Path

from smartsom.config.codec import digest
from smartsom.learning.weights import weights_digest


def rng_state():
    import numpy as np
    import torch

    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_initialized():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng(state):
    import numpy as np
    import torch

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


@contextmanager
def isolated_rng():
    import torch

    state = rng_state()
    numerical_threads = torch.get_num_threads()
    try:
        yield
    finally:
        restore_rng(state)
        torch.set_num_threads(numerical_threads)


def dump_state(path: Path, state):
    with path.open("xb") as stream:
        pickle.dump(state, stream, protocol=5)


def load_state(path: Path):
    with path.open("rb") as stream:
        return pickle.load(stream)


def episode_state(env):
    resource = hasattr(env, "possible_agents")
    return {
        "episode_index": env.episode_index,
        "indices": [s.indices if resource else s.action_index for s in env.steps],
        "trace_sha256": digest(env.simulator.trace),
        "bindings_sha256": digest(env.projection.bindings),
        "steps_sha256": digest(env.steps),
        "return": env.total_reward,
        "reason": env.reason,
    }


def restore_episode(env, state):
    """Replay the current prefix without learner calls or duplicate evidence."""
    notify = env.on_episode
    progress = getattr(env, "step_progress", None)
    env.on_episode = None
    if hasattr(env, "step_progress"):
        env.step_progress = None
    try:
        env.episode_index = state["episode_index"] - 1
        env.reset()
        for action in state["indices"]:
            env.step(dict(action) if hasattr(env, "possible_agents") else action)
        if episode_state(env) != state:
            raise ValueError("restored activity differs from checkpoint episode prefix")
    finally:
        env.on_episode = notify
        if hasattr(env, "step_progress"):
            env.step_progress = progress


class SB3TrainingState:
    def __init__(self, model, env):
        self.model, self.env = model, env

    def hashes(self):
        return {"policy": weights_digest(self.model.policy.state_dict())}

    def initialize(self, directory):
        from sb3_contrib import MaskablePPO

        loaded = MaskablePPO.load(directory / "model.zip", device="cpu")
        self.model.policy.load_state_dict(loaded.policy.state_dict(), strict=True)

    def export(self, directory):
        from sb3_contrib import MaskablePPO

        directory.mkdir()
        self.model.save(directory / "model.zip")
        loaded = MaskablePPO.load(directory / "model.zip", device="cpu")
        if weights_digest(loaded.policy.state_dict()) != self.hashes()["policy"]:
            raise ValueError("SB3 inference export changed weights")

    def save(self, directory):
        directory.mkdir()
        self.model.save(directory / "model.zip")
        # VecEnv/Monitor own counters and last returned buffers, but no physics.
        vec = self.model.get_env()
        wrappers = []
        wrapped = vec.envs[0]
        while wrapped is not self.env:
            wrappers.append(
                {
                    k: copy.deepcopy(v)
                    for k, v in wrapped.__dict__.items()
                    if k not in ("env", "results_writer")
                }
            )
            wrapped = wrapped.env
        dump_state(
            directory / "activity.pkl",
            {
                "episode": episode_state(self.env),
                "wrappers": wrappers,
                "vector": {
                    k: copy.deepcopy(getattr(vec, k))
                    for k in (
                        "buf_obs",
                        "buf_dones",
                        "buf_rews",
                        "buf_infos",
                        "reset_infos",
                        "_seeds",
                        "_options",
                    )
                },
            },
        )

    def restore(self, directory):
        from sb3_contrib import MaskablePPO

        activity = load_state(directory / "activity.pkl")
        restore_episode(self.env, activity["episode"])
        vec = self.model.get_env()
        wrapped = vec.envs[0]
        for state in activity["wrappers"]:
            wrapped.__dict__.update(state)
            wrapped = wrapped.env
        for key, value in activity["vector"].items():
            setattr(vec, key, value)
        restored = MaskablePPO.load(
            directory / "model.zip",
            env=vec,
            device=self.model.device,
            force_reset=False,
        )
        # Keep the opt-in model subclass/hook while restoring SB3's own saved fields.
        self.model.__dict__.update(restored.__dict__)


_RUNNER_FIELDS = (
    "_needs_initial_reset",
    "_cached_to_module",
    "_shared_data",
    "_ongoing_episodes_for_metrics",
    "_done_episodes_for_metrics",
)
_VECTOR_FIELDS = (
    "_observations",
    "_rewards",
    "_terminations",
    "_truncations",
    "_infos",
    "_autoreset_envs",
    "_env_obs",
)


def _ray_episode_state(episode):
    state = episode.get_state()
    if hasattr(episode, "agent_episodes"):
        state["agent_episodes"] = list(
            {
                key: _ray_episode_state(value)
                for key, value in episode.agent_episodes.items()
            }.items()
        )
    else:
        # Ray 2.58 leaves zero-length lookback buffers as objects in get_state,
        # while from_state requires buffer dictionaries, including empty buffers.
        state["extra_model_outputs"] = {
            key: value.get_state() for key, value in episode.extra_model_outputs.items()
        }
    return state


class RayTrainingState:
    def __init__(self, algorithm, env, roles):
        self.algorithm, self.env, self.roles = algorithm, env, tuple(roles)

    def hashes(self):
        return {
            role: weights_digest(self.algorithm.get_module(role).get_state())
            for role in self.roles
        }

    def initialize(self, directory):
        from ray.rllib.core.rl_module.rl_module import RLModule

        for role in self.roles:
            name = "module" if len(self.roles) == 1 else role
            restored = RLModule.from_checkpoint(directory / name)
            self.algorithm.get_module(role).set_state(restored.get_state())
        # Learner and sampler have distinct modules even for a local learner.
        self.algorithm.learner_group.set_weights(
            {role: self.algorithm.get_module(role).get_state() for role in self.roles}
        )

    def export(self, directory):
        from ray.rllib.core.rl_module.rl_module import RLModule

        directory.mkdir()
        for role, expected in self.hashes().items():
            name = "module" if len(self.roles) == 1 else role
            self.algorithm.get_module(role).save_to_path(directory / name)
            restored = RLModule.from_checkpoint(directory / name)
            if weights_digest(restored.get_state()) != expected:
                raise ValueError(f"{role} inference export changed weights")

    def save_core(self, directory):
        learner = self.algorithm.learner_group._learner
        dump_state(directory / "algorithm.pkl", self.algorithm.get_state())
        dump_state(
            directory / "ppo_dynamic.pkl",
            {
                "kl": dict(learner.curr_kl_coeffs_per_module),
                "entropy": {
                    key: value.get_current_value()
                    for key, value in learner.entropy_coeff_schedulers_per_module.items()
                },
            },
        )

    def restore_core(self, directory):
        self.algorithm.set_state(load_state(directory / "algorithm.pkl"))
        dynamic = load_state(directory / "ppo_dynamic.pkl")
        learner = self.algorithm.learner_group._learner
        for role, value in dynamic["kl"].items():
            learner.curr_kl_coeffs_per_module[role] = value
        for role, value in dynamic["entropy"].items():
            learner.entropy_coeff_schedulers_per_module[role]._curr_value = value

    def save(self, directory):
        directory.mkdir()
        runner = self.algorithm.env_runner
        wrappers = []
        wrapped = runner.env.unwrapped.envs[0]
        while wrapped is not wrapped.unwrapped:
            wrappers.append(
                {
                    key: copy.deepcopy(value)
                    for key, value in wrapped.__dict__.items()
                    if key != "env"
                }
            )
            wrapped = wrapped.env
        # Ray's public episode state includes continuation/lookback/mapping state.
        activity = {
            "episode": episode_state(self.env),
            "wrappers": wrappers,
            "episodes": [_ray_episode_state(ep) for ep in runner._ongoing_episodes],
            "runner": {
                key: copy.deepcopy(getattr(runner, key))
                for key in _RUNNER_FIELDS
                if hasattr(runner, key)
            },
            "vector": {
                key: copy.deepcopy(getattr(runner.env.unwrapped, key))
                for key in _VECTOR_FIELDS
                if hasattr(runner.env.unwrapped, key)
            },
        }
        self.save_core(directory)
        dump_state(directory / "activity.pkl", activity)

    def restore(self, directory):
        from ray.rllib.env.multi_agent_episode import MultiAgentEpisode
        from ray.rllib.env.single_agent_episode import SingleAgentEpisode

        runner = self.algorithm.env_runner
        self.restore_core(directory)
        activity = load_state(directory / "activity.pkl")
        restore_episode(self.env, activity["episode"])
        wrapped = runner.env.unwrapped.envs[0]
        for state in activity["wrappers"]:
            wrapped.__dict__.update(state)
            wrapped = wrapped.env
        if hasattr(self.env, "possible_agents"):
            wrapper = runner.env.unwrapped.envs[0].unwrapped
            wrapper.agents = self.env.possible_agents.copy()
        episode_type = (
            MultiAgentEpisode
            if hasattr(self.env, "possible_agents")
            else SingleAgentEpisode
        )
        runner._ongoing_episodes = [
            episode_type.from_state(s) for s in activity["episodes"]
        ]
        for key, value in activity["runner"].items():
            setattr(runner, key, value)
        for key, value in activity["vector"].items():
            setattr(runner.env.unwrapped, key, value)
