"""Locked-framework state adapters for complete single-environment update points.

These adapters extend framework checkpoints, not the PPO loss or simulator. Pickle
members are local trusted training artifacts and are verified before loading.
"""

import pickle
import random
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def backend_cleanup():
    """Close every registered resource without replacing a backend failure."""
    from contextlib import ExitStack

    stack = ExitStack()
    try:
        yield stack
    except BaseException as exc:
        try:
            stack.close()
        except BaseException as cleanup_error:
            exc.add_note(f"backend environment cleanup also failed: {cleanup_error}")
        raise
    else:
        stack.close()


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
        self.algorithm = algorithm

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
