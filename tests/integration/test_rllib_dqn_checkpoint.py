"""Real official MARL DQN recovery; CartPole is an engineering fixture only."""

import copy
import os
import pickle as stdlib_pickle
import subprocess
import sys
from collections import deque
from functools import partial
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import ray
import ray.cloudpickle as pickle
import torch
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.dqn import DQN, DQNConfig
from ray.rllib.env.multi_agent_env import MultiAgentEnv, make_multi_agent
from ray.rllib.env.multi_agent_env_runner import MultiAgentEnvRunner
from ray.rllib.env.multi_agent_episode import MultiAgentEpisode
from ray.rllib.execution.segment_tree import SegmentTree
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.utils.metrics import ENV_RUNNER_RESULTS, NUM_ENV_STEPS_SAMPLED_LIFETIME

from smartsom.learning.rllib_dqn import (
    REPLAY_COMPONENT,
    ReplayCheckpointDQN,
    RestorableMultiAgentPrioritizedEpisodeReplayBuffer,
)


@pytest.mark.parametrize("reward_scale", [0.0, 1.0, -1.0])
@pytest.mark.parametrize("cuts", [(2,), (2, 4, 6)])
def test_fragment_admission_preserves_uncut_agent_trajectories(reward_scale, cuts):
    def new_episode():
        episode = MultiAgentEpisode(
            id_="same-world",
            agent_to_module_mapping_fn=lambda agent, *a, **k: agent,
            observation_space=gym.spaces.Dict(
                {a: gym.spaces.Discrete(32) for a in ("a", "b")}
            ),
            action_space=gym.spaces.Dict(
                {a: gym.spaces.Discrete(2) for a in ("a", "b")}
            ),
        )
        episode.add_env_reset(observations={"a": 0}, infos={"a": {}})
        return episode

    reference, fragment = new_episode(), new_episode()
    buffer = RestorableMultiAgentPrioritizedEpisodeReplayBuffer(
        capacity=64, alpha=0.6, beta=0.4
    )
    previous = "a"
    for tick, current in enumerate(("a", "b", "b", "b", "a", "a", "b", "a"), 1):
        data = dict(
            observations={current: tick},
            actions={previous: tick % 2},
            rewards={previous: reward_scale * tick},
            infos={current: {}},
            terminateds={"__all__": False},
            truncateds={"__all__": False},
            extra_model_outputs={previous: {}},
        )
        reference.add_env_step(**data)
        fragment.add_env_step(**data)
        previous = current
        if tick in cuts or tick == 8:
            successor = fragment.cut(len_lookback_buffer=1)
            # Reproduce the official EnvRunner preprocessing, not a product patch.
            MultiAgentEnvRunner._prune_zero_len_sa_episodes(fragment)
            original = copy.deepcopy(fragment.get_state())
            buffer.add(fragment)
            assert_state_equal(original, fragment.get_state())
            fragment = successor
    combined = buffer.episodes[0]
    assert combined.env_t == reference.env_t
    for agent in ("a", "b"):
        expected, actual = (
            reference.agent_episodes[agent],
            combined.agent_episodes[agent],
        )
        assert_state_equal(expected.get_observations(), actual.get_observations())
        assert_state_equal(expected.get_actions(), actual.get_actions())
        assert_state_equal(expected.get_rewards(), actual.get_rewards())
        assert (expected.is_terminated, expected.is_truncated) == (
            actual.is_terminated,
            actual.is_truncated,
        )
    restored = RestorableMultiAgentPrioritizedEpisodeReplayBuffer(
        capacity=64, alpha=0.6, beta=0.4
    )
    samples = buffer.sample(batch_size_B=8, n_step=1)
    for agent in ("a", "b"):
        count = sum(sample.module_id == agent for sample in samples)
        buffer.update_priorities(np.ones(count), agent)
    restored.set_state(buffer.get_state())
    samples = restored.sample(batch_size_B=8, n_step=1)
    assert samples and all(sample.module_id in ("a", "b") for sample in samples)


class StringAgents(MultiAgentEnv):
    def __init__(self, config):
        super().__init__()
        self.inner = make_multi_agent("CartPole-v1")(config)
        self.possible_agents = [str(x) for x in self.inner.possible_agents]
        self.agents = self.possible_agents.copy()
        self.observation_spaces = {
            str(k): v for k, v in self.inner.observation_spaces.items()
        }
        self.action_spaces = {str(k): v for k, v in self.inner.action_spaces.items()}

    def reset(self, *, seed=None, options=None):
        return tuple(
            {str(k): v for k, v in row.items()}
            for row in self.inner.reset(seed=seed, options=options)
        )

    def step(self, actions):
        return tuple(
            {str(k): v for k, v in row.items()}
            for row in self.inner.step({int(k): v for k, v in actions.items()})
        )

    def close(self):
        for env in self.inner.envs:
            env.close()


def shared(agent_id, *args, **kwargs):
    return "shared"


def independent(agent_id, *args, **kwargs):
    return "agent_" + agent_id


def assert_state_equal(left, right):
    """Compare every exported field, including tree values and episode contents."""
    if isinstance(left, torch.Tensor):
        assert torch.equal(left.cpu(), right.cpu())
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_state_equal(left[key], right[key])
    elif isinstance(left, (list, tuple, deque)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_state_equal(a, b)
    elif isinstance(left, gym.Space):
        assert left == right
    elif isinstance(left, SegmentTree):
        assert type(left) is type(right)
        assert_state_equal(vars(left), vars(right))
    elif isinstance(left, partial):
        assert type(left) is type(right)
        assert_state_equal(left.func, right.func)
        assert_state_equal(left.args, right.args)
        assert_state_equal(left.keywords, right.keywords)
    elif callable(left):
        assert left.__name__ == right.__name__
        if left.__name__ in {"shared", "independent"}:
            assert [left(a) for a in ("0", "1")] == [right(a) for a in ("0", "1")]
    elif hasattr(left, "get_state"):
        assert type(left) is type(right)
        assert_state_equal(left.get_state(), right.get_state())
    else:
        assert left == right


def config(ownership, phase):
    mapping = shared if ownership == "shared" else independent
    policies = {mapping(a): PolicySpec() for a in ("0", "1")}
    return (
        DQNConfig(algo_class=ReplayCheckpointDQN)
        .environment(StringAgents, env_config={"num_agents": 2})
        .env_runners(
            num_env_runners=0, num_envs_per_env_runner=1, rollout_fragment_length=32
        )
        .learners(num_learners=0, num_gpus_per_learner=0)
        .multi_agent(policies=policies, policy_mapping_fn=mapping)
        .fault_tolerance(
            restart_failed_env_runners=False, max_num_env_runner_restarts=0
        )
        .training(
            store_buffer_in_checkpoints=True,
            num_steps_sampled_before_learning_starts=1000 if phase == "warmup" else 32,
            target_network_update_freq=32,
            replay_buffer_config={
                "type": (
                    "smartsom.learning.rllib_dqn."
                    "RestorableMultiAgentPrioritizedEpisodeReplayBuffer"
                ),
                "capacity": 512,
                "alpha": 0.6,
                "beta": 0.4,
            },
        )
        .reporting(min_sample_timesteps_per_iteration=128, min_time_s_per_iteration=0)
        .debugging(seed=101)
    )


def exercise(stage, root, ownership, phase):
    torch.set_num_threads(1)
    assert ray.__version__ == "2.60.0"
    assert ReplayCheckpointDQN.training_step is DQN.training_step
    ray.init(
        address="local",
        num_cpus=2,
        include_dashboard=False,
        object_store_memory=100 * 1024 * 1024,
        log_to_driver=False,
    )
    algorithm = None
    try:
        if stage == "save":
            algorithm = config(ownership, phase).build_algo()
            if phase != "empty":
                algorithm.train()
                algorithm.train()
            state = copy.deepcopy(algorithm.get_state())
            assert type(algorithm) is ReplayCheckpointDQN
            assert REPLAY_COMPONENT in state
            assert (algorithm.local_replay_buffer.get_num_timesteps() > 0) == (
                phase != "empty"
            )
            algorithm.save_to_path(str(root / "checkpoint"))
            with (root / "expected.pkl").open("wb") as stream:
                pickle.dump(state, stream)
            (root / "sampled-counter.txt").write_text(
                str(
                    algorithm.metrics.peek(
                        (ENV_RUNNER_RESULTS, NUM_ENV_STEPS_SAMPLED_LIFETIME), default=0
                    )
                )
            )
            # Loading into a populated instance replaces, rather than mixes, replay.
            algorithm.train()
            algorithm.set_state(copy.deepcopy(state))
            assert_state_equal(
                state[REPLAY_COMPONENT], algorithm.get_state()[REPLAY_COMPONENT]
            )
            partial_state = algorithm.get_state(components="learner_group")
            assert REPLAY_COMPONENT not in partial_state
            algorithm.set_state(partial_state)
            assert_state_equal(
                state[REPLAY_COMPONENT], algorithm.get_state()[REPLAY_COMPONENT]
            )
            missing = dict(state)
            missing.pop(REPLAY_COMPONENT)
            with pytest.raises(ValueError, match="requires checkpoint replay"):
                algorithm.set_state(missing)
            corrupt = dict(state, replay_buffer={"episodes": []})
            with pytest.raises((KeyError, ValueError, TypeError)):
                algorithm.set_state(corrupt)
        else:
            with (root / "expected.pkl").open("rb") as stream:
                expected = pickle.load(stream)
            algorithm = Algorithm.from_checkpoint(str(root / "checkpoint"))
            assert type(algorithm) is ReplayCheckpointDQN
            actual = algorithm.get_state()
            assert_state_equal(expected[REPLAY_COMPONENT], actual[REPLAY_COMPONENT])
            assert_state_equal(expected["learner_group"], actual["learner_group"])
            assert expected["training_iteration"] == algorithm.iteration
            assert algorithm.metrics.peek(
                (ENV_RUNNER_RESULTS, NUM_ENV_STEPS_SAMPLED_LIFETIME), default=0
            ) == float((root / "sampled-counter.txt").read_text())
            buffer = algorithm.local_replay_buffer
            assert (
                buffer.get_num_timesteps()
                == expected[REPLAY_COMPONENT]["_num_timesteps"]
            )
            before = {
                name: copy.deepcopy(algorithm.get_module(name).get_state())
                for name in algorithm.config.policies
            }
            count = buffer.get_added_timesteps()
            evicted = buffer.get_num_episodes_evicted()
            # Include recovery before DQN warmup: allow the normal official
            # threshold to be reached, without lowering it after restoration.
            iterations = 0
            for _ in range(16):
                algorithm.train()
                iterations += 1
                if buffer.get_added_timesteps() - count >= 2 * buffer.capacity and all(
                    any(
                        not np.array_equal(
                            weights[k], algorithm.get_module(name).get_state()[k]
                        )
                        for k in weights
                    )
                    for name, weights in before.items()
                ):
                    break
            assert buffer.get_added_timesteps() > count
            assert buffer.get_added_timesteps() - count >= 2 * buffer.capacity
            assert buffer.get_num_episodes_evicted() >= evicted + 3
            assert algorithm.iteration == expected["training_iteration"] + iterations
            for name, weights in before.items():
                after = algorithm.get_module(name).get_state()
                assert any(not np.array_equal(weights[k], after[k]) for k in weights)
            algorithm.save_to_path(str(root / "continued_checkpoint"))
            continued = copy.deepcopy(algorithm.get_state())
            algorithm.stop()
            algorithm = None
            algorithm = Algorithm.from_checkpoint(str(root / "continued_checkpoint"))
            assert_state_equal(
                continued[REPLAY_COMPONENT], algorithm.get_state()[REPLAY_COMPONENT]
            )
            assert_state_equal(
                continued["learner_group"], algorithm.get_state()["learner_group"]
            )
            assert algorithm.iteration == continued["training_iteration"]
            algorithm.train()
            assert algorithm.local_replay_buffer.get_num_timesteps() > 0
            corrupt_path = root / "corrupt_checkpoint"
            algorithm.save_to_path(str(corrupt_path))
            state_path = corrupt_path / (algorithm.STATE_FILE_NAME + ".pkl")
            with state_path.open("rb") as stream:
                missing = pickle.load(stream)
            missing.pop(REPLAY_COMPONENT)
            with state_path.open("wb") as stream:
                pickle.dump(missing, stream)
            with pytest.raises(ValueError, match="requires checkpoint replay"):
                algorithm.restore_from_path(str(corrupt_path))
            state_path.write_bytes(b"invalid checkpoint pickle")
            algorithm.stop()
            algorithm = None
            with pytest.raises((stdlib_pickle.UnpicklingError, EOFError, ValueError)):
                Algorithm.from_checkpoint(str(corrupt_path))
    finally:
        if algorithm is not None:
            algorithm.stop()
        ray.shutdown()


@pytest.mark.parametrize(
    "ownership,phase",
    [
        ("shared", "trained"),
        ("independent", "trained"),
        ("independent", "warmup"),
        ("shared", "empty"),
    ],
)
def test_official_checkpoint_replay_and_fresh_process_training(
    tmp_path, ownership, phase
):
    repository = Path(__file__).resolve().parents[2]
    env = dict(
        os.environ,
        PYTHONPATH=str(repository / "src"),
        RAY_ENABLE_UV_RUN_RUNTIME_ENV="0",
        RAY_USAGE_STATS_ENABLED="0",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
    )
    for stage in ("save", "restore"):
        result = subprocess.run(
            [sys.executable, __file__, stage, str(tmp_path), ownership, phase],
            cwd=tmp_path,
            env=env,
            text=True,
            capture_output=True,
            timeout=180,
        )
        (tmp_path / f"{stage}.log").write_text(result.stdout + result.stderr)
        assert result.returncode == 0, result.stdout + result.stderr


if __name__ == "__main__":
    exercise(sys.argv[1], Path(sys.argv[2]), sys.argv[3], sys.argv[4])
