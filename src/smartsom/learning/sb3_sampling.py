"""SB3's usual rollout collector over ordered local/process simulation streams."""

import numpy as np
from stable_baselines3.common.vec_env import VecEnv

from smartsom.learning.training_extensions import stack_observations


class OrderedVecEnv(VecEnv):
    def __init__(self, pool):
        self.pool = pool
        observation, action, _ = pool.spaces[0]
        self.actions = None
        super().__init__(pool.num_envs, observation, action)

    def reset(self):
        results = self.pool.reset()
        self.reset_infos = [results[index][1] for index in range(self.num_envs)]
        self._reset_seeds()
        self._reset_options()
        return stack_observations([results[index][0] for index in range(self.num_envs)])

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        results = self.pool.step(self.actions)
        observations, rewards, dones, infos = [], [], [], []
        reset = []
        for index, (observation, reward, terminated, truncated, info) in enumerate(
            results
        ):
            done = terminated or truncated
            info = dict(info)
            info["TimeLimit.truncated"] = truncated and not terminated
            if done:
                snapshot = self.pool.snapshots[index]
                info["terminal_observation"] = observation
                info["episode"] = {
                    "r": snapshot.total_reward,
                    "l": len(snapshot.steps),
                    "t": 0.0,
                }
                reset.append(index)
            observations.append(observation)
            rewards.append(reward * self.pool.learner_scale)
            dones.append(done)
            infos.append(info)
        if reset:
            for index, (observation, info) in self.pool.reset(reset).items():
                observations[index], self.reset_infos[index] = observation, info
        return (
            stack_observations(observations),
            np.asarray(rewards, np.float32),
            np.asarray(dones, bool),
            infos,
        )

    def close(self):
        self.pool.close()

    def get_attr(self, attr_name, indices=None):
        if attr_name == "render_mode":
            return [None for _ in self._get_indices(indices)]
        raise AttributeError(attr_name)

    def has_attr(self, attr_name):
        return attr_name in ("action_masks", "render_mode")

    def set_attr(self, attr_name, value, indices=None):
        raise AttributeError("sampling stream attributes are immutable")

    def env_method(self, method_name, *args, indices=None, **kwargs):
        if method_name != "action_masks" or args or kwargs:
            raise AttributeError(method_name)
        return list(
            self.pool.call(
                "mask", {index: None for index in self._get_indices(indices)}
            ).values()
        )

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False for _ in self._get_indices(indices)]
