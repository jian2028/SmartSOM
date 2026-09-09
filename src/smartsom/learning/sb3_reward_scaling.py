"""Scale only rewards handed to SB3; the wrapped environment records raw/research."""

import gymnasium as gym


class LearnerRewardEnv(gym.Wrapper):
    def __init__(self, env, scale):
        super().__init__(env)
        self.scale = scale

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        return observation, reward * self.scale, terminated, truncated, info

    def action_masks(self):
        return self.env.action_masks()
