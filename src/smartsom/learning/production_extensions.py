"""Research hooks receive detached public observations and semantic choices only."""

from copy import deepcopy
from dataclasses import dataclass

import gymnasium as gym
import numpy as np

from smartsom.learning.extensions import (
    ExtensionsRuntime,
    PublicObservation,
    RewardTransition,
)


@dataclass(frozen=True)
class GridDecisionContext:
    simulation_time: int
    agent_id: str
    candidates: tuple
    action_mask: tuple[bool, ...]


def numpy_observation(value):
    if isinstance(value, dict):
        return {key: numpy_observation(child) for key, child in value.items()}
    return np.asarray(value, dtype=np.float32)


def gym_space(space):
    if space.vector:
        return gym.spaces.Box(-np.inf, np.inf, space.vector, np.float32)
    return gym.spaces.Dict(
        {
            key: gym.spaces.Box(-np.inf, np.inf, shape, np.float32)
            for key, shape in space.fields
        }
    )


class GridHooks:
    def __init__(self, adapter, spec, provider):
        self.adapter = adapter
        self.resource = provider == "rllib.resource_ppo"
        self.roles = (
            ("machine_policy", "agv_policy", "buffer_policy", "quality_policy")
            if self.resource
            else (None,)
        )
        vector = (0.0,) * adapter.feature_count
        layout = PublicObservation(None, vector, (("state", vector),)).layout()
        self.runtime = ExtensionsRuntime(
            spec, provider, dict.fromkeys(self.roles, layout)
        )
        self.spaces = {
            role: gym_space(space) for role, space in self.runtime.spaces.items()
        }
        self.cached = None
        self.fallback_role = self.roles[0]

    def context(self):
        env = self.adapter
        return GridDecisionContext(
            env.sim.tick,
            env.actor,
            tuple(env.current[1]) if env.current else ("WAIT",),
            tuple(bool(value) for value in env.action_masks()),
        )

    def encode(self):
        if self.cached is None:
            env = self.adapter
            role = (
                env.role + "_policy"
                if self.resource and env.current
                else self.fallback_role
            )
            vector = tuple(float(value) for value in env.raw_observation())
            self.cached = numpy_observation(
                self.runtime.encode(
                    PublicObservation(
                        self.context(),
                        vector,
                        (("state", vector),),
                        role,
                        env.actor if self.resource else None,
                    )
                )
            )
        return deepcopy(self.cached)

    def reward(self, before, action, raw):
        env = self.adapter
        transition = RewardTransition(
            before,
            self.context(),
            ((before.agent_id, action),),
            raw,
            env.sim.tick,
            env.reason if env.finished else None,
        )
        if self.resource:
            result = self.runtime.rewards(transition, self.roles)
            return result.team, dict(result.roles)
        return self.runtime.reward(transition), {}
