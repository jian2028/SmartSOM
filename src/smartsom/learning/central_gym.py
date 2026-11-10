"""Optional Gymnasium adapter for custom loops with variable semantic actions.

The custom observation/action spaces intentionally do not imply compatibility
with fixed-Discrete/Box-only learning libraries.
"""

import gymnasium as gym

from smartsom.domain.public_facts import Facts
from smartsom.engine.central_control import ADVANCE
from smartsom.engine.central_environment import CentralEnvironment


class PublicFactsSpace(gym.Space):
    def sample(self, mask=None, probability=None):
        return Facts(
            (
                ("tick", 0),
                ("jobs", ()),
                ("machines", ()),
                ("vehicles", ()),
                ("buffers", ()),
                ("completed", ()),
            )
        )

    def contains(self, x):
        return isinstance(x, Facts) and all(
            k in x
            for k in ("tick", "jobs", "machines", "vehicles", "buffers", "completed")
        )


class SemanticTaskSpace(gym.Space):
    def __init__(self, legal=None):
        super().__init__()
        self.legal = legal

    def sample(self, mask=None, probability=None):
        candidates = self.legal() if self.legal else ()
        if not candidates or ADVANCE in candidates:
            return ADVANCE.identity
        # advance_only_when_idle hides ADVANCE while work exists.
        return candidates[int(self.np_random.integers(len(candidates)))].identity

    def contains(self, x):
        return isinstance(x, str) and bool(x)


class CentralGym(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, scenario_factory, options):
        self.scenario_factory, self.options = scenario_factory, options
        self.observation_space = PublicFactsSpace()
        self.action_space = SemanticTaskSpace(self._legal)
        self.environment = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        world_seed = seed if seed is not None else int(self.np_random.integers(2**31))
        self.environment = CentralEnvironment(
            self.scenario_factory(world_seed), self.options
        )
        return self.environment.observation(), self.environment.info()

    def _legal(self):
        if self.environment is None:
            return ()
        return self.environment.info()["candidates"]

    def step(self, action):
        if self.environment is None:
            raise RuntimeError("call reset before step")
        return self.environment.step(action)
