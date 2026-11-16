"""Single-agent E1a decisions; RLlib owns sampling and PPO updates."""

import gymnasium as gym
import numpy as np

from smartsom.engine.central_control import CentralDecision
from smartsom.engine.decision_facts import DecisionFacts
from smartsom.learning.factory_episode import FactoryEpisode


class CentralVectorEnv(gym.vector.VectorWrapper):
    """Expose Gym Sequence leaves as one-env tensors to RLlib's unbatcher."""

    def __init__(self, config):
        if config.get("num_envs", 1) != 1:
            raise ValueError(
                "central candidate batching requires one factory per runner"
            )
        super().__init__(gym.vector.SyncVectorEnv([lambda: CentralFactoryEnv(config)]))

    @staticmethod
    def _batch(observation):
        return {
            name: np.stack(value)
            if name in {"candidates", "associations", "valid"}
            else value
            for name, value in observation.items()
        }

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._batch(obs), info

    def step(self, actions):
        obs, reward, terminated, truncated, info = self.env.step(actions)
        return self._batch(obs), reward, terminated, truncated, info


class CentralFactoryEnv(gym.Env):
    def __init__(self, config):
        self.scenario, self.options = config["scenario"], config["options"]
        self.action_seed = config.get("action_seed", 0)
        self.training_worlds = config.get("training_worlds")
        self.evaluation_case = None
        self.episode_index = 0
        self.world_identity = {}
        self.episode = FactoryEpisode(self.scenario, self.options)
        request = self.episode.frame.request
        self.layout = request.inputs["layout"]
        self.owners = request.inputs["owners"]
        self.positions = {owner: i for i, owner in enumerate(self.owners)}
        width = len(request.inputs["global"])
        sim = self.episode.sim
        # A conservative finite semantic bound, never an observed quantile or
        # truncation limit. Only actual candidates are stored in each episode.
        destinations = max(
            [
                sum(op in m.operation_types for m in sim.machines.values())
                for op in {
                    op for m in sim.machines.values() for op in m.operation_types
                }
            ]
            + [sum(role == "system_output" for role in sim.roles.values())]
        )
        limit = 1 + len(sim.demands) * (
            len(sim.agvs) * destinations
            + sum(len(m.quality_modes) for m in sim.machines.values())
        )
        self.action_space = gym.spaces.Discrete(limit)
        self.observation_space = gym.spaces.Dict(
            {
                "global": gym.spaces.Box(-np.inf, np.inf, (width,), np.float32),
                "critic": gym.spaces.Box(-np.inf, np.inf, (width,), np.float32),
                "candidates": gym.spaces.Sequence(
                    gym.spaces.Box(-np.inf, np.inf, (15,), np.float32), stack=True
                ),
                "associations": gym.spaces.Sequence(
                    gym.spaces.Box(-1, len(self.owners) - 1, (3,), np.int32), stack=True
                ),
                "valid": gym.spaces.Sequence(
                    gym.spaces.Box(0, 1, (), np.int8), stack=True
                ),
            }
        )
        self.episode.close()

    def _observation(self):
        frame = self.episode.frame
        request = frame.request
        if request is None:
            # A real post-advance public state for truncation bootstrap. This
            # does not commit an action, start services, or resume the simulator.
            request = CentralDecision(
                frame.tick, self.episode.protocol.central.candidates(), None
            )
            with (
                self.episode.protocol.local_facts(),
                DecisionFacts(self.episode.protocol, request) as facts,
            ):
                request = self.episode.protocol.observation_transform(facts, request)
        data = request.inputs
        if data["layout"] != self.layout or data["owners"] != self.owners:
            raise ValueError("central observation changes the frozen entity layout")
        count = len(request.candidates)
        if not 0 < count <= self.action_space.n:
            raise ValueError("central candidate count violates the semantic bound")
        context = np.asarray(data["global"], dtype=np.float32)
        obs = {
            "global": context,
            "critic": context,
            "candidates": np.asarray(data["candidates"], dtype=np.float32),
            "associations": np.asarray(
                [
                    [self.positions.get(o, -1) for o in row]
                    for row in data["associations"]
                ],
                dtype=np.int32,
            ),
            "valid": np.ones(count, dtype=np.int8),
        }
        info = {"physical_tick": frame.tick, **self.world_identity}
        if frame.terminated or frame.truncated:
            info["factory_metrics"] = self.episode.metrics()
        return obs, info

    def set_evaluation_case(self, case):
        if case["mode"] not in {"sample", "argmax"}:
            raise ValueError("evaluation mode must be sample or argmax")
        self._validate_inventory(case["scenario"])
        self.evaluation_case = case

    def _validate_inventory(self, scenario):
        if scenario.factory != self.scenario.factory or len(scenario.demands) != len(
            self.scenario.demands
        ):
            raise ValueError("central world changes the frozen factory inventory")

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        scenario, self.world_identity = self.scenario, {}
        action_seed = self.action_seed if seed is None else seed
        if self.evaluation_case is not None:
            scenario = self.evaluation_case["scenario"]
            action_seed = self.evaluation_case["action_seed"]
            self.world_identity = {"case_id": self.evaluation_case["case_id"]}
        elif self.training_worlds is not None:
            scenario, self.world_identity = self.training_worlds.sample(
                self.np_random, self.episode_index
            )
            self._validate_inventory(scenario)
            self.episode_index += 1
        self.episode.close()
        self.episode = FactoryEpisode(scenario, self.options, action_seed=action_seed)
        return self._observation()

    def step(self, action):
        frame = self.episode.frame
        if frame.request is None or frame.terminated or frame.truncated:
            raise ValueError("episode finished; reset before stepping")
        candidates = frame.request.candidates
        if not self.action_space.contains(action) or action >= len(candidates):
            raise ValueError("action is not a current central candidate")
        policy = self.options.learning_policy
        if (
            policy.network.params.get("advance_only_when_idle", False)
            and candidates[action].kind == "advance"
            and any(c.kind != "advance" for c in candidates)
        ):
            raise ValueError("ADVANCE is masked while work is available")
        self.episode.act(candidates[action].identity)
        if self.episode.frame.request is None:
            self.episode.wait()
        frame = self.episode.frame
        obs, info = self._observation()
        return obs, frame.team_reward, frame.terminated, frame.truncated, info

    def close(self):
        self.episode.close()
