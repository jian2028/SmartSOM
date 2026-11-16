"""RLlib multi-agent adapter over semantic production and learned WAIT frames."""

import json
from hashlib import sha256

import gymnasium as gym
import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from smartsom.algorithms.rule_actions import RULE_CATALOG
from smartsom.domain.training import LearningContext
from smartsom.learning.factory_episode import FactoryEpisode
from smartsom.learning.rllib_observation import V2ObservationCodec
from smartsom.representations.v2.rules import CATALOG
from smartsom.representations.v2.social_config import social_config


class FactoryMultiAgentEnv(MultiAgentEnv):
    def __init__(self, config):
        super().__init__()
        self.scenario, self.options = config["scenario"], config["options"]
        self.algorithm = config["algorithm"]
        if self.algorithm not in {"ppo", "dqn"}:
            raise ValueError("learning environment requires ppo or dqn")
        self.action_seed = config.get("action_seed", 0)
        self.evaluation_case = None
        self.training_worlds = config.get("training_worlds")
        self.episode_index = 0
        self.world_identity = {}
        self.episode = FactoryEpisode(
            self.scenario, self.options, action_seed=self.action_seed
        )
        sim = self.episode.sim
        self.codecs, self.agent_groups = {}, {}
        self.agent_modules = {}
        self.observation_spaces, self.action_spaces = {}, {}
        owners = {
            "machine": tuple(sim.machines),
            "dispatcher": tuple(sim.agvs),
            "pre": tuple(sim.pre.values()),
            "post": tuple(sim.post.values()),
            "input": tuple(
                key for key, role in sim.roles.items() if role == "system_input"
            ),
        }
        jobs, vehicles, stations = len(sim.demands), len(sim.agvs), len(sim.machines)
        for point in CATALOG:
            role, stage = point.split(".")
            policy_name = self.options.production.bindings.get(
                point, self.options.production.default
            )
            policy = self.options.production.policies[policy_name]
            group = point if role in policy.split_stages else role
            if group not in self.codecs:
                social = social_config(**policy.observation.params).model_dump(
                    exclude_none=True
                )
                max_modes = max(len(m.quality_modes) for m in sim.machines.values())
                # Frozen semantic inventory bounds; no performance calibration.
                actions = max(
                    jobs * vehicles,
                    jobs * max_modes,
                    len(sim.storage) + 1,
                    vehicles + 1,
                    max(len(rules) for rules in CATALOG.values()) * max_modes + 1,
                )
                points = [
                    name
                    for name in CATALOG
                    if name == group or name.startswith(group + ".")
                ]
                representation = self.options.action_representation
                if all(representation.mode(name) == "rules" for name in points):
                    # Rule projection emits at most one action per rule (per
                    # quality mode for machines), plus STOP and EMPTY_WAIT.
                    # Physical job×vehicle pairs never enter this action space.
                    actions = max(
                        len(
                            representation.selected_rules(name)
                            or RULE_CATALOG[tuple(name.split("."))]
                        )
                        * (max_modes if role == "machine" else 1)
                        + 2
                        for name in points
                    )
                social_jobs = jobs
                if role == "post":
                    # supply() contains ready post-buffer jobs plus at most
                    # one job on its associated machine. Unbounded storage
                    # retains the full frozen order inventory bound.
                    social_jobs = max(
                        min(jobs, sum(capacities) + 1)
                        if all(cap is not None for cap in capacities)
                        else jobs
                        for owner in owners[role]
                        for capacities in [tuple(sim.capacity[owner].values())]
                    )
                self.codecs[group] = V2ObservationCodec(
                    group,
                    options=self.options,
                    actions=actions,
                    stations=stations,
                    vehicles=vehicles,
                    jobs=social_jobs,
                    social=social,
                )
            codec = self.codecs[group]
            for owner in owners[role]:
                context = LearningContext(0, role, owner, stage)
                agent = self.agent_id(context)
                if agent in self.agent_groups and self.agent_groups[agent] != group:
                    raise ValueError(
                        "an agent cannot change its module group across stages"
                    )
                self.agent_groups[agent] = group
                module = policy_name + "__" + group
                if policy.parameter_sharing.mode(role) == "independent":
                    module += "__" + sha256(agent.encode()).hexdigest()
                if agent in self.agent_modules and self.agent_modules[agent] != module:
                    raise ValueError(
                        "an agent cannot change its policy module across stages"
                    )
                self.agent_modules[agent] = module
                space = codec.observation_space
                if self.algorithm == "ppo":
                    # Existing V2 public critic observation contract is 48 scalars.
                    space = gym.spaces.Dict(
                        {
                            **space.spaces,
                            "critic": gym.spaces.Box(
                                -np.inf, np.inf, (48,), dtype=np.float32
                            ),
                        }
                    )
                self.observation_spaces[agent] = space
                self.action_spaces[agent] = codec.action_space
        self.possible_agents = sorted(self.agent_groups)
        self.agents = []
        self.started = set()
        self.episode.close()

    def agent_id(self, context):
        return json.dumps(self.episode.subject_key(context), separators=(",", ":"))

    def _observations(self):
        frame = self.episode.frame
        observations, infos = {}, {}
        contexts = (frame.request,) if frame.request is not None else frame.waiting
        shared_critic = None
        reuse_critic = getattr(
            self.episode.components.critic, "subject_independent", False
        )
        for context in contexts:
            agent = self.agent_id(context)
            codec = self.codecs[self.agent_groups[agent]]
            obs = (
                codec.encode(context)
                if frame.request is not None
                else codec.encode_wait(self.episode.protocol, context)
            )
            if self.algorithm == "ppo":
                value_context = LearningContext(
                    frame.tick,
                    context.group,
                    context.owner,
                    context.stage,
                    frame.truncated,
                )
                if shared_critic is None:
                    critic = np.asarray(
                        self.episode.components.value_input(
                            self.episode.protocol, value_context
                        ),
                        dtype=np.float32,
                    )
                    if reuse_critic:
                        # Only this observation frame: same-tick actions may
                        # change public state before the next call.
                        shared_critic = critic
                else:
                    critic = shared_critic
                # Keep observations independently writable for RLlib consumers.
                obs["critic"] = critic.copy()
            observations[agent] = obs
            infos[agent] = {
                "physical_tick": frame.tick,
                "waiting": frame.request is None,
            }
        if frame.terminated or frame.truncated:
            if infos:
                infos[next(iter(infos))]["factory_metrics"] = self.episode.metrics()
                infos[next(iter(infos))].update(self.world_identity)
        self.started.update(observations)
        self.agents = (
            sorted(self.started) if not (frame.terminated or frame.truncated) else []
        )
        return observations, infos

    def set_evaluation_case(self, case):
        """Select a frozen case via Gymnasium's public vector-environment call."""
        if case["mode"] not in {"argmax", "sample"}:
            raise ValueError("evaluation mode must be argmax or sample")
        probe = type(self)(
            {
                "scenario": case["scenario"],
                "options": self.options,
                "algorithm": self.algorithm,
                "action_seed": case["action_seed"],
            }
        )
        try:
            if (
                probe.agent_modules != self.agent_modules
                or probe.observation_spaces != self.observation_spaces
                or probe.action_spaces != self.action_spaces
            ):
                raise ValueError(
                    "evaluation case differs from frozen module/space inventory"
                )
        finally:
            probe.close()
        self.evaluation_case = case

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        scenario = self.scenario
        self.world_identity = {}
        if self.evaluation_case is not None:
            scenario = self.evaluation_case["scenario"]
            self.world_identity = {"case_id": self.evaluation_case["case_id"]}
        elif self.training_worlds is not None:
            scenario, self.world_identity = self.training_worlds.sample(
                self.np_random,
                self.episode_index,
            )
            # Model and module inventories are frozen for each trial. Changes
            # need a compiled union of spaces, never clipping or remapping.
            if scenario.factory != self.scenario.factory or len(
                scenario.demands
            ) != len(self.scenario.demands):
                raise ValueError("training world changes the frozen factory inventory")
            self.episode_index += 1
        self.episode.close()
        self.episode = FactoryEpisode(
            scenario,
            self.options,
            action_seed=(self.action_seed if seed is None else seed)
            if self.evaluation_case is None
            else self.evaluation_case["action_seed"],
        )
        self.started = set()
        # No agent transition exists before the first production observation.
        while self.episode.frame.request is None and not self.episode.frame.waiting:
            if self.episode.frame.terminated or self.episode.frame.truncated:
                raise ValueError("factory has no agent decision before its horizon")
            self.episode.wait()
        observations, infos = self._observations()
        if infos:
            infos[next(iter(infos))].update(self.world_identity)
        return observations, infos

    def step(self, actions):
        frame = self.episode.frame
        if frame.terminated or frame.truncated:
            raise ValueError("episode finished; reset before stepping")
        contexts = (frame.request,) if frame.request is not None else frame.waiting
        expected = {self.agent_id(context) for context in contexts}
        if set(actions) != expected:
            raise ValueError("actions must match exactly the currently observed agents")
        if frame.request is not None:
            agent = self.agent_id(frame.request)
            codec = self.codecs[self.agent_groups[agent]]
            self.episode.act(codec.decode(frame.request, actions[agent]))
        else:
            if any(
                not self.action_spaces[agent].contains(action) or action != 0
                for agent, action in actions.items()
            ):
                raise ValueError("waiting agents must select their sole WAIT action")
            self.episode.wait()
        frame = self.episode.frame
        rewards = dict.fromkeys(self.started, frame.team_reward)
        observations, infos = self._observations()
        terminated = {agent: frame.terminated for agent in self.started}
        truncated = {agent: frame.truncated for agent in self.started}
        terminated["__all__"], truncated["__all__"] = frame.terminated, frame.truncated
        return observations, rewards, terminated, truncated, infos

    def close(self):
        self.episode.close()
