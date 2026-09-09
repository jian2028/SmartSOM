"""Optional Parallel API. Physics and proposal arbitration are framework-free."""

import copy
from collections.abc import Callable
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from pettingzoo import ParallelEnv

from smartsom.config.codec import digest
from smartsom.dispatch import DecisionContext
from smartsom.engine import DeadlockError, SimulationResult, Simulator
from smartsom.learning.episode import (
    EpisodeInput,
    EpisodeLimits,
    EpisodeStartFailure,
    resource_outcome,
)
from smartsom.learning.extensions import (
    ExtensionsRuntime,
    RewardBatch,
    RewardTransition,
    resource_observation,
)
from smartsom.learning.gymnasium import extension_gym_space, extension_numpy
from smartsom.learning.joint import JointActionCoordinator, PolicyStalledError
from smartsom.learning.projection import validate_capacity
from smartsom.learning.resources import (
    ResourceDecision,
    ResourceProjection,
    ResourceProjectionSpec,
)


@dataclass(frozen=True, slots=True)
class JointLearningStep:
    decision: ResourceDecision
    indices: tuple[tuple[str, int], ...]
    proposals: tuple
    actions: tuple
    reward: float
    simulation_time: int
    reason: str | None
    trace_start: int
    trace_end: int
    reward_values: RewardBatch | None = None
    encoded_observations_sha256: tuple[tuple[str, str], ...] | None = None


class SmartSOMParallelEnv(ParallelEnv):
    metadata = {
        "name": "smartsom_resource_v1",
        "render_modes": [],
        "is_parallelizable": True,
    }

    def __init__(
        self,
        episode: EpisodeInput,
        projection: ResourceProjectionSpec,
        *,
        limits: EpisodeLimits = EpisodeLimits(),
        episode_source: Callable[[int], EpisodeInput] | None = None,
        on_episode=None,
        extensions=None,
        learner_scale=1.0,
    ):
        validate_capacity(projection, episode.workload, episode.quality)
        self.base_input, self.projection_spec, self.limits = episode, projection, limits
        self.episode_source, self.on_episode = episode_source, on_episode
        self.projection = self._projection(episode)
        self.extensions = None
        if extensions is not None:
            prototype = self.projection.project(DecisionContext(0, (), (), ()))
            layouts = {
                view.role: resource_observation(prototype.context, view).layout()
                for view in prototype.views
            }
            self.extensions = ExtensionsRuntime(
                extensions, "rllib.resource_ppo", layouts, learner_scale=learner_scale
            )
        self.possible_agents = list(self.projection.agents)
        self.agents = []
        self.observation_spaces, self.action_spaces = {}, {}
        for agent in self.possible_agents:
            role = self.policy_for_agent(agent)
            count = self.projection.capacities[role] + 1
            self.action_spaces[agent] = gym.spaces.Discrete(count)
            self.observation_spaces[agent] = gym.spaces.Dict(
                {
                    "observations": gym.spaces.Box(
                        -np.inf,
                        np.inf,
                        (self.projection.observation_sizes[role],),
                        np.float32,
                    ),
                    "action_mask": gym.spaces.Box(0, 1, (count,), np.int8),
                }
            )
            if self.extensions:
                self.observation_spaces[agent] = gym.spaces.Dict(
                    {
                        "observations": extension_gym_space(
                            self.extensions.spaces[role]
                        ),
                        "action_mask": gym.spaces.Box(0, 1, (count,), np.int8),
                    }
                )
        self.episode_index = -1
        self.finished = True
        self.steps = []
        self.simulator = None
        self.result = None
        self.projected = None

    @staticmethod
    def policy_for_agent(agent):
        return "machine_policy" if agent.startswith("machine:") else "agv_policy"

    def _projection(self, episode):
        return ResourceProjection(
            episode.factory,
            self.projection_spec,
            transport_enabled=episode.transport_enabled,
        )

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def reset(self, seed=None, options=None):
        # Parallel API permits framework options; none are scientific overrides.
        if seed is not None:
            # Framework sampling RNG, never scientific materialization.
            for i, agent in enumerate(self.possible_agents):
                self.action_spaces[agent].seed(seed + i)
        self.episode_index += 1
        self.input = (
            self.episode_source(self.episode_index)
            if self.episode_source
            else self.base_input
        )
        if (
            self.input.factory != self.base_input.factory
            or self.input.workload != self.base_input.workload
        ):
            raise ValueError("episode source changed the frozen base structure")
        if self.input.transport_enabled != self.base_input.transport_enabled:
            raise ValueError("episode source changed the resource agent set")
        validate_capacity(self.projection_spec, self.input.workload, self.input.quality)
        self.projection = self._projection(self.input)
        self.agents = self.possible_agents.copy()
        self.steps, self.result, self.reason = [], None, None
        self.total_reward, self.rewarded_tick, self._trace_cursor = 0.0, 0, 0
        self.total_research_reward = self.total_learner_reward = 0.0
        self.encoded_observations = {}
        self.role_reward_totals = {
            role: {"raw": 0.0, "research": 0.0, "learner": 0.0}
            for role in set(map(self.policy_for_agent, self.possible_agents))
        }
        if self.extensions:
            self.episode_extension_initial_state = self.extensions.state_dict()
            self.extensions.begin_episode()
        self.finished = False
        try:
            self.simulator = Simulator(
                self.input.factory, self.input.workload, **self.input.options()
            )
        except DeadlockError as exc:
            self.finished, self.reason, self.agents = True, "deadlock", []
            self.total_reward = -float(self.limits.max_ticks + 1)
            if self.extensions:
                self._extension_rewards(None, None, (), self.total_reward)
            raise EpisodeStartFailure("deadlock") from exc
        self.projected = self.projection.project(self.simulator.current_decision)
        self._encode_observations()
        return self._observations(), self._infos()

    def _encode_observations(self):
        if self.extensions:
            self.encoded_observations = {
                view.agent_id: self.extensions.encode(
                    resource_observation(self.projected.context, view)
                )
                for view in self.projected.views
            }

    def extension_state_dict(self):
        if not self.extensions:
            return None
        return {
            "schema": "smartsom.extension-env-state/v1",
            "runtime": self.extensions.state_dict(),
            "episode_initial_state": copy.deepcopy(
                self.episode_extension_initial_state
            ),
            "observations": copy.deepcopy(self.encoded_observations),
            "research_return": self.total_research_reward,
            "learner_return": self.total_learner_reward,
            "role_returns": copy.deepcopy(self.role_reward_totals),
        }

    def load_extension_state_dict(self, state):
        if not self.extensions:
            if state is not None:
                raise ValueError(
                    "extension state supplied to an unextended environment"
                )
            return
        if state.get("schema") != "smartsom.extension-env-state/v1":
            raise ValueError("unsupported environment extension state schema")
        self.extensions.load_state_dict(state["runtime"])
        self.episode_extension_initial_state = copy.deepcopy(
            state["episode_initial_state"]
        )
        values = state["observations"]
        if not self.finished and set(values) != set(self.possible_agents):
            raise ValueError(
                "active resource extension state lacks observation coverage"
            )
        self.encoded_observations = {
            agent: self.extensions.spaces[self.policy_for_agent(agent)].validate(value)
            for agent, value in values.items()
        }
        self.total_research_reward, self.total_learner_reward = (
            float(state["research_return"]),
            float(state["learner_return"]),
        )
        self.role_reward_totals = copy.deepcopy(state["role_returns"])

    def _extension_rewards(self, before, after, actions, raw):
        result = self.extensions.rewards(
            RewardTransition(
                before, after, tuple(actions), raw, self.rewarded_tick, self.reason
            ),
            self.role_reward_totals,
        )
        self.total_research_reward += result.team.research
        self.total_learner_reward += result.team.learner
        for role, values in result.roles:
            for name in ("raw", "research", "learner"):
                self.role_reward_totals[role][name] += getattr(values, name)
        return result

    def _observations(self, agents=None):
        result = {}
        for aid in self.agents if agents is None else agents:
            view = self.projected.for_agent(aid)
            if self.extensions:
                result[aid] = {
                    "observations": extension_numpy(
                        self.extensions.spaces[view.role].zeros()
                        if self.finished
                        else self.encoded_observations[aid]
                    ),
                    "action_mask": np.asarray(
                        view.action_mask
                        if not self.finished
                        else (1,) + (0,) * (len(view.action_mask) - 1),
                        dtype=np.int8,
                    ),
                }
                continue
            values = np.asarray(view.observations, dtype=np.float32)
            if not np.isfinite(values).all():
                raise ValueError("resource observation exceeds finite float32")
            result[aid] = {
                "observations": values if not self.finished else np.zeros_like(values),
                # Terminal placeholder is NOOP-only; no further agent action is requested.
                "action_mask": np.asarray(
                    view.action_mask
                    if not self.finished
                    else (1,) + (0,) * (len(view.action_mask) - 1),
                    dtype=np.int8,
                ),
            }
        return result

    def _infos(self, agents=None):
        return {
            aid: {
                "simulation_time": self.rewarded_tick
                if self.steps
                else self.projected.context.simulation_time,
                "end_reason": self.reason,
                "rounds": len(self.steps),
                "role": self.policy_for_agent(aid),
                **(
                    {
                        "reward_totals": dict(
                            self.role_reward_totals[self.policy_for_agent(aid)]
                        )
                    }
                    if self.extensions
                    else {}
                ),
            }
            for aid in (self.agents if agents is None else agents)
        }

    def step(self, actions):
        if self.finished:
            if actions:
                raise RuntimeError("resource episode has already ended")
            return {}, {}, {}, {}, {}
        # NumPy scalar indices are framework wire types. Domain decoding stays strict.
        actions = {
            k: int(v)
            if isinstance(v, np.integer) and not isinstance(v, np.bool_)
            else v
            for k, v in actions.items()
        }
        coordinator = JointActionCoordinator(self.projected, actions)
        decision, agents = self.projected, self.agents.copy()
        encoded_sha = (
            tuple((agent, digest(self.encoded_observations[agent])) for agent in agents)
            if self.extensions
            else None
        )
        trace_start = self._trace_cursor
        tick = decision.context.simulation_time
        terminated, truncated = False, False
        after_context = None
        try:
            outcome = coordinator.execute(self.simulator)
            tick = (
                outcome.makespan
                if isinstance(outcome, SimulationResult)
                else outcome.simulation_time
            )
            if isinstance(outcome, SimulationResult):
                self.result, self.reason, terminated = outcome, "completed", True
            else:
                after_context = outcome
        except (DeadlockError, PolicyStalledError) as exc:
            self.reason = (
                "deadlock" if isinstance(exc, DeadlockError) else "policy_stalled"
            )
            terminated = True
            # Failed settle can advance the clock without returning a decision.
            records = self.simulator.trace_since(trace_start)
            tick = max((r.simulation_time for r in records), default=tick)
        self.reason, reward, terminated, truncated = resource_outcome(
            tick=tick,
            rounds=len(self.steps) + 1,
            reason=self.reason,
            limits=self.limits,
            rewarded_tick=self.rewarded_tick,
            total_reward=self.total_reward,
        )
        self.rewarded_tick = tick
        self.total_reward += reward
        self.finished = terminated or truncated
        reward_values = (
            self._extension_rewards(
                decision.context, after_context, coordinator.actions, reward
            )
            if self.extensions
            else None
        )
        self._trace_cursor += len(self.simulator.trace_since(self._trace_cursor))
        self.steps.append(
            JointLearningStep(
                decision,
                tuple(sorted(actions.items())),
                tuple(coordinator.records),
                tuple(coordinator.actions),
                reward,
                tick,
                self.reason,
                trace_start,
                self._trace_cursor,
                reward_values,
                encoded_sha,
            )
        )
        if not self.finished:
            self.projected = self.projection.project(outcome)
            self._encode_observations()
        observations, infos = self._observations(agents), self._infos(agents)
        if self.finished:
            self.agents = []
            if self.on_episode:
                self.on_episode(self)
        return (
            observations,
            {
                agent: dict(reward_values.roles)[self.policy_for_agent(agent)].research
                for agent in agents
            }
            if reward_values
            else dict.fromkeys(agents, reward),
            dict.fromkeys(agents, terminated),
            dict.fromkeys(agents, truncated),
            infos,
        )

    def close(self):
        pass
