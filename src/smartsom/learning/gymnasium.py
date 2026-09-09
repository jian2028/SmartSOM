"""Optional Gymnasium protocol over the sole SmartSOM physics state machine."""

import copy
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import gymnasium as gym
import numpy as np

from smartsom.config.codec import digest
from smartsom.dispatch import DecisionContext
from smartsom.domain.actions import SemanticAction
from smartsom.engine import DeadlockError, SimulationResult, Simulator
from smartsom.learning.episode import EpisodeInput, EpisodeLimits, EpisodeStartFailure
from smartsom.learning.extensions import (
    ExtensionsRuntime,
    RewardTransition,
    central_observation,
)
from smartsom.learning.projection import (
    LearningProjection,
    ProjectedDecision,
    ProjectionSpec,
    validate_capacity,
)


@dataclass(frozen=True, slots=True)
class LearningStep:
    projection: ProjectedDecision
    action_index: int
    action: SemanticAction | None
    reward: float
    simulation_time: int
    reason: str | None
    research_reward: float | None = None
    learner_reward: float | None = None
    encoded_observations_sha256: str | None = None


def extension_gym_space(space):
    if space.vector:
        return gym.spaces.Box(-np.inf, np.inf, space.vector, np.float32)
    return gym.spaces.Dict(
        {
            name: gym.spaces.Box(-np.inf, np.inf, shape, np.float32)
            for name, shape in space.fields
        }
    )


def extension_numpy(value):
    if isinstance(value, dict):
        return {key: extension_numpy(child) for key, child in value.items()}
    result = np.asarray(value, dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError("extension observation exceeds finite float32 representation")
    return result


class SchedulingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        episode: EpisodeInput,
        projection: ProjectionSpec,
        *,
        limits: EpisodeLimits = EpisodeLimits(),
        observation_kind: Literal["plain", "masked"] = "plain",
        episode_source: Callable[[int], EpisodeInput] | None = None,
        on_episode: Callable[["SchedulingEnv"], None] | None = None,
        strict_actions: bool = False,
        extensions=None,
        provider=None,
        learner_scale=1.0,
    ):
        validate_capacity(projection, episode.workload, episode.quality)
        if observation_kind not in ("plain", "masked"):
            raise ValueError("unknown learning observation kind")
        self.base_input = episode
        self.projection_spec = projection
        self.limits = limits
        self.observation_kind = observation_kind
        self.episode_source = episode_source
        self.on_episode = on_episode
        self.strict_actions = strict_actions
        self.projection = LearningProjection(episode.factory, projection)
        self.extensions = None
        if extensions is not None:
            prototype = DecisionContext(0, (), (), ())
            layout = central_observation(
                prototype, self.projection.project(prototype)
            ).layout()
            self.extensions = ExtensionsRuntime(
                extensions,
                provider
                or (
                    "rllib.ppo" if observation_kind == "masked" else "sb3.maskable_ppo"
                ),
                {None: layout},
                learner_scale=learner_scale,
            )
        self.action_space = gym.spaces.Discrete(self.projection.action_count)
        self.state_space = gym.spaces.Box(
            -np.inf, np.inf, (self.projection.observation_size,), np.float32
        )
        if self.extensions:
            self.state_space = extension_gym_space(self.extensions.spaces[None])
        self.observation_space = (
            self.state_space
            if observation_kind == "plain"
            else gym.spaces.Dict(
                {
                    "observations": self.state_space,
                    "action_mask": gym.spaces.Box(
                        0, 1, (self.projection.action_count,), np.float32
                    ),
                }
            )
        )
        self.episode_index = -1
        self.simulator = None
        self.result: SimulationResult | None = None
        self.projected: ProjectedDecision | None = None
        self.steps: list[LearningStep] = []
        self.finished = True

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if options:
            raise ValueError("episode inputs are controlled by the scientific source")
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
        validate_capacity(self.projection_spec, self.input.workload, self.input.quality)
        self.projection = LearningProjection(self.input.factory, self.projection_spec)
        self.steps = []
        self.result = None
        self.projected = None
        self.total_reward = 0.0
        self.total_research_reward = self.total_learner_reward = 0.0
        self.encoded_observation = None
        if self.extensions:
            self.episode_extension_initial_state = self.extensions.state_dict()
            self.extensions.begin_episode()
        self.rewarded_tick = 0
        self._trace_cursor = 0
        self.reason = None
        self.finished = False
        self.simulator = None
        try:
            self.simulator = Simulator(
                self.input.factory, self.input.workload, **self.input.options()
            )
        except DeadlockError as exc:
            self.finished = True
            self.reason = "deadlock"
            self.total_reward = -float(self.limits.max_ticks + 1)
            self._initial_extension_reward(None)
            self._notify()
            raise EpisodeStartFailure("deadlock") from exc
        self.projected = self.projection.project(self.simulator.current_decision)
        if not any(self.projected.action_mask):
            self.finished = True
            self.reason = "policy_stalled"
            self.rewarded_tick = self.simulator.current_decision.simulation_time
            self.total_reward = -float(
                max(
                    self.limits.max_ticks + 1,
                    self.simulator.current_decision.simulation_time,
                )
            )
            self._initial_extension_reward(self.simulator.current_decision)
            self._notify()
            raise EpisodeStartFailure("policy_stalled")
        self._encode_observation()
        return self._observation(), self._info()

    def _initial_extension_reward(self, context):
        if self.extensions:
            values = self.extensions.reward(
                RewardTransition(
                    context,
                    None,
                    (),
                    self.total_reward,
                    self.rewarded_tick,
                    self.reason,
                )
            )
            self.total_research_reward, self.total_learner_reward = (
                values.research,
                values.learner,
            )

    def _encode_observation(self):
        if self.extensions:
            self.encoded_observation = self.extensions.encode(
                central_observation(self.simulator.current_decision, self.projected)
            )

    def extension_state_dict(self):
        if not self.extensions:
            return None
        return {
            "schema": "smartsom.extension-env-state/v1",
            "runtime": self.extensions.state_dict(),
            "episode_initial_state": copy.deepcopy(
                self.episode_extension_initial_state
            ),
            "observations": copy.deepcopy(self.encoded_observation),
            "research_return": self.total_research_reward,
            "learner_return": self.total_learner_reward,
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
        self.encoded_observation = (
            self.extensions.spaces[None].validate(state["observations"])
            if state["observations"] is not None
            else None
        )
        if not self.finished and self.encoded_observation is None:
            raise ValueError("active environment extension state lacks its observation")
        self.total_research_reward, self.total_learner_reward = (
            float(state["research_return"]),
            float(state["learner_return"]),
        )

    def action_masks(self):
        return np.asarray(
            self.projected.action_mask
            if not self.finished
            else (0,) * self.action_space.n,
            dtype=np.bool_,
        )

    def _observation(self):
        if self.extensions:
            state = extension_numpy(
                self.extensions.spaces[None].zeros()
                if self.finished
                else self.encoded_observation
            )
            return (
                state
                if self.observation_kind == "plain"
                else {
                    "observations": state,
                    "action_mask": self.action_masks().astype(np.float32),
                }
            )
        values = (
            self.projected.observations
            if not self.finished
            else (0.0,) * self.state_space.shape[0]
        )
        state = np.asarray(values, dtype=np.float32)
        if not np.isfinite(state).all():
            raise ValueError("observation exceeds finite float32 representation")
        if self.observation_kind == "plain":
            return state
        return {
            "observations": state,
            "action_mask": self.action_masks().astype(np.float32),
        }

    def _info(self):
        result = {
            "simulation_time": self.rewarded_tick
            if self.finished
            else self.simulator.current_decision.simulation_time,
            "end_reason": self.reason,
            "is_success": self.reason == "completed",
            "makespan": self.result.makespan if self.reason == "completed" else None,
        }
        if self.extensions:
            result["reward_totals"] = {
                "raw": self.total_reward,
                "research": self.total_research_reward,
                "learner": self.total_learner_reward,
            }
        return result

    def _notify(self):
        if self.on_episode is not None:
            self.on_episode(self)

    def step(self, action):
        if self.finished:
            raise RuntimeError("reset required after episode termination")
        before = self.projected
        before_context = self.simulator.current_decision
        encoded_sha = digest(self.encoded_observation) if self.extensions else None
        after_context = None
        semantic = None
        reason = None
        truncated = False
        tick = self.simulator.current_decision.simulation_time
        try:
            if isinstance(action, (bool, np.bool_)) or not isinstance(
                action, (int, np.integer)
            ):
                raise ValueError("learning action must be an integer")
            semantic = before.decode(int(action))
        except ValueError:
            if self.strict_actions:
                raise
            # Gym's checker samples without respecting masks. Report a fatal failed
            # episode without changing physics; training always uses strict_actions.
            reason = "invalid_action"
        if reason is None:
            try:
                outcome = self.simulator.step(semantic)
            except DeadlockError:
                reason = "deadlock"
                tick = max(
                    (
                        r.simulation_time
                        for r in self.simulator.trace_since(self._trace_cursor)
                    ),
                    default=tick,
                )
            else:
                if isinstance(outcome, SimulationResult):
                    self.result = outcome
                    tick = outcome.makespan
                    reason = "completed"
                else:
                    after_context = outcome
                    tick = outcome.simulation_time
                    self.projected = self.projection.project(outcome)
                    if not any(self.projected.action_mask):
                        reason = "policy_stalled"
            self._trace_cursor += len(self.simulator.trace_since(self._trace_cursor))
            if reason not in ("deadlock", "policy_stalled") and (
                tick > self.limits.max_ticks
                or (
                    reason != "completed"
                    and (
                        tick >= self.limits.max_ticks
                        or len(self.steps) + 1 >= self.limits.max_decisions
                    )
                )
            ):
                reason, truncated = "budget_exhausted", True
        reward = -(tick - self.rewarded_tick)
        if reason is not None and reason != "completed":
            reward = -max(self.limits.max_ticks + 1, tick) - self.total_reward
        self.total_reward += reward
        self.rewarded_tick = tick
        self.reason = reason
        self.finished = reason is not None
        values = (
            self.extensions.reward(
                RewardTransition(
                    before_context,
                    after_context,
                    (semantic,) if semantic is not None else (),
                    reward,
                    tick,
                    reason,
                )
            )
            if self.extensions
            else None
        )
        if values:
            self.total_research_reward += values.research
            self.total_learner_reward += values.learner
        if not self.finished:
            self._encode_observation()
        self.steps.append(
            LearningStep(
                before,
                int(action) if isinstance(action, (int, np.integer)) else -1,
                semantic,
                reward,
                tick,
                reason,
                values.research if values else None,
                values.learner if values else None,
                encoded_sha,
            )
        )
        if self.finished:
            self._notify()
        return (
            self._observation(),
            float(values.research if values else reward),
            self.finished and not truncated,
            truncated,
            self._info(),
        )
