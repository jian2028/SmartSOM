"""Optional Gymnasium protocol over the sole SmartSOM physics state machine."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import gymnasium as gym
import numpy as np

from smartsom.domain.actions import SemanticAction
from smartsom.engine import DeadlockError, SimulationResult, Simulator
from smartsom.learning.episode import EpisodeInput, EpisodeLimits, EpisodeStartFailure
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
        self.action_space = gym.spaces.Discrete(self.projection.action_count)
        self.state_space = gym.spaces.Box(
            -np.inf, np.inf, (self.projection.observation_size,), np.float32
        )
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
            self._notify()
            raise EpisodeStartFailure("policy_stalled")
        return self._observation(), self._info()

    def action_masks(self):
        return np.asarray(
            self.projected.action_mask
            if not self.finished
            else (0,) * self.action_space.n,
            dtype=np.bool_,
        )

    def _observation(self):
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
        return {
            "simulation_time": self.rewarded_tick
            if self.finished
            else self.simulator.current_decision.simulation_time,
            "end_reason": self.reason,
            "is_success": self.reason == "completed",
            "makespan": self.result.makespan if self.reason == "completed" else None,
        }

    def _notify(self):
        if self.on_episode is not None:
            self.on_episode(self)

    def step(self, action):
        if self.finished:
            raise RuntimeError("reset required after episode termination")
        before = self.projected
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
        self.steps.append(
            LearningStep(
                before,
                int(action) if isinstance(action, (int, np.integer)) else -1,
                semantic,
                reward,
                tick,
                reason,
            )
        )
        if self.finished:
            self._notify()
        return (
            self._observation(),
            float(reward),
            self.finished and not truncated,
            truncated,
            self._info(),
        )
