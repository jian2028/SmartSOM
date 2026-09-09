"""Checkpoint inference queues joint proposals through the existing OnlinePolicy."""

import json
from pathlib import Path

from smartsom.config.codec import digest
from smartsom.dispatch import DecisionContext
from smartsom.engine import DeadlockError, SimulationResult
from smartsom.learning.checkpoint import validate_checkpoint
from smartsom.learning.episode import resource_outcome
from smartsom.learning.extension_evidence import decision_record, reward_record
from smartsom.learning.extensions import (
    EncodedDecision,
    EncodedResourceDecision,
    ExtensionsRuntime,
    RewardTransition,
    resource_observation,
)
from smartsom.learning.joint import JointActionCoordinator, PolicyStalledError
from smartsom.learning.joint_evidence import round_record
from smartsom.learning.resources import ResourceProjection


class ResourceCheckpointPolicy:
    """No simulator access; fresh snapshots and transition acknowledgements only."""

    def __init__(
        self,
        resolved,
        *,
        predictor=None,
        on_round=None,
        deterministic=True,
        on_extension=None,
        on_reward=None,
    ):
        self.manifest = validate_checkpoint(resolved)
        spec = resolved.algorithm.algorithm
        self.checkpoint_sha256 = spec.checkpoint_sha256
        self.on_extension = on_extension
        self.on_reward = on_reward
        self.extension_finished = False
        self.projection = ResourceProjection(
            resolved.factory,
            spec.projection,
            transport_enabled=resolved.transport_enabled,
        )
        self.extensions = None
        if spec.extensions is not None:
            prototype = self.projection.project(DecisionContext(0, (), (), ()))
            layouts = {
                view.role: resource_observation(prototype.context, view).layout()
                for view in prototype.views
            }
            self.extensions = ExtensionsRuntime(
                spec.extensions,
                spec.provider,
                layouts,
                learner_scale=spec.parameters.learner_reward_scale,
            )
            state = json.loads(
                (Path(spec.checkpoint) / self.manifest.extension_state.path).read_text()
            )
            self.extensions.load_state_dict(state)
            self.extensions.begin_episode()
        self.limits = resolved.run.budget.limits()
        if predictor is None:
            from smartsom.learning.rllib_resource import load_predictor

            predictor = (
                load_predictor(Path(spec.checkpoint), self.manifest)
                if deterministic
                else load_predictor(
                    Path(spec.checkpoint),
                    self.manifest,
                    deterministic=False,
                    seed=next(
                        s.value for s in resolved.seeds if s.domain == "algorithm"
                    ),
                )
            )
        self.predict, self.on_round = predictor, on_round
        self.full = (
            resolved.run.recording is None
            or resolved.run.recording.observations == "full"
        )
        self.coordinator = None
        self.rounds = self.trace_cursor = self.rewarded_tick = 0
        self.observed_trace_end = 0
        self.total_reward = 0.0

    def select_action(self, context):
        if self.coordinator is None:
            if self.rounds >= self.limits.max_decisions:
                raise RuntimeError(
                    "budget_exhausted: resource checkpoint episode limit"
                )
            decision = self.projection.project(context)
            state_before_sha256 = (
                digest(self.extensions.state_dict()) if self.extensions else None
            )
            encoded = (
                EncodedResourceDecision(
                    decision,
                    tuple(
                        EncodedDecision(
                            view,
                            self.extensions.encode(resource_observation(context, view)),
                        )
                        for view in decision.views
                    ),
                )
                if self.extensions
                else decision
            )
            self.indices = self.predict(encoded)
            self.coordinator = JointActionCoordinator(decision, self.indices)
            if self.extensions and self.on_extension:
                self.on_extension(
                    decision_record(
                        decision_index=self.rounds,
                        context=context,
                        checkpoint_sha256=self.checkpoint_sha256,
                        runtime=self.extensions,
                        state_before_sha256=state_before_sha256,
                        views=encoded.views,
                        indices=tuple(
                            int(self.indices[v.agent_id]) for v in decision.views
                        ),
                    )
                )
        try:
            action = self.coordinator.next_action(context)
        except PolicyStalledError:
            self._finish(
                context.simulation_time, "policy_stalled", self.observed_trace_end
            )
            raise
        if action is None:
            raise RuntimeError("coordinator ended without acknowledging its outcome")
        return action

    def _finish(self, tick, reason, trace_end, after=None):
        c = self.coordinator
        self.rounds += 1
        reason, reward, _, _ = resource_outcome(
            tick=tick,
            rounds=self.rounds,
            reason=reason,
            limits=self.limits,
            rewarded_tick=self.rewarded_tick,
            total_reward=self.total_reward,
        )
        record = round_record(
            round_index=self.rounds - 1,
            decision=c.decision,
            indices=self.indices,
            proposals=tuple(c.records),
            actions=tuple(c.actions),
            reward=reward,
            tick=tick,
            reason=reason,
            trace_start=self.trace_cursor,
            trace_end=trace_end,
            full=self.full,
        )
        self.trace_cursor, self.rewarded_tick = trace_end, tick
        self.total_reward += reward
        self.coordinator = None
        if self.extensions:
            self._extension_reward(
                c.decision.context,
                after,
                tuple(c.actions),
                reward,
                tick,
                reason,
                self.rounds - 1,
            )
        if self.on_round:
            self.on_round(record)
        if reason == "budget_exhausted":
            raise RuntimeError(
                f"budget_exhausted: actual resource evaluation tick {tick}"
            )

    def check_outcome(self, outcome, *, trace_end):
        self.observed_trace_end = trace_end
        self.coordinator.accept_outcome(outcome)
        tick = (
            outcome.makespan
            if isinstance(outcome, SimulationResult)
            else outcome.simulation_time
        )
        # Limits are checked after each atomic action, as in ParallelEnv; a round
        # stops at that boundary rather than submitting additional queued proposals.
        if tick > self.limits.max_ticks or self.coordinator.finished:
            self._finish(
                tick,
                "completed" if isinstance(outcome, SimulationResult) else None,
                trace_end,
                None if isinstance(outcome, SimulationResult) else outcome,
            )

    def fail(self, exc, *, tick, trace_end):
        if self.coordinator is None:
            if self.extensions and not self.extension_finished and self.rounds == 0:
                raw = -float(max(self.limits.max_ticks + 1, tick))
                self.total_reward = raw
                self.rewarded_tick = tick
                self._extension_reward(
                    None,
                    None,
                    (),
                    raw,
                    tick,
                    "deadlock"
                    if isinstance(exc, DeadlockError)
                    else "execution_failed",
                    None,
                )
            return
        self.coordinator.fail()
        reason = (
            "deadlock"
            if isinstance(exc, DeadlockError)
            else "policy_stalled"
            if isinstance(exc, PolicyStalledError)
            else "execution_failed"
        )
        self._finish(tick, reason, trace_end)

    def _extension_reward(self, before, after, actions, raw, tick, reason, index):
        transition = RewardTransition(before, after, actions, raw, tick, reason)
        state_before = digest(self.extensions.state_dict())
        values = self.extensions.rewards(
            transition, {view[1] for view in self.manifest.role_mapping}
        )
        self.extension_finished = reason is not None
        if self.on_reward:
            self.on_reward(
                reward_record(
                    decision_index=index,
                    checkpoint_sha256=self.checkpoint_sha256,
                    runtime=self.extensions,
                    state_before_sha256=state_before,
                    transition=transition,
                    values=values,
                )
            )
