"""Checkpoint inference queues joint proposals through the existing OnlinePolicy."""

from pathlib import Path

from smartsom.engine import DeadlockError, SimulationResult
from smartsom.learning.checkpoint import validate_checkpoint
from smartsom.learning.episode import resource_outcome
from smartsom.learning.joint import JointActionCoordinator, PolicyStalledError
from smartsom.learning.joint_evidence import round_record
from smartsom.learning.resources import ResourceProjection


class ResourceCheckpointPolicy:
    """No simulator access; fresh snapshots and transition acknowledgements only."""

    def __init__(self, resolved, *, predictor=None, on_round=None, deterministic=True):
        self.manifest = validate_checkpoint(resolved)
        spec = resolved.algorithm.algorithm
        self.projection = ResourceProjection(
            resolved.factory,
            spec.projection,
            transport_enabled=resolved.transport_enabled,
        )
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
            self.indices = self.predict(decision)
            self.coordinator = JointActionCoordinator(decision, self.indices)
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

    def _finish(self, tick, reason, trace_end):
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
            )

    def fail(self, exc, *, tick, trace_end):
        if self.coordinator is None:
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
