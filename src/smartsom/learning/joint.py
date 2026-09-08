"""Deterministic proposal coordination over the unchanged semantic step API."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from smartsom.dispatch import DecisionContext, WaitNextEvent
from smartsom.engine import SimulationResult, Simulator
from smartsom.learning.projection import visible_future_event
from smartsom.learning.resources import (
    PhysicalAction,
    ResourceDecision,
    action_key,
)

COORDINATION_VERSION = "smartsom.resource-coordination/v1"


class PolicyStalledError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ProposalRecord:
    agent_id: str
    index: int
    action: PhysicalAction | None
    job_id: str | None
    disposition: str


class JointActionCoordinator:
    """Queues one round; reads snapshots, never simulator internals.

    next_action/accept_outcome support ordinary online runners. execute uses
    exactly those methods for ParallelEnv and direct joint replay.
    """

    def __init__(self, decision: ResourceDecision, indices: Mapping[str, int]):
        if not isinstance(indices, Mapping) or set(indices) != {
            v.agent_id for v in decision.views
        }:
            raise ValueError("joint actions must contain exactly every resource agent")
        # Decode ALL proposals before any caller can advance the simulator.
        decoded = [
            (v, indices[v.agent_id], v.decode(indices[v.agent_id]))
            for v in decision.views
        ]
        self.decision = decision
        self.records = [
            ProposalRecord(
                v.agent_id,
                i,
                c.action if c else None,
                c.job_id if c else None,
                "pending" if c else "noop",
            )
            for v, i, c in decoded
        ]
        self.pending = sorted(
            [n for n, r in enumerate(self.records) if r.action is not None],
            key=lambda n: (
                action_key(self.records[n].action)[0],
                self.records[n].agent_id,
                action_key(self.records[n].action),
            ),
        )
        self.claimed: set[str] = set()
        self.actions = []
        self.finished = False
        self._waiting = not self.pending
        self._in_flight = False

    def _dispose(self, index, disposition):
        r = self.records[index]
        self.records[index] = ProposalRecord(
            r.agent_id, r.index, r.action, r.job_id, disposition
        )

    def _close(self, reason):
        for n in self.pending:
            self._dispose(
                n, "job_claimed" if self.records[n].job_id in self.claimed else reason
            )
        self.pending.clear()
        self.finished = True

    def next_action(self, context: DecisionContext):
        if self._in_flight:
            raise RuntimeError("coordinator requires the previous step outcome")
        if self.finished:
            return None
        if context.simulation_time != self.decision.context.simulation_time:
            self._close("epoch_advanced")
            return None
        if not self.actions and context != self.decision.context:
            raise ValueError("stale joint decision snapshot")
        if self._waiting:
            self._waiting = False
            if not visible_future_event(context):
                self.finished = True
                raise PolicyStalledError(
                    "policy_stalled: all resources chose NOOP without a public future-event witness"
                )
            action = WaitNextEvent()
        else:
            action = None
            while self.pending:
                n = self.pending.pop(0)
                r = self.records[n]
                if r.job_id in self.claimed:
                    self._dispose(n, "job_claimed")
                elif r.action not in context.feasible_actions:
                    self._dispose(n, "no_longer_feasible")
                else:
                    self._dispose(n, "accepted")
                    self.claimed.add(r.job_id)
                    action = r.action
                    break
            if action is None:
                self.finished = True
                return None
        self.actions.append(action)
        self._in_flight = True
        return action

    def accept_outcome(self, outcome: DecisionContext | SimulationResult):
        if not self._in_flight:
            raise RuntimeError("coordinator has no submitted action")
        self._in_flight = False
        if isinstance(outcome, SimulationResult):
            self._close("terminated")
        elif outcome.simulation_time != self.decision.context.simulation_time:
            self._close("epoch_advanced")
        else:
            # Resolve all stale/conflicting proposals now, without choosing new ones.
            retained = []
            for n in self.pending:
                r = self.records[n]
                if r.job_id in self.claimed:
                    self._dispose(n, "job_claimed")
                elif r.action not in outcome.feasible_actions:
                    self._dispose(n, "no_longer_feasible")
                else:
                    retained.append(n)
            self.pending = retained
            self.finished = not self.pending

    def fail(self):
        self._in_flight = False
        self._close("execution_failed")

    def execute(self, simulator: Simulator, *, on_step: Callable | None = None):
        if simulator.current_decision != self.decision.context:
            raise ValueError("stale joint decision snapshot")
        outcome = simulator.current_decision
        try:
            while not self.finished:
                action = self.next_action(outcome)
                if action is None:
                    break
                if on_step:
                    try:
                        on_step(outcome, action)
                    except BaseException:
                        # A failed pre-step writer did not execute the queued action.
                        self.actions.pop()
                        for n, r in enumerate(self.records):
                            if r.action == action and r.disposition == "accepted":
                                self._dispose(n, "execution_failed")
                        raise
                outcome = simulator.step(action)
                self.accept_outcome(outcome)
        except BaseException:
            self.fail()
            raise
        return outcome
