"""Framework-free verification of joint ledgers through the same coordinator."""

from dataclasses import dataclass

from smartsom.config.codec import primitive
from smartsom.engine import DeadlockError, SimulationResult, Simulator
from smartsom.learning.episode import EpisodeLimits, resource_outcome
from smartsom.learning.joint import JointActionCoordinator, PolicyStalledError
from smartsom.learning.joint_evidence import round_record
from smartsom.learning.resources import ResourceProjection


@dataclass(frozen=True, slots=True)
class JointReplayResult:
    result: SimulationResult | None
    trace: tuple
    actions: tuple
    bindings: tuple
    total_reward: float
    reason: str | None
    rounds: int


def replay_joint(episode, projection, records, *, limits=EpisodeLimits()):
    """Compare every recorded proposal, mapping, transition and reward exactly.

    A nonterminal prefix is returned explicitly for training-budget evidence;
    callers requiring a completed schedule must check result/reason.
    """
    sim = Simulator(episode.factory, episode.workload, **episode.options())
    p = ResourceProjection(
        episode.factory, projection, transport_enabled=episode.transport_enabled
    )
    reason, result = None, None
    total, rewarded_tick, cursor, rounds = 0.0, 0, 0, 0
    actions = []
    for index, expected in enumerate(records):
        if reason is not None:
            raise ValueError("joint replay contains rounds after termination")
        decision = p.project(sim.current_decision)
        indices = dict(expected["indices"])
        if len(indices) != len(expected["indices"]):
            raise ValueError("joint replay contains duplicate agent indices")
        coordinator = JointActionCoordinator(decision, indices)
        tick = decision.context.simulation_time
        try:
            outcome = coordinator.execute(sim)
            if isinstance(outcome, SimulationResult):
                result, reason, tick = outcome, "completed", outcome.makespan
            else:
                tick = outcome.simulation_time
        except (DeadlockError, PolicyStalledError) as exc:
            reason = "deadlock" if isinstance(exc, DeadlockError) else "policy_stalled"
            tick = max(
                (r.simulation_time for r in sim.trace_since(cursor)), default=tick
            )
        reason, reward, _, _ = resource_outcome(
            tick=tick,
            rounds=index + 1,
            reason=reason,
            limits=limits,
            rewarded_tick=rewarded_tick,
            total_reward=total,
        )
        end = cursor + len(sim.trace_since(cursor))
        actual = round_record(
            round_index=index,
            decision=decision,
            indices=indices,
            proposals=tuple(coordinator.records),
            actions=tuple(coordinator.actions),
            reward=reward,
            tick=tick,
            reason=reason,
            trace_start=cursor,
            trace_end=end,
            full=any("observations" in a for a in expected["agents"]),
        )
        if primitive(actual) != primitive(expected):
            raise ValueError(f"joint replay record mismatch at round {index}")
        actions.extend(coordinator.actions)
        total, rewarded_tick, cursor, rounds = total + reward, tick, end, index + 1
    if reason is None:
        # ParallelEnv exposes the next public observation after a nonterminal
        # step, even if the outer training budget ends before another action.
        p.project(sim.current_decision)
    return JointReplayResult(
        result if reason == "completed" else None,
        sim.trace,
        tuple(actions),
        p.bindings,
        total,
        reason,
        rounds,
    )
