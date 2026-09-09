"""Replay public extension inputs and states using retained semantic decisions."""

from smartsom.config.codec import primitive
from smartsom.config.training import episode_input
from smartsom.engine import DeadlockError, Simulator
from smartsom.learning.checkpoint import CheckpointPolicy
from smartsom.learning.joint import PolicyStalledError
from smartsom.learning.resource_policy import ResourceCheckpointPolicy


def replay_extensions(
    resolved, records, trace, *, complete, preexecution=False, reward_records=None
):
    """Verify transformation evidence without loading or executing model weights.

    Model selections are retained inputs to this audit. Their public context,
    float32 observations, masks, semantic decoding and extension state evolution
    must match exactly. The ordinary action/joint audits establish physical replay.
    """
    cursor = 0
    observed = []
    observed_rewards = []
    resource = resolved.algorithm.algorithm.provider == "rllib.resource_ppo"

    def predict(view):
        nonlocal cursor
        if cursor >= len(records):
            raise ValueError("missing extension decision record")
        expected = records[cursor]
        cursor += 1
        views = expected["views"]
        if resource:
            result = {entry["agent_id"]: entry["index"] for entry in views}
            if len(result) != len(views):
                raise ValueError("duplicate extension agent record")
            return result
        if len(views) != 1:
            raise ValueError("central extension decision requires exactly one view")
        return views[0]["index"]

    def observe(actual):
        if primitive(actual) != records[len(observed)]:
            raise ValueError(
                f"extension decision replay mismatch at decision {len(observed)}"
            )
        observed.append(actual)

    def observe_reward(actual):
        if reward_records is not None and (
            len(observed_rewards) >= len(reward_records)
            or primitive(actual) != reward_records[len(observed_rewards)]
        ):
            raise ValueError(
                f"extension reward replay mismatch at transition {len(observed_rewards)}"
            )
        observed_rewards.append(actual)

    policy = (ResourceCheckpointPolicy if resource else CheckpointPolicy)(
        resolved, predictor=predict, on_extension=observe, on_reward=observe_reward
    )
    if policy.extensions is None:
        raise ValueError("extension evidence requires an extended checkpoint")
    if preexecution:
        if records or trace:
            raise ValueError("pre-execution failure has extension decision evidence")
        if not reward_records:
            return {"status": "partial_verified", "decisions": 0, "rewards": 0}
    inp = episode_input(resolved)
    try:
        sim = Simulator(inp.factory, inp.workload, **inp.options())
    except DeadlockError as exc:
        if complete or trace or records:
            raise ValueError("initial extension failure evidence mismatch") from exc
        policy.fail(exc, tick=0, trace_end=0)
        if reward_records is not None and len(observed_rewards) != len(reward_records):
            raise ValueError("extension reward coverage mismatch")
        return {
            "status": "partial_verified",
            "decisions": 0,
            "rewards": len(observed_rewards),
        }
    try:
        while sim.current_decision is not None:
            if cursor == len(records) and (not resource or policy.coordinator is None):
                break
            context = sim.current_decision
            if resource:
                policy.observed_trace_end = len(sim.trace)
            trace_start = len(sim.trace)
            try:
                outcome = sim.step(policy.select_action(context))
                if resource:
                    policy.check_outcome(outcome, trace_end=len(sim.trace))
                else:
                    policy.check_outcome(outcome)
            except (DeadlockError, PolicyStalledError, RuntimeError) as exc:
                policy.fail(
                    exc,
                    tick=max(
                        (row.simulation_time for row in sim.trace_since(trace_start)),
                        default=context.simulation_time,
                    ),
                    trace_end=len(sim.trace),
                )
                raise
    except (DeadlockError, PolicyStalledError) as exc:
        if complete:
            raise ValueError("complete extension replay terminated abnormally") from exc
    except RuntimeError as exc:
        if complete or "budget_exhausted" not in str(exc):
            raise
    if cursor != len(records) or len(observed) != len(records):
        raise ValueError("extension decision coverage mismatch")
    if reward_records is not None and len(observed_rewards) != len(reward_records):
        raise ValueError("extension reward coverage mismatch")
    if primitive(sim.trace) != trace:
        raise ValueError("extension decision physical trace mismatch")
    if complete and sim.current_decision is not None:
        raise ValueError("complete extension evidence ends before physical completion")
    return {
        "status": "passed" if complete else "partial_verified",
        "decisions": len(observed),
        "rewards": len(observed_rewards),
    }
