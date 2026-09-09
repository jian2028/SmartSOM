"""Replay public extension inputs and states using retained semantic decisions."""

from smartsom.config.codec import primitive
from smartsom.config.training import episode_input
from smartsom.engine import DeadlockError, Simulator
from smartsom.learning.checkpoint import CheckpointPolicy
from smartsom.learning.joint import PolicyStalledError
from smartsom.learning.resource_policy import ResourceCheckpointPolicy


def replay_extensions(resolved, records, trace, *, complete, preexecution=False):
    """Verify transformation evidence without loading or executing model weights.

    Model selections are retained inputs to this audit. Their public context,
    float32 observations, masks, semantic decoding and extension state evolution
    must match exactly. The ordinary action/joint audits establish physical replay.
    """
    cursor = 0
    observed = []
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

    policy = (ResourceCheckpointPolicy if resource else CheckpointPolicy)(
        resolved, predictor=predict, on_extension=observe
    )
    if policy.extensions is None:
        raise ValueError("extension evidence requires an extended checkpoint")
    if preexecution:
        if records or trace:
            raise ValueError("pre-execution failure has extension decision evidence")
        return {"status": "partial_verified", "decisions": 0}
    inp = episode_input(resolved)
    sim = Simulator(inp.factory, inp.workload, **inp.options())
    try:
        while sim.current_decision is not None:
            if cursor == len(records) and (not resource or policy.coordinator is None):
                break
            context = sim.current_decision
            if resource:
                policy.observed_trace_end = len(sim.trace)
            outcome = sim.step(policy.select_action(context))
            if resource:
                policy.check_outcome(outcome, trace_end=len(sim.trace))
            else:
                policy.check_outcome(outcome)
    except (DeadlockError, PolicyStalledError) as exc:
        if complete:
            raise ValueError("complete extension replay terminated abnormally") from exc
    except RuntimeError as exc:
        if complete or "budget_exhausted" not in str(exc):
            raise
    if cursor != len(records) or len(observed) != len(records):
        raise ValueError("extension decision coverage mismatch")
    if primitive(sim.trace) != trace:
        raise ValueError("extension decision physical trace mismatch")
    if complete and sim.current_decision is not None:
        raise ValueError("complete extension evidence ends before physical completion")
    return {
        "status": "passed" if complete else "partial_verified",
        "decisions": len(observed),
    }
