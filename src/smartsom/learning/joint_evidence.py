"""Compact, versioned joint records, independent of frameworks and file writers."""

from smartsom.config.codec import digest
from smartsom.learning.joint import COORDINATION_VERSION


def round_record(
    *,
    round_index,
    decision,
    indices,
    proposals,
    actions,
    reward,
    tick,
    reason,
    trace_start,
    trace_end,
    full=False,
):
    return {
        "schema": "smartsom.joint-round/v1",
        "coordination_version": COORDINATION_VERSION,
        "round": round_index,
        "start_tick": decision.context.simulation_time,
        "indices": tuple(sorted(indices.items())),
        "bindings": decision.bindings,
        "agents": tuple(
            {
                "agent_id": v.agent_id,
                "role": v.role,
                "candidates": v.candidates,
                "observations_sha256": digest(v.observations),
                "mask_sha256": digest(v.action_mask),
                **({"observations": v.observations} if full else {}),
            }
            for v in decision.views
        ),
        "proposals": proposals,
        "actions": actions,
        "reward": reward,
        "simulation_time": tick,
        "reason": reason,
        "trace_start": trace_start,
        "trace_end": trace_end,
    }


def step_record(index, step, *, full=False):
    return round_record(
        round_index=index,
        decision=step.decision,
        indices=dict(step.indices),
        proposals=step.proposals,
        actions=step.actions,
        reward=step.reward,
        tick=step.simulation_time,
        reason=step.reason,
        trace_start=step.trace_start,
        trace_end=step.trace_end,
        full=full,
    )
