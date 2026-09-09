"""Framework-free digests of the actual float32 policy input and extension state."""

import math
import struct

from smartsom.config.codec import digest


def wire_observation(value):
    """Match the float32 tensor boundary without importing optional frameworks."""
    if isinstance(value, dict):
        return {key: wire_observation(item) for key, item in value.items()}
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (tuple, list)):
        return tuple(wire_observation(item) for item in value)
    try:
        result = struct.unpack("f", struct.pack("f", float(value)))[0]
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(
            "encoded observation exceeds finite float32 representation"
        ) from exc
    if not math.isfinite(result):
        raise ValueError("encoded observation exceeds finite float32 representation")
    return result


def decision_record(
    *,
    decision_index,
    context,
    checkpoint_sha256,
    runtime,
    state_before_sha256,
    views,
    indices,
):
    return {
        "schema": "smartsom.extension-decision/v1",
        "decision_index": decision_index,
        "context_sha256": digest(context),
        "checkpoint_sha256": checkpoint_sha256,
        "extensions_sha256": digest(runtime.spec),
        "state_before_sha256": state_before_sha256,
        "state_after_sha256": digest(runtime.state_dict()),
        "views": tuple(
            {
                "agent_id": getattr(view.mapping, "agent_id", None),
                "role": getattr(view.mapping, "role", None),
                "observations_sha256": digest(wire_observation(view.observations)),
                "mask_sha256": digest(view.action_mask),
                "index": index,
            }
            for view, index in zip(views, indices, strict=True)
        ),
    }


def reward_record(
    *,
    decision_index,
    checkpoint_sha256,
    runtime,
    state_before_sha256,
    transition,
    values,
):
    """Separate actual raw, research and learner values from the legacy ledgers."""
    from smartsom.learning.extensions import RewardBatch

    return {
        "schema": "smartsom.extension-reward/v1",
        "decision_index": decision_index,
        "checkpoint_sha256": checkpoint_sha256,
        "extensions_sha256": digest(runtime.spec),
        "transition_sha256": digest(transition),
        "simulation_time": transition.simulation_time,
        "reason": transition.reason,
        "state_before_sha256": state_before_sha256,
        "state_after_sha256": digest(runtime.state_dict()),
        "team": values.team if isinstance(values, RewardBatch) else values,
        "roles": values.roles if isinstance(values, RewardBatch) else (),
    }
