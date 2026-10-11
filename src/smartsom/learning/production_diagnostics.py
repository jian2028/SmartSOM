"""Bounded native learner statistics; detached, with no sampling or optimization."""

from math import isfinite

SCHEMA = "smartsom.native-learner-diagnostics/v2"


def _add(learner, values, *, minibatch=True):
    state = getattr(learner, "training_diagnostics", None)
    if state is None:
        state = learner.training_diagnostics = {
            "observed_minibatches": 0,
            "metrics": {},
        }
    state["observed_minibatches"] += int(minibatch)
    for name, (value, weight) in values.items():
        row = state["metrics"].setdefault(
            name, {"weighted_sum": 0.0, "weight": 0, "unavailable_minibatches": 0}
        )
        if weight <= 0 or not isfinite(value):
            row["unavailable_minibatches"] += 1
            continue
        row["weighted_sum"] += value * weight
        row["weight"] += weight


def record_ppo(
    learner,
    actor,
    critic,
    entropy,
    loss,
    log_ratio,
    active,
    value_mask,
    *,
    clip_range=None,
    raw_advantage=None,
    values=None,
    returns=None,
):
    """k3 approximate KL on valid joint actor packets, before this optimizer step."""
    actor_count = int(active.detach().sum().item())
    value_count = int(value_mask.detach().sum().item())
    valid = log_ratio.detach()[active.bool()].double()
    kl = (
        (valid.expm1() - valid).clamp_min(0).mean().item()
        if actor_count
        else float("nan")
    )
    _add(
        learner,
        {
            "actor_loss": (actor.detach().item(), actor_count),
            "value_loss": (critic.detach().item(), value_count),
            "entropy": (entropy.detach().item(), actor_count),
            "approx_kl_k3": (kl, actor_count),
            "total_loss": (loss.detach().item(), len(active)),
        },
    )

    extra = {}
    if clip_range is not None:
        extra["clip_fraction"] = (
            float((valid.exp().sub(1).abs() > clip_range).double().mean())
            if actor_count
            else float("nan"),
            actor_count,
        )
    if raw_advantage is not None:
        selected = raw_advantage.detach()[active.bool()].double()
        extra["raw_advantage_mean"] = (float(selected.mean()), actor_count)
        extra["raw_advantage_abs_mean"] = (float(selected.abs().mean()), actor_count)
    if values is not None and returns is not None:
        target = returns.detach()[value_mask.bool()].double()
        error = target - values.detach()[value_mask.bool()].double()
        moments = learner.training_diagnostics.setdefault(
            "value_moments",
            {
                "count": 0,
                "target_sum": 0.0,
                "target_square_sum": 0.0,
                "error_sum": 0.0,
                "error_square_sum": 0.0,
            },
        )
        if target.isfinite().all() and error.isfinite().all():
            moments["count"] += len(target)
            moments["target_sum"] += float(target.sum())
            moments["target_square_sum"] += float(target.square().sum())
            moments["error_sum"] += float(error.sum())
            moments["error_square_sum"] += float(error.square().sum())
    _add(learner, extra, minibatch=False)


def record_dqn(
    learner,
    loss,
    samples,
    choice_decisions=None,
    *,
    q=None,
    target=None,
    dt=None,
    terminated=None,
    replay_age=None,
):
    _add(learner, {"td_loss": (loss.detach().item(), samples)})
    state = learner.training_diagnostics
    coverage = state.setdefault(
        "q_coverage", {"choice_samples": 0, "forced_samples": 0, "unknown_samples": 0}
    )
    if choice_decisions is None:
        coverage["unknown_samples"] += samples
    else:
        values = choice_decisions.detach()
        coverage["choice_samples"] += int((values == 1).sum().item())
        coverage["forced_samples"] += int((values == 0).sum().item())
        coverage["unknown_samples"] += int((values < 0).sum().item())

    extra = {}
    for name, tensor in {
        "q": q,
        "target": target,
        "dt": dt,
        "terminal_fraction": terminated,
        "replay_age_insertions": replay_age,
        "td_error": target.detach() - q.detach()
        if q is not None and target is not None
        else None,
    }.items():
        if tensor is None:
            continue
        data = tensor.detach().double()
        if name == "replay_age_insertions":
            data = data[data >= 0]
        extra[name + "_mean" if name != "terminal_fraction" else name] = (
            float(data.mean()),
            len(data),
        )
        if name in ("q", "target", "td_error"):
            extra[name + "_abs_mean"] = (float(data.abs().mean()), len(data))
            extra[name + "_rms_square"] = (float(data.square().mean()), len(data))
    _add(learner, extra, minibatch=False)


def explained_variance(moments):
    count = moments.get("count", 0)
    if count < 2:
        return None
    target_variance = (
        moments["target_square_sum"] / count - (moments["target_sum"] / count) ** 2
    )
    if target_variance <= 0:
        return None
    error_variance = max(
        0.0, moments["error_square_sum"] / count - (moments["error_sum"] / count) ** 2
    )
    return 1 - error_variance / target_variance


def summarize(state):
    """Cumulative weighted means, never an average of differently sized batches."""
    state = state or {"observed_minibatches": 0, "metrics": {}}
    return {
        "observed_minibatches": state["observed_minibatches"],
        "value_explained_variance": explained_variance(state.get("value_moments", {})),
        "value_moments": dict(state.get("value_moments", {})),
        "metrics": {
            name: {
                "mean": row["weighted_sum"] / row["weight"] if row["weight"] else None,
                "weight": row["weight"],
                "weighted_sum": row["weighted_sum"],
                "unavailable_minibatches": row["unavailable_minibatches"],
            }
            for name, row in state["metrics"].items()
        },
    }


def learning_coverage(state, algorithm, collector, optimizer_steps):
    """Separate observed optimization from evidence of decision learning.

    Counts are training exposures (including epochs/replay reuse), not unique
    samples or proof of nonzero gradients or improved behavior.
    """
    state = state or {}
    metrics = state.get("metrics", {})
    actor = metrics.get("actor_loss", {})
    value = metrics.get("value_loss", {})
    q = state.get("q_coverage")
    return {
        "schema": "smartsom.learning-coverage/v1",
        "optimizer_steps": optimizer_steps,
        "optimizer_observed": optimizer_steps > 0,
        "actor_training_exposures": actor.get("weight") if algorithm == "ppo" else None,
        "critic_training_exposures": value.get("weight")
        if algorithm == "ppo"
        else None,
        "actor_packets_collected": collector.get("actor_packets")
        if algorithm == "ppo"
        else None,
        "q_training_exposures": metrics.get("td_loss", {}).get("weight")
        if algorithm == "dqn"
        else None,
        "q_choice_training_exposures": q.get("choice_samples") if q else None,
        "q_forced_training_exposures": q.get("forced_samples") if q else None,
        "q_unknown_training_exposures": q.get("unknown_samples") if q else None,
        "q_attribution_complete": bool(q)
        and q["unknown_samples"] == 0
        and sum(q.values()) == metrics.get("td_loss", {}).get("weight"),
        "scope": "policy_group; not individual owners in shared groups",
        "behavior_improvement": "not_established_by_coverage",
    }
