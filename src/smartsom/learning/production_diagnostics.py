"""Bounded native learner statistics; detached, with no sampling or optimization."""

from math import isfinite

SCHEMA = "smartsom.native-learner-diagnostics/v1"


def _add(learner, values):
    state = getattr(learner, "training_diagnostics", None)
    if state is None:
        state = learner.training_diagnostics = {
            "observed_minibatches": 0,
            "metrics": {},
        }
    state["observed_minibatches"] += 1
    for name, (value, weight) in values.items():
        row = state["metrics"].setdefault(
            name, {"weighted_sum": 0.0, "weight": 0, "unavailable_minibatches": 0}
        )
        if weight <= 0 or not isfinite(value):
            row["unavailable_minibatches"] += 1
            continue
        row["weighted_sum"] += value * weight
        row["weight"] += weight


def record_ppo(learner, actor, critic, entropy, loss, log_ratio, active, value_mask):
    """k3 approximate KL on valid joint actor packets, before this optimizer step."""
    actor_count = int(active.detach().sum().item())
    value_count = int(value_mask.detach().sum().item())
    valid = log_ratio.detach()[active.bool()]
    kl = ((valid.exp() - 1) - valid).mean().item() if actor_count else float("nan")
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


def record_dqn(learner, loss, samples):
    _add(learner, {"td_loss": (loss.detach().item(), samples)})


def summarize(state):
    """Cumulative weighted means, never an average of differently sized batches."""
    state = state or {"observed_minibatches": 0, "metrics": {}}
    return {
        "observed_minibatches": state["observed_minibatches"],
        "metrics": {
            name: {
                "mean": row["weighted_sum"] / row["weight"] if row["weight"] else None,
                "weight": row["weight"],
                "unavailable_minibatches": row["unavailable_minibatches"],
            }
            for name, row in state["metrics"].items()
        },
    }
