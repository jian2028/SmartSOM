"""Explicit V3 task objective over privileged committed simulator statistics."""


def configured_task_reward(config):
    """The same explicit objective for learning and rule/control evaluation."""
    training = getattr(config, "training", None)
    legacy = training.reward.task if training is not None else None
    selected = getattr(config, "task_reward", None)
    if selected is not None and legacy is not None and selected != legacy:
        raise ValueError("experiment and training task reward contracts disagree")
    return selected if selected is not None else legacy


def shipment_reward(spec, components):
    return (
        spec.shipment_weight * components["shipments"] / spec.reference_jobs
        + spec.passing_weight * components["passing_change"]
        - spec.tardiness_weight
        * components["overdue_time"]
        / (spec.reference_jobs * spec.reference_ticks)
    )
