"""Explicit V3 task objective over privileged committed simulator statistics."""


def shipment_reward(spec, components):
    return (
        spec.shipment_weight * components["shipments"] / spec.reference_jobs
        + spec.passing_weight * components["passing_change"]
        - spec.tardiness_weight
        * components["overdue_time"]
        / (spec.reference_jobs * spec.reference_ticks)
    )
