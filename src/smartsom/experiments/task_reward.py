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


def starved_machines(sim):
    """Idle, working machines with an empty input buffer while a job that has
    left the system input waits elsewhere for one of their operations next."""
    waiting = set()
    for row in sim.jobs.values():
        location = row["location"]
        if (
            location == "queue"
            or location in sim.machines
            or sim.roles.get(location) in ("system_input", "system_output")
            or row["quality"] == "FAIL"
        ):
            continue
        steps = sim.demands[row["demand"]].steps
        if row["step"] < len(steps):
            waiting.add((steps[row["step"]].operation_type, location))
    count = 0
    for machine, state in sim.machine_state.items():
        pre = sim.pre.get(machine)
        if (
            state["status"] != "IDLE"
            or state["down"]
            or (pre is not None and any(sim.storage[pre].values()))
        ):
            continue
        operations = sim.machines[machine].operation_types
        if any(op in operations and at != pre for op, at in waiting):
            count += 1
    return count


def starvation_penalty(spec, sim):
    if not spec.starvation_weight:
        return 0.0
    return (
        spec.starvation_weight
        * starved_machines(sim)
        / (max(len(sim.machine_state), 1) * spec.reference_ticks)
    )
