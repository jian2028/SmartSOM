"""Deterministic transport baseline, separate from machine dispatch rules."""

from smartsom.dispatch import DecisionContext, Transport, WaitNextEvent


def select_transport(context: DecisionContext) -> Transport | WaitNextEvent:
    machines = {m.machine_id: m for m in context.machines}

    def idle(key):
        m = machines[key]
        return m.operation_id is None and m.availability == "up"

    candidates = tuple(
        x
        for x in context.transport_candidates
        if x.source_machine_id is None
        or (not idle(x.source_machine_id) and idle(x.action.destination.machine_id))
    )
    if not candidates:
        return WaitNextEvent()
    return min(
        candidates,
        key=lambda x: (
            x.empty_ticks + x.loaded_ticks,
            x.action.job_id,
            x.action.agv_id,
            x.action.destination.kind,
            x.action.destination.machine_id or "",
        ),
    ).action
