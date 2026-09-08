"""Deterministic movement rules; conservative admission is a policy, not legality."""

from smartsom.dispatch import DecisionContext, Transfer, Transport, WaitNextEvent


def select_transport(context: DecisionContext) -> Transport | Transfer | WaitNextEvent:
    machines = {m.machine_id: m for m in context.machines}
    buffers = {b.machine_id: b for b in context.buffers}
    positions = {p.job_id: p for p in context.job_positions}

    def idle(key):
        m = machines[key]
        return m.operation_id is None and m.availability == "up"

    def safe(candidate):
        action = candidate.action
        if action.destination.kind == "holding":
            h = context.holding_buffer
            return (
                h is not None
                and (
                    h.capacity is None or len(h.jobs) + len(h.reservations) < h.capacity
                )
                and not any(
                    x.action.job_id == action.job_id
                    and x.action.destination.kind == "machine"
                    and safe(x)
                    for x in context.transport_candidates
                )
            )
        target = action.destination.machine_id
        if candidate.source_machine_id is not None and not (
            not idle(candidate.source_machine_id) and idle(target)
        ):
            return False
        if isinstance(action, Transfer) or target is None or target not in buffers:
            return True
        b = buffers[target]
        if b.pre_capacity is None:
            return True
        if b.pre_capacity > 0:
            return len(b.pre_jobs) + len(b.reservations) < b.pre_capacity
        same_job_leaves = (
            any(
                h.machine_id == target
                and h.job_id == action.job_id
                and h.phase == "blocked"
                for h in context.machine_holdings
            )
            and positions[action.job_id].location.kind == "machine"
        )
        return (
            machines[target].availability == "up"
            and (idle(target) or same_job_leaves)
            and not any(
                a.trip is not None and a.trip.destination.machine_id == target
                for a in context.agvs
            )
        )

    candidates = tuple(
        x
        for x in (*context.transport_candidates, *context.transfer_candidates)
        if safe(x)
    )
    if not candidates:
        return WaitNextEvent()
    return min(
        candidates,
        key=lambda x: (
            getattr(x, "empty_ticks", 0) + getattr(x, "loaded_ticks", 0),
            x.action.job_id,
            getattr(x.action, "agv_id", ""),
            x.action.destination.kind,
            x.action.destination.machine_id or x.action.destination.buffer_id or "",
        ),
    ).action
