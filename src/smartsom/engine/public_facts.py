"""Project current public state by whitelist, never by copying the private core."""

from smartsom.domain.public_facts import freeze


def public_facts(core):
    protocol = core.protocol
    jobs = []
    for identity, row in core.jobs.items():
        demand = core.demands[row["demand"]]
        jobs.append(
            {
                "id": identity,
                "order": row["demand"],
                "location": row["location"],
                "slot": row["slot"],
                "step": row["step"],
                "since": row["since"],
                "due_at": demand.due_at,
                "rush": demand.rush,
                "quality": row["quality"],
                "public_risk": row["risk"],
                "remaining_operations": tuple(
                    {"id": op.operation_id, "type": op.operation_type}
                    for op in demand.steps[row["step"] :]
                ),
            }
        )
    return freeze(
        {
            "tick": core.tick,
            "jobs": jobs,
            "machines": tuple(
                {
                    "id": identity,
                    "job": state["job"],
                    "status": state["status"],
                    "down": state["down"],
                    "elapsed": state["elapsed"],
                    "nominal": state["nominal"],
                    "mode": state["mode"],
                    "capabilities": core.machines[identity].operation_types,
                    "pre": core.pre.get(identity),
                    "post": core.post.get(identity),
                    "legal_processing": tuple(core.machine_choices(identity)),
                }
                for identity, state in core.machine_state.items()
            ),
            "vehicles": tuple(
                {
                    "id": identity,
                    "cell": state["cell"],
                    "job": state["job"],
                    "target": state["target"],
                    "service": state["service"],
                    "reservation_tick": state["reservation_tick"],
                    "arrived_at": state["arrived_at"],
                }
                for identity, state in core.agvs.items()
            ),
            "buffers": tuple(
                {
                    "id": owner,
                    "role": core.roles[owner],
                    "capacity": core.capacity[owner],
                    "inventory": rows,
                    "quota": protocol.quota(owner),
                    "commitments": protocol.commitments(owner),
                    "ready": protocol.ready(owner),
                }
                for owner, rows in core.storage.items()
            ),
            "completed": tuple(sorted(core.completed)),
        }
    )
