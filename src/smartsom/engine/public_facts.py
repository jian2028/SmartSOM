"""Project current public state by whitelist, never by copying the private core."""

from smartsom.domain.public_facts import Facts, freeze


def public_facts(core, *, topology=False):
    protocol = core.protocol
    operations = getattr(protocol, "_public_operation_rows", None)
    if operations is None:
        operations = protocol._public_operation_rows = {}
    job_rows = getattr(protocol, "_public_job_rows", None)
    if job_rows is None:
        job_rows = protocol._public_job_rows = {}
    jobs = [
        _job_row(job_rows, operations, identity, row, core.demands[row["demand"]])
        for identity, row in core.jobs.items()
    ]
    data = {
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
    if topology:
        # Topology is already frozen; adding it before the single freeze avoids
        # re-freezing every other entry while keeping the same key order.
        data["topology"] = protocol.public_topology
    return freeze(data)


_JOB_FIELDS = (
    "id",
    "order",
    "location",
    "slot",
    "step",
    "since",
    "due_at",
    "rush",
    "quality",
    "public_risk",
    "remaining_operations",
)


def _job_row(cache, operations, identity, row, demand):
    # No clock-based invalidation or private realization enters the signature.
    # Fast path: Demand is a frozen dataclass, so the same demand object plus the
    # same mutable row fields implies the same published values (including the
    # remaining operations). Any row field added to _JOB_FIELDS must be added to
    # this key as well. Slow path: equality covers every published field.
    key = (
        row["demand"],
        row["location"],
        row["slot"],
        row["step"],
        row["since"],
        row["quality"],
        row["risk"],
    )
    previous = cache.get(identity)
    if previous is not None and previous[2] is demand and previous[3] == key:
        return previous[1]
    values = (
        identity,
        row["demand"],
        row["location"],
        row["slot"],
        row["step"],
        row["since"],
        demand.due_at,
        demand.rush,
        row["quality"],
        row["risk"],
        tuple(_operation_row(operations, op) for op in demand.steps[row["step"] :]),
    )
    if previous is None or previous[0] != values:
        previous = (values, Facts(zip(_JOB_FIELDS, values, strict=True)))
    cache[identity] = (*previous[:2], demand, key)
    return previous[1]


def _operation_row(cache, operation):
    # These two public strings are immutable design facts. Dynamic job progress
    # still selects its current remaining sequence on every decision.
    key = operation.operation_id, operation.operation_type
    if key not in cache:
        cache[key] = Facts((("id", key[0]), ("type", key[1])))
    return cache[key]


def topology_facts(protocol):
    """Static geometry only; no demand, outage or random realization tables."""
    routes = protocol.routes
    return freeze(
        {
            "width": routes.width,
            "height": routes.height,
            "solids": tuple(sorted(routes.solids)),
            "ports": {
                owner: tuple(sorted(cells))
                for owner, cells in sorted(routes.ports.items())
            },
            "operation_types": tuple(sorted(protocol.core.factory.operation_types)),
        }
    )
