"""Fleet pickup admission using only detached public observations."""

from collections import deque
from math import isqrt

from smartsom.domain.production import MOVES


def active_transport(state):
    return bool(state.get("target") or state.get("job") or state.get("reservation"))


def pickup_plan(view, targets, parameters, *, distance_cache=None):
    """Assign ready sources to nearby idle cars without duplicate reservations.

    Admit at most one inbound car per physical port and never more cars than
    unreserved ready jobs. Both sides of a station may work concurrently.
    Traffic admission uses these local bottlenecks instead of a square-root
    global budget; max_active remains an explicit optional safety limit.
    The plan is deterministic and independent of request order.
    """
    vehicles = view["agvs"]
    limit = parameters.get("max_active")
    if not isinstance(limit, int) or isinstance(limit, bool):
        limit = (
            len(vehicles)
            if parameters.get("fleet_admission") == "traffic"
            else max(1, isqrt(len(vehicles)))
        )
    capacity = max(1, limit) - sum(active_transport(a) for a in vehicles.values())
    if capacity <= 0:
        return {}
    idle = {
        owner: tuple(state["cell"])
        for owner, state in vehicles.items()
        if not active_transport(state) and not state.get("service")
    }
    sources = view.get("sources", {})
    jobs = view.get("jobs", {})
    topology = view["topology"]
    width, height = topology["width"], topology["height"]
    solids = {tuple(cell) for cell in topology["solids"]}
    ports = {key: tuple(cell) for key, cell in topology["ports"].items()}
    port_cells = set(ports.values())
    topology_key = (width, height, frozenset(solids), frozenset(port_cells))
    distance_cache = {} if distance_cache is None else distance_cache
    edges = []
    for target in targets:
        source = sources.get(target.owner, {})
        # A reservation owns one job, not the whole station. Another port can
        # collect a different ready job without queueing at the reserved port.
        available = len(source.get("ready", ())) - source.get("reserved", 0)
        if available <= 0:
            continue
        if any(
            a.get("target") and a["target"]["port"] == target.port
            for a in vehicles.values()
        ):
            continue
        goal = ports[target.port]
        cache_key = topology_key, goal
        distances = distance_cache.get(cache_key)
        if distances is None:
            queue, distances = deque([goal]), {goal: 0}
            while queue:
                cell = queue.popleft()
                for dx, dy in MOVES.values():
                    nxt = cell[0] + dx, cell[1] + dy
                    if (
                        0 <= nxt[0] < width
                        and 0 <= nxt[1] < height
                        and nxt not in solids
                        and nxt not in distances
                        and not (cell in port_cells and nxt in port_cells)
                    ):
                        distances[nxt] = distances[cell] + 1
                        queue.append(nxt)
            distance_cache[cache_key] = distances
        step = max(
            (jobs.get(job, {}).get("step", -1) for job in source["ready"]), default=-1
        )
        priority = -step if parameters.get("work_in_progress_first") else 0
        for owner, cell in idle.items():
            if cell in distances:
                edges.append(
                    (
                        priority,
                        distances[cell],
                        target.owner,
                        target.port,
                        owner,
                        target,
                    )
                )
    assignments, claimed_ports, claimed_sources = {}, set(), {}
    for _, _, source, port, owner, target in sorted(edges, key=lambda edge: edge[:-1]):
        available = len(sources[source]["ready"]) - sources[source].get("reserved", 0)
        if (
            owner in assignments
            or port in claimed_ports
            or claimed_sources.get(source, 0) >= available
        ):
            continue
        assignments[owner] = target
        claimed_ports.add(port)
        claimed_sources[source] = claimed_sources.get(source, 0) + 1
        if len(assignments) >= capacity:
            break
    return assignments


def pickup_intention_score(view, owner, candidate, parameters):
    """Choose useful work for the actual decision owner, not an uncalled AGV.

    Current V3 only requests some empty vehicles at a boundary. Assigning a
    ready job to an arbitrary uncalled vehicle can therefore starve the vehicle
    that actually has permission to retarget. Intention counts are soft queue
    pressure, not reservations or feasibility masks.
    """
    target = candidate.action
    source = view.get("sources", {}).get(target.owner, {})
    ready = source.get("ready", ())
    supply = source.get("supply", 0)
    inbound = [
        state
        for other, state in view["agvs"].items()
        if other != owner
        and not state.get("job")
        and state.get("target")
        and state["target"]["owner"] == target.owner
    ]
    queued = sum(state["target"]["port"] == target.port for state in inbound)
    tier = 0 if ready else (1 if supply else 2)
    excess = max(0, len(inbound) + 1 - len(ready)) if ready else len(inbound)
    step = max(
        (view.get("jobs", {}).get(job, {}).get("step", -1) for job in ready), default=-1
    )
    return (
        tier,
        excess,
        queued,
        -step if ready and parameters.get("work_in_progress_first") else 0,
        candidate.features[1],
        not bool(candidate.features[11]),
        candidate.identity,
    )


def destination_score(view, owner, candidate):
    """Prefer available destinations and preserve an ongoing route on ties."""
    target = candidate.action
    machine = view.get("machines", {}).get(target.owner)
    full = bool(candidate.features[10]) or bool(machine and machine.get("job"))
    job = view.get("jobs", {}).get(view["agvs"][owner].get("job"), {})
    needs_final_inspection = (
        job.get("quality") == "UNKNOWN" and job.get("next_operation") is None
    )
    skips_inspection = needs_final_inspection and target.owner not in view.get(
        "stations", {}
    )
    incoming = sum(
        other != owner
        and bool(state.get("target"))
        and state["target"]["port"] == target.port
        for other, state in view["agvs"].items()
    )
    goal = view["topology"]["ports"].get(target.port)
    occupied = goal is not None and any(
        other != owner and tuple(state["cell"]) == tuple(goal)
        for other, state in view["agvs"].items()
    )
    return (
        skips_inspection,
        full,
        not bool(candidate.features[11]),
        incoming,
        occupied,
        candidate.features[1],
        candidate.identity,
    )
