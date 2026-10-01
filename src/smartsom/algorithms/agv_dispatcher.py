"""Fleet pickup admission using only detached public observations."""

from collections import deque
from math import isqrt

from smartsom.domain.production import MOVES


def active_transport(state):
    return bool(state.get("target") or state.get("job") or state.get("reservation"))


def pickup_plan(view, targets, parameters, *, distance_cache=None):
    """Assign ready sources to nearby idle cars without duplicate reservations.

    Only one inbound car is admitted per source.  A square-root fleet budget
    permits independent production stages to overlap without filling narrow
    port approaches with waiting cars.  The plan is deterministic and independent
    of which idle vehicle asks first.
    """
    vehicles = view["agvs"]
    limit = parameters.get("max_active")
    if not isinstance(limit, int) or isinstance(limit, bool):
        limit = max(1, isqrt(len(vehicles)))
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
        # Pending processing is not ready inventory.  Reserve only when a job
        # can actually be collected, and do not queue behind another pickup.
        if not source.get("ready") or source.get("reserved", 0):
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
    assignments, claimed_sources = {}, set()
    for _, _, source, _, owner, target in sorted(edges, key=lambda edge: edge[:-1]):
        if owner in assignments or source in claimed_sources:
            continue
        assignments[owner] = target
        claimed_sources.add(source)
        if len(assignments) >= capacity:
            break
    return assignments


def destination_score(view, owner, candidate):
    """Prefer available destinations and preserve an ongoing route on ties."""
    target = candidate.action
    machine = view.get("machines", {}).get(target.owner)
    full = bool(candidate.features[10]) or bool(machine and machine.get("job"))
    incoming = sum(
        other != owner
        and bool(state.get("job"))
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
        full,
        incoming,
        occupied,
        not bool(candidate.features[11]),
        candidate.features[1],
        candidate.identity,
    )
