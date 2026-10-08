"""Public-state AGV planning primitives for scalable rule controllers.

The planner is deliberately separate from :mod:`production_rules`: these helpers
only inspect the detached Mover observation and return semantic actions.  The
simulator remains the authority for movement feasibility and collision handling.
"""

from __future__ import annotations

import heapq

from smartsom.domain.production import MOVES


def target_identity(state):
    """Return the stable public target identity, or ``None`` for an idle AGV."""
    target = state.get("target")
    return None if target is None else (target["owner"], target["port"])


def planning_key(view):
    """Return a complete, hashable key for a public movement boundary."""
    topology = view["topology"]
    return (
        view["tick"],
        topology["width"],
        topology["height"],
        tuple(sorted(map(tuple, topology["solids"]))),
        tuple(sorted((key, tuple(cell)) for key, cell in topology["ports"].items())),
        tuple(
            (
                owner,
                tuple(state["cell"]),
                bool(state.get("service")),
                target_identity(state),
                state.get("job"),
                state.get("reservation"),
            )
            for owner, state in sorted(view["agvs"].items())
        ),
        tuple(
            (owner, bool(state.get("ready")))
            for owner, state in sorted(view.get("sources", {}).items())
        ),
    )


def update_history(view, history):
    """Advance durable route commitments once for this public boundary."""
    tick = view["tick"]
    vehicles = view["agvs"]
    for owner in set(history) - set(vehicles):
        del history[owner]
    for owner, state in vehicles.items():
        task = (
            [*target_identity(state), state.get("job")] if state.get("target") else None
        )
        cell = list(state["cell"])
        previous = history.get(owner)
        if previous is None or previous["task"] != task or previous["tick"] > tick:
            prior_cell, blocked_ticks, recent_cells = None, 0, [cell]
        elif previous["cell"] != cell:
            prior_cell, blocked_ticks = previous["cell"], 0
            recent_cells = [
                *previous.get("recent_cells", [previous["cell"]]),
                cell,
            ][-12:]
        else:
            prior_cell = previous["previous"]
            blocked_ticks = previous.get("blocked_ticks", 0) + 1
            recent_cells = previous.get("recent_cells", [cell])
        if previous is None or previous["tick"] != tick:
            history[owner] = {
                "tick": tick,
                "cell": cell,
                "previous": prior_cell,
                "task": task,
                "blocked_ticks": blocked_ticks,
                "recent_cells": recent_cells,
            }


def valid_cell(cell, width, height, solids):
    return 0 <= cell[0] < width and 0 <= cell[1] < height and cell not in solids


def cycling_route(entry):
    """Detect repeated local movement without changing the task identity."""
    cells = entry.get("recent_cells", ())
    return len(cells) >= 8 and len({tuple(cell) for cell in cells}) <= len(cells) // 2


def retarget_scouts(view):
    """Send spare empty cars to reconsider only for uncovered ready sources.

    V3.2 requires arrival at an empty intention before another retarget request.
    Keep that progress path, but do not move the entire spare fleet when useful
    pickup sources already have inbound empty vehicles.
    """
    vehicles, sources = view["agvs"], view.get("sources", {})
    covered = {
        state["target"]["owner"]
        for state in vehicles.values()
        if not state.get("job") and state.get("target")
    }
    missing = sum(
        bool(source.get("ready")) and owner not in covered
        for owner, source in sources.items()
    )
    ports = view["topology"]["ports"]
    candidates = []
    for owner, state in vehicles.items():
        target = state.get("target")
        if state.get("job") or state.get("service") or not target:
            continue
        source = sources.get(target["owner"])
        if source is None or source.get("ready"):
            continue
        cell, goal = state["cell"], ports[target["port"]]
        distance = abs(cell[0] - goal[0]) + abs(cell[1] - goal[1])
        candidates.append((distance, owner))
    return {owner for _, owner in sorted(candidates)[:missing]}


def action_destination(vehicles, owner, action):
    current = tuple(vehicles[owner]["cell"])
    dx, dy = MOVES.get(action, (0, 0))
    return current[0] + dx, current[1] + dy


def legal_actions(owner, vehicles, *, width, height, solids, port_cells, occupied):
    """Reconstruct the protocol's public movement mask for one AGV."""
    state = vehicles[owner]
    if state.get("service"):
        return ("WAIT",)
    current = tuple(state["cell"])
    moves = {
        action: (current[0] + dx, current[1] + dy)
        for action, (dx, dy) in MOVES.items()
        if valid_cell((current[0] + dx, current[1] + dy), width, height, solids)
    }
    exits = {
        action: cell
        for action, cell in moves.items()
        if cell not in occupied and cell not in port_cells
    }
    return tuple(exits) if current in port_cells and exits else (*moves, "WAIT")


def route_distances(owner, vehicles, sources, history, *, width, height, solids, ports):
    """Find public port-aware distances and staging goals for an active AGV."""
    state = vehicles[owner]
    port_cells = set(ports.values())
    target = ports[state["target"]["port"]]
    hard_obstacles = {
        tuple(other["cell"])
        for other_owner, other in vehicles.items()
        if other_owner != owner
        and other.get("service")
        and tuple(other["cell"]) != target
    }
    idle_obstacles = {
        tuple(other["cell"])
        for other_owner, other in vehicles.items()
        if other_owner != owner
        and not other.get("target")
        and not other.get("service")
        and tuple(other["cell"]) != target
    }
    source = sources.get(state["target"]["owner"])
    pickup_pending = (
        state.get("job") is None
        and source is not None
        and not source["ready"]
        and (
            state.get("reservation") == state["target"]["owner"]
            or not any(source.get("ready") for source in sources.values())
        )
    )
    goals = {target}
    target_block = set()
    if pickup_pending:
        goals = {
            (target[0] + dx, target[1] + dy)
            for dx, dy in MOVES.values()
            if valid_cell((target[0] + dx, target[1] + dy), width, height, solids)
            and (target[0] + dx, target[1] + dy) not in port_cells
            and (target[0] + dx, target[1] + dy) not in hard_obstacles | idle_obstacles
        }
        if goals:
            target_block.add(target)
        else:
            goals = {target}
    previous = history[owner]["previous"]
    commitment = set(target_block)
    if (
        previous is not None
        and tuple(previous) not in goals
        and history[owner].get("blocked_ticks", 0) < 3
    ):
        commitment.add(tuple(previous))

    def route(blocked, obstacles):
        queue = [(0, 0, goal) for goal in goals]
        heapq.heapify(queue)
        result = {goal: (0, 0) for goal in goals}
        while queue:
            port_visits, steps, cell = heapq.heappop(queue)
            if result[cell] != (port_visits, steps):
                continue
            for dx, dy in MOVES.values():
                nxt = cell[0] + dx, cell[1] + dy
                if (
                    not valid_cell(nxt, width, height, solids)
                    or nxt in obstacles
                    or nxt in blocked
                    # A port occupant must leave through a non-port cell when
                    # one is available.  Do not score a port-to-port shortcut
                    # that the protocol will reject on the following boundary.
                    or cell in port_cells
                    and nxt in port_cells
                ):
                    continue
                value = (
                    port_visits + int(cell in port_cells and cell not in goals),
                    steps + 1,
                )
                if value < result.get(nxt, (float("inf"), float("inf"))):
                    result[nxt] = value
                    heapq.heappush(queue, (*value, nxt))
        return result

    current = tuple(state["cell"])
    obstacles = hard_obstacles | idle_obstacles
    result = route(commitment, obstacles)
    # Preserve idle parking whenever possible.  Once an AGV has demonstrably
    # entered a two-cell cycle, though, it must plan through an idle blocker so
    # priority inheritance can ask that blocker to yield instead of repeatedly
    # taking the same port-clearance detour.
    if current not in result or cycling_route(history[owner]):
        result = route(commitment, hard_obstacles)
    # A return to the immediately previous cell is a last resort for a genuine
    # cul-de-sac, never merely a reaction to an idle vehicle that can yield.
    if current not in result and previous is not None:
        commitment.discard(tuple(previous))
        result = route(commitment, hard_obstacles)
    return result, goals


def rank_actions(
    owner,
    vehicles,
    sources,
    history,
    *,
    width,
    height,
    solids,
    ports,
    occupied,
    allowed=None,
    yielding=False,
    retargeting=True,
    route_cache=None,
):
    """Rank legal Mover actions without exposing simulator-private state."""
    port_cells = set(ports.values())
    actions = legal_actions(
        owner,
        vehicles,
        width=width,
        height=height,
        solids=solids,
        port_cells=port_cells,
        occupied=occupied,
    )
    if allowed is not None:
        actions = tuple(action for action in actions if action in allowed)
    if not actions:
        raise ValueError(f"no legal coordinated action for {owner}")
    state = vehicles[owner]
    previous = history[owner]["previous"]
    if previous is not None and history[owner].get("blocked_ticks", 0) < 3:
        non_reverse = tuple(
            action
            for action in actions
            if action_destination(vehicles, owner, action) != tuple(previous)
        )
        # Prevent an immediate A→B→A reversal whenever a legal alternative
        # exists.  A single forced exit remains available for a real cul-de-sac.
        if non_reverse:
            actions = non_reverse
    if state.get("service"):
        return actions
    no_pickup_work = (
        not state.get("job")
        and state.get("target")
        and state["target"]["owner"] in sources
        and not sources[state["target"]["owner"]].get("ready")
        and not (
            retargeting and any(source.get("ready") for source in sources.values())
        )
    )
    if not state.get("target") or yielding and not state.get("job") or no_pickup_work:

        def parking_score(action):
            cell = action_destination(vehicles, owner, action)
            clearance = sum(
                valid_cell((cell[0] + dx, cell[1] + dy), width, height, solids)
                and (cell[0] + dx, cell[1] + dy) not in port_cells
                for dx, dy in MOVES.values()
            )
            return (
                cell in port_cells,
                action == "WAIT" if yielding else action != "WAIT",
                -clearance,
                tuple(MOVES).index(action) if action in MOVES else len(MOVES),
            )

        return tuple(sorted(actions, key=parking_score))

    distances, goals = route_distances(
        owner,
        vehicles,
        sources,
        history,
        width=width,
        height=height,
        solids=solids,
        ports=ports,
    )
    if route_cache is not None:
        route_cache[owner] = distances, goals

    def score(action):
        cell = action_destination(vehicles, owner, action)
        port_visits, steps = distances.get(cell, (float("inf"), float("inf")))
        recent_cells = {
            tuple(value) for value in history[owner].get("recent_cells", ())
        }
        return (
            port_visits + int(cell in port_cells and cell not in goals),
            cell in recent_cells and cell not in goals,
            steps,
            action != "WAIT",
            tuple(MOVES).index(action) if action in MOVES else len(MOVES),
        )

    return tuple(sorted(actions, key=score))


def corridor_blockers(
    view, history, *, width, height, solids, ports, occupied, scouts=frozenset()
):
    """Clear idle cars beyond a transit port before its approach is entered.

    The movement mask cannot push an occupied exit from a port when a different
    exit is free: it forces departure through the currently free exit.  Asking
    the blocker to yield one step earlier prevents that forced backtrack.
    """
    vehicles = view["agvs"]
    port_cells = set(ports.values())
    active = {
        owner: state
        for owner, state in vehicles.items()
        if state.get("job")
        or state.get("service")
        or owner in scouts
        or state.get("target")
        and view.get("sources", {}).get(state["target"]["owner"], {}).get("ready")
    }
    blockers = set()
    for owner, state in active.items():
        if state.get("service") or not state.get("target"):
            continue
        distances, goals = route_distances(
            owner,
            active,
            view.get("sources", {}),
            history,
            width=width,
            height=height,
            solids=solids,
            ports=ports,
        )
        current = tuple(state["cell"])
        nearby_ports = (
            {current}
            if current in port_cells
            else {(current[0] + dx, current[1] + dy) for dx, dy in MOVES.values()}
            & port_cells
        )
        for port in nearby_ports - goals:
            for dx, dy in MOVES.values():
                exit_cell = port[0] + dx, port[1] + dy
                other = occupied.get(exit_cell)
                if (
                    other is not None
                    and (
                        other not in active
                        or cycling_route(history[owner])
                        and not vehicles[other].get("job")
                        and not vehicles[other].get("service")
                    )
                    and exit_cell not in port_cells
                    and distances.get(exit_cell, (float("inf"), float("inf")))
                    < distances.get(port, (float("inf"), float("inf")))
                ):
                    blockers.add(other)
    return blockers


def priority_order(view, history, rankings, port_cells, scouts=frozenset()):
    """Return a rotating, ageing priority order for priority inheritance."""
    owners = sorted(view["agvs"])
    rotation = {
        owner: (index - view["tick"]) % len(owners)
        for index, owner in enumerate(owners)
    }

    def key(owner):
        state = view["agvs"][owner]
        forced_exit = (
            tuple(state["cell"]) in port_cells and "WAIT" not in rankings[owner]
        )
        return (
            not forced_exit,
            not bool(
                state.get("job")
                or state.get("target")
                and view.get("sources", {})
                .get(state["target"]["owner"], {})
                .get("ready")
            ),
            not bool(state.get("job")),
            owner not in scouts,
            -history[owner].get("blocked_ticks", 0),
            rotation[owner],
            owner,
        )

    return tuple(sorted(owners, key=key))


def plan_actions(view, history, *, allowed_actions=None):
    """Return one safe, scalable next-action plan using PIBT-style recursion.

    The recursion gives a blocked lower-priority AGV the inherited priority to
    vacate its cell.  Backtracking chooses a different action when that chain
    cannot be made safe.  The resulting work is bounded by the small local
    action degree and the fleet size; it never enumerates the Cartesian product
    of every AGV action.
    """
    update_history(view, history)
    topology, vehicles = view["topology"], view["agvs"]
    width, height = topology["width"], topology["height"]
    solids = {tuple(cell) for cell in topology["solids"]}
    ports = {key: tuple(cell) for key, cell in topology["ports"].items()}
    port_cells = set(ports.values())
    occupied = {tuple(state["cell"]): owner for owner, state in vehicles.items()}
    # Two neighboring ports may share their only currently free departure cell.
    # Once both vehicles occupy those ports the mandatory-exit mask can make
    # every joint action conflict. Serialize entry to overlapping approaches;
    # opposite station sides with independent exits remain concurrent.
    approaches = {
        port: {
            (port[0] + dx, port[1] + dy)
            for dx, dy in MOVES.values()
            if valid_cell((port[0] + dx, port[1] + dy), width, height, solids)
            and (port[0] + dx, port[1] + dy) not in port_cells
        }
        for port in port_cells
    }
    ports_by_exit = {}
    for port, exits in approaches.items():
        for cell in exits:
            ports_by_exit.setdefault(cell, set()).add(port)
    neighboring_ports = {
        tuple(sorted((left, right)))
        for neighbors in ports_by_exit.values()
        for left in neighbors
        for right in neighbors
        if left != right
    }
    scouts = retarget_scouts(view)
    yielding = corridor_blockers(
        view,
        history,
        width=width,
        height=height,
        solids=solids,
        ports=ports,
        occupied=occupied,
        scouts=scouts,
    )
    routes = {}
    rankings = {
        owner: rank_actions(
            owner,
            vehicles,
            view.get("sources", {}),
            history,
            width=width,
            height=height,
            solids=solids,
            ports=ports,
            occupied=occupied,
            allowed=(allowed_actions or {}).get(owner),
            yielding=owner in yielding,
            retargeting=owner in scouts,
            route_cache=routes,
        )
        for owner in sorted(vehicles)
    }
    ordered = priority_order(view, history, rankings, port_cells, scouts)
    proposals, reserved_cells, reserved_edges, planned, visiting = (
        {},
        set(),
        set(),
        set(),
        set(),
    )

    def restore(snapshot):
        proposals.clear()
        proposals.update(snapshot[0])
        reserved_cells.clear()
        reserved_cells.update(snapshot[1])
        reserved_edges.clear()
        reserved_edges.update(snapshot[2])
        planned.clear()
        planned.update(snapshot[3])

    def reserve(owner, action):
        origin = tuple(vehicles[owner]["cell"])
        destination = action_destination(vehicles, owner, action)
        proposals[owner] = action
        reserved_cells.add(destination)
        reserved_edges.add((origin, destination))
        planned.add(owner)

    def creates_port_trap(owner, destination):
        future = {
            action_destination(vehicles, other, proposals[other])
            if other in planned
            else tuple(state["cell"])
            for other, state in vehicles.items()
            if other != owner
        } | {destination}
        for left, right in neighboring_ports:
            # Keep neighboring approaches exclusive until the other port is
            # vacated, even if a temporarily free second exit exists. A later
            # service/queue arrival can otherwise remove that exit.
            if destination in (left, right):
                other = right if destination == left else left
                if other in future:
                    return True
            if left not in future or right not in future:
                continue
            left_exits, right_exits = (
                approaches[left] - future,
                approaches[right] - future,
            )
            if not left_exits or not right_exits or len(left_exits | right_exits) < 2:
                return True
        return False

    def blocks_transit_departure(owner, destination):
        if (
            destination not in port_cells
            or owner not in routes
            or tuple(vehicles[owner]["cell"]) in port_cells
        ):
            return False
        distances, goals = routes[owner]
        if destination in goals:
            return False
        progress = {
            cell
            for cell in approaches[destination]
            if distances.get(cell, (float("inf"), float("inf")))
            < distances.get(destination, (float("inf"), float("inf")))
        }
        if not progress:
            return True
        for cell in sorted(progress):
            blocker = occupied.get(cell)
            if (
                blocker is not None
                and blocker != owner
                and blocker not in planned
                and not vehicles[blocker].get("job")
                and not vehicles[blocker].get("service")
            ):
                assign(blocker, must_vacate=True)
        future = {
            action_destination(vehicles, other, proposals[other])
            if other in planned
            else tuple(state["cell"])
            for other, state in vehicles.items()
            if other != owner
        }
        # Do not enter a transit port only to be forced back out next tick.
        # A currently useful exit must remain free after this joint movement.
        return not (progress - future)

    def assign(owner, *, must_vacate=False, forbidden_destinations=frozenset()):
        if owner in planned:
            return True
        if owner in visiting:
            return False
        visiting.add(owner)
        origin = tuple(vehicles[owner]["cell"])
        for action in rankings[owner]:
            destination = action_destination(vehicles, owner, action)
            if (
                destination in reserved_cells
                or destination in forbidden_destinations
                or must_vacate
                and destination == origin
                or (destination, origin) in reserved_edges
            ):
                continue
            snapshot = (
                dict(proposals),
                set(reserved_cells),
                set(reserved_edges),
                set(planned),
            )
            # Reserve before following the occupancy chain. A longer rotation
            # can then close through an already-planned ancestor, while the
            # reverse-edge check still forbids two-vehicle swaps.
            reserve(owner, action)
            occupant = occupied.get(destination)
            if occupant is not None and occupant != owner:
                if occupant in planned:
                    other_destination = action_destination(
                        vehicles, occupant, proposals[occupant]
                    )
                    if other_destination == tuple(vehicles[owner]["cell"]):
                        restore(snapshot)
                        continue
                elif not assign(
                    occupant,
                    must_vacate=True,
                    forbidden_destinations=frozenset({origin}),
                ):
                    restore(snapshot)
                    continue
                elif (
                    action_destination(vehicles, occupant, proposals[occupant])
                    == origin
                ):
                    restore(snapshot)
                    continue
            if destination in port_cells:
                for left, right in neighboring_ports:
                    if destination not in (left, right):
                        continue
                    other_port = right if destination == left else left
                    blocker = occupied.get(other_port)
                    if blocker is not None and blocker not in planned:
                        assign(blocker, must_vacate=True)
            if creates_port_trap(owner, destination) or blocks_transit_departure(
                owner, destination
            ):
                restore(snapshot)
                continue
            visiting.remove(owner)
            return True
        visiting.remove(owner)
        return False

    for owner in ordered:
        if assign(owner):
            continue
        # A forced port exit can be physically infeasible only when every legal
        # exit is already reserved.  Preserve a semantic proposal so the engine
        # reports that real conflict rather than manufacturing an invalid action.
        reserve(owner, rankings[owner][0])
    return proposals
