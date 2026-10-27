"""Fixed production baselines and the shared, occupancy-aware grid Mover."""

import random


class BaseRule:
    def __init__(self, method="rule", seed=0):
        if method not in ("rule", "random"):
            raise ValueError("unknown Base baseline")
        self.method = method
        self.rng = random.Random(seed)

    def choose(self, request, critic=None):
        candidates = request.candidates
        if self.method == "random":
            return self.rng.choice(candidates).action
        if request.group == "dispatcher" and candidates[-1].action == ("EMPTY_WAIT",):
            candidates = candidates[:-1]
        # EDD for process/service; normal mode; nearest eligible resource with
        # ready supply preferred; accept all feasible requests without STOP.
        if request.group == "machine":
            normal = [c for c in candidates if c.action[1] == "normal"]
            selected = min(
                normal or candidates,
                key=lambda c: (c.features[0], c.features[12], c.identity),
            )
        elif request.group == "dispatcher":
            if request.stage == "pickup":
                selected = min(
                    candidates,
                    key=lambda c: (
                        c.features[2] == 0,
                        c.features[0],
                        c.features[3],
                        c.identity,
                    ),
                )
            else:
                selected = min(
                    candidates,
                    key=lambda c: (c.features[0] + c.features[2], c.identity),
                )
        elif request.stage == "accept":
            selected = min(
                (c for c in candidates if c.action != ("STOP",)),
                key=lambda c: (c.features[0], c.identity),
            )
        else:
            due = 4 if request.group == "pre" else 0
            selected = min(candidates, key=lambda c: (c.features[due], c.identity))
        return selected.action


class SharedMover:
    """One-step space-time reservations with recursive displacement and rollback.

    Uses only current public vehicle/port availability. Joint validation rejects
    vertex conflicts and edge swaps before physics. Production destinations and
    reservations are never changed by traffic coordination.
    """

    def __init__(self, routes):
        self.routes = routes
        self.wait = {}
        self.last = {}
        self.executed = 0

    def moves(self, vehicles, tick):
        routes = self.routes
        current = {v: tuple(a["cell"]) for v, a in vehicles.items()}
        occupant = {cell: v for v, cell in current.items()}
        for v, cell in current.items():
            self.wait[v] = self.wait.get(v, 0) + 1 if self.last.get(v) == cell else 0
        self.last = current.copy()
        order = sorted(
            vehicles,
            key=lambda v: (
                not vehicles[v]["rush"],
                not vehicles[v]["ready"],
                0.02 * vehicles[v]["due"] - self.wait[v],
                (list(vehicles).index(v) - tick) % len(vehicles),
            ),
        )
        port_cells = {c for ports in routes.ports.values() for c in ports}
        chosen_goals = set()
        goals = {}
        for v in order:
            a = vehicles[v]
            if a["service"]:
                goals[v] = current[v]
                chosen_goals.add(current[v])
                continue
            owner = a["owner"]
            ports = routes.ports[owner] if owner else ()
            choices = (
                [
                    c
                    for c in ports
                    if c not in chosen_goals and (c not in occupant or occupant[c] == v)
                ]
                if a["ready"]
                else []
            )
            if choices:
                goal = min(
                    choices, key=lambda c: (routes.distances((c,))[current[v]], c)
                )
            else:
                # A real free road cell, never a virtual waiting capacity. Keep
                # parked positions stable; approaching traffic can displace them.
                choices = [
                    c
                    for c in routes.cells
                    if c not in port_cells
                    and c not in chosen_goals
                    and (c not in occupant or occupant[c] == v)
                ]
                goal = (
                    min(
                        choices,
                        key=lambda c: (
                            routes.distances((current[v],))[c]
                            + (0.15 * routes.to_owner(c, owner) if owner else 0.0),
                            c,
                        ),
                    )
                    if choices
                    else current[v]
                )
            goals[v] = goal
            chosen_goals.add(goal)
        assigned = {v: current[v] for v, a in vehicles.items() if a["service"]}

        def reserve(v, stack):
            if v in assigned:
                return True
            distances = routes.distances((goals[v],))
            choices = (*routes.neighbors[current[v]], current[v])
            choices = sorted(
                choices,
                key=lambda c: (
                    distances[c],
                    c != current[v],
                    (c[0] + 3 * c[1] + tick) % 7,
                    c,
                ),
            )
            for cell in choices:
                holder = next((k for k, c in assigned.items() if c == cell), None)
                parked_holder = (
                    holder is not None
                    and not vehicles[holder]["ready"]
                    and not vehicles[holder]["service"]
                    and assigned[holder] == current[holder]
                )
                if holder is not None and not parked_holder:
                    continue
                other = occupant.get(cell)
                if other is not None and other != v:
                    if other in stack or assigned.get(other) == current[v]:
                        continue
                saved = dict(assigned)
                if parked_holder:
                    # Parking is revocable even for rush cargo with no currently
                    # executable route. This does not displace active urgent traffic
                    # or alter the job's reservation/destination.
                    del assigned[holder]
                assigned[v] = cell
                if (
                    other is None
                    or other == v
                    or other in assigned
                    or reserve(other, (*stack, v))
                ):
                    return True
                assigned.clear()
                assigned.update(saved)
            return False

        for v in order:
            if v not in assigned and not reserve(v, ()):
                # Existing occupants can always wait if no earlier reservation
                # depends on them; failed recursive branches have been rolled back.
                assigned[v] = current[v]
        if len(set(assigned.values())) != len(vehicles):
            raise AssertionError("Mover produced a vertex collision")
        for v, destination in assigned.items():
            other = occupant.get(destination)
            if other and other != v and assigned[other] == current[v]:
                raise AssertionError("Mover produced an edge swap")
        actions = {
            (0, 0): "WAIT",
            (1, 0): "RIGHT",
            (-1, 0): "LEFT",
            (0, 1): "DOWN",
            (0, -1): "UP",
        }
        self.executed += 1
        return {
            v: actions[(c[0] - current[v][0], c[1] - current[v][1])]
            for v, c in assigned.items()
        }
