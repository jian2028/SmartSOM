"""Shared occupancy-aware traffic rules; never choose production targets."""

from collections import OrderedDict
from itertools import chain

from smartsom.engine.base_routes import StaticRoutes


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
        self._parking_orders = OrderedDict()

    def get_state(self):
        return {
            "wait": dict(self.wait),
            "last": dict(self.last),
            "executed": self.executed,
        }

    def set_state(self, state):
        self.wait = dict(state["wait"])
        self.last = {k: tuple(v) for k, v in state["last"].items()}
        self.executed = state["executed"]

    def _parking_order(self, cell, owner, distances, owner_distances):
        """Bounded static rankings; occupancy is filtered anew for every vehicle.

        Public distance maps remain mutable. Compare their contents before reuse,
        not only identities, and retain the public path for custom route objects.
        """
        routes = self.routes
        if (
            distances is None
            or type(routes) is not StaticRoutes
            or "distances" in routes.__dict__
            or "to_owner" in routes.__dict__
            or (owner and owner_distances is None)
        ):
            return None
        cells = tuple(routes.cells)
        key = (cell, owner)
        previous = self._parking_orders.get(key)
        if previous is not None and (
            previous[0] == cells
            and previous[1] == distances
            and previous[2] == owner_distances
        ):
            self._parking_orders.move_to_end(key)
            return previous[3]
        # Do not evaluate excluded/unreachable cells or reorder nonfinite custom
        # scores: the original filtered min remains authoritative in those cases.
        if not all(type(distances.get(c)) is int for c in cells) or (
            owner_distances is not None
            and not all(type(x) is int for x in owner_distances.values())
        ):
            return None
        ranked = tuple(
            sorted(
                cells,
                key=lambda c: (
                    distances[c]
                    + (0.15 * owner_distances.get(c, float("inf")) if owner else 0.0),
                    c,
                ),
            )
        )
        self._parking_orders[key] = (
            cells,
            distances.copy(),
            owner_distances.copy() if owner_distances is not None else None,
            ranked,
        )
        if len(self._parking_orders) > 1024:
            self._parking_orders.popitem(last=False)
        return ranked

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
                native = (
                    type(routes) is StaticRoutes
                    and "distances" not in routes.__dict__
                    and "to_owner" not in routes.__dict__
                )
                choices = (
                    c
                    for c in routes.cells
                    if c not in port_cells
                    and c not in chosen_goals
                    and (c not in occupant or occupant[c] == v)
                )
                # Custom route callbacks keep the original eager candidate list.
                if not native:
                    choices = list(choices)
                first = next(iter(choices), None)
                # The current cell is fixed throughout this scoring pass. Keep
                # the public distance map local so later calls still see edits.
                distances = (
                    routes.distances((current[v],)) if first is not None else None
                )
                # Native owner maps are stable for this scoring pass. Preserve
                # custom routes and instance overrides through their public call.
                owner_distances = (
                    routes._distances_to_owner(owner)
                    if first is not None
                    and owner
                    and type(routes) is StaticRoutes
                    and "to_owner" not in routes.__dict__
                    else None
                )
                ranked = self._parking_order(
                    current[v], owner, distances, owner_distances
                )
                if ranked is not None and first is not None:
                    goal = next(
                        c
                        for c in ranked
                        if c not in port_cells
                        and c not in chosen_goals
                        and (c not in occupant or occupant[c] == v)
                    )
                else:
                    goal = (
                        min(
                            chain((first,), choices) if native else choices,
                            key=lambda c: (
                                distances[c]
                                + (
                                    0.15
                                    * (
                                        owner_distances.get(c, float("inf"))
                                        if owner_distances is not None
                                        else routes.to_owner(c, owner)
                                    )
                                    if owner
                                    else 0.0
                                ),
                                c,
                            ),
                        )
                        if first is not None
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


def create_mover(*, routes, seed=0):
    return SharedMover(routes)


create_mover.CONTRACT = "smartsom.mover/v1"
