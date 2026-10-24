"""Static geometry queries shared by nominal estimates and physical routing."""

from collections import deque

from smartsom.domain.factory_design import occupied_cells, target_owner_id


class StaticRoutes:
    def __init__(self, factory):
        self.width, self.height = factory.grid.width, factory.grid.height
        self.solids = {(c.x, c.y) for c in factory.grid.blocked_cells}
        for entity in (*factory.machines, *factory.buffers):
            self.solids.update((c.x, c.y) for c in occupied_cells(entity.footprint))
        self.cells = tuple(
            (x, y)
            for x in range(self.width)
            for y in range(self.height)
            if (x, y) not in self.solids
        )
        walkable = set(self.cells)
        self.neighbors = {
            c: tuple(
                n
                for n in (
                    (c[0] + 1, c[1]),
                    (c[0] - 1, c[1]),
                    (c[0], c[1] + 1),
                    (c[0], c[1] - 1),
                )
                if n in walkable
            )
            for c in self.cells
        }
        self.ports = {}
        for p in factory.ports:
            for binding in p.bindings:
                self.ports.setdefault(target_owner_id(binding.target), []).append(
                    (p.cell.x, p.cell.y)
                )
        self.cache = {}

    def distances(self, goals):
        key = tuple(sorted(set(goals)))
        if key not in self.cache:
            distances = dict.fromkeys(key, 0)
            queue = deque(key)
            while queue:
                c = queue.popleft()
                for n in self.neighbors[c]:
                    if n not in distances:
                        distances[n] = distances[c] + 1
                        queue.append(n)
            self.cache[key] = distances
        return self.cache[key]

    def to_owner(self, cell, owner):
        return self.distances(self.ports[owner]).get(tuple(cell), float("inf"))

    def between(self, origin, destination):
        return min(self.to_owner(c, destination) for c in self.ports[origin])
