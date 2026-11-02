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
        # Cache only canonical goal keys, not the public mutable distance maps.
        # A fresh ports snapshot detects replacement and in-place changes.
        self._owner_goals = {}

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

    def _distances_to_owner(self, owner):
        ports = tuple(self.ports[owner])
        previous = self._owner_goals.get(owner)
        if previous is None or previous[0] != ports:
            goals = tuple(sorted(set(ports)))
            self._owner_goals[owner] = (ports, goals)
        else:
            goals = previous[1]
        # Honor cache clearing/replacement and direct edits of returned maps.
        if goals not in self.cache:
            return self.distances(ports)
        return self.cache[goals]

    def to_owner(self, cell, owner):
        return self._distances_to_owner(owner).get(tuple(cell), float("inf"))

    def between(self, origin, destination):
        ports = tuple(self.ports[origin])
        if not ports:
            return min(())  # Preserve the original error before destination lookup.
        distances = self._distances_to_owner(destination)
        return min(distances.get(tuple(c), float("inf")) for c in ports)
