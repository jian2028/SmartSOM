"""Public-state weighted-slack dispatch with persistent tasks and traffic plans.

Adapted from the SO-Baseline CentralRulePolicy ideas, not its indexed actions.
Reservations are advisory and replanned each tick; the engine remains authoritative.
"""

from collections import deque
from math import exp

from smartsom.algorithms.production import GreedyProductionPolicy
from smartsom.domain.production import MOVES, JointCommand, MachineCommand


class CoordinatedProductionPolicy(GreedyProductionPolicy):
    def __init__(self, factory, seed=0, *, quality_mode=None):
        super().__init__(factory, seed, quality_mode=quality_mode)
        self.tasks = {}
        self.last_tick = -1
        self.distances = {}

    def distance(self, start, end):
        key = (tuple(start), tuple(end))
        if key not in self.distances:
            path = self.path(*key)
            self.distances[key] = len(path) if path is not None else 10000
        return self.distances[key]

    def priority(self, view, job):
        row = view["jobs"][job]
        remaining = sum(s["nominal_ticks"] for s in row["remaining_steps"])
        # Public nominal processing plus a conservative transfer allowance.
        # Unlike the reference's fixed unique-machine layout, this supports
        # arbitrary capable-machine alternatives without choosing hidden work.
        remaining += 2 * (len(row["remaining_steps"]) + 1)
        slack = row["due_at"] - view["tick"] - remaining
        return (
            row["priority"]
            * exp(-max(slack, 0) / max(2 * remaining, 1))
            / (1 + remaining)
        )

    def rank(self, view):
        if view["tick"] < self.last_tick:
            self.tasks.clear()
        self.last_tick = view["tick"]
        self.scores = {
            owner: {j: self.priority(view, j) for j in jobs}
            for owner, jobs in view["rankings"].items()
        }
        return {
            owner: tuple(sorted(jobs, key=lambda j: (-self.scores[owner][j], j)))
            for owner, jobs in view["rankings"].items()
        }

    def route(self, start, goal, vertices, edges, horizon=8):
        """Space-time BFS with terminal-distance ranking and swap exclusion."""
        queue = deque([(start, ())])
        seen = {(start, 0)}
        candidates = []
        while queue:
            cell, path = queue.popleft()
            t = len(path)
            if cell == goal:
                return path
            if t == horizon:
                candidates.append((self.distance(cell, goal), path))
                continue
            for action, (dx, dy) in (*MOVES.items(), ("WAIT", (0, 0))):
                nxt = (cell[0] + dx, cell[1] + dy)
                if (
                    not 0 <= nxt[0] < self.factory.grid.width
                    or not 0 <= nxt[1] < self.factory.grid.height
                    or nxt in self.solids
                    or (nxt, t + 1) in vertices
                    or (nxt, cell, t + 1) in edges
                    or (nxt, t + 1) in seen
                ):
                    continue
                seen.add((nxt, t + 1))
                queue.append((nxt, (*path, action)))
        return min(candidates)[1] if candidates else None

    def act(self, view):
        machines, claimed = [], set()
        for mid, choices in view["machine_choices"].items():
            if self.quality_mode is not None:
                choices = [c for c in choices if c[1] == self.quality_mode]
            if choices:
                choice = min(
                    choices,
                    key=lambda c: (-self.priority(view, c[0]), c[1] != "normal", c),
                )
                machines.append((mid, MachineCommand(*choice)))
                claimed.add(choice[0])
        quality = tuple(
            (key, "START") for key, ready in view["inspection_choices"].items() if ready
        )
        locked = {key for key, _ in quality}
        vertices, edges, slots = set(), set(), set()
        vehicles = sorted(
            view["agvs"], key=lambda a: (view["agvs"][a]["job"] is None, a)
        )
        unplanned = {a: tuple(view["agvs"][a]["cell"]) for a in vehicles}
        actions = []
        for aid in vehicles:
            cell = unplanned.pop(aid)
            # Unplanned vehicles are stationary obstacles for this plan. Later
            # vehicles may follow a planned departure but cannot swap with it.
            occupied = vertices | {
                (c, t) for c in unplanned.values() for t in range(1, 9)
            }
            options = []
            for port, transfer in view["interactions"][aid].items():
                if transfer is None:
                    continue
                kind, job, owner, slot = transfer
                if job in claimed or owner in locked or (owner, slot) in slots:
                    continue
                role = self.buffer_roles.get(owner)
                if kind == "pickup" and role == "machine_pre":
                    continue  # Do not remove work already staged for its machine.
                if kind == "drop" and owner in self.station_ids:
                    continue  # This deterministic-quality demo requires no inspection detour.
                target = self.ports[port]
                path = self.route(cell, target, occupied, edges)
                if path is None:
                    continue
                task = (job, port, kind)
                options.append(
                    (
                        (
                            task != self.tasks.get(aid),
                            role == "system_input"
                            if kind == "pickup"
                            else role == "storage",
                            -self.priority(view, job),
                            self.distance(cell, target),
                            port,
                        ),
                        path,
                        transfer,
                        task,
                    )
                )
            if options:
                _, path, transfer, task = min(options)
                self.tasks[aid] = task
                claimed.add(transfer[1])
                if transfer[0] == "drop":
                    slots.add((transfer[2], transfer[3]))
                action = path[0] if path else "INTERACT"
                if action == "INTERACT":
                    self.tasks.pop(aid, None)
            else:
                self.tasks.pop(aid, None)
                path, action = (), "WAIT"
                if cell in self.ports.values():
                    parking = [
                        (self.distance(cell, (x, y)), (x, y))
                        for x in range(self.factory.grid.width)
                        for y in range(self.factory.grid.height)
                        if (x, y) not in self.solids
                        and (x, y) not in self.ports.values()
                    ]
                    for _, target in sorted(parking):
                        route = self.route(cell, target, occupied, edges)
                        if route:
                            path, action = route, route[0]
                            break
            cursor = cell
            for t in range(1, 9):
                move = path[t - 1] if t <= len(path) else "WAIT"
                dx, dy = MOVES.get(move, (0, 0))
                nxt = (cursor[0] + dx, cursor[1] + dy)
                vertices.add((nxt, t))
                edges.add((cursor, nxt, t))
                cursor = nxt
            actions.append((aid, action))
        return JointCommand(
            tuple(actions),
            tuple(machines),
            quality,
            tuple((k, tuple(v)) for k, v in view["rankings"].items()),
        )
