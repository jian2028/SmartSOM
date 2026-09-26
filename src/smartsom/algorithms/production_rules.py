"""Role rules using exactly the public semantic candidates used by models."""

import random
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PolicyChoice:
    action: object
    log_probability: float = 0.0
    value: float = 0.0
    model_input: object = None


RULES = {
    "machine": {"spt", "normal_first", "random"},
    "buffer": {"edd", "spt", "random"},
    "dispatcher": {"nearest", "random"},
    "mover": {"shortest_path", "random", "automatic_travel"},
}


class RulePolicy:
    def __init__(self, role, name, seed=0, parameters=None):
        if name not in RULES.get(role, set()):
            raise ValueError(f"unknown {role} rule {name!r}")
        self.role, self.name = role, name
        self.parameters = dict(parameters or {})
        self.rng = random.Random(seed)

    def choose(self, request):
        if self.name == "automatic_travel":
            raise ValueError("automatic transport has no Mover decision")
        if request.role != self.role:
            raise ValueError("rule role mismatch")
        candidates = [c for c in request.candidates if c.legal]
        if not candidates:
            raise ValueError("no legal semantic decision")
        if self.name == "random":
            selected = self.rng.choice(candidates)
        elif request.role == "machine":
            if self.name == "normal_first":
                selected = min(
                    candidates,
                    key=lambda c: (-c.features[11], c.features[8], c.identity),
                )
            else:
                selected = min(
                    candidates,
                    key=lambda c: (c.features[8], -c.features[11], c.identity),
                )
        elif request.role == "buffer":
            position = 1 if self.name == "edd" else 8
            selected = min(candidates, key=lambda c: (c.features[position], c.identity))
        elif request.role == "dispatcher":
            targets = [c for c in candidates if c.action is not None]
            selected = min(
                targets or candidates, key=lambda c: (c.features[1], c.identity)
            )
        else:
            vehicle = request.observation["agvs"][request.owner]
            if vehicle["target"] is None:
                selected = next(
                    (c for c in candidates if c.action == "WAIT"), candidates[0]
                )
            else:
                selected = self.shortest_path(request, candidates)
        return PolicyChoice(selected.action)

    def shortest_path(self, request, candidates):
        from collections import deque

        from smartsom.domain.production import MOVES

        view = request.observation
        topology, vehicles = view["topology"], view["agvs"]
        solids, ports = {tuple(c) for c in topology["solids"]}, topology["ports"]
        occupied = {tuple(v["cell"]) for k, v in vehicles.items() if k != request.owner}
        port_cells = {tuple(c) for c in ports.values()}
        target = tuple(ports[vehicles[request.owner]["target"]["port"]])
        queue, distances = deque([target]), {target: 0}
        while queue:
            cell = queue.popleft()
            for dx, dy in MOVES.values():
                nxt = cell[0] + dx, cell[1] + dy
                if (
                    0 <= nxt[0] < topology["width"]
                    and 0 <= nxt[1] < topology["height"]
                    and nxt not in solids
                    and nxt not in occupied
                    and nxt not in distances
                ):
                    if nxt in port_cells and cell in port_cells:
                        exits = [(nxt[0] + x, nxt[1] + y) for x, y in MOVES.values()]
                        if any(
                            0 <= e[0] < topology["width"]
                            and 0 <= e[1] < topology["height"]
                            and e not in solids
                            and e not in occupied
                            and e not in port_cells
                            for e in exits
                        ):
                            continue
                    distances[nxt] = distances[cell] + 1
                    queue.append(nxt)
        current = tuple(vehicles[request.owner]["cell"])

        def next_cell(candidate):
            dx, dy = MOVES.get(candidate.action, (0, 0))
            return current[0] + dx, current[1] + dy

        free = [
            c for c in candidates if c.action == "WAIT" or next_cell(c) not in occupied
        ]
        choices = free or candidates
        best_distance = min(distances.get(next_cell(c), float("inf")) for c in choices)
        best = [
            c
            for c in choices
            if distances.get(next_cell(c), float("inf")) == best_distance
        ]
        wait = next((c for c in best if c.action == "WAIT"), None)
        selected = wait or self.rng.choice(best)
        # Public, rotating right of way prevents identical shortest-path claims
        # from remaining in permanent collision. Physics still arbitrates all moves.
        selected_cell = next_cell(selected)
        ordered = sorted(vehicles)

        def priority(owner):
            return (ordered.index(owner) - view["tick"]) % len(ordered)

        if selected.action != "WAIT":
            for owner, state in vehicles.items():
                if owner == request.owner or not state["target"] or state["service"]:
                    continue
                cell = tuple(state["cell"])
                if sum(abs(cell[i] - selected_cell[i]) for i in (0, 1)) != 1:
                    continue
                goal = tuple(ports[state["target"]["port"]])
                distance = abs(selected_cell[0] - goal[0]) + abs(
                    selected_cell[1] - goal[1]
                )
                if distance < abs(cell[0] - goal[0]) + abs(
                    cell[1] - goal[1]
                ) and priority(owner) < priority(request.owner):
                    wait = next((c for c in candidates if c.action == "WAIT"), None)
                    if wait:
                        selected = wait
                        break
        return selected

    def state_dict(self):
        return {"rng": self.rng.getstate()}

    def load_state_dict(self, state):
        def tuples(value):
            return tuple(map(tuples, value)) if isinstance(value, list) else value

        self.rng.setstate(tuples(state["rng"]))
