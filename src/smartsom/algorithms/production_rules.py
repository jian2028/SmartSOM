"""Role rules using exactly the public semantic candidates used by models."""

import random
from dataclasses import dataclass

from smartsom.algorithms.rule_registry import (
    _json_copy,
    freeze_rule,
    instantiate_rule,
    public_rule_request,
    rule_registration,
    verify_rule_modules,
)


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
    def __init__(
        self,
        role,
        name,
        seed=0,
        parameters=None,
        *,
        version=None,
        code_sha256=None,
        frozen_identity=None,
    ):
        self.registration = rule_registration(role, name, version, code_sha256)
        self.role, self.name = role, name
        self.parameters = _json_copy(dict(parameters or {}))
        self.seed = seed
        self.rng = random.Random(seed)
        self.rule = None
        self.frozen_identity = None
        if self.registration is not None or frozen_identity is not None or code_sha256:
            self.frozen_identity = freeze_rule(
                role, name, version, self.parameters, code_sha256
            )
            if frozen_identity is not None and self.frozen_identity != frozen_identity:
                raise ValueError("rule identity changed since configuration freeze")
        if self.registration is not None:
            self.rule = instantiate_rule(
                self.registration, parameters=self.parameters, seed=seed
            )

    def choose(self, request):
        if self.name == "automatic_travel":
            raise ValueError("automatic transport has no Mover decision")
        if request.role != self.role:
            raise ValueError("rule role mismatch")
        candidates = [c for c in request.candidates if c.legal]
        if not candidates:
            raise ValueError("no legal semantic decision")
        if self.rule is not None:
            action = self.rule.choose(public_rule_request(request))
            if not any(action == candidate.action for candidate in candidates):
                raise ValueError(
                    f"rule {self.name} returned an illegal semantic action for "
                    f"{request.identity}"
                )
            return PolicyChoice(action)
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
        if self.rule is not None:
            self.registration.verify_sources()
            verify_rule_modules(self.frozen_identity["modules"])
            state = (
                _json_copy(self.rule.state_dict()) if self.registration.stateful else {}
            )
            if not isinstance(state, dict):
                raise ValueError("stateful rule state must be a finite JSON object")
            return {"identity": _json_copy(self.frozen_identity), "state": state}
        return {"rng": self.rng.getstate()}

    def load_state_dict(self, state):
        if self.rule is not None:
            self.registration.verify_sources()
            verify_rule_modules(self.frozen_identity["modules"])
            if state.get("identity") != self.frozen_identity:
                raise ValueError(
                    "saved rule state has a different code or parameter identity"
                )
            saved = _json_copy(state["state"])
            if not isinstance(saved, dict):
                raise ValueError("stateful rule state must be a finite JSON object")
            if self.registration.stateful:
                self.rule.load_state_dict(saved)
            elif saved:
                raise ValueError("stateless rule cannot restore mutable state")
            return

        def tuples(value):
            return tuple(map(tuples, value)) if isinstance(value, list) else value

        self.rng.setstate(tuples(state["rng"]))

    def reset(self):
        self.rng.seed(self.seed)
        if self.rule is not None and callable(getattr(self.rule, "reset", None)):
            self.registration.verify_sources()
            self.rule.reset()
