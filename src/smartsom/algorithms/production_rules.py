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
    "mover": {
        "shortest_path",
        "clearance_shortest_path",
        "random",
        "automatic_travel",
    },
}


def validate_builtin_parameters(role, name, parameters):
    """Reject unused settings and validate the dispatcher admission contract."""
    if not parameters:
        return
    if role != "dispatcher" or name != "nearest":
        raise ValueError(f"builtin rule {role}/{name} exposes no parameters")
    unknown = set(parameters) - {
        "fleet_admission",
        "max_active",
        "work_in_progress_first",
    }
    if unknown:
        raise ValueError(f"unknown dispatcher parameters: {sorted(unknown)}")
    admission = parameters.get("fleet_admission")
    if admission is not None and admission not in ("sqrt", "traffic"):
        raise ValueError("fleet_admission must be sqrt or traffic")
    if "max_active" in parameters:
        limit = parameters["max_active"]
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("max_active must be a positive integer")
        if admission is None:
            raise ValueError("max_active requires fleet_admission")
    if "work_in_progress_first" in parameters and not isinstance(
        parameters["work_in_progress_first"], bool
    ):
        raise ValueError("work_in_progress_first must be a boolean")


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
        if self.registration is None:
            validate_builtin_parameters(role, name, self.parameters)
        self.seed = seed
        self.rng = random.Random(seed)
        self._mover_history = {}
        self._mover_plan_key = None
        self._mover_plan = {}
        self._dispatcher_plan_key = None
        self._dispatcher_plan = {}
        self._dispatcher_distance_cache = {}
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
            selected = self.dispatch_target(request, candidates)
        else:
            vehicle = request.observation["agvs"][request.owner]
            if self.name == "clearance_shortest_path":
                selected = self.coordinated_shortest_path(request, candidates)
            elif vehicle["target"] is None:
                selected = next(
                    (c for c in candidates if c.action == "WAIT"), candidates[0]
                )
            else:
                selected = self.shortest_path(request, candidates)
        return PolicyChoice(selected.action)

    def dispatch_target(self, request, candidates):
        """Choose a pickup admission or a stable loaded destination."""
        targets = [c for c in candidates if c.action is not None]
        idle = next((c for c in candidates if c.action is None), None)
        if (
            self.parameters.get("fleet_admission") == "traffic"
            and request.observation.get("transport", {}).get("mode")
            != "travel_time_matrix"
        ):
            from smartsom.algorithms.agv_dispatcher import (
                destination_score,
                pickup_plan,
            )

            if idle is None:
                selected = min(
                    targets,
                    key=lambda c: destination_score(
                        request.observation, request.owner, c
                    ),
                )
            else:
                key = (
                    request.observation,
                    tuple((c.action.owner, c.action.port) for c in targets),
                )
                if key != self._dispatcher_plan_key:
                    self._dispatcher_plan = pickup_plan(
                        request.observation,
                        [c.action for c in targets],
                        self.parameters,
                        distance_cache=self._dispatcher_distance_cache,
                    )
                    self._dispatcher_plan_key = key
                target = self._dispatcher_plan.get(request.owner)
                selected = next((c for c in candidates if c.action == target), idle)
        elif (
            self.parameters.get("fleet_admission") in ("sqrt", "traffic")
            and idle is not None
        ):
            vehicles = request.observation["agvs"]
            active = sum(
                bool(
                    state.get("target") or state.get("job") or state.get("reservation")
                )
                for state in vehicles.values()
            )
            unassigned = sorted(
                owner
                for owner, state in vehicles.items()
                if not (
                    state.get("target") or state.get("job") or state.get("reservation")
                )
            )
            fleet_limit = self.parameters.get("max_active")
            if not isinstance(fleet_limit, int) or isinstance(fleet_limit, bool):
                fleet_limit = max(1, int(len(vehicles) ** 0.5))
            capacity = max(1, fleet_limit) - active
            if request.owner not in unassigned or unassigned.index(
                request.owner
            ) >= max(0, capacity):
                selected = idle
            else:
                selected = self.nearest_dispatch_target(request, targets, candidates)
        else:
            selected = self.nearest_dispatch_target(request, targets, candidates)
        return selected

    def nearest_dispatch_target(self, request, targets, candidates):
        """Prefer the most advanced ready work when scale admission requests it."""
        if self.parameters.get("work_in_progress_first"):
            steps_by_source = {}
            for job in request.observation.get("jobs", {}).values():
                source = job.get("location")
                if source is not None:
                    steps_by_source[source] = max(
                        steps_by_source.get(source, -1), job.get("step", -1)
                    )
            return min(
                targets or candidates,
                key=lambda candidate: (
                    -steps_by_source.get(candidate.action.owner, -1)
                    if candidate.action is not None
                    else 0,
                    candidate.features[1],
                    candidate.identity,
                ),
            )
        return min(targets or candidates, key=lambda c: (c.features[1], c.identity))

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

    def coordinated_shortest_path(self, request, candidates):
        """Reuse a public-state PIBT plan for every Mover request this tick."""
        from smartsom.algorithms.agv_planner import plan_actions, planning_key

        view = request.observation
        key = planning_key(view)
        supplied = {candidate.action for candidate in candidates}
        if key != self._mover_plan_key:
            self._mover_plan = plan_actions(view, self._mover_history)
            self._mover_plan_key = key
        selected = self._mover_plan[request.owner]
        if selected not in supplied:
            selected = plan_actions(
                view,
                self._mover_history,
                allowed_actions={request.owner: supplied},
            )[request.owner]
        return next(
            candidate for candidate in candidates if candidate.action == selected
        )

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
        state = {"rng": self.rng.getstate()}
        if self.name == "clearance_shortest_path":
            state["mover_history"] = _json_copy(self._mover_history)
        return state

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
        self._dispatcher_plan_key = None
        self._dispatcher_plan = {}
        self._dispatcher_distance_cache = {}
        if self.name == "clearance_shortest_path":
            history = _json_copy(state.get("mover_history", {}))
            if not isinstance(history, dict):
                raise ValueError("mover history must be a finite JSON object")
            self._mover_history = history
            self._mover_plan_key = None
            self._mover_plan = {}

    def reset(self):
        self.rng.seed(self.seed)
        self._mover_history = {}
        self._mover_plan_key = None
        self._mover_plan = {}
        self._dispatcher_plan_key = None
        self._dispatcher_plan = {}
        self._dispatcher_distance_cache = {}
        if self.rule is not None and callable(getattr(self.rule, "reset", None)):
            self.registration.verify_sources()
            self.rule.reset()
