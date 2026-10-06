"""Staged decision physics owned by ProductionSimulator, with no policy imports."""

import copy
from collections import Counter, deque
from dataclasses import asdict

from smartsom.domain.production import MOVES, JointCommand
from smartsom.domain.production_decisions import (
    MOVEMENT_ACTIONS,
    BoundaryCommand,
    Candidate,
    DecisionRequest,
    DispatchTarget,
)


class ProductionProtocol:
    """One transactional boundary; all physical advancement uses the existing kernel."""

    def __init__(self, core):
        self.core = core
        self.stage = "closed"
        self.episode = 0
        self.requests = ()
        self.log = []
        for vehicle in core.agvs.values():
            vehicle.update(
                target=None,
                reservation=None,
                reservation_tick=None,
                reservation_version=0,
                service=None,
                feedback=None,
                empty_notified=False,
                arrived_at=0,
            )
        self.ports = {p.port_id: p for p in core.factory.ports}
        self.sources = tuple(
            owner
            for owner in (*core.storage, *core.machines)
            if self.ports_for(owner, "pickup")
            and core.roles.get(owner) not in ("machine_pre", "system_output")
            and (owner not in core.machines or owner not in core.post)
        )
        self.supply = {}
        self.paths = {}
        self.backup = None
        self.matrix = core.scenario.transport_matrix
        self.travel_times = (
            {(a, b): v for a, b, v in self.matrix.times} if self.matrix else {}
        )
        if self.matrix:
            for key, state in core.agvs.items():
                state.update(point="initial:" + key, travel=None, arrived_at=0)
            if self.matrix.source == "auto":
                points = {p: (x, y) for p, x, y in self.matrix.points}
                for a, b, ticks in self.matrix.times:
                    distance = self.distance(points[a], points[b])
                    expected = None if distance == float("inf") else distance
                    if ticks != expected:
                        raise ValueError(
                            "automatic travel overrides must match geometry"
                        )

    def route(self, start, target):
        """Deterministic unit-step BFS; AGVs do not occupy abstract roads."""
        start, target = tuple(start), tuple(target)
        queue, previous = deque([start]), {start: None}
        while queue:
            cell = queue.popleft()
            if cell == target:
                path = []
                while previous[cell] is not None:
                    path.append(cell)
                    cell = previous[cell]
                return tuple(reversed(path))
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nxt = cell[0] + dx, cell[1] + dy
                if (
                    0 <= nxt[0] < self.core.factory.grid.width
                    and 0 <= nxt[1] < self.core.factory.grid.height
                    and nxt not in self.core.solids
                    and nxt not in previous
                ):
                    previous[nxt] = cell
                    queue.append(nxt)
        raise ValueError("unreachable travel proposal")

    def distance(self, start, target):
        start, target = tuple(start), tuple(target)
        key = start, target
        if key not in self.paths:
            core = self.core
            queue, seen = deque([(start, 0)]), {start}
            result = float("inf")
            while queue:
                cell, length = queue.popleft()
                if cell == target:
                    result = length
                    break
                for dx, dy in MOVES.values():
                    nxt = cell[0] + dx, cell[1] + dy
                    if (
                        0 <= nxt[0] < core.factory.grid.width
                        and 0 <= nxt[1] < core.factory.grid.height
                        and nxt not in core.solids
                        and nxt not in seen
                    ):
                        seen.add(nxt)
                        queue.append((nxt, length + 1))
            self.paths[key] = result
        return self.paths[key]

    def travel_cost(self, vehicle, port):
        state = self.core.agvs[vehicle]
        if self.matrix:
            if self.matrix.source == "auto":
                return self.distance(state["cell"], (port.cell.x, port.cell.y))
            value = self.travel_times[(state["point"], port.port_id)]
            return float("inf") if value is None else value
        return self.distance(state["cell"], (port.cell.x, port.cell.y))

    def arrived(self, state, port):
        if self.matrix:
            return state["travel"] is None and state["point"] == port.port_id
        return tuple(state["cell"]) == (port.cell.x, port.cell.y)

    def depart(self, vehicle, port):
        state = self.core.agvs[vehicle]
        if self.arrived(state, port):
            return
        if state["travel"] and self.matrix.source != "auto":
            raise ValueError("manual matrix cannot retarget mid-trip without geometry")
        duration = self.travel_cost(vehicle, port)
        if duration == float("inf"):
            raise ValueError("unreachable travel proposal")
        state["travel"] = {
            "from": state["point"],
            "to": port.port_id,
            "departed_at": self.core.tick,
            "arrival_tick": self.core.tick + duration,
        }
        if self.matrix.source == "auto":
            state["travel"].update(
                path=self.route(state["cell"], (port.cell.x, port.cell.y)),
                progress=0,
            )
        self.core._emit("travel_started", agv=vehicle, **state["travel"])
        self.complete_travel()

    def complete_travel(self):
        if not self.matrix:
            return
        for vehicle, state in self.core.agvs.items():
            travel = state["travel"]
            if travel and self.matrix.source == "auto":
                progress = min(
                    len(travel["path"]), self.core.tick - travel["departed_at"]
                )
                if progress:
                    state["cell"] = list(travel["path"][progress - 1])
                travel["progress"] = progress
            if travel and travel["arrival_tick"] <= self.core.tick:
                port = self.ports[travel["to"]]
                state.update(
                    point=port.port_id,
                    cell=[port.cell.x, port.cell.y],
                    travel=None,
                    arrived_at=self.core.tick,
                    feedback="travel_arrived",
                )
                self.core._emit("travel_arrived", agv=vehicle, point=port.port_id)

    def port_admission(self):
        """Abstract queues have no road occupancy; only eligible heads get service."""
        import hashlib
        import json
        import random

        # Queue ties use a private, stateless environment stream.
        waiting = {}
        busy = {a["target"]["port"] for a in self.core.agvs.values() if a["service"]}
        for vehicle, state in self.core.agvs.items():
            if state["service"] or not state["target"]:
                continue
            port = self.ports[state["target"]["port"]]
            if port.port_id in busy or not self.arrived(state, port):
                continue
            owner = state["target"]["owner"]
            eligible = (
                bool(self.ready(owner))
                if state["job"] is None
                else self.drop_slot(owner, Counter(), port) is not None
            )
            if eligible:
                waiting.setdefault(port.port_id, []).append(vehicle)
        admitted = set()
        for port, vehicles in sorted(waiting.items()):
            earliest = min(self.core.agvs[v]["arrived_at"] for v in vehicles)
            tied = sorted(
                v for v in vehicles if self.core.agvs[v]["arrived_at"] == earliest
            )
            seed = hashlib.sha256(
                json.dumps(
                    [
                        "port-queue/v1",
                        self.core.scenario.seed,
                        self.episode,
                        port,
                        earliest,
                        tied,
                    ]
                ).encode()
            ).digest()
            rng = random.Random(int.from_bytes(seed, "big"))
            admitted.add(rng.choice(tied))
        return admitted

    def ports_for(self, owner, operation):
        return tuple(
            p
            for p in self.ports.values()
            if any(
                self.core._target(b.target)[0] == owner and operation in b.operations
                for b in p.bindings
            )
        )

    def ready(self, owner):
        core = self.core
        bound = {
            a["service"]["job"]
            for a in core.agvs.values()
            if a["service"] and a["service"]["kind"] == "pickup"
        }
        if owner in core.storage:
            if core.roles[owner] in ("machine_pre", "system_output"):
                return ()
            return tuple(
                sorted(
                    j
                    for rows in core.storage[owner].values()
                    for j in rows
                    if j not in bound and not core._inspection_busy(j)
                )
            )
        if owner in core.machines:
            state = core.machine_state[owner]
            if owner not in core.post and state["status"] == "BLOCKED":
                return (state["job"],) if state["job"] not in bound else ()
        return ()

    def source_supply(self, owner):
        core = self.core
        supply = len(self.ready(owner))
        for machine, post in core.post.items():
            state = core.machine_state[machine]
            if (
                post == owner
                and state["status"] == "PROCESSING"
                and core._free(post) is not None
            ):
                supply += 1
        return supply

    def reserved(self, owner):
        return sum(
            a["job"] is None
            and a["target"] is not None
            and a["target"]["owner"] == owner
            for a in self.core.agvs.values()
        )

    def source_empty(self, owner):
        """Only present physical work; future arrivals and external FIFO are hidden."""
        core = self.core
        machine = next(
            (m for m, post in core.post.items() if post == owner),
            owner if owner in core.machines else None,
        )
        if machine is not None:
            pre = core.pre.get(machine)
            post = core.post.get(machine)
            return (
                core.machine_state[machine]["job"] is None
                and not any(core.storage.get(pre, {}).values())
                and not any(core.storage.get(post, {}).values())
            )
        return not any(
            core.storage.get(owner, {}).values()
        ) and not core.station_state.get(owner, {}).get("jobs")

    def public_view(self):
        core = self.core
        view = core.snapshot(public=True)
        for job, row in view["jobs"].items():
            demand = core.demands[row["demand"]]
            if demand.rush:
                row["rush"] = True
            steps = demand.steps[row["step"] :]
            row.update(
                next_operation=steps[0].operation_type if steps else None,
                remaining_steps=[asdict(s) for s in steps],
                due_at=demand.due_at,
                priority=demand.priority,
                attempt=core.attempts[row["demand"]],
                machine_nominal_ticks={
                    m: steps[0].ticks_on(m)
                    for m in core.machines
                    if steps
                    and steps[0].operation_type in core.machines[m].operation_types
                },
            )
        view["sources"] = {
            owner: {
                "ready": list(self.ready(owner)),
                "supply": self.supply.get(owner, self.source_supply(owner)),
                "reserved": self.reserved(owner),
            }
            for owner in self.sources
        }
        view["topology"] = {
            "width": core.factory.grid.width,
            "height": core.factory.grid.height,
            "solids": sorted(core.solids),
            "ports": {p.port_id: [p.cell.x, p.cell.y] for p in self.ports.values()},
        }
        view["feature_time_scale"] = core.scenario.reward_time_scale
        if self.matrix:
            view["transport"] = {
                "mode": "travel_time_matrix",
                "source": self.matrix.source,
            }
            for state in view["agvs"].values():
                state["travel_remaining"] = (
                    max(0, state["travel"]["arrival_tick"] - core.tick)
                    if state["travel"]
                    else 0
                )
        view["state_version"] = f"{core.tick}:{self.stage}"
        return view

    def job_features(self, job):
        core = self.core
        row = core.jobs[job]
        demand = core.demands[row["demand"]]
        scale = core.scenario.reward_time_scale
        return (
            (len(demand.steps) - row["step"]) / 10,
            (demand.due_at - core.tick) / scale,
            demand.priority / 10,
            (core.tick - row["since"]) / scale,
            core.attempts[row["demand"]] / 10,
            float(row["quality"] == "UNKNOWN"),
            float(row["quality"] == "PASS"),
            float(row["quality"] == "FAIL"),
        )

    def request(self, role, owner, candidates, *, prefix=(), count=1):
        return DecisionRequest(
            self.core.tick,
            self.stage,
            role,
            owner,
            tuple(candidates),
            self.public_view(),
            tuple(prefix),
            count,
        )

    def begin(self):
        if self.stage != "closed" or self.core.done:
            raise ValueError("boundary is open or simulation has ended")
        self.backup = copy.deepcopy(
            {k: v for k, v in self.core.__dict__.items() if k != "protocol"}
        )
        self.core.events = []
        self.stage, self.log = "proposals", []
        self.sources = tuple(
            owner
            for owner in (*self.core.storage, *self.core.machines)
            if self.ports_for(owner, "pickup")
            and self.core.roles.get(owner) not in ("machine_pre", "system_output")
            and (owner not in self.core.machines or owner not in self.core.post)
        )
        self.supply = {owner: self.source_supply(owner) for owner in self.sources}
        self.machine_commands, self.dispatch_commands = {}, {}
        self.dispatch_rejections = {}
        self.prefixes, self.pairs, self.service_vehicles = {}, (), {}
        requests = []
        for machine in self.core.machines:
            candidates = []
            for job, mode_id in self.core.machine_choices(machine):
                mode = next(
                    m
                    for m in self.core.machines[machine].quality_modes
                    if m.quality_mode_id == mode_id
                )
                nominal = (
                    self.core.demands[self.core.jobs[job]["demand"]]
                    .steps[self.core.jobs[job]["step"]]
                    .ticks_on(machine)
                )
                features = (
                    *self.job_features(job),
                    nominal
                    * float(mode.time_scale)
                    / float(self.core.machines[machine].processing_rate_multiplier)
                    / self.core.scenario.reward_time_scale,
                    float(mode.time_scale),
                    float(mode.error_rate)
                    if self.core.scenario.quality_probability_visibility == "public"
                    else -1,
                    float(mode_id == "normal"),
                )
                candidates.append(
                    Candidate(f"START:{job}:{mode_id}", (job, mode_id), features)
                )
            if candidates:
                requests.append(self.request("machine", machine, candidates))
        for vehicle, state in self.core.agvs.items():
            if state["service"]:
                continue
            if state["job"] is None and state["target"]:
                empty = self.source_empty(state["target"]["owner"])
                if not empty:
                    state["empty_notified"] = False
                if not empty or state["empty_notified"]:
                    continue
                state["empty_notified"] = True
            elif self.matrix and state["travel"]:
                continue
            candidates = self.dispatch_candidates(vehicle)
            if candidates:
                requests.append(self.request("dispatcher", vehicle, candidates))
        self.requests = tuple(requests)
        return self.requests

    def destination_owners(self, job):
        core, row = self.core, self.core.jobs[job]
        steps = core.demands[row["demand"]].steps
        if row["quality"] == "FAIL":
            return tuple(core.scrap)
        if row["step"] < len(steps):
            destinations = tuple(
                core.pre.get(m, m)
                for m, machine in core.machines.items()
                if steps[row["step"]].operation_type in machine.operation_types
            )
        else:
            destinations = tuple(
                owner for owner in core.storage if core.roles[owner] == "system_output"
            )
        if row["quality"] == "UNKNOWN":
            destinations += tuple(core.stations)
        return destinations

    def dispatch_candidates(self, vehicle):
        core, state = self.core, self.core.agvs[vehicle]
        result = []
        if state["job"] is None:
            owners = self.sources
            operation = "pickup"
        else:
            owners, operation = self.destination_owners(state["job"]), "drop_off"
        for owner in sorted(owners):
            for port in self.ports_for(owner, operation):
                distance = self.travel_cost(vehicle, port)
                if distance == float("inf"):
                    continue
                slots = core.storage.get(owner, {})
                occupied = sum(len(v) for v in slots.values())
                cap_values = core.capacity.get(owner, {})
                finite = all(v is not None for v in cap_values.values())
                cap = sum(v for v in cap_values.values() if v is not None)
                features = (
                    1.0,
                    distance / core.scenario.reward_time_scale,
                    port.cell.x / core.factory.grid.width,
                    port.cell.y / core.factory.grid.height,
                    float(state["job"] is not None),
                    self.supply.get(owner, 0) / 100,
                    self.reserved(owner) / 100,
                    occupied / 100,
                    cap / 100,
                    float(finite),
                    float(finite and occupied >= cap) if cap_values else 0.0,
                    float(state["target"] == {"owner": owner, "port": port.port_id}),
                )
                result.append(
                    Candidate(
                        f"TARGET:{owner}:{port.port_id}",
                        DispatchTarget(owner, port.port_id),
                        features,
                    )
                )
        return tuple(result)

    def accept_proposals(self, machines, dispatchers):
        if self.stage != "proposals":
            raise ValueError("wrong boundary stage")
        known = {(r.role, r.owner): r for r in self.requests}
        for key, action in machines.items():
            request = known.get(("machine", key))
            if not request or action not in [c.action for c in request.candidates]:
                raise ValueError("invalid machine START proposal")
        for request in self.requests:
            if request.role == "machine" and request.owner not in machines:
                raise ValueError("a legal machine START cannot be omitted")
            if request.role == "dispatcher" and request.owner not in dispatchers:
                raise ValueError("a legal Dispatcher target cannot be omitted")
        for key, target in dispatchers.items():
            request = known.get(("dispatcher", key))
            if not request or target not in [c.action for c in request.candidates]:
                raise ValueError("invalid Dispatcher target proposal")
            state = self.core.agvs[key]
            if (
                self.matrix
                and self.matrix.source != "auto"
                and state["travel"]
                and state["target"] != asdict(target)
            ):
                raise ValueError(
                    "manual matrix cannot retarget mid-trip without geometry"
                )
        self.machine_commands = dict(machines)
        self.dispatch_commands = dict(dispatchers)
        for key, target in sorted(dispatchers.items()):
            state = self.core.agvs[key]
            if state["target"] == asdict(target):
                continue
            if state["target"]:
                self.core.metrics["reroutes"] += 1
            state["target"] = asdict(target)
            state["empty_notified"] = False
            state["arrived_at"] = self.core.tick
            if self.matrix:
                self.depart(key, self.ports[target.port])
        # START occurs at t, after the source supply was frozen.
        for key, action in sorted(machines.items()):
            from smartsom.domain.production import MachineCommand

            self.core._start(key, MachineCommand(*action))
        self.stage = "buffer"
        self.service_vehicles = {}
        self.admitted = self.port_admission() if self.matrix else set(self.core.agvs)
        for vehicle, state in self.core.agvs.items():
            if state["service"] or not state["target"]:
                continue
            target = state["target"]
            port = self.ports[target["port"]]
            if not self.arrived(state, port):
                continue
            if vehicle in self.admitted and state["job"] is None:
                self.service_vehicles.setdefault(target["owner"], []).append(vehicle)
        self.requests = tuple(
            self.buffer_request(owner, ())
            for owner in sorted(self.service_vehicles)
            if min(len(self.ready(owner)), len(self.service_vehicles[owner])) > 0
        )
        return self.requests

    def buffer_request(self, owner, prefix):
        count = min(len(self.ready(owner)), len(self.service_vehicles[owner]))
        candidates = tuple(
            Candidate(
                job,
                job,
                (
                    *self.job_features(job),
                    min(
                        (
                            self.core.demands[self.core.jobs[job]["demand"]]
                            .steps[self.core.jobs[job]["step"]]
                            .ticks_on(machine)
                            for machine in self.core.machines
                            if self.core.jobs[job]["step"]
                            < len(
                                self.core.demands[self.core.jobs[job]["demand"]].steps
                            )
                            and self.core.demands[self.core.jobs[job]["demand"]]
                            .steps[self.core.jobs[job]["step"]]
                            .operation_type
                            in self.core.machines[machine].operation_types
                        ),
                        default=0,
                    )
                    / self.core.scenario.reward_time_scale,
                    0.0,
                    0.0,
                    0.0,
                ),
            )
            for job in self.ready(owner)
            if job not in prefix
        )
        return self.request("buffer", owner, candidates, prefix=prefix, count=count)

    def matching_cost(self, job, vehicle):
        distances = [
            self.travel_cost(vehicle, p)
            for owner in self.destination_owners(job)
            for p in self.ports_for(owner, "drop_off")
        ]
        return min(distances, default=float("inf"))

    def inspection_tie(self, job, vehicle):
        row = self.core.jobs[job]
        if row["step"] >= len(self.core.demands[row["demand"]].steps):
            return 0
        return min(
            (
                self.travel_cost(vehicle, p)
                for owner in self.core.stations
                for p in self.ports_for(owner, "drop_off")
            ),
            default=float("inf"),
        )

    def prepare_services(self, prefixes, matching):
        if self.stage != "buffer":
            raise ValueError("wrong boundary stage")
        expected = {}
        for owner, vehicles in self.service_vehicles.items():
            k = min(len(self.ready(owner)), len(vehicles))
            prefix = tuple(prefixes.get(owner, ()))
            if (
                len(prefix) != k
                or len(set(prefix)) != k
                or not set(prefix) <= set(self.ready(owner))
            ):
                raise ValueError("Buffer proposal must be the exact legal K-prefix")
            expected[owner] = prefix
        pairs = tuple(matching)
        fifo = tuple(
            pair
            for owner, prefix in expected.items()
            for pair in self.fifo_matching(owner, prefix)
        )
        if set(pairs) != set(fifo):
            raise ValueError("pickup matching must preserve source-local first arrival")
        if len({a for a, _ in pairs}) != len(pairs) or len(
            {j for _, j in pairs}
        ) != len(pairs):
            raise ValueError("pickup matching must be one-to-one")
        jobs_by_source = {j: o for o, prefix in expected.items() for j in prefix}
        if {j for _, j in pairs} != set(jobs_by_source):
            raise ValueError("matching must preserve the Buffer prefix")
        for vehicle, job in pairs:
            owner = jobs_by_source[job]
            if vehicle not in self.service_vehicles[owner]:
                raise ValueError("in-transit or incompatible vehicle in matching")
        self.prefixes, self.pairs = expected, pairs
        occupied_ports = set()
        for vehicle, job in pairs:
            state = self.core.agvs[vehicle]
            owner = jobs_by_source[job]
            port = state["target"]["port"]
            if port in occupied_ports:
                raise ValueError("two services on one physical port")
            occupied_ports.add(port)
            slot = self.core.jobs[job]["slot"] or "machine"
            self.start_service(vehicle, "pickup", job, owner, slot)
            state["reservation"] = None
            state["reservation_tick"] = None
        slots_used = Counter()
        for vehicle, state in sorted(self.core.agvs.items()):
            if state["service"] or not state["job"] or not state["target"]:
                continue
            target, job = state["target"], state["job"]
            port = self.ports[target["port"]]
            if not self.arrived(state, port):
                continue
            owner = target["owner"]
            if (
                owner not in self.destination_owners(job)
                or port.port_id in occupied_ports
                or vehicle not in self.admitted
            ):
                continue
            slot = self.drop_slot(owner, slots_used, port)
            if slot is None:
                self.core.metrics["destination_waiting"] += 1
                continue
            slots_used[(owner, slot)] += 1
            occupied_ports.add(port.port_id)
            self.start_service(vehicle, "drop", job, owner, slot)
        if self.matrix:
            self.core.metrics["destination_waiting"] += sum(
                bool(
                    a["job"]
                    and a["target"]
                    and not a["travel"]
                    and not a["service"]
                    and self.drop_slot(
                        a["target"]["owner"], Counter(), self.ports[a["target"]["port"]]
                    )
                    is None
                )
                for a in self.core.agvs.values()
            )
        self.stage = "mover"
        self.requests = (
            ()
            if self.matrix
            else tuple(self.mover_request(vehicle) for vehicle in self.core.agvs)
        )
        return self.requests

    def fifo_matching(self, owner, prefix):
        from smartsom.algorithms.pickup_matching import first_arrival, matching_rng

        vehicles = self.service_vehicles[owner]
        rng = matching_rng(
            self.core.scenario.seed,
            self.episode,
            owner,
            self.core.tick,
            prefix,
            vehicles,
        )
        return first_arrival(
            prefix, vehicles, lambda v: self.core.agvs[v]["arrived_at"], rng
        )

    def drop_slot(self, owner, used, port=None):
        core = self.core
        if owner in core.scrap:
            cap = core.scrap[owner].capacity
            return (
                "sink"
                if cap is None
                or core.metrics[f"scrap:{owner}"] + used[(owner, "sink")] < cap
                else None
            )
        if owner in core.machines:
            state = core.machine_state[owner]
            return (
                "machine"
                if state["job"] is None and not used[(owner, "machine")]
                else None
            )
        allowed = (
            {
                core._target(binding.target)[1]
                for binding in port.bindings
                if "drop_off" in binding.operations
                and core._target(binding.target)[0] == owner
            }
            if port
            else None
        )
        for slot, jobs in sorted(core.storage[owner].items()):
            if allowed is not None and slot not in allowed:
                continue
            cap = core.capacity[owner][slot]
            if cap is None or len(jobs) + used[(owner, slot)] < cap:
                return slot
        return None

    def start_service(self, vehicle, kind, job, owner, slot):
        state = self.core.agvs[vehicle]
        state["service"] = {
            "kind": kind,
            "job": job,
            "owner": owner,
            "slot": slot,
            "started_at": self.core.tick,
            "remaining": 1,
        }
        self.core.metrics[f"{kind}_services"] += 1
        self.core._emit(f"{kind}_started", agv=vehicle, job=job, owner=owner, slot=slot)

    def mover_mask(self, vehicle):
        core, state = self.core, self.core.agvs[vehicle]
        if self.matrix or state["service"]:
            return (False, False, False, False, True)
        x, y = state["cell"]
        occupied = {tuple(a["cell"]) for a in core.agvs.values()}
        legal = []
        exits = []
        for action, (dx, dy) in MOVES.items():
            nxt = x + dx, y + dy
            valid = (
                0 <= nxt[0] < core.factory.grid.width
                and 0 <= nxt[1] < core.factory.grid.height
                and nxt not in core.solids
            )
            legal.append(valid)
            exits.append(valid and nxt not in occupied and nxt not in core.ports)
        must_clear = (x, y) in core.ports and any(exits)
        return (*exits, False) if must_clear else (*legal, True)

    def mover_request(self, vehicle):
        core, state = self.core, self.core.agvs[vehicle]
        target = self.ports[state["target"]["port"]] if state["target"] else None
        goal = (target.cell.x, target.cell.y) if target else None
        candidates = []
        mask = self.mover_mask(vehicle)
        for action, legal in zip(MOVEMENT_ACTIONS, mask, strict=True):
            dx, dy = MOVES.get(action, (0, 0))
            nxt = state["cell"][0] + dx, state["cell"][1] + dy
            distance = self.distance(nxt, goal) if goal and legal else 0
            if distance == float("inf"):
                distance = core.factory.grid.width * core.factory.grid.height
            features = (
                dx,
                dy,
                distance / core.scenario.reward_time_scale,
                float(goal is not None),
                float(state["job"] is not None),
                float(state["service"] is not None),
                float(nxt in core.ports),
                float(any(tuple(a["cell"]) == nxt for a in core.agvs.values())),
                0.0,
                0.0,
                0.0,
                0.0,
            )
            candidates.append(Candidate(action, action, features, legal))
        return self.request("mover", vehicle, candidates)

    def commit(self, movers):
        if self.stage != "mover":
            raise ValueError("wrong boundary stage")
        if self.matrix and movers:
            raise ValueError("automatic matrix transport does not accept Mover actions")
        if not self.matrix and set(movers) != set(self.core.agvs):
            raise ValueError("every Mover must return a semantic action")
        for vehicle, action in movers.items():
            if (
                action not in MOVEMENT_ACTIONS
                or not self.mover_mask(vehicle)[MOVEMENT_ACTIONS.index(action)]
            ):
                raise ValueError("illegal Mover proposal")
        command = BoundaryCommand(
            tuple(sorted(self.machine_commands.items())),
            tuple(sorted(self.dispatch_commands.items())),
            tuple(sorted(self.prefixes.items())),
            self.pairs,
            tuple(sorted(movers.items())),
        )
        if self.matrix:
            self.meter_matrix_tick()
            for state in self.core.agvs.values():
                if state["travel"]:
                    self.core.metrics["matrix_travel_ticks"] += 1
                    self.core.metrics["matrix_loaded_travel_ticks"] += int(
                        state["job"] is not None
                    )
                elif state["target"] and not state["service"]:
                    self.core.metrics["matrix_port_wait_ticks"] += 1
        before = self.core.snapshot()
        self.core._phase_events = list(self.core.events)
        result = self.core.step(JointCommand(agvs=tuple(sorted(movers.items()))))
        if not self.matrix:
            for vehicle, state in self.core.agvs.items():
                if not state["target"]:
                    continue
                port = self.ports[state["target"]["port"]]
                previous = before["agvs"][vehicle]
                if self.arrived(state, port) and not self.arrived(previous, port):
                    state["arrived_at"] = self.core.tick
            result["state"] = self.core.snapshot()
        result["rejections"].update(self.dispatch_rejections)
        result["actions"] = asdict(command)
        result["boundary_state"] = before
        result["decisions"] = copy.deepcopy(self.log)
        self.stage, self.backup = "closed", None
        self.core._check()
        return result

    def meter_matrix_tick(self):
        """Diagnostics only; never modify rewards or consume randomness."""
        core = self.core
        active = set(core.queue)
        working = set()
        for slots in core.storage.values():
            active.update(j for jobs in slots.values() for j in jobs)
        for key, state in core.machine_state.items():
            status = state["status"].lower()
            if state["down"]:
                status = "down"
            core.metrics[f"machine_{status}_ticks:{key}"] += 1
            if state["job"]:
                active.add(state["job"])
                if status == "processing":
                    working.add(state["job"])
        for state in core.agvs.values():
            if state["job"]:
                active.add(state["job"])
                if state["travel"] or state["service"]:
                    working.add(state["job"])
            if state["service"]:
                working.add(state["service"]["job"])
        for state in core.station_state.values():
            working.update(state["jobs"])
        active = {
            j
            for j in active
            if core.roles.get(core.jobs[j]["location"]) != "system_output"
        }
        core.metrics["wip_ticks"] += len(active)
        core.metrics["job_waiting_ticks"] += len(active - working)

    def complete_services(self):
        for vehicle, state in self.core.agvs.items():
            service = state["service"]
            if not service:
                continue
            service["remaining"] -= 1
            if service["remaining"]:
                continue
            transfer = tuple(service[k] for k in ("kind", "job", "owner", "slot"))
            self.core._transfer(vehicle, transfer)
            state["service"] = None
            state["target"] = None
            state["empty_notified"] = False
            state["feedback"] = f"{service['kind']}_completed"
            self.core._emit(
                f"{service['kind']}_completed", agv=vehicle, job=service["job"]
            )

    def abort(self):
        if self.backup is not None:
            protocol = self.core.protocol
            self.core.__dict__.clear()
            self.core.__dict__.update(self.backup)
            self.core.protocol = protocol
        self.stage, self.backup = "closed", None

    def replay(self, command):
        if not isinstance(command, BoundaryCommand):
            raise TypeError("v3 replay requires BoundaryCommand")
        try:
            self.begin()
            self.accept_proposals(dict(command.machines), dict(command.dispatchers))
            self.prepare_services(dict(command.prefixes), command.matching)
            return self.commit(dict(command.movers))
        except BaseException:
            self.abort()
            raise
