"""Integer-tick production physics. No GUI, policy framework or artifact I/O."""

import copy
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import asdict
from fractions import Fraction

from smartsom.domain.factory_design import (
    PoolStorage,
    occupied_cells,
)
from smartsom.domain.production import (
    AGV_ACTIONS,
    MOVES,
    JointCommand,
    ProductionScenario,
    validate_production_scenario,
)


def rounded(value):
    value = Fraction(value) + Fraction(1, 2)
    return max(1, value.numerator // value.denominator)


class ProductionSimulator:
    def __init__(self, scenario):
        if not isinstance(scenario, ProductionScenario):
            raise TypeError(
                "Simulator requires a grid ProductionScenario; migrate matrix inputs with explicit grid and ports"
            )
        validate_production_scenario(scenario)
        self.scenario, self.factory = scenario, scenario.factory
        self.tick = 0
        self.events = []
        self.jobs = {}
        self.completed = set()
        self.attempts = Counter()
        self.released = set()
        self.queue = []
        self.total_reward = 0.0
        self.metrics = Counter()
        self.demands = {
            d.demand_id: d for d in sorted(scenario.demands, key=lambda d: d.demand_id)
        }
        self.processing_samples = {
            (
                f"{sample.demand_id}/attempt/{sample.attempt}",
                sample.operation_id,
                sample.machine_id,
            ): sample.actual_ticks
            for sample in scenario.processing_samples
        }
        self.quality_samples = {
            (
                f"{sample.demand_id}/attempt/{sample.attempt}",
                sample.operation_id,
            ): sample.draw
            for sample in scenario.quality_samples
        }
        self.machines = {
            m.machine_id: m
            for m in sorted(self.factory.machines, key=lambda m: m.machine_id)
        }
        self.stations = {
            s.inspection_station_id: s
            for s in sorted(
                self.factory.inspection_stations, key=lambda s: s.inspection_station_id
            )
        }
        self.buffers = {
            b.buffer_id: b
            for b in sorted(self.factory.buffers, key=lambda b: b.buffer_id)
        }
        self.scrap = {
            b.scrap_bin_id: b
            for b in sorted(self.factory.scrap_bins, key=lambda b: b.scrap_bin_id)
        }
        self.storage, self.capacity, self.roles = {}, {}, {}
        for b in self.buffers.values():
            if isinstance(b.storage, PoolStorage):
                slots = {"pool": b.storage.capacity}
            else:
                slots = {s.slot_id: s.capacity for s in b.storage.slots}
            self.capacity[b.buffer_id] = slots
            self.roles[b.buffer_id] = b.role
        for s in self.stations.values():
            self.capacity[s.inspection_station_id] = {x.slot_id: 1 for x in s.slots}
            self.roles[s.inspection_station_id] = "inspection"
        for key, slots in self.capacity.items():
            self.storage[key] = {slot: [] for slot in slots}
        self.pre, self.post = {}, {}
        for b in self.buffers.values():
            if b.machine_id:
                target = self.pre if b.role == "machine_pre" else self.post
                if b.machine_id in target:
                    raise ValueError("a machine can own at most one buffer per role")
                target[b.machine_id] = b.buffer_id
        self.machine_state = {
            m: {
                "job": None,
                "status": "IDLE",
                "remaining": 0,
                "elapsed": 0,
                "nominal": 0,
                "mode": None,
                "down": False,
            }
            for m in self.machines
        }
        self.station_state = {
            s: {"batch": [], "remaining": 0, "status": "IDLE"} for s in self.stations
        }
        self.agvs = {
            a.agv_id: {"cell": [a.initial_cell.x, a.initial_cell.y], "job": None}
            for a in sorted(self.factory.agvs, key=lambda a: a.agv_id)
        }
        self.rankings = {
            k: [] for k in self.storage if self.roles[k] != "system_output"
        }
        self.ports = {(p.cell.x, p.cell.y): p for p in self.factory.ports}
        self.solids = {(c.x, c.y) for c in self.factory.grid.blocked_cells}
        for group in (
            self.factory.machines,
            self.factory.buffers,
            self.factory.inspection_stations,
            self.factory.scrap_bins,
            self.factory.chargers,
        ):
            for resource in group:
                self.solids.update(
                    (c.x, c.y) for c in occupied_cells(resource.footprint)
                )
        self._boundary()
        self._check()

    @staticmethod
    def _target(target):
        data = asdict(target)
        key = next(v for k, v in data.items() if k.endswith("_id") and k != "slot_id")
        return key, data.get("slot_id", "pool")

    def _emit(self, kind, **fields):
        self.events.append({"tick": self.tick, "kind": kind, **fields})

    def _draw(self, domain, job, operation):
        if domain == "quality" and (job, operation) in self.quality_samples:
            return Fraction(self.quality_samples[(job, operation)], 2**53)
        key = json.dumps(
            [domain, self.scenario.seed, job, operation], separators=(",", ":")
        )
        return Fraction(
            int.from_bytes(hashlib.sha256(key.encode()).digest(), "big"), 2**256
        )

    def _input(self, demand):
        return demand.input_id or next(
            k for k in self.storage if self.roles[k] == "system_input"
        )

    def _enqueue(self, demand_id):
        self.attempts[demand_id] += 1
        key = f"{demand_id}/attempt/{self.attempts[demand_id]}"
        self.jobs[key] = {
            "demand": demand_id,
            "step": 0,
            "quality": "UNKNOWN",
            "defective": False,
            "risk": 0.0,
            "location": "queue",
            "slot": None,
            "since": self.tick,
            "confirmed": False,
        }
        self.queue.append(key)
        self._emit("attempt_queued", job=key, demand=demand_id)

    def _free(self, owner, allowed=None):
        for slot in sorted(self.storage[owner]):
            if allowed is not None and slot not in allowed:
                continue
            cap = self.capacity[owner][slot]
            if cap is None or len(self.storage[owner][slot]) < cap:
                return slot
        return None

    def _place(self, job, owner, slot):
        self.storage[owner][slot].append(job)
        self.jobs[job].update(location=owner, slot=slot, since=self.tick)

    def _remove(self, job):
        row = self.jobs[job]
        if row["location"] in self.storage:
            self.storage[row["location"]][row["slot"]].remove(job)
        elif row["location"] in self.machines:
            state = self.machine_state[row["location"]]
            state.update(job=None, status="IDLE", remaining=0, mode=None)
        elif row["location"] in self.agvs:
            self.agvs[row["location"]]["job"] = None

    def _locked(self, owner):
        return (
            owner in self.station_state
            and self.station_state[owner]["status"] != "IDLE"
        )

    def _boundary(self):
        for key, state in self.machine_state.items():
            down = any(
                x.machine_id == key and x.start <= self.tick < x.end
                for x in self.scenario.outages
            )
            if state["down"] != down:
                state["down"] = down
                self._emit("breakdown" if down else "repair", machine=key)
        for d in sorted(
            self.demands.values(), key=lambda d: (d.release_at, d.demand_id)
        ):
            if d.release_at <= self.tick and d.demand_id not in self.released:
                self.released.add(d.demand_id)
                self._enqueue(d.demand_id)
                self._emit("demand_released", demand=d.demand_id)
        remaining = []
        blocked_inputs = set()
        for job in self.queue:
            owner = self._input(self.demands[self.jobs[job]["demand"]])
            slot = self._free(owner)
            if slot is None or owner in blocked_inputs:
                blocked_inputs.add(owner)
                remaining.append(job)
            else:
                self._place(job, owner, slot)
                self._emit("input_admitted", job=job, owner=owner, slot=slot)
        self.queue = remaining
        for key, state in self.machine_state.items():
            if state["status"] == "BLOCKED" and key in self.post:
                owner = self.post[key]
                slot = self._free(owner)
                if slot is not None:
                    job = state["job"]
                    self._remove(job)
                    self._place(job, owner, slot)
                    self._emit(
                        "machine_released", job=job, machine=key, owner=owner, slot=slot
                    )

    def selectable(self, owner):
        if self._locked(owner):
            return []
        accessible = set()
        for p in self.factory.ports:
            for binding in p.bindings:
                target, slot = self._target(binding.target)
                if target == owner and "pickup" in binding.operations:
                    accessible.add(slot)
        return sorted(
            job
            for slot, jobs in self.storage[owner].items()
            if slot in accessible
            for job in jobs
        )

    def prepare_rankings(self, rankings):
        """Validate a full permutation without advancing physical time."""
        unknown = set(rankings) - self.rankings.keys()
        if unknown:
            raise ValueError(f"unknown pickup facilities: {unknown}")
        prepared = {}
        for owner in self.rankings:
            eligible = self.selectable(owner)
            proposed = list(rankings.get(owner, eligible))
            if len(proposed) != len(set(proposed)) or set(proposed) != set(eligible):
                raise ValueError(
                    f"ranking for {owner} must cover each selectable job once"
                )
            prepared[owner] = proposed
        return prepared

    def machine_choices(self, machine):
        state = self.machine_state[machine]
        if state["down"] or state["status"] not in ("IDLE", "READY"):
            return []
        if machine in self.pre:
            jobs = [
                j for rows in self.storage[self.pre[machine]].values() for j in rows
            ]
        else:
            jobs = [state["job"]] if state["job"] else []
        result = []
        for job in sorted(jobs):
            row = self.jobs[job]
            steps = self.demands[row["demand"]].steps
            if row["quality"] != "FAIL" and row["step"] < len(steps):
                if (
                    steps[row["step"]].operation_type
                    in self.machines[machine].operation_types
                ):
                    result.extend(
                        (job, mode.quality_mode_id)
                        for mode in self.machines[machine].quality_modes
                    )
        return sorted(result)

    def _admit(self, job, owner):
        row = self.jobs[job]
        steps = self.demands[row["demand"]].steps
        role = self.roles.get(owner)
        if owner in self.machines or role == "machine_pre":
            machine = (
                owner if owner in self.machines else self.buffers[owner].machine_id
            )
            return (
                row["quality"] != "FAIL"
                and row["step"] < len(steps)
                and (
                    steps[row["step"]].operation_type
                    in self.machines[machine].operation_types
                )
            )
        if owner in self.stations:
            return row["quality"] == "UNKNOWN" and not self._locked(owner)
        if role == "system_output":
            return row["step"] == len(steps) and row["quality"] != "FAIL"
        if owner in self.scrap:
            return row["quality"] == "FAIL" and (
                self.scrap[owner].capacity is None
                or self.metrics[f"scrap:{owner}"] < self.scrap[owner].capacity
            )
        return role == "storage"

    def interaction(self, agv, rankings=None, *, cell=None):
        vehicle = self.agvs[agv]
        port = self.ports.get(tuple(vehicle["cell"] if cell is None else cell))
        if port is None:
            return None
        ranks = self.prepare_rankings({}) if rankings is None else rankings
        cargo = vehicle["job"]
        access = defaultdict(set)
        for binding in port.bindings:
            if ("drop_off" if cargo else "pickup") in binding.operations:
                owner, slot = self._target(binding.target)
                access[owner].add(slot)
        for owner, slots in access.items():
            if self._locked(owner):
                continue
            if cargo:
                if not self._admit(cargo, owner):
                    continue
                if owner in self.scrap:
                    return ("drop", cargo, owner, "sink")
                if owner in self.machines:
                    s = self.machine_state[owner]
                    if owner not in self.pre and s["job"] is None and not s["down"]:
                        return ("drop", cargo, owner, "machine")
                elif owner in self.storage:
                    slot = self._free(owner, slots)
                    if slot is not None:
                        return ("drop", cargo, owner, slot)
            elif owner in self.storage:
                for job in ranks.get(owner, []):
                    if self.jobs[job]["slot"] in slots:
                        return ("pickup", job, owner, self.jobs[job]["slot"])
            elif owner in self.machines:
                s = self.machine_state[owner]
                if owner not in self.post and s["status"] == "BLOCKED":
                    return ("pickup", s["job"], owner, "machine")
        return None

    def agv_mask(self, agv, rankings=None):
        cell = self.agvs[agv]["cell"]
        mask = []
        for dx, dy in MOVES.values():
            x, y = cell[0] + dx, cell[1] + dy
            mask.append(
                0 <= x < self.factory.grid.width
                and 0 <= y < self.factory.grid.height
                and (x, y) not in self.solids
            )
        return [*mask, self.interaction(agv, rankings) is not None, True]

    def inspection_jobs(self, station):
        if self._locked(station):
            return []
        jobs = [
            j
            for slot in self.storage[station].values()
            for j in slot
            if self.jobs[j]["quality"] == "UNKNOWN"
        ]
        jobs.sort(key=lambda j: (self.jobs[j]["since"], j))
        cap = self.stations[station].parallel_capacity
        return jobs if cap == "max" else jobs[:cap]

    @property
    def done(self):
        return self.tick >= self.scenario.tick_limit or (
            self.scenario.mode == "static" and len(self.completed) == len(self.demands)
        )

    @property
    def status(self):
        if not self.done:
            return "running"
        if self.scenario.mode == "static" and len(self.completed) < len(self.demands):
            return "truncated"
        return "completed"

    def step(self, command: JointCommand):
        if self.done:
            raise ValueError("simulation has ended")
        rankings = self.prepare_rankings(dict(command.rankings))
        agvs, machines, quality = (
            dict(command.agvs),
            dict(command.machines),
            dict(command.quality),
        )
        for supplied, known in (
            (agvs, self.agvs),
            (machines, self.machines),
            (quality, self.stations),
        ):
            if set(supplied) - known.keys():
                raise ValueError("command references an unknown resource")
        if any(a not in AGV_ACTIONS for a in agvs.values()):
            raise ValueError("unknown AGV action")
        if any(a not in ("START", "WAIT") for a in quality.values()):
            raise ValueError("unknown quality action")
        self.events = []
        outstanding = [self.demands[d] for d in self.released - self.completed]
        waiting = sum(d.priority for d in outstanding)
        late = sum(d.priority for d in outstanding if self.tick >= d.due_at)
        previous_pass = set(self.completed)
        rejected, interactions, starts, batches, moves = {}, {}, {}, {}, {}
        claims, station_claims = defaultdict(list), defaultdict(list)
        for key, action in machines.items():
            actor = f"machine:{key}"
            if action.job_id is None:
                continue
            if (action.job_id, action.mode_id) not in self.machine_choices(key):
                rejected[actor] = "invalid"
            else:
                starts[key] = action
                claims[("job", action.job_id)].append(actor)
        for key, action in quality.items():
            if action == "START":
                jobs = self.inspection_jobs(key)
                if not jobs:
                    rejected[f"quality:{key}"] = "invalid"
                else:
                    batches[key] = jobs
        for key in self.agvs:
            action = agvs.get(key, "WAIT")
            actor = f"agv:{key}"
            if not self.agv_mask(key, rankings)[AGV_ACTIONS.index(action)]:
                rejected[actor] = "invalid"
            elif action in MOVES:
                dx, dy = MOVES[action]
                x, y = self.agvs[key]["cell"]
                moves[key] = (x + dx, y + dy)
            elif action == "INTERACT":
                transfer = self.interaction(key, rankings)
                interactions[key] = transfer
                kind, job, owner, slot = transfer
                claims[("job", job)].append(actor)
                if kind == "drop":
                    claims[("slot", owner, slot)].append(actor)
                if owner in self.stations:
                    station_claims[owner].append(actor)
        for claim, actors in claims.items():
            available = 1
            if claim[0] == "slot":
                _, owner, slot = claim
                if owner in self.scrap:
                    cap = self.scrap[owner].capacity
                    available = (
                        len(actors)
                        if cap is None
                        else cap - self.metrics[f"scrap:{owner}"]
                    )
                elif owner in self.storage:
                    cap = self.capacity[owner][slot]
                    available = (
                        len(actors)
                        if cap is None
                        else cap - len(self.storage[owner][slot])
                    )
            if len(actors) > available:
                rejected.update((a, "conflict") for a in actors)
        for key in batches:
            if station_claims[key]:
                rejected[f"quality:{key}"] = "conflict"
                rejected.update((a, "conflict") for a in station_claims[key])
        occupants = {tuple(a["cell"]): key for key, a in self.agvs.items()}
        counts = Counter(moves.values())
        for key, dest in moves.items():
            other = occupants.get(dest)
            if counts[dest] > 1 or (
                other in moves and moves[other] == tuple(self.agvs[key]["cell"])
            ):
                rejected[f"agv:{key}"] = "conflict"
        changed = True
        while changed:
            changed = False
            for key, dest in moves.items():
                other = occupants.get(dest)
                if (
                    f"agv:{key}" not in rejected
                    and other is not None
                    and (other not in moves or f"agv:{other}" in rejected)
                ):
                    rejected[f"agv:{key}"] = "conflict"
                    changed = True
        self.rankings = copy.deepcopy(rankings)
        for key, action in starts.items():
            if f"machine:{key}" not in rejected:
                self._start(key, action)
        for key, jobs in batches.items():
            if f"quality:{key}" not in rejected:
                self.station_state[key].update(
                    batch=jobs,
                    remaining=self.stations[key].inspection_ticks,
                    status="INSPECTING",
                )
                self._emit("inspection_started", station=key, jobs=jobs)
        for key, transfer in interactions.items():
            if f"agv:{key}" not in rejected:
                self._transfer(key, transfer)
        for key, dest in moves.items():
            if f"agv:{key}" not in rejected:
                before = self.agvs[key]["cell"]
                self.agvs[key]["cell"] = list(dest)
                self._emit("move", agv=key, before=before, after=list(dest))
        for actor, reason in sorted(rejected.items()):
            self._emit("rejected", actor=actor, reason=reason)
        self.tick += 1
        self._advance()
        self._boundary()
        for owner, rank in self.rankings.items():
            eligible = set(self.selectable(owner))
            self.rankings[owner] = [j for j in rank if j in eligible]
        delivered = sum(
            self.demands[d].priority for d in self.completed - previous_pass
        )
        n_batches = sum(e["kind"] == "inspection_started" for e in self.events)
        n_fail = sum(
            e["kind"] == "quality_revealed" and e["quality"] == "FAIL" and e["first"]
            for e in self.events
        )
        n_conflicts = sum(
            a.startswith("agv:") and why == "conflict" for a, why in rejected.items()
        )
        reward = 10 * delivered - (waiting + 5 * late) / self.scenario.reward_time_scale
        reward -= 0.1 * n_batches + n_fail + 0.02 * n_conflicts
        if self.tick == self.scenario.tick_limit and self.scenario.mode == "dynamic":
            reward -= 10 * sum(
                self.demands[d].priority for d in self.released - self.completed
            )
        self.total_reward += reward
        self._check()
        return {
            "tick": self.tick,
            "actions": asdict(command),
            "rankings": rankings,
            "rejections": rejected,
            "events": copy.deepcopy(self.events),
            "reward": reward,
            "state": self.snapshot(),
            "status": self.status,
        }

    def _start(self, machine, action):
        job = action.job_id
        row = self.jobs[job]
        op = self.demands[row["demand"]].steps[row["step"]]
        mode = next(
            x
            for x in self.machines[machine].quality_modes
            if x.quality_mode_id == action.mode_id
        )
        lo, hi = (
            Fraction(self.scenario.processing_low),
            Fraction(self.scenario.processing_high),
        )
        nominal_base = op.ticks_on(machine)
        base = rounded(
            nominal_base
            * (lo + (hi - lo) * self._draw("processing", job, op.operation_id))
        )
        base = self.processing_samples.get((job, op.operation_id, machine), base)
        actual = rounded(base * Fraction(mode.time_scale))
        nominal = rounded(nominal_base * Fraction(mode.time_scale))
        self._remove(job)
        row.update(location=machine, slot=None, since=self.tick)
        self.machine_state[machine].update(
            job=job,
            status="PROCESSING",
            remaining=actual,
            elapsed=0,
            nominal=nominal,
            mode=action.mode_id,
        )
        self._emit(
            "processing_started",
            machine=machine,
            job=job,
            mode=action.mode_id,
            nominal_ticks=nominal,
            actual_ticks=actual,
        )

    def _reveal(self, job):
        row = self.jobs[job]
        quality = "FAIL" if row["defective"] else "PASS"
        first = quality == "FAIL" and not row["confirmed"]
        row.update(
            quality=quality,
            risk=float(row["defective"]),
            confirmed=row["confirmed"] or first,
        )
        self._emit("quality_revealed", job=job, quality=quality, first=first)

    def _transfer(self, agv, transfer):
        kind, job, owner, slot = transfer
        self._remove(job)
        row = self.jobs[job]
        if kind == "pickup":
            self.agvs[agv]["job"] = job
            row.update(location=agv, slot=None, since=self.tick)
        elif owner in self.machines:
            self.machine_state[owner].update(job=job, status="READY")
            row.update(location=owner, slot=None, since=self.tick)
        elif slot == "sink" or self.roles.get(owner) == "system_output":
            row.update(location=owner, slot=None, since=self.tick)
            if owner in self.scrap:
                self.metrics[f"scrap:{owner}"] += 1
                self.metrics["pre_output_scrap"] += 1
                self._enqueue(row["demand"])
            else:
                self.metrics["submitted"] += 1
                if row["quality"] == "UNKNOWN":
                    self._reveal(job)
                if row["quality"] == "PASS":
                    self._place(job, owner, slot)
                    self.completed.add(row["demand"])
                    self.metrics["passed"] += 1
                else:
                    self.metrics["output_rejected"] += 1
                    self._enqueue(row["demand"])
        else:
            self._place(job, owner, slot)
        self._emit(kind, agv=agv, job=job, owner=owner, slot=slot)

    def _advance(self):
        for key, state in self.machine_state.items():
            if state["status"] != "PROCESSING" or state["down"]:
                continue
            state["remaining"] -= 1
            state["elapsed"] += 1
            if state["remaining"]:
                continue
            job = state["job"]
            row = self.jobs[job]
            op = self.demands[row["demand"]].steps[row["step"]]
            mode = next(
                x
                for x in self.machines[key].quality_modes
                if x.quality_mode_id == state["mode"]
            )
            defect = self._draw("quality", job, op.operation_id) < Fraction(
                mode.error_rate
            )
            row["defective"] |= defect
            row["quality"] = "UNKNOWN"
            row["risk"] += (1 - row["risk"]) * float(mode.error_rate)
            row["step"] += 1
            state["status"] = "BLOCKED"
            self._emit(
                "processing_completed",
                job=job,
                machine=key,
                operation=op.operation_id,
                defect=defect,
                elapsed=state["elapsed"],
            )
        for key, state in self.station_state.items():
            if state["status"] != "INSPECTING":
                continue
            state["remaining"] -= 1
            if not state["remaining"]:
                for job in state["batch"]:
                    self._reveal(job)
                self._emit("inspection_completed", station=key, jobs=state["batch"])
                state.update(status="IDLE", batch=[])

    def snapshot(self, *, public=False):
        jobs = copy.deepcopy(self.jobs)
        if public:
            jobs = {j: r for j, r in jobs.items() if r["location"] != "queue"}
            for row in jobs.values():
                row.pop("defective")
                if (
                    self.scenario.quality_probability_visibility == "hidden"
                    and row["quality"] == "UNKNOWN"
                ):
                    row["risk"] = -1.0
        machines = copy.deepcopy(self.machine_state)
        if public:
            for row in machines.values():
                row.pop("remaining")
        return {
            "tick": self.tick,
            "jobs": jobs,
            "agvs": copy.deepcopy(self.agvs),
            "machines": machines,
            "stations": copy.deepcopy(self.station_state),
            "storage": copy.deepcopy(self.storage),
            "rankings": copy.deepcopy(self.rankings),
            "completed": sorted(self.completed),
            "released": sorted(self.released),
            "announced": [
                {
                    **asdict(d),
                    "steps": [
                        {
                            **asdict(s),
                            "machine_nominal_ticks": dict(s.machine_nominal_ticks),
                        }
                        for s in d.steps
                    ],
                }
                for d in self.demands.values()
                if d.reveal_at <= self.tick < d.release_at
            ],
            "queue": [] if public else list(self.queue),
            "metrics": dict(self.metrics),
            "return": self.total_reward,
            "status": self.status,
        }

    def decision(self, rankings=None):
        """Detached public input; legal actions never include hidden future state."""
        ranks = self.prepare_rankings(rankings or {})
        view = self.snapshot(public=True)
        view["rankings"] = ranks
        view["machine_choices"] = {m: self.machine_choices(m) for m in self.machines}
        view["inspection_choices"] = {
            s: bool(self.inspection_jobs(s)) for s in self.stations
        }
        view["agv_masks"] = {a: self.agv_mask(a, ranks) for a in self.agvs}
        view["interactions"] = {
            a: {
                p.port_id: self.interaction(a, ranks, cell=(p.cell.x, p.cell.y))
                for p in self.factory.ports
            }
            for a in self.agvs
        }
        for job, row in view["jobs"].items():
            demand = self.demands[row["demand"]]
            row.update(
                remaining_steps=[
                    {
                        "operation_type": step.operation_type,
                        "nominal_ticks": step.nominal_ticks,
                    }
                    for step in demand.steps[row["step"] :]
                ],
                priority=demand.priority,
                due_at=demand.due_at,
                machine_nominal_ticks={
                    machine_id: demand.steps[row["step"]].ticks_on(machine_id)
                    for machine_id, machine in self.machines.items()
                    if row["step"] < len(demand.steps)
                    and demand.steps[row["step"]].operation_type
                    in machine.operation_types
                },
                next_operation=(
                    demand.steps[row["step"]].operation_type
                    if row["step"] < len(demand.steps)
                    else None
                ),
            )
        return view

    def _check(self):
        cells = [tuple(a["cell"]) for a in self.agvs.values()]
        if len(cells) != len(set(cells)) or any(c in self.solids for c in cells):
            raise AssertionError("AGV collision")
        held = []
        for owner, slots in self.storage.items():
            for slot, jobs in slots.items():
                cap = self.capacity[owner][slot]
                if cap is not None and len(jobs) > cap:
                    raise AssertionError("storage overflow")
                held.extend(jobs)
                if any(
                    self.jobs[j]["location"] != owner or self.jobs[j]["slot"] != slot
                    for j in jobs
                ):
                    raise AssertionError("storage identity mismatch")
        for resources in (self.agvs, self.machine_state):
            for owner, row in resources.items():
                if row["job"]:
                    if self.jobs[row["job"]]["location"] != owner:
                        raise AssertionError("resource identity mismatch")
                    held.append(row["job"])
        held.extend(self.queue)
        if len(held) != len(set(held)):
            raise AssertionError("job has multiple physical owners")
        active = Counter(
            self.jobs[j]["demand"]
            for j in held
            if self.roles.get(self.jobs[j]["location"]) != "system_output"
        )
        if any(n != 1 for n in active.values()) or set(active) & self.completed:
            raise AssertionError(
                "duplicate active attempt or completed demand still active"
            )
        if set(active) | self.completed != self.released:
            raise AssertionError("lost demand")
