"""Integer-tick production physics. No GUI, policy framework or artifact I/O."""

import copy
import hashlib
import json
from collections import Counter
from dataclasses import asdict
from fractions import Fraction

from smartsom._copy import detached_copy
from smartsom.domain.factory_design import PoolStorage, occupied_cells
from smartsom.domain.production import (
    AGV_ACTIONS,
    MOVES,
    JointCommand,
    ProductionScenario,
    validate_production_scenario,
)
from smartsom.engine.port_lookup import decode_target


def rounded(value):
    value = Fraction(value) + Fraction(1, 2)
    return max(1, value.numerator // value.denominator)


class ProductionSimulator:
    def __init__(self, scenario, *, contract="base"):
        if not isinstance(scenario, ProductionScenario):
            raise TypeError(
                "Simulator requires a grid ProductionScenario; migrate matrix inputs with explicit grid and ports"
            )
        if contract != "base" or scenario.transport_matrix is not None:
            raise ValueError("historical simulation contract requires archived source")
        if scenario.factory.inspection_stations or scenario.factory.scrap_bins:
            raise ValueError(
                "current runtime uses final Output inspection without rework"
            )
        validate_production_scenario(scenario)
        self.scenario, self.factory = (scenario, scenario.factory)
        self.tick = 0
        self.events = []
        self.jobs = {}
        self.completed = set()
        self.shipped = self.completed
        self.shipment_times = {}
        self._qualified_shipments = set()
        self._overdue_time = 0
        self._last_reward_components = None
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
        self.storage, self.capacity, self.roles = ({}, {}, {})
        for b in self.buffers.values():
            if isinstance(b.storage, PoolStorage):
                slots = {"pool": b.storage.capacity}
            else:
                slots = {s.slot_id: s.capacity for s in b.storage.slots}
            self.capacity[b.buffer_id] = slots
            self.roles[b.buffer_id] = b.role
        for s in self.stations.values():
            self.capacity[s.inspection_station_id] = {
                x.slot_id: x.capacity for x in s.slots
            }
            self.roles[s.inspection_station_id] = "inspection"
        for key, slots in self.capacity.items():
            self.storage[key] = {slot: [] for slot in slots}
        self.pre, self.post = ({}, {})
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
            s: {"batch": [], "remaining": 0, "status": "IDLE", "jobs": {}}
            for s in self.stations
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
                    ((c.x, c.y) for c in occupied_cells(resource.footprint))
                )
        self._boundary()
        from smartsom.engine.base_protocol import BaseProtocol

        self.protocol = BaseProtocol(self)
        self._check()

    _target = staticmethod(decode_target)

    def _emit(self, kind, **fields):
        self.events.append({"tick": self.tick, "kind": kind, **fields})

    def _draw(self, domain, job, operation):
        if domain == "quality" and (job, operation) in self.quality_samples:
            return Fraction(self.quality_samples[job, operation], 2**53)
        key = json.dumps(
            [domain, self.scenario.seed, job, operation], separators=(",", ":")
        )
        return Fraction(
            int.from_bytes(hashlib.sha256(key.encode()).digest(), "big"), 2**256
        )

    def _input(self, demand):
        return demand.input_id or next(
            (k for k in self.storage if self.roles[k] == "system_input")
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

    def _boundary(self):
        for key, state in self.machine_state.items():
            if getattr(self, "protocol", None) is not None:
                down = self.protocol.down_at(key, self.tick)
            else:
                down = any(
                    (
                        x.machine_id == key and x.start <= self.tick < x.end
                        for x in self.scenario.outages
                    )
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
                row = self.jobs[state["job"]]
                steps = self.demands[row["demand"]].steps
                if (
                    row["step"] < len(steps)
                    and steps[row["step"]].operation_type
                    in self.machines[key].operation_types
                ):
                    state["status"] = "READY"
                    self._emit("same_machine_retained", machine=key, job=state["job"])
                    continue
                owner = self.post[key]
                slot = self._free(owner)
                if slot is not None:
                    job = state["job"]
                    self._remove(job)
                    self._place(job, owner, slot)
                    self._emit(
                        "machine_released", job=job, machine=key, owner=owner, slot=slot
                    )

    def machine_choices(self, machine):
        state = self.machine_state[machine]
        if state["down"] or state["status"] not in ("IDLE", "READY"):
            return []
        if state["job"] is not None:
            jobs = [state["job"]]
        elif machine in self.pre:
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
                        (
                            (job, mode.quality_mode_id)
                            for mode in self.machines[machine].quality_modes
                        )
                    )
        return sorted(result)

    def agv_mask(self, agv, rankings=None):
        mask = self.protocol.mover_mask(agv)
        return [*mask[:4], False, mask[4]]

    @property
    def done(self):
        return self.tick >= self.scenario.tick_limit or (
            self.scenario.mode != "dynamic" and len(self.completed) == len(self.demands)
        )

    @property
    def status(self):
        if not self.done:
            return "running"
        if self.scenario.mode != "dynamic" and len(self.completed) < len(self.demands):
            return "truncated"
        return "completed"

    def step(self, command: JointCommand, *, compact=False, elapsed=1):
        if not isinstance(command, JointCommand):
            raise ValueError("historical command requires archived source")
        if command.machines or command.rankings or command.quality:
            raise ValueError(
                "submit production choices through the current control interface"
            )
        if self.done:
            raise ValueError("simulation has ended")
        if type(elapsed) is not int or elapsed < 1:
            raise ValueError("physical duration must be a positive integer")
        agvs = dict(command.agvs)
        if len(agvs) != len(command.agvs) or set(agvs) - self.agvs.keys():
            raise ValueError("duplicate or unknown movement resource")
        if any((a not in (*MOVES, "WAIT") for a in agvs.values())):
            raise ValueError("physical movement requires a direction or WAIT")
        if elapsed > 1 and (
            any((a != "WAIT" for a in agvs.values()))
            or elapsed > self.protocol.idle_interval(agvs)
        ):
            raise ValueError("idle jump would cross a required physical event")
        self.events = getattr(self, "_phase_events", [])
        self._phase_events = []
        outstanding = [self.demands[d] for d in self.released - self.completed]
        previous_pass = set(self.completed)
        previous_good = len(self._qualified_shipments)
        previous_passing = previous_good / len(previous_pass) if previous_pass else 0.0
        interval_start = self.tick
        rejected, moves = ({}, {})
        for key in self.agvs:
            action = agvs.get(key, "WAIT")
            if not self.agv_mask(key)[AGV_ACTIONS.index(action)]:
                rejected[f"agv:{key}"] = "invalid"
            elif action in MOVES:
                dx, dy = MOVES[action]
                x, y = self.agvs[key]["cell"]
                moves[key] = (x + dx, y + dy)
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
        self.rankings = {}
        for key, dest in moves.items():
            if f"agv:{key}" not in rejected:
                before = self.agvs[key]["cell"]
                self.agvs[key]["cell"] = list(dest)
                self._emit("move", agv=key, before=before, after=list(dest))
        for actor, reason in sorted(rejected.items()):
            self._emit("rejected", actor=actor, reason=reason)
        self.tick += elapsed
        self._advance(elapsed)
        self._boundary()
        overdue = sum(
            (
                max(0, self.shipment_times.get(d.demand_id, self.tick) - d.due_at)
                - max(0, interval_start - d.due_at)
                for d in outstanding
            )
        )
        self._overdue_time += overdue
        passing = (
            len(self._qualified_shipments) / len(self.shipped) if self.shipped else 0.0
        )
        failed = len(self.shipped - previous_pass) - (
            len(self._qualified_shipments) - previous_good
        )
        self._last_reward_components = {
            "shipments": len(self.shipped - previous_pass),
            "passing_change": passing - previous_passing,
            "overdue_time": overdue,
            "failed": failed,
        }
        reward = -(overdue + self.protocol.quality_penalty * failed) / len(self.demands)
        self.total_reward += reward
        self._check()
        return {
            "tick": self.tick,
            "actions": asdict(command),
            "rankings": {},
            "rejections": rejected,
            "events": copy.deepcopy(self.events),
            "reward": reward,
            "state": None if compact else self.snapshot(),
            "status": self.status,
        }

    def _start(self, machine, action):
        job = action.job_id
        row = self.jobs[job]
        op = self.demands[row["demand"]].steps[row["step"]]
        mode = next(
            (
                x
                for x in self.machines[machine].quality_modes
                if x.quality_mode_id == action.mode_id
            )
        )
        lo, hi = (
            Fraction(self.scenario.processing_low),
            Fraction(self.scenario.processing_high),
        )
        nominal_base = op.work_ticks_on(machine)
        base = rounded(
            nominal_base
            * (lo + (hi - lo) * self._draw("processing", job, op.operation_id))
        )
        base = self.processing_samples.get((job, op.operation_id, machine), base)
        scale = Fraction(mode.time_scale) / Fraction(
            self.machines[machine].processing_rate_multiplier
        )
        if self.scenario.processing_rounding == "ceil":
            import math

            actual = max(
                1,
                math.ceil(
                    nominal_base
                    * (lo + (hi - lo) * self._draw("processing", job, op.operation_id))
                    * scale
                ),
            )
            if (job, op.operation_id, machine) in self.processing_samples:
                actual = max(1, math.ceil(base * scale))
            nominal = max(1, math.ceil(nominal_base * scale))
        else:
            actual = rounded(base * scale)
            nominal = rounded(nominal_base * scale)
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
        first = quality == "FAIL" and (not row["confirmed"])
        row.update(
            quality=quality,
            risk=float(row["defective"]),
            confirmed=row["confirmed"] or first,
        )
        self._emit("quality_revealed", job=job, quality=quality, first=first)

    def _transfer(self, agv, transfer):
        kind, job, owner, slot = transfer
        if (
            kind != "pickup"
            and self.roles.get(owner) == "system_output"
            and (self.jobs[job]["demand"] in self.shipped)
        ):
            raise ValueError("original demand already shipped")
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
            demand = row["demand"]
            if demand in self.shipped:
                raise AssertionError("original demand shipped twice")
            self._place(job, owner, slot)
            self.shipped.add(demand)
            self.shipment_times[demand] = self.tick
            self.metrics["submitted"] += 1
            if not row["defective"]:
                self._qualified_shipments.add(demand)
            self._reveal(job)
        else:
            self._place(job, owner, slot)
        self._emit(kind, agv=agv, job=job, owner=owner, slot=slot)

    def _advance(self, elapsed=1):
        self.protocol.complete_services()
        self.protocol.complete_travel()
        for key, state in self.machine_state.items():
            if state["status"] != "PROCESSING" or state["down"]:
                continue
            state["remaining"] -= elapsed
            state["elapsed"] += elapsed
            if state["remaining"]:
                continue
            job = state["job"]
            row = self.jobs[job]
            op = self.demands[row["demand"]].steps[row["step"]]
            mode = next(
                (
                    x
                    for x in self.machines[key].quality_modes
                    if x.quality_mode_id == state["mode"]
                )
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
                **{},
                elapsed=state["elapsed"],
            )

    def privileged_reward_components(self):
        """Trainer-only per-interval deltas; never added to an event or snapshot."""
        if self._last_reward_components is None:
            raise ValueError("no committed reward interval")
        return copy.deepcopy(self._last_reward_components)

    def privileged_task_totals(self):
        quality = self.privileged_output_quality()
        return {
            "shipments": quality["shipped"],
            "passing_change": quality["passing_rate"] or 0.0,
            "overdue_time": self._overdue_time,
        }

    def privileged_output_quality(self):
        """Oracle evaluation/trainer channel; never attach to public observations."""
        shipped = len(self.shipped)
        good = len(self._qualified_shipments)
        return {
            "shipped": shipped,
            "good_shipped": good,
            "bad_shipped": shipped - good,
            "passing_rate": good / shipped if shipped else None,
        }

    def snapshot(self, *, public=False):
        jobs = detached_copy(self.jobs)
        for row in jobs.values():
            row.pop("defective", None)
        if public:
            jobs = {j: r for j, r in jobs.items() if r["location"] != "queue"}
            for row in jobs.values():
                row.pop("defective", None)
                if (
                    self.scenario.quality_probability_visibility == "hidden"
                    and row["quality"] == "UNKNOWN"
                ):
                    row["risk"] = -1.0
        machines = detached_copy(self.machine_state)
        if public:
            for row in machines.values():
                row.pop("remaining")
        return {
            "tick": self.tick,
            "jobs": jobs,
            "agvs": detached_copy(self.agvs),
            "machines": machines,
            "stations": detached_copy(self.station_state),
            "storage": detached_copy(self.storage),
            "rankings": detached_copy(self.rankings),
            "completed": sorted(self.completed),
            **{
                "output_semantics": "blind-shipment/v1",
                "shipped": sorted(self.shipped),
                "shipment_times": dict(self.shipment_times),
            },
            "released": sorted(self.released),
            "announced": [
                {
                    **{
                        key: value
                        for key, value in asdict(d).items()
                        if key != "rush" or value
                    },
                    "steps": [
                        {
                            **{
                                key: value
                                for key, value in asdict(s).items()
                                if key != "reference_ticks" or value is not None
                            },
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
            **{
                "source_supply": {
                    owner: {
                        "ready": list(self.protocol.ready(owner)),
                        "supply": self.protocol.source_supply(owner),
                        "reserved": self.protocol.reserved(owner),
                    }
                    for owner in self.protocol.sources
                }
            },
        }

    def _check(self):
        cells = [tuple(a["cell"]) for a in self.agvs.values()]
        if self.scenario.transport_matrix is None and (
            len(cells) != len(set(cells)) or any((c in self.solids for c in cells))
        ):
            raise AssertionError("AGV collision")
        held = []
        for owner, slots in self.storage.items():
            for slot, jobs in slots.items():
                cap = self.capacity[owner][slot]
                if cap is not None and len(jobs) > cap:
                    raise AssertionError("storage overflow")
                held.extend(jobs)
                if any(
                    (
                        self.jobs[j]["location"] != owner
                        or self.jobs[j]["slot"] != slot
                        for j in jobs
                    )
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
            (
                self.jobs[j]["demand"]
                for j in held
                if self.roles.get(self.jobs[j]["location"]) != "system_output"
            )
        )
        if any((n != 1 for n in active.values())) or set(active) & self.completed:
            raise AssertionError(
                "duplicate active attempt or completed demand still active"
            )
        if not self._qualified_shipments <= self.shipped:
            raise AssertionError("qualified shipment has no shipped identity")
        if set(active) | self.completed != self.released:
            raise AssertionError("lost demand")
