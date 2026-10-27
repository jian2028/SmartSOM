"""Reservation and service state machine; policies only answer detached requests.

The generator serializes zero-time decisions. It does not optimize allocations,
select policy implementations, read model weights or import a learning framework.
"""

import math
from bisect import bisect_right
from collections import Counter

from smartsom.domain.base_decisions import BaseCandidate, BaseDecision, vector
from smartsom.domain.production import MOVES, JointCommand, MachineCommand
from smartsom.domain.social import SOCIAL_ARMS
from smartsom.engine.base_routes import StaticRoutes
from smartsom.engine.production_protocol import ProductionProtocol
from smartsom.engine.social_observation import project_social


class BaseProtocol(ProductionProtocol):
    def __init__(self, core):
        super().__init__(core)
        self.routes = StaticRoutes(core.factory)
        self.station = {
            b: m for m in core.machines for b in (core.pre[m], core.post[m])
        }
        self.time_scale, self.distance_scale, self.count_scale = 200.0, 50.0, 20.0
        self.reevaluate_ticks = 10
        self.quality_penalty = 1000.0
        self.reservation_stop = True
        self.empty_wait = False
        self.reconstructed_inputs = False
        self.social_arm = "base"
        self.wait_until = {}
        self.wake = {}
        self.rejected = {}
        self.metrics = Counter()

    def down_at(self, machine, tick):
        outages = self.core.scenario.outages
        if getattr(self, "_outage_source", None) is not outages:
            self._outage_source = outages
            self._outage_intervals = {
                m: tuple(sorted((o.start, o.end) for o in outages if o.machine_id == m))
                for m in self.core.machines
            }
            self._outage_starts = {
                m: tuple(t for t, _ in rows)
                for m, rows in self._outage_intervals.items()
            }
        index = bisect_right(self._outage_starts[machine], tick) - 1
        return index >= 0 and tick < self._outage_intervals[machine][index][1]

    def configure(self, options):
        for name in (
            "time_scale",
            "distance_scale",
            "count_scale",
            "reevaluate_ticks",
            "quality_penalty",
            "reservation_stop",
            "empty_wait",
            "reconstructed_inputs",
            "social_arm",
        ):
            setattr(self, name, getattr(options, name))

    def items(self, owner):
        return tuple(
            j for rows in self.core.storage.get(owner, {}).values() for j in rows
        )

    def station_items(self, machine):
        state = self.core.machine_state[machine]
        return (
            *self.items(self.core.pre[machine]),
            *self.items(self.core.post[machine]),
            *((state["job"],) if state["job"] else ()),
        )

    def commitments(self, owner):
        return tuple(
            v
            for v, a in self.core.agvs.items()
            if a["target"] and a["target"]["owner"] == owner
        )

    def quota(self, owner):
        core = self.core
        role = core.roles[owner]
        machine = self.station.get(owner)
        if role == "system_output":
            return len(core.agvs)
        if role == "system_input":
            supply = len(self.items(owner))
        elif role == "machine_post":
            supply = len(self.station_items(machine))
        else:
            capacity = (
                1
                + sum(core.capacity[core.pre[machine]].values())
                + sum(core.capacity[core.post[machine]].values())
            )
            supply = capacity - len(self.station_items(machine))
        return max(0, supply - len(self.commitments(owner)))

    def eligible_owner(self, owner):
        machine = self.station.get(owner)
        if machine and self.core.machine_state[machine]["down"]:
            return self.core.roles[owner] == "machine_post" and bool(self.items(owner))
        return True

    def cancel(self, vehicle, reason):
        a = self.core.agvs[vehicle]
        if a["target"] is None:
            return
        self.core._emit(
            "reservation_cancelled",
            agv=vehicle,
            owner=a["target"]["owner"],
            reason=reason,
        )
        if a["service"]:
            self.core._emit(
                "service_interrupted", agv=vehicle, service=dict(a["service"])
            )
        a.update(target=None, reservation=None, service=None)
        self.metrics["cancellations"] += 1

    def complete_services(self):
        # Look only inside physics at the current timestamp. Processing below
        # still uses the previous interval's availability, so simultaneous
        # process completion succeeds, while unfinished unloading is cancelled.
        for vehicle, a in self.core.agvs.items():
            service = a["service"]
            if service and service["kind"] == "drop_off":
                machine = self.station.get(service["owner"])
                if machine and self.down_at(machine, self.core.tick):
                    self.cancel(vehicle, "target_breakdown")
        super().complete_services()
        for a in self.core.agvs.values():
            if a["target"] is None:
                a["reservation"] = None

    def nominal(self, job, machine, mode="normal"):
        row = self.core.jobs[job]
        steps = self.core.demands[row["demand"]].steps
        if row["step"] >= len(steps):
            return 0
        resource = self.core.machines[machine]
        factor = next(
            float(m.time_scale)
            for m in resource.quality_modes
            if m.quality_mode_id == mode
        )
        return max(
            1,
            math.ceil(
                float(steps[row["step"]].work_ticks_on(machine))
                * factor
                / float(resource.processing_rate_multiplier)
            ),
        )

    def remaining_work(self, job):
        core, row = self.core, self.core.jobs[job]
        if getattr(self, "_work_tick", None) != core.tick:
            self._work_tick, self._work_cache = core.tick, {}
        if job in self._work_cache:
            return self._work_cache[job]
        steps = core.demands[row["demand"]].steps[row["step"] :]
        work = 0.0
        for index, op in enumerate(steps):
            machines = [
                m
                for m, spec in core.machines.items()
                if op.operation_type in spec.operation_types
            ]
            value = min(
                max(
                    1,
                    math.ceil(
                        float(op.work_ticks_on(m))
                        / float(core.machines[m].processing_rate_multiplier)
                    ),
                )
                for m in machines
            )
            if index == 0 and row["location"] in core.machines:
                state = core.machine_state[row["location"]]
                if state["status"] == "PROCESSING":
                    value = self.nominal(job, row["location"]) * max(
                        0.0, 1 - state["elapsed"] / state["nominal"]
                    )
            work += value
        self._work_cache[job] = work
        return work

    def job_vector(self, job):
        core, row = self.core, self.core.jobs[job]
        demand = core.demands[row["demand"]]
        steps = demand.steps[row["step"] :]
        locked = any(
            a["service"] and a["service"]["job"] == job for a in core.agvs.values()
        )
        queued = row["location"] in core.storage and not locked
        types = tuple(core.factory.operation_types)
        operation = steps[0].operation_type if steps else None
        following = next(
            (step.operation_type for step in steps if step.operation_type != operation),
            None,
        )
        return vector(
            (demand.due_at - core.tick) / self.time_scale,
            self.remaining_work(job) / self.time_scale,
            (core.tick - row["since"]) / self.time_scale if queued else 0.0,
            demand.rush,
            len(steps) / 5.0,
            row["risk"],
            *(float(operation == t) for t in types),
            float(locked),
            float(queued),
            *(float(following == t) for t in types),
        )

    def summary(self, jobs):
        vectors = [self.job_vector(j) for j in jobs]
        if not vectors:
            return (0.0,) * 12
        n = len(vectors)
        return (
            n / self.count_scale,
            min(v[0] for v in vectors),
            sum(v[0] for v in vectors) / n,
            sum(v[0] < 0 for v in vectors) / self.count_scale,
            sum(v[1] for v in vectors),
            sum(v[1] for v in vectors) / n,
            max(v[1] for v in vectors),
            sum(v[3] for v in vectors) / self.count_scale,
            sum(v[3] for v in vectors) / n,
            sum(v[2] for v in vectors) / n,
            max(v[2] for v in vectors),
            sum(v[11] for v in vectors) / self.count_scale,
        )

    def current_standard_work(self, machine):
        state = self.core.machine_state[machine]
        if state["status"] != "PROCESSING" or not state["job"]:
            return 0.0
        return self.nominal(state["job"], machine) * max(
            0.0, 1 - state["elapsed"] / state["nominal"]
        )

    def station_vectors(self, owner):
        core, machine = self.core, self.station.get(owner)
        quota = self.quota(owner) / self.count_scale
        commitments = len(self.commitments(owner)) / self.count_scale
        if machine is None:
            return (
                vector(
                    0,
                    quota,
                    commitments,
                    *self.summary(self.ready(owner))
                    if core.roles[owner] != "system_output"
                    else (),
                ),
            )
        state = core.machine_state[machine]
        status = (
            state["down"],
            state["status"] == "PROCESSING",
            state["status"] == "BLOCKED",
        )
        pre = self.items(core.pre[machine])
        processing = (state["job"],) if state["job"] else ()
        post = self.ready(core.post[machine])
        pre_work = sum(self.nominal(j, machine) for j in pre) / self.time_scale
        return tuple(
            vector(
                tag,
                quota,
                commitments,
                *status,
                pre_work,
                *self.summary(jobs),
                sum(core.capacity[core.pre[machine]].values()) / self.count_scale,
                sum(core.capacity[core.post[machine]].values()) / self.count_scale,
                *(
                    float(t in core.machines[machine].operation_types)
                    for t in core.factory.operation_types
                ),
                self.current_standard_work(machine) / self.time_scale
                if self.reconstructed_inputs
                else 0.0,
            )
            for tag, jobs in ((1, pre), (2, processing), (3, post))
        )

    def vehicle_vector(self, vehicle, owner=None):
        a = self.core.agvs[vehicle]
        distance = (
            self.routes.to_owner(a["cell"], owner) / self.distance_scale
            if owner
            else 0.0
        )
        return vector(
            distance,
            a["job"] is not None,
            a["service"] is not None,
            (self.core.tick - a["arrived_at"]) / self.time_scale,
            *(self.job_vector(a["job"])[:12] if a["job"] else (0.0,) * 12),
            a["cell"][0] / self.core.factory.grid.width,
            a["cell"][1] / self.core.factory.grid.height,
            *(self.job_vector(a["job"])[12:16] if a["job"] else (0.0,) * 4),
        )

    def make_request(
        self, group, owner, stage, candidates, *, applicants=(), destinations=()
    ):
        core = self.core
        machine = owner if group == "machine" else self.station.get(owner)
        if group == "dispatcher":
            job = core.agvs[owner]["job"]
            jobs = (job,) if job else ()
            commitments = ()
            context = list(self.vehicle_vector(owner))
            context[22 if self.reconstructed_inputs else 18] = (
                len(self.rejected.get(owner, ())) / self.count_scale
            )
            context = tuple(context)
            stations = tuple(
                v for target in destinations for v in self.station_vectors(target)
            )
        else:
            jobs = self.station_items(machine) if machine else self.items(owner)
            if group == "machine":
                jobs = tuple(sorted({j for j, _ in core.machine_choices(machine)}))
            owners = (core.pre[machine], core.post[machine]) if machine else (owner,)
            commitments = tuple(
                v for target in owners for v in self.commitments(target)
            )
            local = self.station_vectors(core.pre[machine] if machine else owner)
            context = local[0]
            if self.reconstructed_inputs and group == "post":
                context = list(context)
                context[26] = self.quota(owner) / self.count_scale
                context[27] = len(self.commitments(owner)) / self.count_scale
                context = tuple(context)
            # Machine uses only its station; Buffer may read direct downstream summaries.
            downstream = set()
            if group != "machine":
                related_jobs = set(jobs)
                related_jobs.update(
                    core.agvs[v]["job"]
                    for v in (*applicants, *commitments)
                    if core.agvs[v]["job"]
                )
                for j in related_jobs:
                    row = core.jobs[j]
                    steps = core.demands[row["demand"]].steps
                    index = row["step"]
                    if machine and row["location"] != core.post[machine]:
                        while (
                            index < len(steps)
                            and steps[index].operation_type
                            in core.machines[machine].operation_types
                        ):
                            index += 1
                    if index < len(steps):
                        downstream.update(
                            core.pre[m]
                            for m, spec in core.machines.items()
                            if steps[index].operation_type in spec.operation_types
                        )
            stations = (
                *local,
                *(
                    v
                    for target in sorted(downstream)
                    for v in self.station_vectors(target)
                ),
            )
        sets = (
            tuple(self.job_vector(j) for j in jobs),
            tuple(self.vehicle_vector(v, owner) for v in applicants),
            tuple(
                self.vehicle_vector(v, core.agvs[v]["target"]["owner"])
                for v in commitments
            ),
            tuple(stations),
        )
        request = BaseDecision(
            core.tick,
            group,
            owner,
            stage,
            tuple(candidates),
            context,
            sets,
            tuple(sorted(self.rejected.get(owner, ())))
            if group == "dispatcher"
            else (),
        )
        if (group, stage) in SOCIAL_ARMS[self.social_arm]:
            from dataclasses import replace

            request = replace(request, social=project_social(self, request))
        return request

    def candidate_owners(self, vehicle, rejected=()):
        a = self.core.agvs[vehicle]
        if a["job"]:
            owners = self.destination_owners(a["job"])
        else:
            owners = self.sources
        return tuple(
            o
            for o in owners
            if o not in rejected
            and self.eligible_owner(o)
            and self.quota(o) > 0
            and math.isfinite(self.routes.to_owner(a["cell"], o))
        )

    def accept(self, vehicle, owner):
        if self.core.agvs[vehicle]["target"] or self.quota(owner) <= 0:
            raise ValueError("invalid reservation transition")
        a = self.core.agvs[vehicle]
        ports = self.ports_for(owner, "drop_off" if a["job"] else "pickup")
        port = min(
            ports,
            key=lambda p: (
                self.routes.distances(((p.cell.x, p.cell.y),)).get(
                    tuple(a["cell"]), math.inf
                ),
                p.port_id,
            ),
        )
        a.update(
            target={"owner": owner, "port": port.port_id},
            reservation=owner,
            reservation_tick=self.core.tick,
            arrived_at=self.core.tick,
        )
        self.core._emit("reservation_accepted", agv=vehicle, owner=owner)
        self.metrics["accepted"] += 1

    @staticmethod
    def checked(request, action):
        if action not in [c.action for c in request.candidates]:
            raise ValueError("illegal semantic Base action")
        return action

    def decisions(self):
        core = self.core
        core.events = []
        for v, a in core.agvs.items():
            if a["target"] and not self.eligible_owner(a["target"]["owner"]):
                self.cancel(v, "source_or_target_breakdown")
        for m in core.machines:
            candidates = []
            for job, mode in core.machine_choices(m):
                spec = next(
                    x
                    for x in core.machines[m].quality_modes
                    if x.quality_mode_id == mode
                )
                candidates.append(
                    BaseCandidate(
                        f"start:{job}:{mode}",
                        (job, mode),
                        vector(
                            *self.job_vector(job)[:12],
                            self.nominal(job, m, mode) / self.time_scale,
                            float(spec.error_rate),
                            float(spec.time_scale),
                            mode == "normal",
                        ),
                    )
                )
            if candidates:
                request = self.make_request("machine", m, "process", candidates)
                action = self.checked(request, (yield request))
                core._start(m, MachineCommand(*action))
                self.metrics["machine_choices"] += 1
        # Opportunity signatures exclude clocks and nominal progress. New supply,
        # freed capacity and repair wake relevant vehicles; otherwise positive retry.
        active = []
        waited = set()
        for v, a in core.agvs.items():
            if a["target"] or a["service"]:
                continue
            owners = self.candidate_owners(v)
            signature = tuple((o, self.quota(o)) for o in owners)
            old_tick, old_signature = self.wake.get(v, (-self.reevaluate_ticks, None))
            can_wait = self.empty_wait and a["job"] is None
            if can_wait or (
                owners
                and (
                    signature != old_signature
                    or core.tick - old_tick >= self.reevaluate_ticks
                )
            ):
                active.append(v)
                self.wake[v] = (core.tick, signature)
        rejected = {v: set() for v in active}
        self.rejected = rejected
        while active:
            applications = {}
            for v in active:
                owners = self.candidate_owners(v, rejected[v])
                can_wait = self.empty_wait and core.agvs[v]["job"] is None
                if not owners and not can_wait:
                    continue
                candidates = []
                for owner in owners:
                    stats = self.station_vectors(owner)
                    ready = (
                        self.summary(self.ready(owner))
                        if core.agvs[v]["job"] is None
                        else stats[0][7:19]
                    )
                    machine = self.station.get(owner)
                    pre_summary = (
                        self.summary(self.items(core.pre[machine]))
                        if machine
                        else (0.0,) * 12
                    )
                    state = core.machine_state[machine] if machine else None
                    machine_summary = (
                        self.summary((state["job"],))
                        if state and state["job"]
                        else (0.0,) * 12
                    )
                    local_status = (
                        (
                            state["down"],
                            state["status"] == "PROCESSING",
                            state["status"] == "BLOCKED",
                            sum(
                                self.nominal(j, machine)
                                for j in self.items(core.pre[machine])
                            )
                            / self.time_scale,
                            len(self.items(core.post[machine])) / self.count_scale,
                        )
                        if state
                        else (0.0,) * 5
                    )
                    candidates.append(
                        BaseCandidate(
                            f"request:{owner}",
                            (owner,),
                            vector(
                                self.routes.to_owner(core.agvs[v]["cell"], owner)
                                / self.distance_scale,
                                self.quota(owner) / self.count_scale,
                                *ready,
                                len(self.commitments(owner)) / self.count_scale,
                                *pre_summary,
                                *machine_summary,
                                *local_status,
                                self.current_standard_work(machine) / self.time_scale
                                if self.reconstructed_inputs and machine
                                else 0.0,
                            ),
                        )
                    )
                if can_wait:
                    candidates.append(
                        BaseCandidate("EMPTY_WAIT", ("EMPTY_WAIT",), vector())
                    )
                stage = "delivery" if core.agvs[v]["job"] else "pickup"
                request = self.make_request(
                    "dispatcher", v, stage, candidates, destinations=owners
                )
                (owner,) = self.checked(request, (yield request))
                if owner == "EMPTY_WAIT":
                    waited.add(v)
                    self.wait_until[v] = core.tick + 1
                    self.metrics["empty_wait"] += 1
                    continue
                applications.setdefault(owner, []).append(v)
                self.metrics["requests"] += 1
            if not applications:
                break
            self.metrics["negotiation_rounds"] += 1
            for owner, waiting in sorted(applications.items()):
                role = core.roles[owner]
                group = {
                    "system_input": "input",
                    "machine_pre": "pre",
                    "machine_post": "post",
                }.get(role)
                if role == "system_output" or (
                    role == "system_input" and len(waiting) <= self.quota(owner)
                ):
                    for v in waiting:
                        self.accept(v, owner)
                    waiting = []
                while waiting and self.quota(owner) > 0:
                    candidates = [
                        BaseCandidate(
                            f"accept:{v}", (v,), self.vehicle_vector(v, owner)
                        )
                        for v in waiting
                    ]
                    if group != "input" and self.reservation_stop:
                        candidates.append(BaseCandidate("STOP", ("STOP",), vector()))
                    request = self.make_request(
                        group, owner, "accept", candidates, applicants=waiting
                    )
                    (selected,) = self.checked(request, (yield request))
                    if selected == "STOP":
                        self.metrics["stop"] += 1
                        break
                    self.accept(selected, owner)
                    waiting.remove(selected)
                for v in waiting:
                    rejected[v].add(owner)
                    self.metrics["rejected"] += 1
            active = [
                v for v in active if core.agvs[v]["target"] is None and v not in waited
            ]
        # On-site selection is independent of reservation acceptance order.
        for owner in sorted(core.storage):
            while True:
                vehicles = [
                    v
                    for v in self.commitments(owner)
                    if not core.agvs[v]["service"]
                    and tuple(core.agvs[v]["cell"]) in self.routes.ports[owner]
                ]
                candidates = []
                if core.roles[owner] in ("machine_post", "system_input"):
                    for v in vehicles:
                        for j in self.ready(owner):
                            candidates.append(
                                BaseCandidate(
                                    f"load:{j}:{v}",
                                    (j, v),
                                    vector(
                                        *self.job_vector(j)[:16],
                                        *self.vehicle_vector(v, owner)[:4],
                                        *self.vehicle_vector(v, owner)[16:18],
                                    ),
                                )
                            )
                elif self.free_drop(owner):
                    candidates = [
                        BaseCandidate(
                            f"unload:{v}", (v,), self.vehicle_vector(v, owner)
                        )
                        for v in vehicles
                    ]
                if not candidates:
                    break
                role = core.roles[owner]
                if role == "system_output":
                    action = min(
                        candidates,
                        key=lambda c: (
                            core.agvs[c.action[0]]["arrived_at"],
                            c.identity,
                        ),
                    ).action
                else:
                    group = {
                        "system_input": "input",
                        "machine_pre": "pre",
                        "machine_post": "post",
                    }[role]
                    request = self.make_request(group, owner, "service", candidates)
                    action = self.checked(request, (yield request))
                if len(action) == 2:
                    job, vehicle = action
                    kind, slot = "pickup", core.jobs[job]["slot"]
                else:
                    (vehicle,) = action
                    job = core.agvs[vehicle]["job"]
                    kind, slot = "drop_off", self.free_drop(owner)
                actual_port = next(
                    p
                    for p in self.ports_for(owner, kind)
                    if (p.cell.x, p.cell.y) == tuple(core.agvs[vehicle]["cell"])
                )
                core.agvs[vehicle]["target"]["port"] = actual_port.port_id
                self.start_service(vehicle, kind, job, owner, slot)
                self.metrics["site_pairs"] += 1
        self.check_reservations()

    def free_drop(self, owner):
        locked = Counter(
            a["service"]["slot"]
            for a in self.core.agvs.values()
            if a["service"]
            and a["service"]["kind"] == "drop_off"
            and a["service"]["owner"] == owner
        )
        for slot, cap in self.core.capacity[owner].items():
            if cap is None or len(self.core.storage[owner][slot]) + locked[slot] < cap:
                return slot
        return None

    def check_reservations(self):
        for m in self.core.machines:
            total = len(self.station_items(m)) + len(self.commitments(self.core.pre[m]))
            capacity = (
                1
                + sum(self.core.capacity[self.core.pre[m]].values())
                + sum(self.core.capacity[self.core.post[m]].values())
            )
            assert total <= capacity, (m, total, capacity)
        for owner in self.sources:
            supply = (
                len(self.station_items(self.station[owner]))
                if owner in self.station
                else len(self.items(owner))
            )
            assert len(self.commitments(owner)) <= supply
        for owner in self.core.storage:
            locked = Counter(
                a["service"]["slot"]
                for a in self.core.agvs.values()
                if a["service"]
                and a["service"]["kind"] == "drop_off"
                and a["service"]["owner"] == owner
            )
            for slot, cap in self.core.capacity[owner].items():
                assert (
                    cap is None
                    or len(self.core.storage[owner][slot]) + locked[slot] <= cap
                )

    def advance(self, movers):
        self.core._phase_events = list(self.core.events)
        elapsed = self.idle_interval(movers)
        before = {v: tuple(a["cell"]) for v, a in self.core.agvs.items()}
        for m, s in self.core.machine_state.items():
            status = "down" if s["down"] else s["status"].lower()
            self.metrics[f"machine_{status}_ticks"] += elapsed
        result = self.core.step(
            JointCommand(agvs=tuple(sorted(movers.items()))),
            compact=True,
            elapsed=elapsed,
        )
        for v, a in self.core.agvs.items():
            if a["target"]:
                ports = self.routes.ports[a["target"]["owner"]]
                if tuple(a["cell"]) in ports and before[v] not in ports:
                    a["arrived_at"] = self.core.tick
        result["elapsed"] = elapsed
        self.metrics["physical_ticks"] += elapsed
        self.metrics["boundaries"] += 1
        self.check_reservations()
        return result

    def idle_interval(self, movers):
        core = self.core
        if any(t > core.tick for t in self.wait_until.values()):
            return 1
        if any(a != "WAIT" for a in movers.values()) or any(
            a["service"] for a in core.agvs.values()
        ):
            return 1
        if any(v["ready"] for v in self.traffic_view().values()):
            return 1
        # The event calendar is private to physics, never a policy observation.
        times = [core.scenario.tick_limit]
        times.extend(
            d.release_at for d in core.demands.values() if d.release_at > core.tick
        )
        times.extend(
            t for o in core.scenario.outages for t in (o.start, o.end) if t > core.tick
        )
        times.extend(
            core.tick + s["remaining"]
            for s in core.machine_state.values()
            if s["status"] == "PROCESSING" and not s["down"]
        )
        times.extend(
            t + self.reevaluate_ticks
            for t, _ in self.wake.values()
            if t + self.reevaluate_ticks > core.tick
        )
        times.append(core.tick + self.reevaluate_ticks)
        return max(1, min(times) - core.tick)

    def traffic_view(self):
        result = {}
        for v, a in self.core.agvs.items():
            owner = a["target"]["owner"] if a["target"] else None
            demand = (
                self.core.demands[self.core.jobs[a["job"]]["demand"]]
                if a["job"]
                else None
            )
            ready = bool(
                owner and (self.free_drop(owner) if a["job"] else self.ready(owner))
            )
            result[v] = {
                "cell": tuple(a["cell"]),
                "owner": owner,
                "ready": ready,
                "service": a["service"] is not None,
                "rush": bool(demand and demand.rush),
                "due": demand.due_at - self.core.tick if demand else 0.0,
            }
        return result

    def mover_mask(self, vehicle):
        a = self.core.agvs[vehicle]
        if a["service"]:
            return (False, False, False, False, True)
        x, y = a["cell"]
        return (
            *((x + dx, y + dy) in self.routes.neighbors for dx, dy in MOVES.values()),
            True,
        )

    def critic_features(self):
        # Only released orders, public elapsed progress and current machine status.
        # No future release/fault times, latent quality or accurate actual remaining.
        core = self.core
        active = [
            j for j, row in core.jobs.items() if row["demand"] not in core.completed
        ]
        states = list(core.machine_state.values())
        return vector(
            *self.summary(active),
            len(core.completed) / self.count_scale,
            sum(s["down"] for s in states) / self.count_scale,
            sum(s["status"] == "BLOCKED" for s in states) / self.count_scale,
            sum(s["status"] == "PROCESSING" for s in states) / self.count_scale,
            sum(a["job"] is not None for a in core.agvs.values()) / self.count_scale,
            sum(a["target"] is not None for a in core.agvs.values()) / self.count_scale,
        )
