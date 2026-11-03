"""Reservation and service state machine; policies only answer detached requests.

The generator serializes zero-time decisions. It does not optimize allocations,
select policy implementations, read model weights or import a learning framework.
"""

import math
from bisect import bisect_right
from collections import Counter
from functools import cached_property

from smartsom.domain.base_decisions import BaseCandidate, vector
from smartsom.domain.base_observation import rejected_target_index
from smartsom.domain.production import MOVES, JointCommand, MachineCommand
from smartsom.engine.base_routes import StaticRoutes
from smartsom.engine.physical_services import PhysicalServices


class BaseProtocol(PhysicalServices):
    @cached_property
    def public_topology(self):
        from smartsom.engine.public_facts import topology_facts

        return topology_facts(self)

    def __init__(self, core):
        super().__init__(core)
        self.routes = StaticRoutes(core.factory)
        self.station = {
            b: m for m in core.machines for b in (core.pre[m], core.post[m])
        }
        self.time_scale, self.distance_scale, self.count_scale = 200.0, 50.0, 20.0
        self.reevaluate_ticks = 10
        self.quality_penalty = 1000.0
        self.rejected_target_index = rejected_target_index("native-v2")
        self.social_arm = "base"
        self.idle_since = {}
        self._credit_deficits = {}
        self.central = None
        self.observation_transform = None
        self.stop_permissions, self.allow_empty_wait = (
            {"pre": True, "post": True},
            False,
        )
        self.empty_wait_due = {}
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
        self.rejected_target_index = rejected_target_index(options.observation_schema)
        self.stop_permissions = (
            {
                g: getattr(options.decision_protocol, g).accept.allow_stop
                for g in ("pre", "post")
            }
            if options.control == "base"
            else {"pre": True, "post": True}
        )
        for name in (
            "social_arm",
            "allow_empty_wait",
            "time_scale",
            "distance_scale",
            "count_scale",
            "reevaluate_ticks",
            "quality_penalty",
        ):
            setattr(self, name, getattr(options, name))
        self.sync_idle_clocks()

    def sync_idle_clocks(self):
        """Continuous empty/unassigned time, including unassigned yield travel.

        This is not an application queue or a station waiting-time clock.
        Only physical core.tick supplies time; negotiation never advances it.
        """
        if self.core.done:
            self.idle_since.clear()
            return
        for vehicle, state in self.core.agvs.items():
            if state["job"] is None and not state["target"] and not state["service"]:
                self.idle_since.setdefault(vehicle, self.core.tick)
            else:
                self.idle_since.pop(vehicle, None)

    def post_credit_components(self, owner):
        """Physical POST stock + distinct started machine job - commitments.

        Loading locks remain physical stock until transfer, and their pickup
        commitments remain charged over that same interval. PRE jobs never count.
        A buffered machine's local READY job is a retained started continuation.
        """
        machine = self.station[owner]
        physical = set(self.items(owner))
        state = self.core.machine_state[machine]
        job = state["job"]
        started = int(
            job is not None
            and job not in physical
            and self.core.jobs[job]["location"] == machine
            and state["status"] in ("PROCESSING", "BLOCKED", "READY")
        )
        committed = len(self.commitments(owner))
        return len(physical), started, committed, len(physical) + started - committed

    def diagnose_credit_deficit(self, owner, components):
        if components[3] >= 0:
            self._credit_deficits.pop(owner, None)
        elif self._credit_deficits.get(owner) != components:
            self._credit_deficits[owner] = components
            self.metrics["pickup_credit_deficits"] += 1
            self.core._emit(
                "pickup_credit_deficit",
                owner=owner,
                physical_post=components[0],
                started_machine_job=components[1],
                commitments=components[2],
                credits=components[3],
            )

    def acceptance_vector(self, vehicle, owner):
        features = list(self.vehicle_vector(vehicle, owner))
        if self.core.roles[owner] in (
            "system_input",
            "machine_post",
        ):
            self.sync_idle_clocks()
            since = self.idle_since.get(vehicle, self.core.tick)
            # Slot 22 intentionally stays zero: no logical request-age fiction.
            features[23] = (self.core.tick - since) / self.time_scale
        return tuple(features)

    def record_accept_opportunity(self, group, owner, waiting, *, decision=False):
        """Bounded passive counters, never an allocation or clock transition.

        A round is one nonempty owner's application batch in a negotiation
        iteration, before any acceptance (including Input automatic admission).
        A decision is one yielded accept request; sequential picks in one round
        are separate decisions, not extra rounds. Counts exclude STOP and Output
        (which has no accept actor). Ages use the round's real applicants only.
        All-equal includes all-zero; equal + distinct = rounds for POST/Input.
        Waiting/quota totals are sums over rounds, not unique vehicles or jobs.
        """
        if group not in ("pre", "post", "input"):
            return
        prefix = f"accept_{group}_"
        n = len(waiting)
        if decision:
            self.metrics[prefix + "decisions"] += 1
            self.metrics[prefix + "decision_real_candidates_total"] += n
            self.metrics[prefix + "decisions_with_multiple_candidates"] += int(n > 1)
            return
        quota = self.quota(owner)
        self.metrics[prefix + "rounds"] += 1
        self.metrics[prefix + "round_waiting_total"] += n
        self.metrics[prefix + "round_quota_total"] += quota
        self.metrics[prefix + "oversubscribed_rounds"] += int(n > quota)
        if group in ("post", "input"):
            ages = {
                self.core.tick - self.idle_since.get(v, self.core.tick) for v in waiting
            }
            self.metrics[prefix + "idle_all_zero_rounds"] += int(ages == {0})
            self.metrics[prefix + "idle_all_equal_rounds"] += int(len(ages) == 1)
            self.metrics[prefix + "idle_distinct_rounds"] += int(len(ages) > 1)

    def pickup_pair_vector(self, job, vehicle, owner):
        features = list(
            vector(
                *self.job_vector(job)[:16],
                *self.vehicle_vector(vehicle, owner)[:4],
                *self.vehicle_vector(vehicle, owner)[16:18],
            )
        )
        row = self.core.jobs[job]
        steps = self.core.demands[row["demand"]].steps
        terminal = row["step"] >= len(steps)
        feasible = []
        if not terminal:
            operation = steps[row["step"]].operation_type
            for machine, spec in self.core.machines.items():
                pre = self.core.pre[machine]
                if (
                    operation in spec.operation_types
                    and not self.core.machine_state[machine]["down"]
                    and self.quota(pre) > 0
                    and math.isfinite(
                        self.routes.to_owner(self.core.agvs[vehicle]["cell"], pre)
                    )
                ):
                    feasible.append(
                        sum(self.nominal(j, machine) for j in self.items(pre))
                    )
        features[22:26] = (
            len(feasible) / self.count_scale,
            min(feasible, default=0) / self.time_scale,
            float(bool(feasible)),
            float(terminal),
        )
        return tuple(features)

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
            if (a["target"] and a["target"]["owner"] == owner)
            or (
                self.central is not None
                and v in self.central.destinations_reserved(owner)
            )
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
            components = self.post_credit_components(owner)
            self.diagnose_credit_deficit(owner, components)
            return max(0, components[3])
        else:
            capacity = (
                1
                + sum(core.capacity[core.pre[machine]].values())
                + sum(core.capacity[core.post[machine]].values())
            )
            supply = capacity - len(self.station_items(machine))
        return max(0, supply - len(self.commitments(owner)))

    def source_supply(self, owner):
        if self.core.roles[owner] == "machine_post":
            physical, started, _, _ = self.post_credit_components(owner)
            return physical + started
        return super().source_supply(owner)

    def eligible_owner(self, owner):
        machine = self.station.get(owner)
        if machine and self.core.machine_state[machine]["down"]:
            return self.core.roles[owner] == "machine_post" and bool(self.items(owner))
        return True

    def cancel(self, vehicle, reason):
        if self.central is not None:
            self.central.cancel(vehicle)
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
        self.sync_idle_clocks()

    def complete_services(self):
        if self.central is not None:
            self.central.before_services()
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
        if self.central is not None:
            self.central.after_services()
        self.sync_idle_clocks()

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
        from smartsom.observations.base import job_vector

        return job_vector(self, job)

    def summary(self, jobs):
        from smartsom.observations.base import summary

        return summary(self, jobs)

    def station_vectors(self, owner):
        from smartsom.observations.base import station_vectors

        return station_vectors(self, owner)

    def vehicle_vector(self, vehicle, owner=None):
        from smartsom.observations.base import vehicle_vector

        return vehicle_vector(self, vehicle, owner)

    def make_request(
        self, group, owner, stage, candidates, *, applicants=(), destinations=()
    ):
        from smartsom.observations.base import make_request

        return make_request(
            self,
            group,
            owner,
            stage,
            candidates,
            applicants=applicants,
            destinations=destinations,
        )

    def current_standard_work(self, machine):
        state = self.core.machine_state[machine]
        if state["status"] != "PROCESSING":
            return 0.0
        return self.nominal(state["job"], machine) * max(
            0.0, 1 - state["elapsed"] / state["nominal"]
        )

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
        self.sync_idle_clocks()

    @staticmethod
    def checked(request, action):
        if action not in [c.action for c in request.candidates]:
            raise ValueError("illegal semantic Base action")
        return action

    def decisions(self):
        if self.central is not None:
            yield from self.central.decisions()
            return
        core = self.core
        core.events = []
        self.sync_idle_clocks()
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
        empty_waited = {v for v, due in self.empty_wait_due.items() if due > core.tick}
        for v, a in core.agvs.items():
            if v in empty_waited or a["target"] or a["service"]:
                continue
            owners = self.candidate_owners(v)
            signature = tuple((o, self.quota(o)) for o in owners)
            old_tick, old_signature = self.wake.get(v, (-self.reevaluate_ticks, None))
            wait_due = self.empty_wait_due.pop(v, None)
            if owners and (
                (wait_due is not None and wait_due <= core.tick)
                or signature != old_signature
                or core.tick - old_tick >= self.reevaluate_ticks
            ):
                active.append(v)
                self.wake[v] = (core.tick, signature)
        rejected = {v: set() for v in active}
        self.rejected = rejected
        while active:
            applications = {}
            for v in active:
                owners = self.candidate_owners(v, rejected[v])
                if not owners:
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
                                if machine
                                else 0.0,
                            ),
                        )
                    )
                if self.allow_empty_wait and core.agvs[v]["job"] is None:
                    candidates.append(
                        BaseCandidate("wait:empty", ("EMPTY_WAIT",), vector())
                    )
                stage = "delivery" if core.agvs[v]["job"] else "pickup"
                request = self.make_request(
                    "dispatcher", v, stage, candidates, destinations=owners
                )
                (owner,) = self.checked(request, (yield request))
                if owner == "EMPTY_WAIT":
                    empty_waited.add(v)
                    self.empty_wait_due[v] = core.tick + 1
                    self.metrics["empty_wait_choices"] += 1
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
                self.record_accept_opportunity(group, owner, waiting)
                if role == "system_output" or (
                    role == "system_input" and len(waiting) <= self.quota(owner)
                ):
                    for v in waiting:
                        self.accept(v, owner)
                    waiting = []
                while waiting and self.quota(owner) > 0:
                    candidates = [
                        BaseCandidate(
                            f"accept:{v}", (v,), self.acceptance_vector(v, owner)
                        )
                        for v in waiting
                    ]
                    if self.stop_permissions.get(group, False):
                        candidates.append(BaseCandidate("STOP", ("STOP",), vector()))
                    request = self.make_request(
                        group, owner, "accept", candidates, applicants=waiting
                    )
                    self.record_accept_opportunity(group, owner, waiting, decision=True)
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
                v
                for v in active
                if core.agvs[v]["target"] is None and v not in empty_waited
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
                                    self.pickup_pair_vector(j, v, owner),
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
            if self.core.roles[owner] == "machine_post":
                components = self.post_credit_components(owner)
                self.diagnose_credit_deficit(owner, components)
                assert components[3] >= 0, ("pickup_credit_deficit", owner, components)
                continue
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

    def advance(self, movers, *, on_tick=None):
        self.core._phase_events = list(self.core.events)
        elapsed = self.idle_interval(movers)
        before = {v: tuple(a["cell"]) for v, a in self.core.agvs.items()}
        for m, s in self.core.machine_state.items():
            status = "down" if s["down"] else s["status"].lower()
            if status == "ready":
                # A retained job awaiting its next processing assignment is idle
                # time; centralized controllers may explicitly advance here.
                status = "idle"
            self.metrics[f"machine_{status}_ticks"] += elapsed
        command = JointCommand(agvs=tuple(sorted(movers.items())))
        if on_tick is None:
            result = self.core.step(command, compact=True, elapsed=elapsed)
        else:
            # Subdivide only the already selected physical interval. Recording
            # never introduces additional controller decisions or RNG draws.
            events, reward = [], 0.0
            components = Counter()
            for _ in range(elapsed):
                result = self.core.step(command, compact=True, elapsed=1)
                events.extend(result["events"])
                reward += result["reward"]
                components.update(self.core._last_reward_components)
                on_tick(result, movers)
            result.update(events=events, reward=reward)
            self.core.events = events
            self.core._last_reward_components = dict(components)
        for v, a in self.core.agvs.items():
            if a["target"]:
                ports = self.routes.ports[a["target"]["owner"]]
                if tuple(a["cell"]) in ports and before[v] not in ports:
                    a["arrived_at"] = self.core.tick
        self.sync_idle_clocks()
        result["elapsed"] = elapsed
        self.metrics["physical_ticks"] += elapsed
        self.metrics["boundaries"] += 1
        self.check_reservations()
        return result

    def idle_interval(self, movers):
        core = self.core
        if any(a != "WAIT" for a in movers.values()) or any(
            a["service"] for a in core.agvs.values()
        ):
            return 1
        if any(v["ready"] for v in self.traffic_view().values()):
            return 1
        # The event calendar is private to physics, never a policy observation.
        times = [core.scenario.tick_limit]
        times.extend(due for due in self.empty_wait_due.values() if due > core.tick)
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
