"""Built-in Base observation fields and explicit SL augmentation.

The numeric projection is separate from physical transitions. Custom files receive
only immutable public facts through the observation interface.
"""

import math

from smartsom.domain.observation_data import BaseInputs, vector


def job_summary(self, job):
    core, row = self.core, self.core.jobs[job]
    demand = core.demands[row["demand"]]
    queued = row["location"] in core.storage and job not in core.service_jobs
    return (
        float((demand.due_at - core.tick) / self.time_scale),
        float(self.remaining_work(job) / self.time_scale),
        float((core.tick - row["since"]) / self.time_scale if queued else 0.0),
        float(demand.rush),
        float(queued),
    )


def job_vector(self, job):
    core, row = self.core, self.core.jobs[job]
    demand = core.demands[row["demand"]]
    steps = demand.steps[row["step"] :]
    types = core.factory.operation_types
    operation = steps[0].operation_type if steps else None
    following = next(
        (step.operation_type for step in steps if step.operation_type != operation),
        None,
    )
    values = self.job_summary(job)
    return vector(
        *values[:4],
        len(steps) / 5.0,
        row["risk"],
        *(float(operation == t) for t in types),
        float(job in core.service_jobs),
        values[4],
        *(float(following == t) for t in types),
    )


def summary(self, jobs):
    vectors = [self.job_summary(j) for j in jobs]
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
        sum(v[4] for v in vectors) / self.count_scale,
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
    pre_capacity = sum(core.capacity[core.pre[machine]].values()) / self.count_scale
    post_capacity = sum(core.capacity[core.post[machine]].values()) / self.count_scale
    capabilities = tuple(
        float(t in core.machines[machine].operation_types)
        for t in core.factory.operation_types
    )
    current_work = self.current_standard_work(machine) / self.time_scale
    return tuple(
        vector(
            tag,
            quota,
            commitments,
            *status,
            pre_work,
            *self.summary(jobs),
            pre_capacity,
            post_capacity,
            *capabilities,
            current_work,
        )
        for tag, jobs in ((1, pre), (2, processing), (3, post))
    )


def vehicle_vector(self, vehicle, owner=None):
    a = self.core.agvs[vehicle]
    job_features = self.job_vector(a["job"]) if a["job"] else None
    distance = (
        self.routes.to_owner(a["cell"], owner) / self.distance_scale if owner else 0.0
    )
    return vector(
        distance,
        a["job"] is not None,
        a["service"] is not None,
        (self.core.tick - a["arrived_at"]) / self.time_scale,
        *(job_features[:12] if a["job"] else (0.0,) * 12),
        a["cell"][0] / self.core.factory.grid.width,
        a["cell"][1] / self.core.factory.grid.height,
        *(job_features[12:16] if a["job"] else (0.0,) * 4),
    )


def _make_request(
    self,
    group,
    owner,
    stage,
    candidates,
    *,
    applicants=(),
    destinations=(),
    station_rows=None,
):
    core = self.core
    machine = owner if group == "machine" else self.station.get(owner)
    if group == "dispatcher":
        job = core.agvs[owner]["job"]
        jobs = (job,) if job else ()
        commitments = ()
        context = list(self.vehicle_vector(owner))
        context[self.rejected_target_index] = (
            len(self.rejected.get(owner, ())) / self.count_scale
        )
        context = tuple(context)
        # These rows were computed while constructing this request's candidates,
        # before any action is yielded/committed. Never retain them across requests.
        stations = tuple(
            v
            for target in destinations
            for v in (
                station_rows[target]
                if station_rows is not None and target in station_rows
                else self.station_vectors(target)
            )
        )
    else:
        jobs = self.station_items(machine) if machine else self.items(owner)
        if group == "machine":
            jobs = tuple(sorted({j for j, _ in core.machine_choices(machine)}))
        owners = (core.pre[machine], core.post[machine]) if machine else (owner,)
        commitments = tuple(v for target in owners for v in self.commitments(target))
        local = self.station_vectors(core.pre[machine] if machine else owner)
        context = local[0]
        if group == "post":
            context = list(context)
            context[26] = self.quota(owner) / self.count_scale
            context[27] = len(self.commitments(owner)) / self.count_scale
            if any(context[28:32]):
                raise ValueError("POST credit fields overlap existing context")
            context[28:32] = tuple(
                x / self.count_scale for x in self.post_credit_components(owner)
            )
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
            *(v for target in sorted(downstream) for v in self.station_vectors(target)),
        )
    sets = (
        tuple(self.job_vector(j) for j in jobs),
        tuple(self.vehicle_vector(v, owner) for v in applicants),
        tuple(
            self.vehicle_vector(v, core.agvs[v]["target"]["owner"]) for v in commitments
        ),
        tuple(stations),
    )
    request = self.request
    social = None

    from smartsom.domain.social import SOCIAL_ARMS

    if (group, stage) in SOCIAL_ARMS[self.social_arm]:
        from smartsom.observations.social import project_social

        physical = self.facts.decision
        rows = project_social(self, physical)
        by_action = {c.action: row for c, row in zip(physical.candidates, rows)}
        from dataclasses import replace

        social = tuple(
            replace(by_action[c.action], identity=c.identity)
            for c in request.candidates
        )
    return BaseInputs(context, sets, tuple(candidates), social)


class Projection:
    """Numeric feature construction from the current public query interface."""

    def __init__(self, facts, request, *, arm="base"):
        self.facts = facts
        self.core = facts
        self.request = request
        self.station = facts.station
        self.routes = facts.routes
        self.time_scale, self.distance_scale, self.count_scale = facts.scales
        self.social_arm = arm
        self.rejected_target_index = 22
        self.rejected = {request.owner: request.rejected_owners}
        self.idle_since = facts.idle_since
        self._jobs, self._job_summaries, self._summaries, self._stations = (
            {},
            {},
            {},
            {},
        )

    def items(self, owner):
        return self.facts.inventory(owner)

    def commitments(self, owner):
        return self.facts.commitments(owner)

    def ready(self, owner):
        return self.facts.ready(owner)

    def quota(self, owner):
        return self.facts.quota(owner)

    def nominal(self, job, machine, mode="normal"):
        return self.facts.nominal(job, machine, mode)

    def remaining_work(self, job):
        return self.facts.remaining_work(job)

    def current_standard_work(self, machine):
        return self.facts.current_standard_work(machine)

    def post_credit_components(self, owner):
        return self.facts.post_credit_components(owner)

    def station_items(self, machine):
        return self.facts.station_items(machine)

    def sync_idle_clocks(self):
        pass

    def job_summary(self, job):
        if job not in self._job_summaries:
            self._job_summaries[job] = job_summary(self, job)
        return self._job_summaries[job]

    def job_vector(self, job):
        if job not in self._jobs:
            self._jobs[job] = job_vector(self, job)
        return self._jobs[job]

    def summary(self, jobs):
        key = tuple(jobs)
        if key not in self._summaries:
            self._summaries[key] = summary(self, key)
        return self._summaries[key]

    def station_vectors(self, owner):
        if owner not in self._stations:
            self._stations[owner] = station_vectors(self, owner)
        return self._stations[owner]

    def vehicle_vector(self, vehicle, owner=None):
        return vehicle_vector(self, vehicle, owner)

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

    def pickup_pair_vector(self, job, vehicle, owner):
        vehicle_features = self.vehicle_vector(vehicle, owner)
        features = list(
            vector(
                *self.job_vector(job)[:16],
                *vehicle_features[:4],
                *vehicle_features[16:18],
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

    def candidate_vector(self, candidate):
        core, request = self.core, self.request
        group, stage = request.group, request.stage
        if candidate.action in (("STOP",), ("EMPTY_WAIT",)):
            return vector()
        if group == "machine":
            job, mode = candidate.action
            machine = request.owner
            spec = next(
                x
                for x in core.machines[machine].quality_modes
                if x.quality_mode_id == mode
            )
            result = vector(
                *self.job_vector(job)[:12],
                self.nominal(job, machine, mode) / self.time_scale,
                float(spec.error_rate),
                float(spec.time_scale),
                mode == "normal",
            )
        elif group == "dispatcher":
            (owner,) = candidate.action
            v = request.owner
            stats = self.station_vectors(owner)
            machine = self.station.get(owner)
            # Native station rows already contain these exact summaries.
            # Reuse only within this request, before yielding an action.
            ready = (
                (stats[2][7:19] if machine else stats[0][3:15])
                if core.agvs[v]["job"] is None
                else stats[0][7:19]
            )
            pre_summary = stats[0][7:19] if machine else (0.0,) * 12
            state = core.machine_state[machine] if machine else None
            machine_summary = stats[1][7:19] if machine else (0.0,) * 12
            local_status = (
                (
                    state["down"],
                    state["status"] == "PROCESSING",
                    state["status"] == "BLOCKED",
                    stats[0][6],
                    len(self.items(core.post[machine])) / self.count_scale,
                )
                if state
                else (0.0,) * 5
            )
            result = vector(
                self.routes.to_owner(core.agvs[v]["cell"], owner) / self.distance_scale,
                self.quota(owner) / self.count_scale,
                *ready,
                len(self.commitments(owner)) / self.count_scale,
                *pre_summary,
                *machine_summary,
                *local_status,
                self.current_standard_work(machine) / self.time_scale
                if machine
                else 0.0,
            )
        elif stage == "accept":
            result = self.acceptance_vector(candidate.action[0], request.owner)
        elif group in ("input", "post"):
            result = self.pickup_pair_vector(*candidate.action, request.owner)
        else:
            result = self.vehicle_vector(candidate.action[0], request.owner)
        code = candidate.rule_code
        return (*result[:45], float(code & 1), float((code >> 1) & 1), float(code >> 2))

    def build(self):
        raw = self.facts.decision
        applicants = (
            tuple(c.action[0] for c in raw.candidates if c.action != ("STOP",))
            if raw.stage == "accept"
            else ()
        )
        destinations = (
            tuple(c.action[0] for c in raw.candidates if c.action != ("EMPTY_WAIT",))
            if raw.group == "dispatcher"
            else ()
        )
        candidates = tuple(self.candidate_vector(c) for c in self.request.candidates)
        return _make_request(
            self,
            self.request.group,
            self.request.owner,
            self.request.stage,
            candidates,
            applicants=applicants,
            destinations=destinations,
        )
