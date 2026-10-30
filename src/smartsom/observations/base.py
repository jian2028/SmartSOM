"""Built-in Base observation fields and explicit SL augmentation.

The numeric projection is separate from physical transitions. Custom files receive
only immutable public facts through the observation interface.
"""

from smartsom.domain.base_decisions import BaseDecision, vector


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
        )
        for tag, jobs in ((1, pre), (2, processing), (3, post))
    )


def vehicle_vector(self, vehicle, owner=None):
    a = self.core.agvs[vehicle]
    distance = (
        self.routes.to_owner(a["cell"], owner) / self.distance_scale if owner else 0.0
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
        context[self.rejected_target_index] = (
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
        commitments = tuple(v for target in owners for v in self.commitments(target))
        local = self.station_vectors(core.pre[machine] if machine else owner)
        context = local[0]
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
    request = BaseDecision(
        core.tick,
        group,
        owner,
        stage,
        tuple(candidates),
        context,
        sets,
        tuple(sorted(self.rejected.get(owner, ()))) if group == "dispatcher" else (),
    )

    from smartsom.domain.social import SOCIAL_ARMS

    if (group, stage) in SOCIAL_ARMS[self.social_arm]:
        from dataclasses import replace

        from smartsom.engine.social_observation import project_social

        request = replace(request, social=project_social(self, request))
    if (
        self.observation_transform is not None
        and self.observation_transform.needs_facts(request)
    ):
        from smartsom.engine.public_facts import public_facts

        request = self.observation_transform(public_facts(core), request)
    return request
