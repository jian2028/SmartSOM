"""V2 named observations; all scalars derive from the current public callback."""

from smartsom.domain.public_facts import freeze
from smartsom.representations.v2.facts import view
from smartsom.representations.v2.rules import CATALOG
from smartsom.representations.v2.social_config import social_config

SCHEMA = "smartsom.v2.rule-observation/v1"
CONTEXT = {
    "machine": 8,
    "dispatcher.pickup": 0,
    "dispatcher.delivery": 3,
    "pre": 11,
    "post": 11,
    "input": 6,
}
FEATURES = {
    "machine.process": 4,
    "dispatcher.pickup": 18,
    "dispatcher.delivery": 14,
    "pre.accept": 5,
    "pre.service": 5,
    "post.accept": 4,
    "input.accept": 4,
    "post.service": 6,
    "input.service": 6,
}


def _mean(values):
    return sum(values) / len(values) if values else 0.0


def context(v, request):
    f, owner = v.f, request.owner
    t, n = v.time, v.count
    if request.group == "dispatcher":
        if request.stage == "pickup":
            return ()
        due, work, _ = v.job(f.agvs[owner]["job"])
        return due / t, work / t, v.loaded_age(owner) / t
    if request.group == "machine":
        pre, post = f.pre[owner], f.post[owner]
        jobs = f.inventory(pre)
        return (
            v.occupancy(pre),
            v.occupancy(post),
            len(f.commitments(pre)) / n,
            len(f.commitments(post)) / n,
            v.pre_work(owner) / t,
            _mean([v.job(j)[1] for j in jobs]) / t,
            _mean([float(v.job(j)[0] < v.job(j)[1]) for j in jobs]),
            _mean([v.job(j)[2] for j in jobs]) / t,
        )
    traveling, arrived = v.counts(owner)
    ready = f.ready(owner)
    min_slack = min((v.job(j)[0] - v.job(j)[1] for j in ready), default=0) / t
    longest = max((v.job(j)[2] for j in ready), default=0) / t
    if request.group == "input":
        return (
            len(ready) / n,
            f.quota(owner) / n,
            traveling / n,
            arrived / n,
            min_slack,
            longest,
        )
    machine = f.station[owner]
    pre, post = f.pre[machine], f.post[machine]
    state = f.machine_state[machine]
    current = f.current_standard_work(machine) / t
    down, blocked = float(state["down"]), float(state["status"] == "BLOCKED")
    if request.group == "post":
        return (
            v.occupancy(post),
            len(ready) / n,
            f.quota(post) / (v.capacity(post) + 1),
            traveling / n,
            arrived / n,
            current,
            down,
            blocked,
            v.pre_work(machine) / t,
            min_slack,
            longest,
        )
    jobs = f.inventory(pre)
    return (
        v.occupancy(pre),
        f.free_drop_count(pre) / v.capacity(pre),
        v.occupancy(post),
        f.quota(pre) / (v.capacity(pre) + 1 + v.capacity(post)),
        traveling / n,
        arrived / n,
        v.pre_work(machine) / t,
        current,
        down,
        blocked,
        _mean([float(v.job(j)[0] < v.job(j)[1]) for j in jobs]),
    )


def candidate_values(v, request, candidate, physical):
    f = v.f
    t, n, d = v.time, v.count, v.distance
    action, owner = candidate.action, request.owner
    if request.group == "machine":
        due, work, wait = v.job(action[0])
        return due / t, f.nominal(action[0], owner) / t, work / t, wait / t
    if request.group == "dispatcher":
        destination = action[0]
        machine = f.station.get(destination)
        traveling, arrived = v.counts(destination)
        distance = f.routes.to_owner(f.agvs[owner]["cell"], destination) / d
        state = f.machine_state[machine] if machine else None
        current = f.current_standard_work(machine) / t if machine else 0.0
        pre_work = v.pre_work(machine) / t if machine else 0.0
        blocked = float(state["status"] == "BLOCKED") if state else 0.0
        processing = float(state["status"] == "PROCESSING") if state else 0.0
        if request.stage == "pickup":
            supply, ready = v.supply(destination), f.ready(destination)
            is_post = f.roles[destination] == "machine_post"
            return (
                float(not is_post),
                float(is_post),
                distance,
                len(ready) / n,
                traveling / n,
                arrived / n,
                f.quota(destination) / (v.capacity(destination) + 1 if is_post else n),
                min((v.job(j)[0] - v.job(j)[1] for j in supply), default=0) / t,
                _mean([float(v.job(j)[0] < v.job(j)[1]) for j in supply]),
                max((v.job(j)[2] for j in ready), default=0) / t,
                v.occupancy(destination) if is_post else 0.0,
                pre_work,
                processing,
                current,
                blocked,
                float(state["down"]) if state else 0.0,
                v.density(destination, owner),
                float(bool(supply)),
            )
        job = f.agvs[owner]["job"]
        capacity = (
            v.capacity(f.pre[machine]) + 1 + v.capacity(f.post[machine])
            if machine
            else n
        )
        return (
            float(machine is not None),
            float(machine is None),
            distance,
            f.nominal(job, machine) / t if machine else 0.0,
            processing,
            current,
            pre_work,
            v.occupancy(destination) if machine else 0.0,
            f.quota(destination) / capacity,
            traveling / n,
            arrived / n,
            v.occupancy(f.post[machine]) if machine else 0.0,
            blocked,
            v.density(destination, owner),
        )
    if request.group == "pre":
        vehicle = action[0]
        state = f.agvs[vehicle]
        due, work, _ = v.job(state["job"])
        fourth = (
            f.routes.to_owner(state["cell"], owner) / d
            if request.stage == "accept"
            else (f.tick - state["arrived_at"]) / t
        )
        return (
            due / t,
            f.nominal(state["job"], f.station[owner]) / t,
            work / t,
            fourth,
            v.loaded_age(vehicle) / t,
        )
    if request.stage == "accept":
        vehicle = action[0]
        state = f.agvs[vehicle]
        return (
            f.routes.to_owner(state["cell"], owner) / d,
            (f.tick - f.idle_since.get(vehicle, f.tick)) / t,
            state["task_ticks"] / f.tick if f.tick else 0.0,
            v.density(vehicle=vehicle),
        )
    job, vehicle = action
    due, work, wait = v.job(job)
    _, distance = v.pair(job, physical)
    _, feasible, terminal = v.destinations(job, vehicle)
    workload = min(
        (v.pre_work(f.station[o]) for o, _ in feasible if o in f.station), default=0
    )
    return (
        due / t,
        work / t,
        wait / t,
        float(bool(feasible)),
        (workload / t if not terminal else 0.0),
        distance / d,
    )


def relation(v, job, machine=None, *, ordered=False):
    stations, terminal = v.next_stations(job, machine)
    if ordered:
        stations = sorted(stations)
    return {"peers": tuple(v.station(m) for m in stations), "terminal": float(terminal)}


def social_relations(v, request, candidate, *, ordered=False):
    f, action = v.f, candidate.action
    if request.group == "dispatcher":
        vehicles = v.vehicles(action[0])
        if ordered:
            vehicles = sorted(vehicles, key=lambda row: row[0])
        peers = tuple(
            (float(arrived), distance / v.distance, wait / v.time)
            for vehicle, arrived, distance, wait in vehicles
            if vehicle != request.owner
        )
        return {"peers": peers}
    if request.group == "machine":
        return {"downstream": relation(v, action[0], request.owner, ordered=ordered)}
    if request.group == "pre":
        job = f.agvs[action[0]]["job"]
        machine = f.station[request.owner]
        result = {"downstream": relation(v, job, machine, ordered=ordered)}
        if request.stage == "accept":
            alternatives, _ = v.next_stations(job)
            if ordered:
                alternatives = sorted(alternatives)
            result["alternatives"] = {
                "peers": tuple(v.station(m) for m in alternatives if m != machine)
            }
        return result
    if request.stage == "service":
        return {"downstream": relation(v, action[0], ordered=ordered)}
    jobs = []
    supply = v.supply(request.owner)
    if ordered:
        supply = sorted(supply)
    for job in supply:
        row = f.jobs[job]
        machine = f.station.get(request.owner)
        on_machine = machine is not None and row["location"] == machine
        condition = tuple(x / v.time for x in v.job(job))
        if request.group == "post":
            condition += (float(on_machine),)
        jobs.append(
            {
                "condition": condition,
                "downstream": relation(
                    v, job, machine if on_machine else None, ordered=ordered
                ),
            }
        )
    return {"jobs": tuple(jobs)}


def create_observation(*, social_enabled=None, social=None):
    settings = social_config(social, social_enabled)
    enabled = settings.kind == "implicit"
    ordered = settings.fusion == "concat"

    def observe(facts, request):
        v = view(facts)
        point = request.group + "." + request.stage
        rules = CATALOG[point]
        if request.rule_audit is None:
            raise ValueError("V2 observation requires explicit rule projection")
        # Service pairing is derived from the original legal pairs, retained by
        # the rule component in this same callback's local cache.
        physical = facts.component_cache.get(
            "smartsom.v2.service-candidates", request.candidates
        )
        rows, modes, relations = [], [], []
        for candidate in request.candidates:
            rule = request.rule_audit["rules"][candidate.identity]["rule"]
            prefix = "v2_"
            rule = rule[len(prefix) :] if rule.startswith(prefix) else rule
            code = rules.index(rule)
            values = candidate_values(v, request, candidate, physical)
            if len(values) != FEATURES[point]:
                raise ValueError(f"V2 field count mismatch: {point}")
            rows.append((*values, *(float(i == code) for i in range(len(rules)))))
            modes.append(
                tuple(
                    float(candidate.action[1] == m) for m in ("slow", "normal", "fast")
                )
                if request.group == "machine"
                else ()
            )
            if enabled:
                relations.append(
                    freeze(social_relations(v, request, candidate, ordered=ordered))
                )
        return {
            "context": context(v, request),
            "candidates": tuple(rows),
            "modes": tuple(modes),
            "social": tuple(relations),
            "stage": request.stage,
        }

    return observe


create_observation.OUTPUT_SCHEMA = SCHEMA
