"""Request-local derived public facts shared by V2 rules and observations."""

import math
from functools import wraps


def request_cache(method):
    """Memoize on the request-owned instance without retaining it globally."""

    @wraps(method)
    def wrapped(self, *args, **kwargs):
        key = (method.__name__, args, tuple(sorted(kwargs.items())))
        if key not in self._memo:
            self._memo[key] = method(self, *args, **kwargs)
        return self._memo[key]

    return wrapped


class View:
    def __init__(self, facts):
        self._memo = {}
        self.f = facts
        self.time, self.distance, self.count = facts.scales

    @request_cache
    def job(self, job):
        f = self.f
        row = f.jobs[job]
        due = f.demands[row["demand"]].due_at - f.tick
        work = f.remaining_work(job)
        wait = (
            f.tick - row["since"]
            if row["location"] in f.storage and job not in f.service_jobs
            else 0
        )
        return due, work, wait

    @request_cache
    def capacity(self, owner):
        values = self.f.capacity[owner].values()
        return None if any(v is None for v in values) else sum(values)

    @request_cache
    def occupancy(self, owner):
        cap = self.capacity(owner)
        return len(self.f.inventory(owner)) / cap if cap else 0.0

    @request_cache
    def pre_work(self, machine):
        return sum(
            self.f.nominal(j, machine) for j in self.f.inventory(self.f.pre[machine])
        )

    @request_cache
    def vehicles(self, owner):
        f = self.f
        rows = []
        for vehicle in f.commitments(owner):
            state = f.agvs[vehicle]
            if state["service"]:
                continue
            distance = f.routes.to_owner(state["cell"], owner)
            arrived = distance == 0
            rows.append(
                (
                    vehicle,
                    int(arrived),
                    distance,
                    f.tick - state["arrived_at"] if arrived else 0,
                )
            )
        return tuple(rows)

    @request_cache
    def counts(self, owner):
        rows = self.vehicles(owner)
        arrived = sum(r[1] for r in rows)
        return len(rows) - arrived, arrived

    @request_cache
    def supply(self, owner):
        f = self.f
        jobs = list(f.ready(owner))
        if f.roles[owner] == "machine_post" and f.post_credit_components(owner)[1]:
            job = f.machine_state[f.station[owner]]["job"]
            if job not in jobs and job not in f.service_jobs:
                jobs.append(job)
        return tuple(jobs)

    @request_cache
    def loaded_age(self, vehicle):
        state = self.f.agvs[vehicle]
        return self.f.tick - self.f.jobs[state["job"]]["since"] if state["job"] else 0

    @request_cache
    def density(self, owner=None, vehicle=None):
        f = self.f
        centers = (
            f.routes.ports[owner]
            if owner is not None
            else (tuple(f.agvs[vehicle]["cell"]),)
        )
        roads = set().union(*(f.routes.local_roads(c) for c in centers))
        if vehicle is not None:
            roads.discard(tuple(f.agvs[vehicle]["cell"]))
        return (
            sum(tuple(s["cell"]) in roads for v, s in f.agvs.items() if v != vehicle)
            / len(roads)
            if roads
            else 0.0
        )

    @request_cache
    def next_stations(self, job, after_machine=None):
        f = self.f
        row = f.jobs[job]
        steps = f.demands[row["demand"]].steps
        index = row["step"]
        if after_machine is not None:
            capabilities = f.machines[after_machine].operation_types
            while index < len(steps) and steps[index].operation_type in capabilities:
                index += 1
        if index == len(steps):
            return (), True
        operation = steps[index].operation_type
        return tuple(
            m for m, spec in f.machines.items() if operation in spec.operation_types
        ), False

    @request_cache
    def destinations(self, job, vehicle):
        f = self.f
        machines, terminal = self.next_stations(job)
        owners = (
            tuple(o for o, role in f.roles.items() if role == "system_output")
            if terminal
            else tuple(f.pre[m] for m in machines)
        )
        reachable = tuple(
            (o, f.routes.to_owner(f.agvs[vehicle]["cell"], o)) for o in owners
        )
        reachable = tuple((o, d) for o, d in reachable if math.isfinite(d))
        feasible = tuple(
            (o, d)
            for o, d in reachable
            if f.quota(o) > 0
            and (o not in f.station or not f.machine_state[f.station[o]]["down"])
        )
        return reachable, feasible, terminal

    @request_cache
    def pair(self, job, candidates):
        """Fixed closest downstream vehicle, never a learned vehicle selector."""
        f = self.f
        rows = [
            (c, *self.destinations(job, c.action[1])[:2])
            for c in candidates
            if c.action[0] == job
        ]
        use_feasible = any(feasible for _, _, feasible in rows)
        distances = [
            (
                c,
                min(
                    (d for _, d in (feasible if use_feasible else reachable)),
                    default=math.inf,
                ),
            )
            for c, reachable, feasible in rows
        ]
        c, distance = min(
            distances,
            key=lambda row: (
                row[1],
                f.agvs[row[0].action[1]]["arrived_at"],
                row[0].action[1],
            ),
        )
        if not math.isfinite(distance):
            raise ValueError("job has no compatible reachable downstream destination")
        return c, distance

    @request_cache
    def station(self, machine):
        f = self.f
        pre, post = f.pre[machine], f.post[machine]
        state = f.machine_state[machine]
        capacity = self.capacity(pre) + 1 + self.capacity(post)
        return (
            self.occupancy(pre),
            self.occupancy(post),
            f.quota(pre) / capacity,
            self.pre_work(machine) / self.time,
            f.current_standard_work(machine) / self.time,
            float(state["down"]),
            float(state["status"] == "BLOCKED"),
            len(f.commitments(pre)) / self.count,
            len(f.commitments(post)) / self.count,
        )


def view(facts):
    key = "smartsom.v2.public-derived/v1"
    if key not in facts.component_cache:
        facts.component_cache[key] = View(facts)
    return facts.component_cache[key]
