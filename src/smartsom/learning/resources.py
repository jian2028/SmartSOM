"""IDETC-style resource views reconstructed exclusively from public information."""

from dataclasses import dataclass
from math import isfinite

from smartsom.dispatch import DecisionContext, Dispatch, Transfer, Transport
from smartsom.domain import FactorySpec
from smartsom.domain.destinations import TransportDestination
from smartsom.learning.projection import (
    HOLDINGS,
    LOCATIONS,
    PHASES,
    JobBinding,
    ProjectionSpec,
    _chain,
)

RESOURCE_VERSION = "smartsom.resource-projection/v1"
type PhysicalAction = Dispatch | Transport | Transfer


@dataclass(frozen=True, slots=True)
class ResourceProjectionSpec:
    max_jobs: int
    max_operations_per_job: int
    max_modes_per_operation: int
    time_scale: float = 100.0
    count_scale: float = 100.0
    version: str = RESOURCE_VERSION

    def __post_init__(self):
        ProjectionSpec(
            self.max_jobs,
            self.max_operations_per_job,
            self.max_modes_per_operation,
            self.time_scale,
            self.count_scale,
        )
        if self.version != RESOURCE_VERSION:
            raise ValueError("unsupported resource projection version")


def action_key(action: PhysicalAction) -> tuple:
    if isinstance(action, Dispatch):
        return (0, action.operation_id, action.processing_mode_id)
    d = action.destination
    return (
        1,
        "transport" if isinstance(action, Transport) else "transfer",
        action.job_id,
        d.kind,
        d.machine_id or "",
        d.buffer_id or "",
    )


@dataclass(frozen=True, slots=True)
class ResourceCandidate:
    action: PhysicalAction
    job_id: str


@dataclass(frozen=True, slots=True)
class ResourceView:
    agent_id: str
    role: str
    global_features: tuple[float, ...]
    local_features: tuple[float, ...]
    candidate_features: tuple[tuple[float, ...], ...]
    candidates: tuple[ResourceCandidate, ...]

    @property
    def observations(self) -> tuple[float, ...]:
        return (
            self.global_features
            + self.local_features
            + tuple(x for row in self.candidate_features for x in row)
        )

    @property
    def action_mask(self) -> tuple[int, ...]:
        return (
            (1,)
            + (1,) * len(self.candidates)
            + (0,) * (len(self.candidate_features) - len(self.candidates))
        )

    def decode(self, index: int) -> ResourceCandidate | None:
        if type(index) is not int or not 0 <= index <= len(self.candidates):
            raise ValueError(
                f"invalid or masked resource action {self.agent_id}: {index!r}"
            )
        return self.candidates[index - 1] if index else None


@dataclass(frozen=True, slots=True)
class ResourceDecision:
    context: DecisionContext
    views: tuple[ResourceView, ...]
    bindings: tuple[JobBinding, ...]

    def for_agent(self, agent_id: str) -> ResourceView:
        return next(v for v in self.views if v.agent_id == agent_id)


class ResourceProjection:
    """Owns reveal-bound slots, never a workload, simulator or hidden event queue."""

    def __init__(
        self,
        factory: FactorySpec,
        spec: ResourceProjectionSpec,
        *,
        transport_enabled=False,
    ):
        if not isinstance(spec, ResourceProjectionSpec):
            raise TypeError("resource projection requires ResourceProjectionSpec")
        self.spec = spec
        self.machines = tuple(sorted(m.machine_id for m in factory.machines))
        self.agvs = (
            tuple(sorted(a.agv_id for a in factory.transport.agvs))
            if factory.transport
            else ()
        )
        self.nodes = tuple(sorted(factory.transport.nodes)) if factory.transport else ()
        self.destinations = tuple(
            TransportDestination("machine", m) for m in self.machines
        ) + (TransportDestination("output"),)
        if factory.holding_buffer:
            self.destinations += (
                TransportDestination(
                    "holding", buffer_id=factory.holding_buffer.buffer_id
                ),
            )
        self.quality_ids = tuple(
            sorted(
                {
                    q.quality_mode_id
                    for m in self.machines
                    for q in (
                        factory.quality_speed.for_machine(m)
                        if factory.quality_speed
                        else ()
                    )
                }
            )
        )
        self.agents = tuple(f"machine:{m}" for m in self.machines) + (
            tuple(f"agv:{a}" for a in self.agvs) if transport_enabled else ()
        )
        self.bindings: tuple[JobBinding, ...] = ()
        self.capacities = {
            "machine_policy": spec.max_jobs * (spec.max_modes_per_operation + 1),
            "agv_policy": spec.max_jobs * len(self.destinations),
        }
        empty = self.project(DecisionContext(0, (), (), ()))
        self.observation_sizes = {v.role: len(v.observations) for v in empty.views}

    def project(self, context: DecisionContext) -> ResourceDecision:
        s = self.spec
        known = {b.job_id for b in self.bindings}
        additions = []
        for job in sorted(context.jobs, key=lambda j: (j.reveal_at, j.job_id)):
            if job.job_id in known:
                continue
            chain = _chain(job.operations)
            if len(chain) > s.max_operations_per_job or any(
                len(o.modes) > s.max_modes_per_operation for o in chain
            ):
                raise ValueError("visible job exceeds resource projection capacity")
            additions.append(
                JobBinding(
                    job.job_id,
                    tuple(
                        (
                            o.operation_id,
                            tuple(sorted(m.processing_mode_id for m in o.modes)),
                        )
                        for o in chain
                    ),
                )
            )
        if len(self.bindings) + len(additions) > s.max_jobs:
            raise ValueError("visible jobs exceed resource projection capacity")
        bindings = self.bindings + tuple(additions)
        enc = _Encoder(self, context, bindings)
        by_agent = {a: [] for a in self.agents}
        for candidate in context.candidates:
            a = candidate.action
            by_agent[f"machine:{candidate.machine_id}"].append(
                ResourceCandidate(a, enc.op_jobs[a.operation_id])
            )
        for candidate in context.transport_candidates:
            a = candidate.action
            by_agent[f"agv:{a.agv_id}"].append(ResourceCandidate(a, a.job_id))
        for candidate in context.transfer_candidates:
            a = candidate.action
            owner = a.destination.machine_id
            if a.destination.kind == "output":
                owner = enc.positions[a.job_id].location.resource_id
            if owner not in self.machines:
                raise ValueError("Transfer has no unique machine owner")
            by_agent[f"machine:{owner}"].append(ResourceCandidate(a, a.job_id))
        shared = tuple(enc.shared())
        views = []
        for agent in self.agents:
            kind, resource = agent.split(":", 1)
            role = "machine_policy" if kind == "machine" else "agv_policy"
            candidates = tuple(
                sorted(by_agent[agent], key=lambda c: action_key(c.action))
            )
            capacity = self.capacities[role]
            if len(candidates) > capacity:
                raise ValueError("resource candidate capacity overflow; no truncation")
            empty = tuple(0.0 for _ in enc.candidate(None))
            rows = tuple(tuple(enc.candidate(c)) for c in candidates) + (empty,) * (
                capacity - len(candidates)
            )
            local = (
                enc.machine(resource) if kind == "machine" else enc.vehicle(resource)
            )
            view = ResourceView(agent, role, shared, tuple(local), rows, candidates)
            if not all(isfinite(x) for x in view.observations):
                raise ValueError("nonfinite resource observation")
            views.append(view)
        self.bindings = bindings
        return ResourceDecision(context, tuple(views), bindings)


class _Encoder:
    """Numerical fields with explicit presence flags and configured scales."""

    def __init__(self, projection, context, bindings):
        self.p, self.s, self.c, self.bindings = (
            projection,
            projection.spec,
            context,
            bindings,
        )
        self.jobs = {j.job_id: j for j in context.jobs}
        self.states = {o.operation_id: o for o in context.operations}
        self.ops = {o.operation_id: o for j in context.jobs for o in j.operations}
        self.op_jobs = {
            o.operation_id: j.job_id for j in context.jobs for o in j.operations
        }
        self.positions = {p.job_id: p for p in context.job_positions}
        self.machines = {m.machine_id: m for m in context.machines}
        self.buffers = {b.machine_id: b for b in context.buffers}
        self.holdings = {h.machine_id: h for h in context.machine_holdings}
        self.vehicles = {a.agv_id: a for a in context.agvs}
        self.quality = {
            (q.operation_id, q.processing_mode_id): q for q in context.quality_modes
        }
        self.inspections = {q.job_id: q for q in context.job_quality}
        self.trips = {t.action: t for t in context.transport_candidates}

    @staticmethod
    def one(value, keys):
        return [float(value == k) for k in keys]

    def tick(self, value):
        return [
            float(value is not None),
            value / self.s.time_scale if value is not None else 0.0,
        ]

    def job_ref(self, job):
        return [
            float(i < len(self.bindings) and self.bindings[i].job_id == job)
            for i in range(self.s.max_jobs)
        ]

    def location(self, location):
        kind, resource = (
            (location.kind, location.resource_id) if location else (None, None)
        )
        return (
            self.one(kind, LOCATIONS)
            + self.one(resource, self.p.machines)
            + self.one(resource, self.p.agvs)
        )

    def capacity(self, value):
        return [
            float(value is None),
            value / (self.s.count_scale + value) if value is not None else 0.0,
        ]

    def buffer(self, machine_id):
        b = self.buffers.get(machine_id)
        row = (
            [float(b is not None)]
            + self.capacity(b.pre_capacity if b else None)
            + self.capacity(b.post_capacity if b else None)
        )
        for members in (b.pre_jobs if b else (), b.post_jobs if b else ()):
            row += [len(members) / self.s.count_scale]
            row += [
                float(i < len(self.bindings) and self.bindings[i].job_id in members)
                for i in range(self.s.max_jobs)
            ]
            row += self.job_ref(members[0] if members else None)
        for agv in self.p.agvs:
            row += self.job_ref(
                next((r.job_id for r in b.reservations if r.agv_id == agv), None)
                if b
                else None
            )
        return row if b else [0.0] * len(row)

    def shared(self):
        row = [self.c.simulation_time / self.s.time_scale]
        visits = {m: 0 for m in self.p.machines}
        for i in range(self.s.max_jobs):
            binding = self.bindings[i] if i < len(self.bindings) else None
            job = self.jobs.get(binding.job_id) if binding else None
            states = (
                [self.states[o.operation_id] for o in job.operations] if job else []
            )
            done = [x for x in states if x.status == "completed"]
            busy = any(x.status in ("processing", "paused") for x in states)
            released = job is not None and job.release_at <= self.c.simulation_time
            used = {
                self.ops[x.operation_id].mode(x.processing_mode_id).machine_id
                for x in done
            }
            for m in used:
                visits[m] += 1
            fields = [
                float(job is not None),
                float(released),
                float(busy),
                len(done) / self.s.count_scale,
                float(released and not busy and len(done) < len(states)),
            ]
            fields += [float(m in used) for m in self.p.machines]
            fields += self.tick(job.release_at if job else None)
            pos = self.positions.get(job.job_id) if job else None
            fields += self.location(pos.location if pos else None)
            fields += self.one(pos.bound_agv_id if pos else None, self.p.agvs)
            q = self.inspections.get(job.job_id) if job else None
            fields += [
                float(q is not None and q.passed is not None),
                float(q is not None and q.passed is True),
            ]
            fields += self.tick(q.inspection_time if q else None)
            row += fields if job else [0.0] * len(fields)
        starving = 0
        for mid in self.p.machines:
            m, b, h = (
                self.machines.get(mid),
                self.buffers.get(mid),
                self.holdings.get(mid),
            )
            row += [
                float(m is not None),
                float(m is not None and m.operation_id is not None),
                float(m is not None and m.availability == "down"),
                visits[mid] / self.s.count_scale,
            ]
            starving += int(
                m is not None
                and m.availability == "up"
                and m.operation_id is None
                and h is None
                and (b is None or not b.pre_jobs)
            )
        row += [
            sum(p.location.kind == kind for p in self.positions.values())
            / self.s.count_scale
            for kind in ("input", "output")
        ]
        row += [starving / len(self.p.machines)]
        h = self.c.holding_buffer
        row += [float(h is not None)] + self.capacity(h.capacity if h else None)
        row += [
            float(
                i < len(self.bindings)
                and h is not None
                and self.bindings[i].job_id in h.jobs
            )
            for i in range(self.s.max_jobs)
        ]
        for agv in self.p.agvs:
            row += self.job_ref(
                next((r.job_id for r in h.reservations if r.agv_id == agv), None)
                if h
                else None
            )
        return row

    def machine(self, mid):
        m, h = self.machines.get(mid), self.holdings.get(mid)
        opid = m.operation_id if m else None
        state = self.states.get(opid)
        mode = (
            self.ops[opid].mode(state.processing_mode_id)
            if state and state.processing_mode_id
            else None
        )
        return (
            self.one(mid, self.p.machines)
            + [float(m is not None), float(m is not None and m.availability == "up")]
            + self.one(h.phase if h else state.status if state else None, HOLDINGS)
            + self.job_ref(h.job_id if h else self.op_jobs.get(opid))
            + self.tick(mode.nominal_ticks if mode else None)
            + self.tick(
                self.c.simulation_time - state.start_time
                if state and state.start_time is not None
                else None
            )
            + self.buffer(mid)
        )

    def vehicle(self, aid):
        a = self.vehicles.get(aid)
        t = a.trip if a else None
        arrival = getattr(t, "arrival_time", getattr(t, "delivery_time", None))
        return (
            self.one(aid, self.p.agvs)
            + [float(a is not None)]
            + self.one(a.phase if a else None, PHASES)
            + self.one(a.node_id if a else None, self.p.nodes)
            + self.job_ref(t.job_id if t else None)
            + self.one(t.destination if t else None, self.p.destinations)
            + self.location(t.source if t else None)
            + self.tick(
                max(0, t.pickup_time - self.c.simulation_time)
                if t and a.phase == "empty"
                else None
            )
            + self.tick(
                max(0, arrival - self.c.simulation_time)
                if t and a.phase in ("empty", "loaded")
                else None
            )
        )

    def candidate(self, candidate):
        a = candidate.action if candidate else None
        jid = candidate.job_id if candidate else None
        binding = next((b for b in self.bindings if b.job_id == jid), None)
        current = (
            next(
                (
                    (oid, modes)
                    for oid, modes in binding.operation_modes
                    if self.states[oid].status != "completed"
                ),
                None,
            )
            if binding
            else None
        )
        opid = (
            a.operation_id
            if isinstance(a, Dispatch)
            else current[0]
            if current
            else None
        )
        op = self.ops.get(opid)
        mode = op.mode(a.processing_mode_id) if isinstance(a, Dispatch) else None
        qm = self.quality.get((opid, mode.processing_mode_id)) if mode else None
        mode_ids = current[1] if current else ()
        d = (
            a.destination
            if isinstance(a, (Transport, Transfer))
            else TransportDestination("machine", mode.machine_id)
            if mode
            else None
        )
        row = (
            [float(a is not None)]
            + self.one(type(a), (Dispatch, Transport, Transfer))
            + self.job_ref(jid)
        )
        row += [
            float(
                binding is not None
                and i < len(binding.operation_modes)
                and binding.operation_modes[i][0] == opid
            )
            for i in range(self.s.max_operations_per_job)
        ]
        row += [
            float(
                mode is not None
                and i < len(mode_ids)
                and mode_ids[i] == mode.processing_mode_id
            )
            for i in range(self.s.max_modes_per_operation)
        ]
        row += self.one(d, self.p.destinations)
        remaining = (
            sum(
                self.states[oid].status != "completed"
                for oid, _ in binding.operation_modes
            )
            if binding
            else 0
        )
        row += [remaining / self.s.count_scale] + self.tick(
            mode.nominal_ticks if mode else None
        )
        durations = (
            [m.nominal_ticks for m in op.modes if d and m.machine_id == d.machine_id]
            if op
            else []
        )
        # AGV routing does not choose a processing mode. Expose the destination's
        # nominal range, with missing flags for output/holding destinations.
        row += self.tick(min(durations) if durations else None)
        row += self.tick(max(durations) if durations else None)
        row += [
            float(qm is not None),
            float(qm.time_scale) if qm else 0.0,
            float(qm is not None and qm.error_rate is not None),
            float(qm.error_rate) if qm and qm.error_rate is not None else 0.0,
        ]
        row += self.one(qm.quality_mode_id if qm else None, self.p.quality_ids)
        base_ids = (
            sorted(
                {
                    self.quality[(opid, m.processing_mode_id)].base_processing_mode_id
                    if (opid, m.processing_mode_id) in self.quality
                    else m.processing_mode_id
                    for m in op.modes
                }
            )
            if op
            else []
        )
        base = (
            qm.base_processing_mode_id
            if qm
            else mode.processing_mode_id
            if mode
            else None
        )
        row += [
            float(i < len(base_ids) and base_ids[i] == base)
            for i in range(self.s.max_modes_per_operation)
        ]
        trip = self.trips.get(a)
        row += self.tick(trip.empty_ticks if trip else None) + self.tick(
            trip.loaded_ticks if trip else None
        )
        pos = self.positions.get(jid)
        row += self.location(pos.location if pos else None)
        target = d.machine_id if d else None
        row += self.machine(target)
        return row
