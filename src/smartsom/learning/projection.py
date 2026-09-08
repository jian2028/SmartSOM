"""Versioned, reveal-bound numerical projection of public decision information.

This module deliberately receives no simulator, workload or event calendar.
The separate capacity preflight is privileged input validation, not observation.
"""

from dataclasses import dataclass
from math import isfinite

from smartsom.dispatch.contracts import DecisionContext
from smartsom.domain import FactorySpec, WorkloadInstance
from smartsom.domain.actions import (
    SemanticAction,
    WaitNextEvent,
)
from smartsom.domain.destinations import TransportDestination
from smartsom.domain.quality import QualityPlan

VERSION = "smartsom.learning-projection/v1"
STATUSES = ("pending", "processing", "paused", "completed")
LOCATIONS = (
    "unreleased",
    "input",
    "prebuffer",
    "machine",
    "postbuffer",
    "agv",
    "output",
    "holding",
)
HOLDINGS = ("awaiting_dispatch", "processing", "paused", "blocked")
PHASES = ("idle", "empty", "loaded", "waiting")


@dataclass(frozen=True, slots=True)
class ProjectionSpec:
    max_jobs: int
    max_operations_per_job: int
    max_modes_per_operation: int
    time_scale: float = 100.0
    count_scale: float = 100.0
    version: str = VERSION

    def __post_init__(self):
        for name in ("max_jobs", "max_operations_per_job", "max_modes_per_operation"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("time_scale", "count_scale"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.version != VERSION:
            raise ValueError("unsupported learning projection version")


def validate_capacity(
    spec: ProjectionSpec, workload: WorkloadInstance, quality: QualityPlan | None = None
):
    """Reject overflow before simulation; never pass these counts to the learner."""
    jobs = tuple(j for order in workload.orders for j in order.jobs)
    counts = {}
    if quality is not None:
        for mode in quality.modes:
            counts[mode.operation_id] = counts.get(mode.operation_id, 0) + 1
    if (
        len(jobs) > spec.max_jobs
        or any(len(j.operations) > spec.max_operations_per_job for j in jobs)
        or any(
            counts.get(op.operation_id, len(op.modes)) > spec.max_modes_per_operation
            for op in workload.operations
        )
    ):
        raise ValueError("workload exceeds configured learning projection capacity")


def _chain(operations):
    successors = {
        op.predecessor_ids[0] if op.predecessor_ids else None: op for op in operations
    }
    ordered = []
    op = successors.get(None)
    while op is not None:
        ordered.append(op)
        op = successors.get(op.operation_id)
    if len(ordered) != len(operations):
        raise ValueError("invalid visible operation chain")
    return tuple(ordered)


@dataclass(frozen=True, slots=True)
class JobBinding:
    job_id: str
    operation_modes: tuple[tuple[str, tuple[str, ...]], ...]


@dataclass(frozen=True, slots=True)
class ProjectedDecision:
    observations: tuple[float, ...]
    actions: tuple[SemanticAction | None, ...]
    bindings: tuple[JobBinding, ...]

    @property
    def action_mask(self) -> tuple[int, ...]:
        return tuple(int(a is not None) for a in self.actions)

    def decode(self, index: int) -> SemanticAction:
        if (
            type(index) is not int
            or not 0 <= index < len(self.actions)
            or self.actions[index] is None
        ):
            raise ValueError(f"invalid or masked learning action {index!r}")
        return self.actions[index]


def visible_future_event(context: DecisionContext) -> bool:
    """A sufficient public witness, never a query into hidden event times."""
    return (
        any(x.status in ("processing", "paused") for x in context.operations)
        or any(x.availability == "down" for x in context.machines)
        or any(x.phase in ("empty", "loaded") for x in context.agvs)
        or any(x.release_at > context.simulation_time for x in context.jobs)
    )


class LearningProjection:
    def __init__(self, factory: FactorySpec, spec: ProjectionSpec):
        self.spec = spec
        self.machines = tuple(sorted(m.machine_id for m in factory.machines))
        self.agvs = (
            tuple(x.agv_id for x in factory.transport.agvs) if factory.transport else ()
        )
        self.nodes = factory.transport.nodes if factory.transport else ()
        self.destinations = tuple(
            TransportDestination("machine", m) for m in self.machines
        ) + (TransportDestination("output"),)
        if factory.holding_buffer:
            self.destinations += (
                TransportDestination(
                    "holding", buffer_id=factory.holding_buffer.buffer_id
                ),
            )
        self.quality_ids = (
            tuple(
                sorted(
                    {
                        mode.quality_mode_id
                        for m in self.machines
                        for mode in factory.quality_speed.for_machine(m)
                    }
                )
            )
            if factory.quality_speed
            else ()
        )
        self._bindings: list[JobBinding] = []
        self._dispatch_size = (
            spec.max_jobs * spec.max_operations_per_job * spec.max_modes_per_operation
        )
        self._transport_size = len(self.agvs) * spec.max_jobs * len(self.destinations)
        self.action_count = (
            self._dispatch_size
            + self._transport_size
            + spec.max_jobs * len(self.destinations)
            + 1
        )
        # Known factory topology is public and constant, independent of an episode.
        self._factory_features = (
            tuple(x.ticks / spec.time_scale for x in factory.transport.travel_times)
            if factory.transport
            else ()
        )
        if factory.transport:
            self._factory_features += tuple(
                float(loc.node_id == n)
                for m in self.machines
                for loc in factory.transport.machine_locations
                if loc.machine_id == m
                for n in self.nodes
            )
            self._factory_features += tuple(
                float(node == n)
                for node in (
                    factory.transport.input_node_id,
                    factory.transport.output_node_id,
                )
                for n in self.nodes
            )
        self.observation_size = len(
            self.project(DecisionContext(0, (), (), ())).observations
        )

    @property
    def bindings(self) -> tuple[JobBinding, ...]:
        return tuple(self._bindings)

    def project(self, context: DecisionContext) -> ProjectedDecision:
        s = self.spec
        jobs = {j.job_id: j for j in context.jobs}
        known = {b.job_id for b in self._bindings}
        additions = []
        for j in sorted(context.jobs, key=lambda j: (j.reveal_at, j.job_id)):
            if j.job_id not in known:
                chain = _chain(j.operations)
                if len(chain) > s.max_operations_per_job or any(
                    len(op.modes) > s.max_modes_per_operation for op in chain
                ):
                    raise ValueError("visible job exceeds projection capacity")
                additions.append(
                    JobBinding(
                        j.job_id,
                        tuple(
                            (
                                op.operation_id,
                                tuple(sorted(m.processing_mode_id for m in op.modes)),
                            )
                            for op in chain
                        ),
                    )
                )
        if len(self._bindings) + len(additions) > s.max_jobs:
            raise ValueError("visible jobs exceed projection capacity")
        self._bindings.extend(additions)
        job_ids = tuple(b.job_id for b in self._bindings)
        op_slots = {
            op: (j, o)
            for j, b in enumerate(self._bindings)
            for o, (op, _) in enumerate(b.operation_modes)
        }
        mode_slots = {
            (op, mode): (j * s.max_operations_per_job + o) * s.max_modes_per_operation
            + k
            for j, b in enumerate(self._bindings)
            for o, (op, modes) in enumerate(b.operation_modes)
            for k, mode in enumerate(modes)
        }
        actions = [None] * self.action_count
        for c in context.candidates:
            actions[
                mode_slots[(c.action.operation_id, c.action.processing_mode_id)]
            ] = c.action
        for c in context.transport_candidates:
            a = c.action
            index = (
                self._dispatch_size
                + (self.agvs.index(a.agv_id) * s.max_jobs + job_ids.index(a.job_id))
                * len(self.destinations)
                + self.destinations.index(a.destination)
            )
            actions[index] = a
        for c in context.transfer_candidates:
            a = c.action
            index = (
                self._dispatch_size
                + self._transport_size
                + job_ids.index(a.job_id) * len(self.destinations)
                + self.destinations.index(a.destination)
            )
            actions[index] = a
        if visible_future_event(context):
            actions[-1] = WaitNextEvent()

        def one(value, keys):
            return [float(value == k) for k in keys]

        def job_ref(value):
            return [
                float(i < len(job_ids) and job_ids[i] == value)
                for i in range(s.max_jobs)
            ]

        def op_ref(value):
            slot = op_slots.get(value)
            return [
                float(slot == (j, o))
                for j in range(s.max_jobs)
                for o in range(s.max_operations_per_job)
            ]

        def tick(value):
            return [
                float(value is not None),
                0.0 if value is None else value / s.time_scale,
            ]

        def capacity(value):
            return [
                float(value is None),
                0.0 if value is None else value / (s.count_scale + value),
            ]

        def location(loc):
            kind, resource = (loc.kind, loc.resource_id) if loc else (None, None)
            return (
                one(kind, LOCATIONS)
                + one(resource, self.machines)
                + one(resource, self.agvs)
            )

        data = [context.simulation_time / s.time_scale, *self._factory_features]
        states = {x.operation_id: x for x in context.operations}
        positions = {x.job_id: x for x in context.job_positions}
        quality = {x.job_id: x for x in context.job_quality}
        q_modes = {
            (x.operation_id, x.processing_mode_id): x for x in context.quality_modes
        }
        for j in range(s.max_jobs):
            binding = self._bindings[j] if j < len(self._bindings) else None
            job = jobs.get(binding.job_id) if binding else None
            pos = positions.get(job.job_id) if job else None
            q = quality.get(job.job_id) if job else None
            row = [float(job is not None)]
            row += tick(job.release_at if job else None) + tick(
                job.reveal_at if job else None
            )
            row += location(pos.location if pos else None) + one(
                pos.bound_agv_id if pos else None, self.agvs
            )
            row += [
                float(q is not None and q.passed is not None),
                float(q is not None and q.passed is True),
            ] + tick(q.inspection_time if q else None)
            data += row if job else [0.0] * len(row)
            operations = {op.operation_id: op for op in job.operations} if job else {}
            for o in range(s.max_operations_per_job):
                pair = (
                    binding.operation_modes[o]
                    if job and o < len(binding.operation_modes)
                    else None
                )
                op = operations[pair[0]] if pair else None
                state = states.get(op.operation_id) if op else None
                row = [float(op is not None)] + one(
                    state.status if state else None, STATUSES
                )
                row += (
                    tick(state.start_time if state else None)
                    + tick(state.completion_time if state else None)
                    + tick(state.actual_processing_ticks if state else None)
                )
                data += row if op else [0.0] * len(row)
                base_ids = (
                    sorted(
                        {
                            q_modes[
                                (op.operation_id, m.processing_mode_id)
                            ].base_processing_mode_id
                            if (op.operation_id, m.processing_mode_id) in q_modes
                            else m.processing_mode_id
                            for m in op.modes
                        }
                    )
                    if op
                    else []
                )
                for k in range(s.max_modes_per_operation):
                    mode = op.mode(pair[1][k]) if pair and k < len(pair[1]) else None
                    qm = (
                        q_modes.get((op.operation_id, mode.processing_mode_id))
                        if mode
                        else None
                    )
                    base = (
                        qm.base_processing_mode_id
                        if qm
                        else mode.processing_mode_id
                        if mode
                        else None
                    )
                    row = [
                        float(mode is not None),
                        float(
                            state is not None
                            and mode is not None
                            and state.processing_mode_id == mode.processing_mode_id
                        ),
                    ]
                    row += one(mode.machine_id if mode else None, self.machines)
                    row += [
                        mode.nominal_ticks / s.time_scale if mode else 0.0,
                        float(qm.time_scale) if qm else 1.0,
                        float(qm is not None and qm.error_rate is not None),
                        float(qm.error_rate)
                        if qm and qm.error_rate is not None
                        else 0.0,
                    ]
                    row += one(qm.quality_mode_id if qm else None, self.quality_ids)
                    row += [
                        float(i < len(base_ids) and base_ids[i] == base)
                        for i in range(s.max_modes_per_operation)
                    ]
                    data += row if mode else [0.0] * len(row)
        machines = {m.machine_id: m for m in context.machines}
        holdings = {m.machine_id: m for m in context.machine_holdings}
        buffers = {b.machine_id: b for b in context.buffers}
        for machine_id in self.machines:
            m, h, b = (
                machines.get(machine_id),
                holdings.get(machine_id),
                buffers.get(machine_id),
            )
            data += [
                float(m is not None),
                float(m is not None and m.availability == "up"),
            ]
            data += (
                op_ref(m.operation_id if m else None)
                + job_ref(h.job_id if h else None)
                + one(h.phase if h else None, HOLDINGS)
            )
            data += (
                [float(b is not None)]
                + capacity(b.pre_capacity if b else None)
                + capacity(b.post_capacity if b else None)
            )
            for members in (b.pre_jobs if b else (), b.post_jobs if b else ()):
                data += [
                    float(i < len(job_ids) and job_ids[i] in members)
                    for i in range(s.max_jobs)
                ]
            for agv_id in self.agvs:
                data += job_ref(
                    next((r.job_id for r in b.reservations if r.agv_id == agv_id), None)
                    if b
                    else None
                )
        h = context.holding_buffer
        data += [float(h is not None)] + capacity(h.capacity if h else None)
        data += [
            float(i < len(job_ids) and h is not None and job_ids[i] in h.jobs)
            for i in range(s.max_jobs)
        ]
        for agv_id in self.agvs:
            data += job_ref(
                next((r.job_id for r in h.reservations if r.agv_id == agv_id), None)
                if h
                else None
            )
        vehicles = {a.agv_id: a for a in context.agvs}
        for agv_id in self.agvs:
            a = vehicles.get(agv_id)
            trip = a.trip if a else None
            data += (
                [float(a is not None)]
                + one(a.phase if a else None, PHASES)
                + one(a.node_id if a else None, self.nodes)
            )
            data += job_ref(trip.job_id if trip else None) + one(
                trip.destination if trip else None, self.destinations
            )
            data += location(trip.source if trip else None)
            for field in ("from_node_id", "pickup_node_id", "delivery_node_id"):
                data += one(getattr(trip, field, None), self.nodes)
            for field in ("start_time", "pickup_time"):
                data += tick(getattr(trip, field, None))
            data += tick(
                getattr(trip, "arrival_time", getattr(trip, "delivery_time", None))
            )
            data += [trip.transport_sequence / s.count_scale if trip else 0.0]
        trips = {c.action: c for c in context.transport_candidates}
        # Duration features align with the fixed action blocks, never candidate order.
        for a in actions[
            self._dispatch_size : self._dispatch_size + self._transport_size
        ]:
            c = trips.get(a)
            data += (
                [c.empty_ticks / s.time_scale, c.loaded_ticks / s.time_scale]
                if c
                else [0.0, 0.0]
            )
        if not all(isfinite(x) for x in data):
            raise ValueError("nonfinite learning observation")
        return ProjectedDecision(tuple(data), tuple(actions), self.bindings)
