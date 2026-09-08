"""Read-only resource indexes and transport feasibility; no runtime writer."""

from collections.abc import Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Literal

from smartsom.dispatch import (
    DecisionContext,
    Transfer,
    TransferCandidate,
    Transport,
    TransportCandidate,
)
from smartsom.domain import (
    ActiveTransport,
    AGVState,
    FactorySpec,
    JobLocation,
    JobPosition,
    MachineHolding,
    Operation,
    OperationStatus,
    TransportDestination,
    TransportSpec,
    WorkloadInstance,
)
from smartsom.modules.buffers import BufferModule


@dataclass(frozen=True, slots=True)
class TransportEvent:
    simulation_time: int
    agv_id: str
    job_id: str
    transport_sequence: int
    kind: Literal["pickup", "delivery"]


@dataclass(frozen=True, slots=True, init=False)
class TransportModule:
    spec: TransportSpec | None
    buffers: BufferModule
    detailed: bool
    machine_nodes: Mapping[str, str]
    travel_times: Mapping[tuple[str, str], int]
    chains: Mapping[str, tuple[Operation, ...]]
    operation_jobs: Mapping[str, str]

    def __init__(
        self,
        factory: FactorySpec,
        workload: WorkloadInstance,
        *,
        transport_enabled=True,
        buffers_enabled=False,
        holding_buffer_enabled=False,
    ):
        if transport_enabled and factory.transport is None:
            raise ValueError("enabled transport requires factory transport resources")
        if holding_buffer_enabled and (
            not transport_enabled or factory.holding_buffer is None
        ):
            raise ValueError(
                "enabled holding buffer requires AGV transport and factory holding buffer"
            )
        spec = factory.transport if transport_enabled else None
        buffers = BufferModule(factory, buffers_enabled, holding_buffer_enabled)
        object.__setattr__(self, "buffers", buffers)
        object.__setattr__(
            self, "detailed", spec is None or buffers.finite or holding_buffer_enabled
        )
        chains = {}
        for order in workload.orders:
            for job in order.jobs:
                successors = {op.predecessor_ids: op for op in job.operations}
                chain = []
                op = successors[()]
                while op is not None:
                    chain.append(op)
                    op = successors.get((op.operation_id,))
                chains[job.job_id] = tuple(chain)
        object.__setattr__(self, "spec", spec)
        object.__setattr__(
            self,
            "machine_nodes",
            MappingProxyType(
                {
                    x.machine_id: x.node_id
                    for x in (spec.machine_locations if spec else ())
                }
            ),
        )
        object.__setattr__(
            self,
            "travel_times",
            MappingProxyType(
                {
                    (x.from_node_id, x.to_node_id): x.ticks
                    for x in (spec.travel_times if spec else ())
                }
            ),
        )
        object.__setattr__(self, "chains", MappingProxyType(chains))
        object.__setattr__(
            self,
            "operation_jobs",
            MappingProxyType(
                {op.operation_id: job for job, chain in chains.items() for op in chain}
            ),
        )

    def node(self, location: JobLocation) -> str:
        if location.kind == "input":
            return self.spec.input_node_id
        if location.kind == "output":
            return self.spec.output_node_id
        if location.kind == "holding":
            return self.buffers.holding.node_id
        return self.machine_nodes[location.resource_id]

    def destination_location(self, destination: TransportDestination) -> JobLocation:
        if destination.kind == "holding":
            return JobLocation("holding", destination.buffer_id)
        return (
            JobLocation("output")
            if destination.kind == "output"
            else JobLocation(
                "machine"
                if self.buffers.limits[destination.machine_id].pre_capacity == 0
                else "prebuffer",
                destination.machine_id,
            )
        )

    def project(
        self,
        context: DecisionContext,
        positions: Mapping[str, JobPosition],
        agvs: tuple[AGVState, ...],
        reservations=(),
    ) -> DecisionContext:
        states = {op.operation_id: op for op in context.operations}
        visible_ids = {job.job_id for job in context.jobs}
        visible_positions = tuple(positions[key] for key in sorted(visible_ids))
        dispatches = tuple(
            candidate
            for candidate in context.candidates
            if positions[self.operation_jobs[candidate.action.operation_id]].location
            in (
                JobLocation("prebuffer", candidate.machine_id),
                JobLocation("machine", candidate.machine_id),
            )
            and positions[
                self.operation_jobs[candidate.action.operation_id]
            ].bound_agv_id
            is None
        )
        candidates = []
        transfers = []
        machines = {m.machine_id: m for m in context.machines}
        for position in visible_positions:
            location = position.location
            if position.bound_agv_id is not None or location.kind not in (
                "input",
                "prebuffer",
                "postbuffer",
                "machine",
                "holding",
            ):
                continue
            operation = next(
                (
                    op
                    for op in self.chains[position.job_id]
                    if states[op.operation_id].status != OperationStatus.COMPLETED
                ),
                None,
            )
            if operation is None:
                destinations = (
                    (TransportDestination("output"),)
                    if location.kind in ("postbuffer", "machine")
                    else ()
                )
            elif states[operation.operation_id].status != OperationStatus.PENDING:
                destinations = ()
            else:
                destinations = tuple(
                    TransportDestination("machine", key)
                    for key in sorted({m.machine_id for m in operation.modes})
                    if location != JobLocation("prebuffer", key)
                )
            if (
                self.buffers.holding is not None
                and operation is not None
                and states[operation.operation_id].status == OperationStatus.PENDING
                and (
                    location.kind == "postbuffer"
                    or location.kind == "machine"
                    and states[machines[location.resource_id].operation_id].status
                    == OperationStatus.COMPLETED
                )
            ):
                destinations += (
                    TransportDestination(
                        "holding", buffer_id=self.buffers.holding.buffer_id
                    ),
                )
            rerouting = location.kind == "prebuffer" or (
                location.kind == "machine"
                and states[machines[location.resource_id].operation_id].status
                == OperationStatus.PENDING
            )
            if rerouting:
                destinations = tuple(
                    d for d in destinations if d.machine_id != location.resource_id
                )
            if self.spec is None:
                for destination in destinations:
                    if self.can_receive(
                        destination,
                        context,
                        positions,
                        reservations,
                        vacating_job=position.job_id,
                    ):
                        transfers.append(
                            TransferCandidate(
                                Transfer(position.job_id, destination),
                                location.resource_id if rerouting else None,
                            )
                        )
            for agv in sorted(agvs, key=lambda x: x.agv_id):
                if agv.phase != "idle":
                    continue
                for destination in destinations:
                    pickup_node = self.node(location)
                    candidates.append(
                        TransportCandidate(
                            Transport(agv.agv_id, position.job_id, destination),
                            self.travel_times[(agv.node_id, pickup_node)],
                            self.travel_times[
                                (
                                    pickup_node,
                                    self.node(self.destination_location(destination)),
                                )
                            ],
                            location.resource_id if rerouting else None,
                        )
                    )
        return replace(
            context,
            candidates=dispatches,
            agvs=tuple(
                replace(a, trip=a.trip.completed(a.trip.arrival_time))
                if not self.detailed and isinstance(a.trip, ActiveTransport)
                else a
                for a in sorted(agvs, key=lambda x: x.agv_id)
            ),
            buffers=self.buffers.snapshots(positions, reservations),
            holding_buffer=self.buffers.holding_snapshot(positions, reservations),
            machine_holdings=tuple(
                MachineHolding(
                    m.machine_id,
                    self.operation_jobs[m.operation_id],
                    m.operation_id,
                    {
                        OperationStatus.PENDING: "awaiting_dispatch",
                        OperationStatus.COMPLETED: "blocked",
                    }.get(
                        states[m.operation_id].status,
                        states[m.operation_id].status.value,
                    ),
                )
                for m in context.machines
                if m.operation_id is not None
            )
            if self.buffers.enabled
            else (),
            transfer_candidates=tuple(transfers),
            job_positions=visible_positions,
            transport_candidates=tuple(candidates),
        )

    def can_receive(
        self, destination, context, positions, reservations=(), *, vacating_job=None
    ):
        if destination.kind == "output":
            return True
        if destination.kind == "holding":
            return self.destination_space(destination, positions, reservations)
        key = destination.machine_id
        if self.buffers.limits[key].pre_capacity != 0:
            return self.buffers.space(key, "prebuffer", positions, reservations)
        machine = next(m for m in context.machines if m.machine_id == key)
        vacating = (
            machine.operation_id is not None
            and self.operation_jobs[machine.operation_id] == vacating_job
            and next(
                o for o in context.operations if o.operation_id == machine.operation_id
            ).status
            == OperationStatus.COMPLETED
        )
        return machine.availability == "up" and (
            machine.operation_id is None or vacating
        )

    def waiting_capacity(self, destination):
        if destination.kind == "output":
            return None
        if destination.kind == "holding":
            return self.buffers.holding.capacity
        return self.buffers.limits[destination.machine_id].pre_capacity

    def destination_space(self, destination, positions, reservations=()):
        if destination.kind == "output":
            return True
        return self.buffers.space(
            destination.buffer_id
            if destination.kind == "holding"
            else destination.machine_id,
            "holding" if destination.kind == "holding" else "prebuffer",
            positions,
            reservations,
        )
