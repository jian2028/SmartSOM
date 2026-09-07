"""Read-only resource indexes and transport feasibility; no runtime writer."""

from collections.abc import Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Literal

from smartsom.dispatch import DecisionContext, Transport, TransportCandidate
from smartsom.domain import (
    AGVState,
    FactorySpec,
    JobLocation,
    JobPosition,
    Operation,
    OperationStatus,
    TransportDestination,
    TransportSpec,
    WorkloadInstance,
)


@dataclass(frozen=True, slots=True)
class TransportEvent:
    simulation_time: int
    agv_id: str
    job_id: str
    transport_sequence: int
    kind: Literal["pickup", "delivery"]


@dataclass(frozen=True, slots=True, init=False)
class TransportModule:
    spec: TransportSpec
    machine_nodes: Mapping[str, str]
    travel_times: Mapping[tuple[str, str], int]
    chains: Mapping[str, tuple[Operation, ...]]
    operation_jobs: Mapping[str, str]

    def __init__(self, factory: FactorySpec, workload: WorkloadInstance):
        if factory.transport is None:
            raise ValueError("enabled transport requires factory transport resources")
        spec = factory.transport
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
            MappingProxyType({x.machine_id: x.node_id for x in spec.machine_locations}),
        )
        object.__setattr__(
            self,
            "travel_times",
            MappingProxyType(
                {(x.from_node_id, x.to_node_id): x.ticks for x in spec.travel_times}
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
        return self.machine_nodes[location.resource_id]

    def destination_location(self, destination: TransportDestination) -> JobLocation:
        return (
            JobLocation("output")
            if destination.kind == "output"
            else JobLocation("prebuffer", destination.machine_id)
        )

    def project(
        self,
        context: DecisionContext,
        positions: Mapping[str, JobPosition],
        agvs: tuple[AGVState, ...],
    ) -> DecisionContext:
        states = {op.operation_id: op for op in context.operations}
        visible_ids = {job.job_id for job in context.jobs}
        visible_positions = tuple(positions[key] for key in sorted(visible_ids))
        dispatches = tuple(
            candidate
            for candidate in context.candidates
            if positions[self.operation_jobs[candidate.action.operation_id]].location
            == JobLocation("prebuffer", candidate.machine_id)
            and positions[
                self.operation_jobs[candidate.action.operation_id]
            ].bound_agv_id
            is None
        )
        candidates = []
        for position in visible_positions:
            location = position.location
            if position.bound_agv_id is not None or location.kind not in (
                "input",
                "prebuffer",
                "postbuffer",
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
                    if location.kind == "postbuffer"
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
                            location.resource_id
                            if location.kind == "prebuffer"
                            else None,
                        )
                    )
        return replace(
            context,
            candidates=dispatches,
            agvs=tuple(sorted(agvs, key=lambda x: x.agv_id)),
            job_positions=visible_positions,
            transport_candidates=tuple(candidates),
        )
