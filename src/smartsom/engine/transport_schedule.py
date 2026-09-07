"""Static checks of a transport timetable's interval and position constraints."""

from smartsom.domain import (
    ArrivalPlan,
    ExecutionSchedule,
    FactorySpec,
    JobLocation,
    ScheduledOperation,
    ScheduledTransport,
    TransportDestination,
    WorkloadInstance,
)
from smartsom.engine.replay import ReplayError
from smartsom.modules.arrivals import ArrivalModule
from smartsom.modules.transport import TransportModule


def validate_transports(
    factory: FactorySpec,
    workload: WorkloadInstance,
    execution: ExecutionSchedule,
    operations: tuple[ScheduledOperation, ...],
    arrivals: ArrivalPlan | None,
) -> tuple[ScheduledTransport, ...]:
    module = TransportModule(factory, workload)
    timing = ArrivalModule(workload, arrivals)
    trips = execution.transports
    for trip in trips:
        if not isinstance(trip, ScheduledTransport):
            raise ReplayError("expected ScheduledTransport")
        if not isinstance(trip.source, JobLocation) or not isinstance(
            trip.destination, TransportDestination
        ):
            raise ReplayError(
                "transport requires typed source and destination locations"
            )
        if any(
            not isinstance(value, str)
            for value in (
                trip.agv_id,
                trip.job_id,
                trip.from_node_id,
                trip.pickup_node_id,
                trip.delivery_node_id,
            )
        ):
            raise ReplayError("transport identities must be strings")
        if any(
            type(v) is not int or v < 0
            for v in (
                trip.transport_sequence,
                trip.start_time,
                trip.pickup_time,
                trip.delivery_time,
            )
        ):
            raise ReplayError("transport times/order must be nonnegative integers")
        if (
            trip.agv_id not in {a.agv_id for a in module.spec.agvs}
            or trip.job_id not in module.chains
        ):
            raise ReplayError("unknown transport vehicle or job")
        if (
            trip.from_node_id not in module.spec.nodes
            or trip.pickup_node_id not in module.spec.nodes
            or trip.delivery_node_id not in module.spec.nodes
        ):
            raise ReplayError("unknown transport node")
        if trip.source.kind not in ("input", "prebuffer", "postbuffer"):
            raise ReplayError("invalid transport pickup location")
        if (
            trip.source.kind != "input"
            and trip.source.resource_id not in module.machine_nodes
        ):
            raise ReplayError("unknown pickup machine")
        if (
            trip.destination.kind == "machine"
            and trip.destination.machine_id not in module.machine_nodes
        ):
            raise ReplayError("unknown destination machine")
        if trip.pickup_node_id != module.node(
            trip.source
        ) or trip.delivery_node_id != module.node(
            module.destination_location(trip.destination)
        ):
            raise ReplayError("transport nodes disagree with locations")
        if (
            trip.pickup_time - trip.start_time
            != module.travel_times[(trip.from_node_id, trip.pickup_node_id)]
            or trip.delivery_time - trip.pickup_time
            != module.travel_times[(trip.pickup_node_id, trip.delivery_node_id)]
        ):
            raise ReplayError("transport time disagrees with matrix")
    trips = tuple(sorted(trips, key=lambda t: t.transport_sequence))
    if [t.transport_sequence for t in trips] != list(range(1, len(trips) + 1)):
        raise ReplayError("transport sequence has missing or duplicate occurrences")
    if any(a.start_time > b.start_time for a, b in zip(trips, trips[1:])):
        raise ReplayError("transport order goes backwards in time")
    vehicles = {a.agv_id: (0, a.initial_node_id) for a in module.spec.agvs}
    for trip in trips:
        end, node = vehicles[trip.agv_id]
        if trip.start_time < end or trip.from_node_id != node:
            raise ReplayError("AGV overlap or disconnected position")
        vehicles[trip.agv_id] = trip.delivery_time, trip.delivery_node_id
    entries = {op.operation_id: op for op in operations}
    for job, chain in module.chains.items():
        transfers = [t for t in trips if t.job_id == job]
        index = 0
        available = timing.release_at(chain[0].operation_id)
        location = JobLocation("input")
        for op in chain:
            entry = entries[op.operation_id]
            while (
                index < len(transfers)
                and transfers[index].start_time <= entry.start_time
            ):
                trip = transfers[index]
                if trip.source != location or trip.start_time < available:
                    raise ReplayError(
                        "job pickup precedes availability or breaks its position chain"
                    )
                if trip.destination.kind != "machine" or not any(
                    m.machine_id == trip.destination.machine_id for m in op.modes
                ):
                    raise ReplayError(
                        "delivery is incompatible with the pending operation"
                    )
                destination = module.destination_location(trip.destination)
                if location == destination:
                    raise ReplayError("prebuffer self-reroute is not allowed")
                location, available = destination, trip.delivery_time
                index += 1
            if (
                location != JobLocation("prebuffer", entry.machine_id)
                or entry.start_time < available
            ):
                raise ReplayError(
                    "processing starts before delivery or on another machine"
                )
            location, available = (
                JobLocation("postbuffer", entry.machine_id),
                entry.completion_time,
            )
        remaining = transfers[index:]
        if len(remaining) != 1:
            raise ReplayError("job must have exactly one final output delivery")
        final = remaining[0]
        if (
            final.source != location
            or final.start_time < available
            or final.destination.kind != "output"
        ):
            raise ReplayError("invalid final output transport")
    return trips
