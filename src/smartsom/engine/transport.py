"""Engine-owned logistics state and transitions, sharing the simulator calendar."""

from dataclasses import replace

from smartsom.dispatch import TransportCandidate
from smartsom.domain import (
    AGVState,
    JobLocation,
    JobPosition,
    OperationStatus,
    ScheduledTransport,
)
from smartsom.engine.calendar import Event, EventCalendar
from smartsom.engine.invariants import _require
from smartsom.engine.state import RuntimeState
from smartsom.modules.arrivals import ArrivalEvent, ArrivalModule
from smartsom.modules.transport import TransportEvent, TransportModule
from smartsom.trace import TraceRecord, TransportRecord


class TransportExecution:
    def __init__(self, module: TransportModule, arrivals: ArrivalModule) -> None:
        self.module = module
        self.positions = {
            job: JobPosition(
                job,
                JobLocation(
                    "input"
                    if arrivals.release_at(chain[0].operation_id) == 0
                    else "unreleased"
                ),
            )
            for job, chain in module.chains.items()
        }
        self.agvs = {
            agv.agv_id: AGVState(agv.agv_id, agv.initial_node_id)
            for agv in module.spec.agvs
        }
        self.schedule: list[ScheduledTransport] = []
        self.started = 0

    @property
    def finished(self) -> bool:
        return all(p.location.kind == "output" for p in self.positions.values())

    @property
    def makespan(self) -> int:
        return max(
            x.delivery_time for x in self.schedule if x.destination.kind == "output"
        )

    def release(self, job_id: str) -> None:
        self.positions[job_id] = JobPosition(job_id, JobLocation("input"))

    def processing(
        self, operation_id: str, machine_id: str, *, complete: bool = False
    ) -> None:
        job = self.module.operation_jobs[operation_id]
        self.positions[job] = JobPosition(
            job, JobLocation("postbuffer" if complete else "machine", machine_id)
        )

    def dispatch(
        self,
        candidate: TransportCandidate,
        tick: int,
        calendar: EventCalendar,
        trace: list[TraceRecord],
    ) -> None:
        action = candidate.action
        position = self.positions[action.job_id]
        agv = self.agvs[action.agv_id]
        _require(agv.node_id is not None, "dispatch from busy AGV")
        self.started += 1
        trip = ScheduledTransport(
            self.started,
            action.agv_id,
            action.job_id,
            action.destination,
            position.location,
            agv.node_id,
            self.module.node(position.location),
            self.module.node(self.module.destination_location(action.destination)),
            tick,
            tick + candidate.empty_ticks,
            tick + candidate.empty_ticks + candidate.loaded_ticks,
        )
        self.positions[action.job_id] = replace(position, bound_agv_id=action.agv_id)
        self.agvs[action.agv_id] = AGVState(action.agv_id, None, "empty", trip)
        calendar.schedule(
            TransportEvent(
                trip.pickup_time,
                trip.agv_id,
                trip.job_id,
                trip.transport_sequence,
                "pickup",
            )
        )
        trace.append(TransportRecord(len(trace), tick, trip, "empty_start"))

    def handle(
        self, event: TransportEvent, calendar: EventCalendar, trace: list[TraceRecord]
    ) -> None:
        agv = self.agvs[event.agv_id]
        trip = agv.trip
        _require(
            trip is not None
            and trip.transport_sequence == event.transport_sequence
            and trip.job_id == event.job_id,
            "stale transport event",
        )
        if event.kind == "pickup":
            self.positions[trip.job_id] = JobPosition(
                trip.job_id, JobLocation("agv", trip.agv_id), trip.agv_id
            )
            self.agvs[trip.agv_id] = replace(agv, phase="loaded")
            trace.append(
                TransportRecord(len(trace), event.simulation_time, trip, "pickup")
            )
            trace.append(
                TransportRecord(len(trace), event.simulation_time, trip, "loaded_start")
            )
            calendar.schedule(
                TransportEvent(
                    trip.delivery_time,
                    trip.agv_id,
                    trip.job_id,
                    trip.transport_sequence,
                    "delivery",
                )
            )
        else:
            self.positions[trip.job_id] = JobPosition(
                trip.job_id, self.module.destination_location(trip.destination)
            )
            self.agvs[trip.agv_id] = AGVState(trip.agv_id, trip.delivery_node_id)
            self.schedule.append(trip)
            trace.append(
                TransportRecord(len(trace), event.simulation_time, trip, "delivery")
            )

    def check(
        self,
        state: RuntimeState,
        pending_events: tuple[Event, ...],
        handled_arrivals: set[ArrivalEvent],
        arrivals: ArrivalModule,
    ) -> None:
        """Cross-check ownership, pending phase and every recorded vehicle interval."""
        _require(
            set(self.positions) == set(self.module.chains),
            "job position identity changed",
        )
        _require(
            set(self.agvs) == {a.agv_id for a in self.module.spec.agvs},
            "AGV identity changed",
        )
        expected = []
        active_trips = []
        for key, agv in self.agvs.items():
            _require(key == agv.agv_id, "AGV state identity changed")
            if agv.phase == "idle":
                _require(
                    agv.trip is None and agv.node_id in self.module.spec.nodes,
                    "invalid idle AGV",
                )
                continue
            trip = agv.trip
            _require(
                agv.phase in ("empty", "loaded")
                and trip is not None
                and trip.agv_id == key
                and agv.node_id is None,
                "invalid busy AGV",
            )
            left, right = (
                (trip.start_time, trip.pickup_time)
                if agv.phase == "empty"
                else (trip.pickup_time, trip.delivery_time)
            )
            _require(
                left <= state.simulation_time <= right, "AGV phase disagrees with clock"
            )
            active_trips.append(trip)
            expected.append(
                TransportEvent(
                    trip.pickup_time if agv.phase == "empty" else trip.delivery_time,
                    key,
                    trip.job_id,
                    trip.transport_sequence,
                    "pickup" if agv.phase == "empty" else "delivery",
                )
            )
            position = self.positions[trip.job_id]
            _require(position.bound_agv_id == key, "trip lost its bound job")
            _require(
                position.location
                == (trip.source if agv.phase == "empty" else JobLocation("agv", key)),
                "job disagrees with AGV phase",
            )
        actual = tuple(e for e in pending_events if isinstance(e, TransportEvent))
        _require(
            len(actual) == len(expected) and set(actual) == set(expected),
            "transport calendar disagrees with busy AGVs",
        )
        _require(
            all(e.simulation_time >= state.simulation_time for e in actual),
            "transport event in the past",
        )
        trips = sorted(
            (*self.schedule, *active_trips), key=lambda t: t.transport_sequence
        )
        _require(
            [t.transport_sequence for t in trips] == list(range(1, self.started + 1)),
            "missing or duplicate transport occurrence",
        )
        previous = {a.agv_id: (0, a.initial_node_id) for a in self.module.spec.agvs}
        for trip in trips:
            end, node = previous[trip.agv_id]
            _require(
                trip.start_time >= end and trip.from_node_id == node,
                "AGV overlap or disconnected position",
            )
            _require(
                trip.pickup_node_id == self.module.node(trip.source)
                and trip.delivery_node_id
                == self.module.node(self.module.destination_location(trip.destination)),
                "transport nodes disagree with job locations",
            )
            _require(
                trip.pickup_time - trip.start_time
                == self.module.travel_times[(node, trip.pickup_node_id)]
                and trip.delivery_time - trip.pickup_time
                == self.module.travel_times[
                    (trip.pickup_node_id, trip.delivery_node_id)
                ],
                "transport duration disagrees with matrix",
            )
            previous[trip.agv_id] = (trip.delivery_time, trip.delivery_node_id)
        for key, agv in self.agvs.items():
            if agv.phase == "idle":
                _require(
                    agv.node_id == previous[key][1],
                    "idle AGV left its last delivery node",
                )
        _require(
            all(t.delivery_time <= state.simulation_time for t in self.schedule),
            "recorded future delivery",
        )
        delivered = [t.job_id for t in self.schedule if t.destination.kind == "output"]
        _require(
            len(delivered) == len(set(delivered))
            and set(delivered)
            == {k for k, p in self.positions.items() if p.location.kind == "output"},
            "output jobs disagree with deliveries",
        )
        for job, position in self.positions.items():
            _require(job == position.job_id, "job position identity changed")
            chain = self.module.chains[job]
            current = next(
                (
                    state.operations[op.operation_id]
                    for op in chain
                    if state.operations[op.operation_id].status
                    != OperationStatus.COMPLETED
                ),
                None,
            )
            release = arrivals.release_at(chain[0].operation_id)
            released = release == 0 or any(
                e.job_id == job and e.kind == "release" for e in handled_arrivals
            )
            _require(
                (position.location.kind == "unreleased") == (not released),
                "job location disagrees with release",
            )
            if position.bound_agv_id is not None:
                agv = self.agvs[position.bound_agv_id]
                _require(
                    agv.trip is not None and agv.trip.job_id == job,
                    "job binding lacks its vehicle",
                )
            if current is not None and current.status in (
                OperationStatus.PROCESSING,
                OperationStatus.PAUSED,
            ):
                mode = next(
                    op for op in chain if op.operation_id == current.operation_id
                ).mode(current.processing_mode_id)
                _require(
                    position.location == JobLocation("machine", mode.machine_id)
                    and position.bound_agv_id is None,
                    "processing job is not held by its machine",
                )
            else:
                _require(
                    position.location.kind != "machine",
                    "machine holds a nonprocessing job",
                )
            if position.location.kind == "agv":
                _require(
                    position.bound_agv_id == position.location.resource_id,
                    "loaded job lacks its AGV",
                )
            if position.location.kind == "prebuffer":
                _require(
                    current is not None and current.status == OperationStatus.PENDING,
                    "prebuffer job has no pending operation",
                )
                op = next(op for op in chain if op.operation_id == current.operation_id)
                _require(
                    any(
                        m.machine_id == position.location.resource_id for m in op.modes
                    ),
                    "job delivered to incompatible machine",
                )
            if position.location.kind in ("input", "unreleased"):
                _require(
                    all(
                        state.operations[op.operation_id].status
                        == OperationStatus.PENDING
                        for op in chain
                    ),
                    "processed job returned to input",
                )
            if position.location.kind == "postbuffer":
                completed = [
                    op
                    for op in chain
                    if state.operations[op.operation_id].status
                    == OperationStatus.COMPLETED
                ]
                _require(bool(completed), "postbuffer job has no completed operation")
                last = completed[-1]
                mode = last.mode(state.operations[last.operation_id].processing_mode_id)
                _require(
                    mode.machine_id == position.location.resource_id,
                    "postbuffer disagrees with last processing machine",
                )
            if position.location.kind == "output":
                _require(current is None, "unfinished job at output")
