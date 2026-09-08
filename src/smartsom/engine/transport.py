"""Engine-owned logistics shared by AGV transport and instantaneous transfers."""

from collections.abc import Callable, Collection
from dataclasses import replace

from smartsom.dispatch import Transfer, TransportCandidate
from smartsom.domain import (
    ActiveTransport,
    AGVState,
    BufferReservation,
    JobLocation,
    JobPosition,
    OperationStatus,
    ScheduledTransfer,
    ScheduledTransport,
    TransportArrival,
    TransportDestination,
)
from smartsom.domain.buffers import HoldingReservation
from smartsom.engine.calendar import Event, EventCalendar
from smartsom.engine.invariants import _require
from smartsom.engine.state import RuntimeState
from smartsom.modules.arrivals import ArrivalEvent, ArrivalModule
from smartsom.modules.transport import TransportEvent, TransportModule
from smartsom.trace import TraceRecord, TransportRecord
from smartsom.trace.records import (
    BufferRecord,
    HoldingBufferRecord,
    TransferRecord,
    VehicleRecord,
)


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
            a.agv_id: AGVState(a.agv_id, a.initial_node_id)
            for a in (module.spec.agvs if module.spec else ())
        }
        self.schedule: list[ScheduledTransport] = []
        self.arrivals: list[TransportArrival] = []
        self.transfers: list[ScheduledTransfer] = []
        self.reservations: dict[int, BufferReservation | HoldingReservation] = {}
        self.started = 0
        self._waiting_logged: set[int] = set()

    @property
    def finished(self) -> bool:
        return all(p.location.kind == "output" for p in self.positions.values())

    @property
    def makespan(self) -> int:
        return max(
            [t.delivery_time for t in self.schedule if t.destination.kind == "output"]
            + [
                t.simulation_time
                for t in self.transfers
                if t.destination.kind == "output"
            ]
        )

    def release(self, job_id: str) -> None:
        self.positions[job_id] = JobPosition(job_id, JobLocation("input"))

    def processing(
        self,
        operation_id: str,
        machine_id: str,
        state: RuntimeState,
        trace: list[TraceRecord],
        *,
        complete: bool = False,
    ) -> None:
        job = self.module.operation_jobs[operation_id]
        if not complete:
            self.positions[job] = JobPosition(job, JobLocation("machine", machine_id))
        elif self.module.buffers.space(machine_id, "postbuffer", self.positions):
            self.positions[job] = JobPosition(
                job, JobLocation("postbuffer", machine_id)
            )
            state.machine_occupants[machine_id] = None
        else:
            trace.append(
                BufferRecord(
                    len(trace), state.simulation_time, machine_id, job, "block"
                )
            )

    def _record_vehicle(
        self, trace: list[TraceRecord], tick: int, trip: ActiveTransport, kind: str
    ) -> None:
        if self.module.detailed:
            trace.append(VehicleRecord(len(trace), tick, trip, kind))
        else:
            trace.append(
                TransportRecord(
                    len(trace), tick, trip.completed(trip.arrival_time), kind
                )
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
        trip = ActiveTransport(
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
        destination = action.destination
        if self.module.waiting_capacity(destination) not in (
            None,
            0,
        ) and self.module.destination_space(
            destination, self.positions, self.reservations.values()
        ):
            reservation = (
                HoldingReservation(
                    destination.buffer_id,
                    trip.agv_id,
                    trip.job_id,
                    trip.transport_sequence,
                )
                if destination.kind == "holding"
                else BufferReservation(
                    destination.machine_id,
                    trip.agv_id,
                    trip.job_id,
                    trip.transport_sequence,
                )
            )
            self.reservations[trip.transport_sequence] = reservation
            self._record_reservation(trace, tick, reservation, consume=False)
        calendar.schedule(
            TransportEvent(
                trip.pickup_time,
                trip.agv_id,
                trip.job_id,
                trip.transport_sequence,
                "pickup",
            )
        )
        self._record_vehicle(trace, tick, trip, "empty_start")

    def _record_reservation(self, trace, tick, reservation, *, consume):
        if isinstance(reservation, HoldingReservation):
            record = HoldingBufferRecord(
                len(trace),
                tick,
                reservation.buffer_id,
                reservation.job_id,
                "holding_consume_reservation" if consume else "holding_reserve",
                reservation.transport_sequence,
            )
        else:
            record = BufferRecord(
                len(trace),
                tick,
                reservation.machine_id,
                reservation.job_id,
                "consume_reservation" if consume else "reserve",
                reservation.transport_sequence,
            )
        trace.append(record)

    def _release_source(
        self, job: str, state: RuntimeState, trace: list[TraceRecord], reason: str
    ) -> None:
        location = self.positions[job].location
        if location.kind == "machine":
            key = location.resource_id
            operation = state.operations[state.machine_occupants[key]]
            _require(
                operation.status
                in (OperationStatus.PENDING, OperationStatus.COMPLETED),
                "cannot move processing/paused work",
            )
            state.machine_occupants[key] = None
            if operation.status == OperationStatus.COMPLETED:
                trace.append(
                    BufferRecord(
                        len(trace),
                        state.simulation_time,
                        key,
                        job,
                        "unblock",
                        reason=reason,
                    )
                )

    def _place(
        self, job: str, destination: TransportDestination, state: RuntimeState
    ) -> None:
        location = self.module.destination_location(destination)
        self.positions[job] = JobPosition(job, location)
        if location.kind == "machine":
            _require(
                state.machine_occupants[location.resource_id] is None,
                "unload onto occupied processing position",
            )
            operation = next(
                op
                for op in self.module.chains[job]
                if state.operations[op.operation_id].status != OperationStatus.COMPLETED
            )
            state.machine_occupants[location.resource_id] = operation.operation_id

    def transfer(
        self, action: Transfer, state: RuntimeState, trace: list[TraceRecord]
    ) -> None:
        source = self.positions[action.job_id].location
        self._release_source(action.job_id, state, trace, "transfer")
        self._place(action.job_id, action.destination, state)
        transfer = ScheduledTransfer(
            len(self.transfers) + 1,
            action.job_id,
            action.destination,
            source,
            state.simulation_time,
        )
        self.transfers.append(transfer)
        trace.append(TransferRecord(len(trace), state.simulation_time, transfer))

    def _can_deliver(self, trip: ActiveTransport, state: RuntimeState) -> bool:
        destination = trip.destination
        if destination.kind == "output":
            return True
        key = destination.machine_id
        if (
            destination.kind == "machine"
            and self.module.waiting_capacity(destination) == 0
        ):
            return (
                state.machine_occupants[key] is None and key not in state.down_machines
            )
        return (
            trip.transport_sequence in self.reservations
            or self.module.destination_space(
                destination, self.positions, self.reservations.values()
            )
        )

    def _deliver(
        self, trip: ActiveTransport, state: RuntimeState, trace: list[TraceRecord]
    ) -> None:
        reservation = self.reservations.pop(trip.transport_sequence, None)
        if reservation is not None:
            self._record_reservation(
                trace, state.simulation_time, reservation, consume=True
            )
        self._place(trip.job_id, trip.destination, state)
        self.agvs[trip.agv_id] = AGVState(trip.agv_id, trip.delivery_node_id)
        self.schedule.append(trip.completed(state.simulation_time))
        self._record_vehicle(trace, state.simulation_time, trip, "delivery")

    def handle(
        self,
        event: TransportEvent,
        calendar: EventCalendar,
        trace: list[TraceRecord],
        state: RuntimeState,
    ) -> None:
        trip = self.agvs[event.agv_id].trip
        _require(
            trip is not None
            and trip.transport_sequence == event.transport_sequence
            and trip.job_id == event.job_id,
            "stale transport event",
        )
        if event.kind == "pickup":
            self._release_source(trip.job_id, state, trace, "pickup")
            self.positions[trip.job_id] = JobPosition(
                trip.job_id, JobLocation("agv", trip.agv_id), trip.agv_id
            )
            self.agvs[trip.agv_id] = AGVState(trip.agv_id, None, "loaded", trip)
            self._record_vehicle(trace, event.simulation_time, trip, "pickup")
            self._record_vehicle(trace, event.simulation_time, trip, "loaded_start")
            calendar.schedule(
                TransportEvent(
                    trip.arrival_time,
                    trip.agv_id,
                    trip.job_id,
                    trip.transport_sequence,
                    "delivery",
                )
            )
        else:
            self.arrivals.append(
                TransportArrival(trip.transport_sequence, event.simulation_time)
            )
            if self.module.detailed:
                self.agvs[trip.agv_id] = AGVState(
                    trip.agv_id, trip.delivery_node_id, "waiting", trip
                )
                self._record_vehicle(trace, event.simulation_time, trip, "arrival")
            else:
                self._deliver(trip, state, trace)

    def settle(
        self, state: RuntimeState, trace: list[TraceRecord], check: Callable[[], None]
    ) -> None:
        """Drain automatic handoffs after calendar phases; never advance the clock."""
        while True:
            changed = False
            for key in sorted(state.machine_occupants):
                op_id = state.machine_occupants[key]
                if (
                    op_id is None
                    or state.operations[op_id].status != OperationStatus.COMPLETED
                ):
                    continue
                job = self.module.operation_jobs[op_id]
                if self.positions[
                    job
                ].bound_agv_id is None and self.module.buffers.space(
                    key, "postbuffer", self.positions
                ):
                    state.machine_occupants[key] = None
                    self.positions[job] = JobPosition(
                        job, JobLocation("postbuffer", key)
                    )
                    trace.append(
                        BufferRecord(
                            len(trace),
                            state.simulation_time,
                            key,
                            job,
                            "unblock",
                            reason="postbuffer",
                        )
                    )
                    changed = True
                    check()
            waiting = sorted(
                (a.trip for a in self.agvs.values() if a.phase == "waiting"),
                key=lambda t: (
                    t.destination.machine_id or t.destination.buffer_id or "",
                    t.arrival_time,
                    t.agv_id,
                    t.job_id,
                    t.transport_sequence,
                ),
            )
            for trip in waiting:
                if self._can_deliver(trip, state):
                    self._deliver(trip, state, trace)
                    changed = True
                    check()
            if not changed:
                break
        for a in sorted(self.agvs.values(), key=lambda x: x.agv_id):
            if (
                a.phase == "waiting"
                and a.trip.transport_sequence not in self._waiting_logged
            ):
                self._waiting_logged.add(a.trip.transport_sequence)
                self._record_vehicle(
                    trace, state.simulation_time, a.trip, "wait_for_unload"
                )

    def held_operations(self, state: RuntimeState) -> frozenset[str]:
        return frozenset(
            x
            for x in state.machine_occupants.values()
            if x is not None
            and state.operations[x].status
            in (OperationStatus.PENDING, OperationStatus.COMPLETED)
        )

    def diagnostic(self, state: RuntimeState) -> str:
        return (
            f"positions={tuple(self.positions[k] for k in sorted(self.positions))}; "
            f"vehicles={tuple(self.agvs[k] for k in sorted(self.agvs))}; "
            f"buffers={self.module.buffers.snapshots(self.positions, self.reservations.values())}; "
            f"holding={self.module.buffers.holding_snapshot(self.positions, self.reservations.values())}"
        )

    def check(
        self,
        state: RuntimeState,
        pending_events: tuple[Event, ...],
        handled_arrivals: Collection[ArrivalEvent],
        arrivals: ArrivalModule,
    ) -> None:
        _require(
            set(self.positions) == set(self.module.chains),
            "job position identity changed",
        )
        spec = self.module.spec
        initial = {a.agv_id: a.initial_node_id for a in (spec.agvs if spec else ())}
        _require(set(self.agvs) == set(initial), "AGV identity changed")
        expected, active = [], []
        arrived = {x.transport_sequence: x.arrival_time for x in self.arrivals}
        _require(len(arrived) == len(self.arrivals), "duplicate transport arrival")
        for key, agv in self.agvs.items():
            _require(agv.agv_id == key, "AGV identity mismatch")
            if agv.phase == "idle":
                _require(
                    agv.trip is None and agv.node_id in spec.nodes, "invalid idle AGV"
                )
                continue
            trip = agv.trip
            _require(
                isinstance(trip, ActiveTransport)
                and trip.agv_id == key
                and agv.phase in ("empty", "loaded", "waiting"),
                "invalid busy AGV",
            )
            active.append(trip)
            if agv.phase == "waiting":
                _require(
                    agv.node_id == trip.delivery_node_id
                    and arrived.get(trip.transport_sequence)
                    == trip.arrival_time
                    <= state.simulation_time,
                    "waiting AGV has not arrived",
                )
            else:
                _require(agv.node_id is None, "travelling vehicle has a parked node")
                left, right = (
                    (trip.start_time, trip.pickup_time)
                    if agv.phase == "empty"
                    else (trip.pickup_time, trip.arrival_time)
                )
                _require(
                    left <= state.simulation_time <= right,
                    "AGV phase disagrees with clock",
                )
                _require(
                    trip.transport_sequence not in arrived,
                    "travelling AGV already arrived",
                )
                expected.append(
                    TransportEvent(
                        right,
                        key,
                        trip.job_id,
                        trip.transport_sequence,
                        "pickup" if agv.phase == "empty" else "delivery",
                    )
                )
            pos = self.positions[trip.job_id]
            _require(
                pos.bound_agv_id == key
                and pos.location
                == (trip.source if agv.phase == "empty" else JobLocation("agv", key)),
                "job disagrees with AGV binding/phase",
            )
        actual = tuple(e for e in pending_events if isinstance(e, TransportEvent))
        _require(
            len(actual) == len(expected) and set(actual) == set(expected),
            "transport calendar disagrees with busy AGVs",
        )
        trips = sorted((*self.schedule, *active), key=lambda t: t.transport_sequence)
        _require(
            [t.transport_sequence for t in trips] == list(range(1, self.started + 1)),
            "missing or duplicate transport occurrence",
        )
        previous = {key: (0, node) for key, node in initial.items()}
        for t in trips:
            end, node = previous[t.agv_id]
            at = (
                t.arrival_time
                if isinstance(t, ActiveTransport)
                else arrived.get(t.transport_sequence)
            )
            _require(type(at) is int, "delivery lacks arrival")
            _require(
                t.start_time >= end and t.from_node_id == node,
                "AGV overlap or disconnected position",
            )
            _require(
                t.pickup_node_id == self.module.node(t.source)
                and t.delivery_node_id
                == self.module.node(self.module.destination_location(t.destination)),
                "transport nodes disagree with locations",
            )
            _require(
                t.pickup_time - t.start_time
                == self.module.travel_times[node, t.pickup_node_id]
                and at - t.pickup_time
                == self.module.travel_times[t.pickup_node_id, t.delivery_node_id],
                "transport duration disagrees with matrix",
            )
            end = (
                state.simulation_time
                if isinstance(t, ActiveTransport)
                else t.delivery_time
            )
            _require(
                at <= end <= state.simulation_time
                if not isinstance(t, ActiveTransport)
                else t.start_time <= state.simulation_time,
                "recorded future delivery",
            )
            previous[t.agv_id] = end, t.delivery_node_id
        for key, a in self.agvs.items():
            if a.phase == "idle":
                _require(
                    a.node_id == previous[key][1],
                    "idle AGV left its last delivery node",
                )
        _require(
            set(arrived)
            == {t.transport_sequence for t in self.schedule}
            | {
                a.trip.transport_sequence
                for a in self.agvs.values()
                if a.phase == "waiting"
            },
            "arrival coverage mismatch",
        )
        for seq, r in self.reservations.items():
            a = self.agvs[r.agv_id]
            destination = (
                TransportDestination("holding", buffer_id=r.buffer_id)
                if isinstance(r, HoldingReservation)
                else TransportDestination("machine", r.machine_id)
            )
            _require(
                a.trip is not None
                and a.trip.transport_sequence == seq == r.transport_sequence
                and a.trip.job_id == r.job_id
                and a.trip.destination == destination,
                "orphan reservation",
            )
            _require(
                self.module.waiting_capacity(destination) not in (None, 0),
                "invalid reservation capacity",
            )
        holding = self.module.buffers.holding_snapshot(
            self.positions, self.reservations.values()
        )
        if holding is not None:
            _require(
                holding.capacity is None
                or len(holding.jobs) + len(holding.reservations) <= holding.capacity,
                "holding buffer overflow",
            )
        for b in self.module.buffers.snapshots(
            self.positions, self.reservations.values()
        ):
            _require(
                b.pre_capacity is None
                or len(b.pre_jobs) + len(b.reservations) <= b.pre_capacity,
                "prebuffer overflow",
            )
            _require(
                b.post_capacity is None or len(b.post_jobs) <= b.post_capacity,
                "postbuffer overflow",
            )
        holders = {}
        for job, pos in self.positions.items():
            _require(job == pos.job_id, "job position identity changed")
            chain = self.module.chains[job]
            current = next(
                (
                    op
                    for op in chain
                    if state.operations[op.operation_id].status
                    != OperationStatus.COMPLETED
                ),
                None,
            )
            released = arrivals.release_at(chain[0].operation_id) == 0 or any(
                e.job_id == job and e.kind == "release" for e in handled_arrivals
            )
            _require(
                (pos.location.kind == "unreleased") == (not released),
                "job location disagrees with release",
            )
            if pos.bound_agv_id is not None:
                a = self.agvs[pos.bound_agv_id]
                _require(
                    a.trip is not None and a.trip.job_id == job,
                    "job binding lacks its vehicle",
                )
            if pos.location.kind == "machine":
                key = pos.location.resource_id
                _require(key not in holders, "two jobs on processing position")
                op_id = state.machine_occupants[key]
                _require(
                    op_id in {op.operation_id for op in chain},
                    "machine/job occupancy mismatch",
                )
                holders[key] = op_id
                status = state.operations[op_id].status
                if status == OperationStatus.PENDING:
                    _require(
                        self.module.buffers.limits[key].pre_capacity == 0
                        and current.operation_id == op_id
                        and any(m.machine_id == key for m in current.modes),
                        "invalid pending machine load",
                    )
                elif status == OperationStatus.COMPLETED:
                    _require(
                        op_id
                        == [
                            op.operation_id
                            for op in chain
                            if state.operations[op.operation_id].status
                            == OperationStatus.COMPLETED
                        ][-1],
                        "machine holds an obsolete completion",
                    )
                    completed = next(op for op in chain if op.operation_id == op_id)
                    _require(
                        completed.mode(
                            state.operations[op_id].processing_mode_id
                        ).machine_id
                        == key,
                        "completed job held by wrong machine",
                    )
                else:
                    _require(
                        pos.bound_agv_id is None, "processing job bound to vehicle"
                    )
            if current is not None and state.operations[
                current.operation_id
            ].status in (OperationStatus.PROCESSING, OperationStatus.PAUSED):
                mode = current.mode(
                    state.operations[current.operation_id].processing_mode_id
                )
                _require(
                    pos.location == JobLocation("machine", mode.machine_id)
                    and pos.bound_agv_id is None,
                    "processing job is not held by its machine",
                )
            if pos.location.kind == "agv":
                _require(
                    pos.bound_agv_id == pos.location.resource_id,
                    "loaded job lacks its AGV",
                )
            if pos.location.kind == "holding":
                _require(
                    holding is not None
                    and pos.location.resource_id == holding.buffer_id
                    and current is not None
                    and state.operations[current.operation_id].status
                    == OperationStatus.PENDING
                    and any(
                        state.operations[op.operation_id].status
                        == OperationStatus.COMPLETED
                        for op in chain
                    ),
                    "invalid holding buffer job",
                )
            if pos.location.kind == "prebuffer":
                _require(
                    current is not None
                    and state.operations[current.operation_id].status
                    == OperationStatus.PENDING
                    and any(
                        m.machine_id == pos.location.resource_id for m in current.modes
                    ),
                    "incompatible prebuffer job",
                )
            if pos.location.kind in ("input", "unreleased"):
                _require(
                    all(
                        state.operations[op.operation_id].status
                        == OperationStatus.PENDING
                        for op in chain
                    ),
                    "processed job returned to input",
                )
            if pos.location.kind == "postbuffer":
                done = [
                    op
                    for op in chain
                    if state.operations[op.operation_id].status
                    == OperationStatus.COMPLETED
                ]
                _require(
                    bool(done)
                    and done[-1]
                    .mode(state.operations[done[-1].operation_id].processing_mode_id)
                    .machine_id
                    == pos.location.resource_id,
                    "postbuffer disagrees with last machine",
                )
            if pos.location.kind == "output":
                _require(current is None, "unfinished job at output")
        _require(
            holders
            == {k: v for k, v in state.machine_occupants.items() if v is not None},
            "machine occupancy disagrees with positions",
        )
        _require(
            [t.transfer_sequence for t in self.transfers]
            == list(range(1, len(self.transfers) + 1)),
            "transfer occurrence mismatch",
        )
        _require(
            all(t.simulation_time <= state.simulation_time for t in self.transfers),
            "future transfer recorded",
        )
        delivered = [
            t.job_id
            for t in (*self.schedule, *self.transfers)
            if t.destination.kind == "output"
        ]
        _require(
            len(delivered) == len(set(delivered))
            and set(delivered)
            == {k for k, p in self.positions.items() if p.location.kind == "output"},
            "output jobs disagree with deliveries",
        )
