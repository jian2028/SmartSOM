"""Validate and follow v2 action timing, delegating all physical transitions to step."""

from dataclasses import replace

from smartsom.dispatch import (
    DecisionContext,
    Dispatch,
    SemanticAction,
    Transfer,
    Transport,
    WaitNextEvent,
    WaitUntil,
)
from smartsom.domain import (
    ExecutionSchedule,
    FactorySpec,
    JobLocation,
    ScheduledOperation,
    TransportDestination,
    WorkloadInstance,
)
from smartsom.engine.replay import ReplayError
from smartsom.engine.result import SimulationResult
from smartsom.modules.transport import TransportModule


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReplayError(message)


def _ticks(*values: object) -> bool:
    return all(type(x) is int and x >= 0 for x in values)


class ExecutionReplay:
    def __init__(
        self,
        factory: FactorySpec,
        workload: WorkloadInstance,
        schedule: ExecutionSchedule,
        operations: tuple[ScheduledOperation, ...],
        *,
        transport_enabled: bool,
    ) -> None:
        _require(
            isinstance(schedule, ExecutionSchedule) and schedule.version == 2,
            "finite/direct logistics replay requires a v2 ExecutionSchedule",
        )
        module = TransportModule(
            factory, workload, transport_enabled=transport_enabled, buffers_enabled=True
        )
        for entries, field in (
            (schedule.transports, "transport_sequence"),
            (schedule.transfers, "transfer_sequence"),
            (schedule.action_order, "sequence"),
            (schedule.arrivals, "transport_sequence"),
        ):
            values = [getattr(t, field) for t in entries]
            _require(
                _ticks(*values) and sorted(values) == list(range(1, len(entries) + 1)),
                "missing, duplicate or invalid execution sequence",
            )
        trips = tuple(sorted(schedule.transports, key=lambda t: t.transport_sequence))
        transfers = tuple(sorted(schedule.transfers, key=lambda t: t.transfer_sequence))
        actions = tuple(sorted(schedule.action_order, key=lambda t: t.sequence))
        arrivals = tuple(sorted(schedule.arrivals, key=lambda t: t.transport_sequence))
        _require(bool(actions), "v2 schedule needs explicit action order")
        _require(len(arrivals) == len(trips), "arrival coverage mismatch")
        _require(
            transport_enabled or not trips,
            "AGV schedule supplied while transport is disabled",
        )
        _require(
            not transport_enabled or not transfers,
            "direct transfer supplied while AGV is enabled",
        )
        by_arrival = {t.transport_sequence: t.arrival_time for t in arrivals}
        machines = {m.machine_id for m in factory.machines}

        def location(value):
            return isinstance(value, JobLocation) and (
                (value.kind == "input" and value.resource_id is None)
                or (
                    value.kind in ("prebuffer", "machine", "postbuffer")
                    and value.resource_id in machines
                )
            )

        def destination(value):
            return isinstance(value, TransportDestination) and (
                (value.kind == "output" and value.machine_id is None)
                or (value.kind == "machine" and value.machine_id in machines)
            )

        previous = {
            v.agv_id: (0, v.initial_node_id)
            for v in (module.spec.agvs if module.spec else ())
        }
        for t in trips:
            at = by_arrival[t.transport_sequence]
            _require(
                _ticks(t.start_time, t.pickup_time, at, t.delivery_time),
                "invalid transport time",
            )
            _require(
                isinstance(t.agv_id, str)
                and t.agv_id in previous
                and isinstance(t.job_id, str)
                and t.job_id in module.chains,
                "unknown AGV or job",
            )
            _require(
                location(t.source) and destination(t.destination),
                "invalid transport location or destination",
            )
            _require(
                all(
                    isinstance(n, str) and n in module.spec.nodes
                    for n in (t.from_node_id, t.pickup_node_id, t.delivery_node_id)
                ),
                "unknown transport node",
            )
            end, node = previous[t.agv_id]
            _require(
                t.start_time >= end and t.from_node_id == node,
                "vehicle overlap or position discontinuity",
            )
            _require(
                t.pickup_node_id == module.node(t.source)
                and t.delivery_node_id
                == module.node(module.destination_location(t.destination)),
                "transport node/location mismatch",
            )
            _require(
                t.pickup_time - t.start_time
                == module.travel_times[node, t.pickup_node_id]
                and at - t.pickup_time
                == module.travel_times[t.pickup_node_id, t.delivery_node_id]
                and t.delivery_time >= at,
                "wrong travel or unloading time",
            )
            previous[t.agv_id] = t.delivery_time, t.delivery_node_id
        for t in transfers:
            _require(
                _ticks(t.simulation_time)
                and isinstance(t.job_id, str)
                and t.job_id in module.chains
                and location(t.source)
                and destination(t.destination),
                "invalid direct transfer",
            )
        dispatched = set()
        entries = {op.operation_id: op for op in operations}
        ti = fi = 0
        last = 0
        for timed in actions:
            _require(
                _ticks(timed.simulation_time) and timed.simulation_time >= last,
                "invalid action tick or action order",
            )
            last = timed.simulation_time
            action = timed.action
            if isinstance(action, Dispatch):
                _require(
                    isinstance(action.operation_id, str)
                    and action.operation_id in entries
                    and action.operation_id not in dispatched,
                    "missing, duplicate or unknown processing action",
                )
                entry = entries[action.operation_id]
                _require(
                    action.processing_mode_id == entry.processing_mode_id
                    and timed.simulation_time == entry.start_time,
                    "dispatch disagrees with processing timetable",
                )
                dispatched.add(action.operation_id)
            elif isinstance(action, Transport):
                _require(ti < len(trips), "extra transport action")
                t = trips[ti]
                ti += 1
                _require(
                    action == Transport(t.agv_id, t.job_id, t.destination)
                    and timed.simulation_time == t.start_time,
                    "transport action order/timing mismatch",
                )
            elif isinstance(action, Transfer):
                _require(fi < len(transfers), "extra transfer action")
                t = transfers[fi]
                fi += 1
                _require(
                    action == Transfer(t.job_id, t.destination)
                    and timed.simulation_time == t.simulation_time,
                    "transfer action order/timing mismatch",
                )
            else:
                raise ReplayError(
                    "action_order must contain only dispatch, transport or transfer"
                )
        _require(
            dispatched == set(entries) and ti == len(trips) and fi == len(transfers),
            "incomplete action coverage",
        )
        outputs = [
            t.job_id for t in (*trips, *transfers) if t.destination.kind == "output"
        ]
        _require(
            len(outputs) == len(set(outputs)) and set(outputs) == set(module.chains),
            "missing or duplicate output transfer",
        )
        self.expected = replace(
            schedule,
            operations=operations,
            transports=trips,
            arrivals=arrivals,
            transfers=transfers,
            action_order=actions,
        )
        self.actions = actions
        self.index = 0

    def select_action(self, context: DecisionContext) -> SemanticAction:
        if self.index == len(self.actions):
            if not context.feasible_actions:
                return WaitNextEvent()
            raise ReplayError("execution action order ended before termination")
        timed = self.actions[self.index]
        tick = context.simulation_time
        if tick < timed.simulation_time:
            return WaitUntil(timed.simulation_time)
        _require(
            tick == timed.simulation_time,
            f"missed execution action at tick {timed.simulation_time}",
        )
        _require(
            timed.action in context.feasible_actions,
            f"execution action {timed.sequence} is not feasible at tick {tick}",
        )
        self.index += 1
        return timed.action

    def verify_result(self, result: SimulationResult) -> None:
        _require(self.index == len(self.actions), "unused execution actions")
        _require(
            result.execution_schedule == self.expected,
            "actual execution schedule differs from requested schedule",
        )
        times = [
            t.delivery_time
            for t in self.expected.transports
            if t.destination.kind == "output"
        ] + [
            t.simulation_time
            for t in self.expected.transfers
            if t.destination.kind == "output"
        ]
        _require(
            result.makespan == max(times), "actual makespan differs from output timing"
        )
