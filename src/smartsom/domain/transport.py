"""Immutable fixed-matrix resources, public positions and executed timetables."""

from dataclasses import dataclass
from typing import Literal

from smartsom.domain.state import ScheduledOperation
from smartsom.domain.validation import (
    DomainValidationError,
    _identifier,
    _items,
    _unique,
)


@dataclass(frozen=True, slots=True)
class AGV:
    agv_id: str
    initial_node_id: str

    def __post_init__(self):
        _identifier(self.agv_id, "agv_id")
        _identifier(self.initial_node_id, "initial_node_id")


@dataclass(frozen=True, slots=True)
class MachineLocation:
    machine_id: str
    node_id: str

    def __post_init__(self):
        _identifier(self.machine_id, "machine_id")
        _identifier(self.node_id, "node_id")


@dataclass(frozen=True, slots=True)
class TravelTime:
    from_node_id: str
    to_node_id: str
    ticks: int

    def __post_init__(self):
        _identifier(self.from_node_id, "from_node_id")
        _identifier(self.to_node_id, "to_node_id")
        if type(self.ticks) is not int or self.ticks < 0:
            raise DomainValidationError("travel ticks must be nonnegative integers")


@dataclass(frozen=True, slots=True)
class TransportSpec:
    nodes: tuple[str, ...]
    machine_locations: tuple[MachineLocation, ...]
    input_node_id: str
    output_node_id: str
    agvs: tuple[AGV, ...]
    travel_times: tuple[TravelTime, ...]

    def __post_init__(self):
        for name, item_type, key in (
            ("nodes", str, lambda x: x),
            ("machine_locations", MachineLocation, lambda x: x.machine_id),
            ("agvs", AGV, lambda x: x.agv_id),
            ("travel_times", TravelTime, lambda x: (x.from_node_id, x.to_node_id)),
        ):
            items = _items(getattr(self, name), item_type, name)
            _unique(tuple(key(item) for item in items), name)
            object.__setattr__(self, name, tuple(sorted(items, key=key)))
        for node in self.nodes:
            _identifier(node, "node_id")
        references = (
            self.input_node_id,
            self.output_node_id,
            *(x.node_id for x in self.machine_locations),
            *(x.initial_node_id for x in self.agvs),
        )
        if any(not isinstance(x, str) or x not in self.nodes for x in references):
            raise DomainValidationError("unknown transport node reference")
        expected = {(a, b) for a in self.nodes for b in self.nodes}
        if {(x.from_node_id, x.to_node_id) for x in self.travel_times} != expected:
            raise DomainValidationError(
                "travel matrix must cover every node pair exactly once"
            )
        if any(
            x.ticks != 0 for x in self.travel_times if x.from_node_id == x.to_node_id
        ):
            raise DomainValidationError("travel matrix diagonal must be zero")


@dataclass(frozen=True, slots=True)
class TransportDestination:
    kind: Literal["machine", "output"]
    machine_id: str | None = None

    def __post_init__(self):
        if self.kind == "machine":
            _identifier(self.machine_id, "destination machine_id")
        elif self.kind != "output" or self.machine_id is not None:
            raise DomainValidationError("destination must be a machine or output")


@dataclass(frozen=True, slots=True)
class JobLocation:
    kind: Literal[
        "unreleased", "input", "prebuffer", "machine", "postbuffer", "agv", "output"
    ]
    resource_id: str | None = None

    def __post_init__(self):
        if self.kind in ("prebuffer", "machine", "postbuffer", "agv"):
            _identifier(self.resource_id, "location resource_id")
        elif (
            self.kind not in ("unreleased", "input", "output")
            or self.resource_id is not None
        ):
            raise DomainValidationError("invalid job location")


@dataclass(frozen=True, slots=True)
class JobPosition:
    job_id: str
    location: JobLocation
    bound_agv_id: str | None = None


@dataclass(frozen=True, slots=True)
class ScheduledTransport:
    # An explicit occurrence order, not a container/candidate position.
    transport_sequence: int
    agv_id: str
    job_id: str
    destination: TransportDestination
    source: JobLocation
    from_node_id: str
    pickup_node_id: str
    delivery_node_id: str
    start_time: int
    pickup_time: int
    delivery_time: int


@dataclass(frozen=True, slots=True)
class AGVState:
    agv_id: str
    node_id: str | None
    phase: Literal["idle", "empty", "loaded"] = "idle"
    trip: ScheduledTransport | None = None


@dataclass(frozen=True, slots=True)
class ExecutionSchedule:
    operations: tuple[ScheduledOperation, ...]
    transports: tuple[ScheduledTransport, ...]

    def __post_init__(self):
        for name, item_type in (
            ("operations", ScheduledOperation),
            ("transports", ScheduledTransport),
        ):
            values = getattr(self, name)
            if not isinstance(values, (tuple, list)) or any(
                not isinstance(x, item_type) for x in values
            ):
                raise DomainValidationError(f"invalid execution schedule {name}")
            object.__setattr__(self, name, tuple(values))
