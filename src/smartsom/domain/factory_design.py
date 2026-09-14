"""Immutable factory authoring data, independent of executable FactorySpec.

Geometry and capabilities here describe an editor document. Validation establishes
its internal consistency, not runtime support, reachability, or simulation behavior.
Footprint dimensions and slot coordinates describe the CURRENT orientation; rotation
is retained as metadata and is never applied a second time by geometry helpers.
"""

import re
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Literal

from smartsom.domain.quality import QualityMode, _decimal
from smartsom.domain.validation import DomainValidationError

type Heading = Literal["north", "east", "south", "west"]
type PortOperation = Literal["pickup", "drop_off", "charge"]
type BufferRole = Literal[
    "machine_pre", "machine_post", "storage", "system_input", "system_output"
]

_ID = re.compile(r"[A-Za-z][A-Za-z0-9_-]*\Z", re.ASCII)


def operation_type_key(identifier: str):
    match = re.fullmatch(r"operation_([0-9]+)", identifier)
    return (0, int(match[1]), identifier) if match else (1, 0, identifier)


_HEADINGS = ("north", "east", "south", "west")
_OPERATIONS = ("pickup", "drop_off", "charge")
_ROLES = ("machine_pre", "machine_post", "storage", "system_input", "system_output")


def _id(value, name):
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise DomainValidationError(
            f"{name} must start with an ASCII letter and contain only "
            "ASCII letters, digits, underscores or hyphens"
        )


def _integer(value, name, minimum=None):
    if type(value) is not int or minimum is not None and value < minimum:
        suffix = "" if minimum is None else f" >= {minimum}"
        raise DomainValidationError(f"{name} must be an integer{suffix}")


def _capacity(value, name="capacity"):
    if value is not None:
        _integer(value, name, 0)


def _tuple(obj, name, item_type):
    items = getattr(obj, name)
    if not isinstance(items, (tuple, list)) or any(
        not isinstance(item, item_type) for item in items
    ):
        raise DomainValidationError(f"{name} contains invalid items")
    object.__setattr__(obj, name, tuple(items))


def _tag(value, expected, name):
    if value != expected:
        raise DomainValidationError(f"{name} must be {expected!r}")


def _name(value):
    if not isinstance(value, str) or not value.strip():
        raise DomainValidationError("name must be a non-empty string")


def _resource(obj, id_field):
    _id(getattr(obj, id_field), id_field)
    _name(obj.name)
    if not isinstance(obj.footprint, Footprint):
        raise DomainValidationError("footprint must be a Footprint")


def _energy(obj, name, *, positive=False):
    value = _decimal(getattr(obj, name), name)
    if value < 0 or positive and value == 0:
        raise DomainValidationError(
            f"{name} must be {'positive' if positive else 'nonnegative'}"
        )
    object.__setattr__(obj, name, value)


@dataclass(frozen=True, slots=True)
class Cell:
    x: int
    y: int

    def __post_init__(self):
        _integer(self.x, "x")
        _integer(self.y, "y")


@dataclass(frozen=True, slots=True)
class Footprint:
    x: int
    y: int
    width: int
    height: int
    rotation: int = 0

    def __post_init__(self):
        _integer(self.x, "x")
        _integer(self.y, "y")
        _integer(self.width, "width", 1)
        _integer(self.height, "height", 1)
        if type(self.rotation) is not int or self.rotation not in (0, 90, 180, 270):
            raise DomainValidationError("rotation must be 0, 90, 180 or 270")


@dataclass(frozen=True, slots=True)
class GridDesign:
    width: int
    height: int
    blocked_cells: tuple[Cell, ...] = ()

    def __post_init__(self):
        _integer(self.width, "width", 1)
        _integer(self.height, "height", 1)
        _tuple(self, "blocked_cells", Cell)


@dataclass(frozen=True, slots=True)
class SlotDesign:
    slot_id: str
    local_cell: Cell
    capacity: int = 1

    def __post_init__(self):
        _id(self.slot_id, "slot_id")
        if not isinstance(self.local_cell, Cell):
            raise DomainValidationError("local_cell must be a Cell")
        _integer(self.capacity, "capacity", 1)


@dataclass(frozen=True, slots=True)
class PoolStorage:
    capacity: int | None = 4
    mode: Literal["pool"] = field(default="pool", kw_only=True)

    def __post_init__(self):
        _capacity(self.capacity)
        _tag(self.mode, "pool", "mode")


@dataclass(frozen=True, slots=True)
class SlotStorage:
    slots: tuple[SlotDesign, ...] = ()
    mode: Literal["slots"] = field(default="slots", kw_only=True)

    def __post_init__(self):
        _tuple(self, "slots", SlotDesign)
        _tag(self.mode, "slots", "mode")


type StorageDesign = PoolStorage | SlotStorage


@dataclass(frozen=True, slots=True)
class MachineDesign:
    machine_id: str
    name: str
    operation_types: tuple[str, ...] = field(default=(), kw_only=True)
    footprint: Footprint
    quality_modes: tuple[QualityMode, ...] = (
        QualityMode("normal", Decimal(1), Decimal(0)),
    )

    def __post_init__(self):
        _resource(self, "machine_id")
        _tuple(self, "operation_types", str)
        for operation_type in self.operation_types:
            _id(operation_type, "operation_type")
        if len(self.operation_types) != len(set(self.operation_types)):
            raise DomainValidationError("duplicate operation_type")
        _tuple(self, "quality_modes", QualityMode)
        if not self.quality_modes:
            raise DomainValidationError("quality_modes must not be empty")
        for mode in self.quality_modes:
            _id(mode.quality_mode_id, "quality_mode_id")
        ids = [mode.quality_mode_id for mode in self.quality_modes]
        if len(ids) != len(set(ids)):
            raise DomainValidationError("duplicate quality_mode_id")


@dataclass(frozen=True, slots=True)
class BufferDesign:
    buffer_id: str
    name: str
    footprint: Footprint
    role: BufferRole = "storage"
    storage: StorageDesign = field(default_factory=PoolStorage)
    machine_id: str | None = None

    def __post_init__(self):
        _resource(self, "buffer_id")
        if self.role not in _ROLES:
            raise DomainValidationError("unknown buffer role")
        if not isinstance(self.storage, (PoolStorage, SlotStorage)):
            raise DomainValidationError("storage must be PoolStorage or SlotStorage")
        if self.machine_id is not None:
            _id(self.machine_id, "machine_id")


@dataclass(frozen=True, slots=True)
class InspectionStationDesign:
    inspection_station_id: str
    name: str
    footprint: Footprint
    slots: tuple[SlotDesign, ...] = ()
    inspection_ticks: int = 2
    parallel_capacity: int | Literal["max"] = "max"

    def __post_init__(self):
        _resource(self, "inspection_station_id")
        _tuple(self, "slots", SlotDesign)
        _integer(self.inspection_ticks, "inspection_ticks", 1)
        if self.parallel_capacity != "max":
            _integer(self.parallel_capacity, "parallel_capacity", 1)
        if any(slot.capacity != 1 for slot in self.slots):
            raise DomainValidationError("inspection slots must have capacity 1")


@dataclass(frozen=True, slots=True)
class ScrapBinDesign:
    scrap_bin_id: str
    name: str
    footprint: Footprint
    capacity: int | None = None

    def __post_init__(self):
        _resource(self, "scrap_bin_id")
        _capacity(self.capacity)


@dataclass(frozen=True, slots=True)
class ChargerDesign:
    charger_id: str
    name: str
    footprint: Footprint
    agv_capacity: int = 1
    charge_energy_per_tick: Decimal = Decimal(10)

    def __post_init__(self):
        _resource(self, "charger_id")
        _integer(self.agv_capacity, "agv_capacity", 1)
        _energy(self, "charge_energy_per_tick", positive=True)


@dataclass(frozen=True, slots=True)
class BatteryDesign:
    energy_capacity: Decimal = Decimal(100)
    initial_energy: Decimal = Decimal(100)
    move_energy_per_cell: Decimal = Decimal(1)
    idle_energy_per_tick: Decimal = Decimal(0)

    def __post_init__(self):
        _energy(self, "energy_capacity", positive=True)
        for name in ("initial_energy", "move_energy_per_cell", "idle_energy_per_tick"):
            _energy(self, name)
        if self.initial_energy > self.energy_capacity:
            raise DomainValidationError("initial_energy exceeds energy_capacity")


@dataclass(frozen=True, slots=True)
class AGVDesign:
    agv_id: str
    name: str
    initial_cell: Cell
    job_capacity: int = 1
    initial_heading: Heading = "east"
    move_cells_per_tick: int = 1
    battery: BatteryDesign | None = None

    def __post_init__(self):
        _id(self.agv_id, "agv_id")
        _name(self.name)
        if not isinstance(self.initial_cell, Cell):
            raise DomainValidationError("initial_cell must be a Cell")
        _integer(self.job_capacity, "job_capacity", 1)
        _integer(self.move_cells_per_tick, "move_cells_per_tick", 1)
        if self.initial_heading not in _HEADINGS:
            raise DomainValidationError("unknown initial_heading")
        if self.battery is not None and not isinstance(self.battery, BatteryDesign):
            raise DomainValidationError("battery must be a BatteryDesign or None")


@dataclass(frozen=True, slots=True)
class MachineTarget:
    machine_id: str
    kind: Literal["machine"] = field(default="machine", kw_only=True)

    def __post_init__(self):
        _id(self.machine_id, "machine_id")
        _tag(self.kind, "machine", "kind")


@dataclass(frozen=True, slots=True)
class BufferTarget:
    buffer_id: str
    kind: Literal["buffer"] = field(default="buffer", kw_only=True)

    def __post_init__(self):
        _id(self.buffer_id, "buffer_id")
        _tag(self.kind, "buffer", "kind")


@dataclass(frozen=True, slots=True)
class BufferSlotTarget:
    buffer_id: str
    slot_id: str
    kind: Literal["buffer_slot"] = field(default="buffer_slot", kw_only=True)

    def __post_init__(self):
        _id(self.buffer_id, "buffer_id")
        _id(self.slot_id, "slot_id")
        _tag(self.kind, "buffer_slot", "kind")


@dataclass(frozen=True, slots=True)
class InspectionSlotTarget:
    inspection_station_id: str
    slot_id: str
    kind: Literal["inspection_slot"] = field(default="inspection_slot", kw_only=True)

    def __post_init__(self):
        _id(self.inspection_station_id, "inspection_station_id")
        _id(self.slot_id, "slot_id")
        _tag(self.kind, "inspection_slot", "kind")


@dataclass(frozen=True, slots=True)
class ScrapBinTarget:
    scrap_bin_id: str
    kind: Literal["scrap_bin"] = field(default="scrap_bin", kw_only=True)

    def __post_init__(self):
        _id(self.scrap_bin_id, "scrap_bin_id")
        _tag(self.kind, "scrap_bin", "kind")


@dataclass(frozen=True, slots=True)
class ChargerTarget:
    charger_id: str
    kind: Literal["charger"] = field(default="charger", kw_only=True)

    def __post_init__(self):
        _id(self.charger_id, "charger_id")
        _tag(self.kind, "charger", "kind")


type PortTarget = (
    MachineTarget
    | BufferTarget
    | BufferSlotTarget
    | InspectionSlotTarget
    | ScrapBinTarget
    | ChargerTarget
)
_TARGET_TYPES = (
    MachineTarget,
    BufferTarget,
    BufferSlotTarget,
    InspectionSlotTarget,
    ScrapBinTarget,
    ChargerTarget,
)
_TARGET_RESOURCE_TYPES = {
    MachineTarget: MachineDesign,
    BufferTarget: BufferDesign,
    BufferSlotTarget: BufferDesign,
    InspectionSlotTarget: InspectionStationDesign,
    ScrapBinTarget: ScrapBinDesign,
    ChargerTarget: ChargerDesign,
}


@dataclass(frozen=True, slots=True)
class PortBinding:
    target: PortTarget
    operations: tuple[PortOperation, ...] = ("pickup", "drop_off")

    def __post_init__(self):
        if not isinstance(self.target, _TARGET_TYPES):
            raise DomainValidationError("target must be a typed port target")
        _tuple(self, "operations", str)
        if not self.operations or any(op not in _OPERATIONS for op in self.operations):
            raise DomainValidationError(
                "operations must contain pickup, drop_off or charge"
            )
        if len(self.operations) != len(set(self.operations)):
            raise DomainValidationError("duplicate port operation")


@dataclass(frozen=True, slots=True)
class PortDesign:
    port_id: str
    name: str
    cell: Cell
    allowed_headings: tuple[Heading, ...] = _HEADINGS
    bindings: tuple[PortBinding, ...] = ()

    def __post_init__(self):
        _id(self.port_id, "port_id")
        _name(self.name)
        if not isinstance(self.cell, Cell):
            raise DomainValidationError("cell must be a Cell")
        _tuple(self, "allowed_headings", str)
        _tuple(self, "bindings", PortBinding)
        if not self.allowed_headings or any(
            h not in _HEADINGS for h in self.allowed_headings
        ):
            raise DomainValidationError(
                "allowed_headings must contain cardinal headings"
            )
        if len(set(self.allowed_headings)) != len(self.allowed_headings):
            raise DomainValidationError("duplicate allowed heading")


@dataclass(frozen=True, slots=True)
class FactoryDesign:
    factory_id: str
    name: str
    grid: GridDesign
    operation_types: tuple[str, ...] = field(default=(), kw_only=True)
    machines: tuple[MachineDesign, ...] = ()
    buffers: tuple[BufferDesign, ...] = ()
    inspection_stations: tuple[InspectionStationDesign, ...] = ()
    scrap_bins: tuple[ScrapBinDesign, ...] = ()
    chargers: tuple[ChargerDesign, ...] = ()
    ports: tuple[PortDesign, ...] = ()
    agvs: tuple[AGVDesign, ...] = ()

    def __post_init__(self):
        _id(self.factory_id, "factory_id")
        _name(self.name)
        _tuple(self, "operation_types", str)
        for operation_type in self.operation_types:
            _id(operation_type, "operation_type")
        if len(set(self.operation_types)) != len(self.operation_types):
            raise DomainValidationError("duplicate factory operation_type")
        if not isinstance(self.grid, GridDesign):
            raise DomainValidationError("grid must be a GridDesign")
        for name, cls in (
            ("machines", MachineDesign),
            ("buffers", BufferDesign),
            ("inspection_stations", InspectionStationDesign),
            ("scrap_bins", ScrapBinDesign),
            ("chargers", ChargerDesign),
            ("ports", PortDesign),
            ("agvs", AGVDesign),
        ):
            _tuple(self, name, cls)


@dataclass(frozen=True, slots=True)
class DesignIssue:
    severity: Literal["error", "warning"]
    code: str
    message: str
    entity_id: str | None = None
    field: str | None = None


type SolidResource = (
    MachineDesign
    | BufferDesign
    | InspectionStationDesign
    | ScrapBinDesign
    | ChargerDesign
)


def iter_resources(
    design: FactoryDesign,
) -> tuple[SolidResource | PortDesign | AGVDesign, ...]:
    """Return every top-level resource in document order, including ports and AGVs."""
    return (
        *design.machines,
        *design.buffers,
        *design.inspection_stations,
        *design.scrap_bins,
        *design.chargers,
        *design.ports,
        *design.agvs,
    )


def entity_id(entity: SolidResource | PortDesign | AGVDesign) -> str:
    for cls, name in (
        (MachineDesign, "machine_id"),
        (BufferDesign, "buffer_id"),
        (InspectionStationDesign, "inspection_station_id"),
        (ScrapBinDesign, "scrap_bin_id"),
        (ChargerDesign, "charger_id"),
        (PortDesign, "port_id"),
        (AGVDesign, "agv_id"),
    ):
        if isinstance(entity, cls):
            return getattr(entity, name)
    raise TypeError("unknown design entity")


def occupied_cells(footprint: Footprint) -> tuple[Cell, ...]:
    """Enumerate current world cells; do not rotate current dimensions again."""
    return tuple(
        Cell(x, y)
        for y in range(footprint.y, footprint.y + footprint.height)
        for x in range(footprint.x, footprint.x + footprint.width)
    )


def world_cell(footprint: Footprint, local_cell: Cell) -> Cell:
    return Cell(footprint.x + local_cell.x, footprint.y + local_cell.y)


def target_owner_id(target: PortTarget) -> str:
    for name in (
        "machine_id",
        "buffer_id",
        "inspection_station_id",
        "scrap_bin_id",
        "charger_id",
    ):
        if hasattr(target, name):
            return getattr(target, name)
    raise TypeError("unknown port target")


def _target_resource(design, target):
    collections = {
        MachineTarget: design.machines,
        BufferTarget: design.buffers,
        BufferSlotTarget: design.buffers,
        InspectionSlotTarget: design.inspection_stations,
        ScrapBinTarget: design.scrap_bins,
        ChargerTarget: design.chargers,
    }
    return next(
        (
            r
            for r in collections[type(target)]
            if entity_id(r) == target_owner_id(target)
        ),
        None,
    )


def target_cell(design: FactoryDesign, target: PortTarget) -> Cell | None:
    """Resolve a target's world cell, or None for a missing/incompatible target.

    Whole-resource targets use their central cell as a drawing anchor; the
    anchor imposes no service distance, path, or adjacency rule.
    """
    return _target_cell(_target_resource(design, target), target)


def _target_cell(resource, target):
    if resource is None:
        return None
    if isinstance(target, BufferTarget) and not isinstance(
        resource.storage, PoolStorage
    ):
        return None
    if isinstance(target, (BufferSlotTarget, InspectionSlotTarget)):
        if isinstance(resource, BufferDesign):
            if not isinstance(resource.storage, SlotStorage):
                return None
            slots = resource.storage.slots
        else:
            slots = resource.slots
        slot = next((slot for slot in slots if slot.slot_id == target.slot_id), None)
        return None if slot is None else world_cell(resource.footprint, slot.local_cell)
    footprint = resource.footprint
    return Cell(footprint.x + footprint.width // 2, footprint.y + footprint.height // 2)


def validate_factory_design(design: FactoryDesign) -> tuple[DesignIssue, ...]:
    """Check references and geometry while retaining incomplete authoring drafts.

    Constructors enforce local field types/ranges. This function reports design
    errors without mutating or compiling the document into simulator resources.
    """
    if not isinstance(design, FactoryDesign):
        raise TypeError("expected FactoryDesign")
    issues = []

    def issue(code, message, owner=None, field=None, severity="error"):
        issues.append(DesignIssue(severity, code, message, owner, field))

    def inside(cell):
        return 0 <= cell.x < design.grid.width and 0 <= cell.y < design.grid.height

    for machine in design.machines:
        if not machine.operation_types:
            issue(
                "unspecified_machine_capability",
                "Machine capability needs setup.",
                machine.machine_id,
                "operation_types",
                "warning",
            )
        for operation_type in machine.operation_types:
            if operation_type not in design.operation_types:
                issue(
                    "unknown_operation_type",
                    f"Unknown operation type {operation_type!r}.",
                    machine.machine_id,
                    "operation_types",
                )

    resources = tuple(r for r in iter_resources(design) if hasattr(r, "footprint"))
    resource_index = {(type(r), entity_id(r)): r for r in resources}
    seen_ids = set()
    for item in (*resources, *design.ports, *design.agvs):
        key = entity_id(item)
        if key in seen_ids:
            issue("duplicate_entity_id", f"Duplicate entity ID {key!r}.", key)
        seen_ids.add(key)

    blocked = set()
    for cell in design.grid.blocked_cells:
        if not inside(cell):
            issue(
                "blocked_out_of_bounds",
                f"Blocked cell {cell} is outside the grid.",
                field="grid.blocked_cells",
            )
        if cell in blocked:
            issue(
                "duplicate_blocked_cell",
                f"Blocked cell {cell} is repeated.",
                field="grid.blocked_cells",
            )
        blocked.add(cell)

    solids = {}
    for resource in resources:
        key = entity_id(resource)
        footprint = resource.footprint
        if (
            footprint.x < 0
            or footprint.y < 0
            or footprint.x + footprint.width > design.grid.width
            or footprint.y + footprint.height > design.grid.height
        ):
            issue(
                "footprint_out_of_bounds",
                "Footprint extends outside the grid.",
                key,
                "footprint",
            )
            continue
        cells = occupied_cells(footprint)
        if any(cell in blocked for cell in cells):
            issue(
                "blocked_overlap", "Footprint overlaps blocked cells.", key, "footprint"
            )
        other_ids = sorted({solids[cell] for cell in cells if cell in solids})
        if other_ids:
            issue(
                "resource_overlap",
                f"Footprint overlaps {', '.join(other_ids)}.",
                key,
                "footprint",
            )
        for cell in cells:
            solids.setdefault(cell, key)

    machine_ids = {m.machine_id for m in design.machines}
    role_owners = set()
    system_roles = set()
    for buffer in design.buffers:
        key = buffer.buffer_id
        if buffer.role in ("machine_pre", "machine_post"):
            if buffer.machine_id is None:
                issue(
                    "orphan_machine_buffer",
                    "Machine buffer has no owning machine yet.",
                    key,
                    "machine_id",
                    "warning",
                )
            else:
                pair = (buffer.role, buffer.machine_id)
                if pair in role_owners:
                    issue(
                        "duplicate_machine_buffer",
                        "A machine may have only one buffer of each pre/post role.",
                        key,
                        "machine_id",
                    )
                role_owners.add(pair)
        elif buffer.machine_id is not None:
            issue(
                "unexpected_machine_owner",
                "Only machine_pre and machine_post buffers may name a machine.",
                key,
                "machine_id",
            )
        if buffer.machine_id is not None and buffer.machine_id not in machine_ids:
            issue(
                "unknown_machine",
                f"Unknown machine {buffer.machine_id!r}.",
                key,
                "machine_id",
            )
        if buffer.role in ("system_input", "system_output"):
            if buffer.role in system_roles:
                issue(
                    "duplicate_system_buffer",
                    f"Only one {buffer.role} buffer is allowed.",
                    key,
                    "role",
                )
            system_roles.add(buffer.role)

    for resource in (*design.buffers, *design.inspection_stations):
        if isinstance(resource, BufferDesign):
            if not isinstance(resource.storage, SlotStorage):
                continue
            slots = resource.storage.slots
        else:
            slots = resource.slots
        key = entity_id(resource)
        if not slots:
            issue(
                "empty_slots",
                "No storage slots have been defined yet.",
                key,
                "slots",
                "warning",
            )
        slot_ids = set()
        slot_cells = set()
        for slot in slots:
            if slot.slot_id in slot_ids:
                issue(
                    "duplicate_slot_id",
                    f"Duplicate local slot ID {slot.slot_id!r}.",
                    key,
                    "slots",
                )
            slot_ids.add(slot.slot_id)
            if slot.local_cell in slot_cells:
                issue(
                    "slot_overlap",
                    "Two slots occupy the same local cell.",
                    key,
                    "slots",
                )
            slot_cells.add(slot.local_cell)
            if not (
                0 <= slot.local_cell.x < resource.footprint.width
                and 0 <= slot.local_cell.y < resource.footprint.height
            ):
                issue(
                    "slot_out_of_bounds",
                    f"Slot {slot.slot_id!r} is outside the current footprint.",
                    key,
                    "slots",
                )
        if (
            isinstance(resource, InspectionStationDesign)
            and resource.parallel_capacity != "max"
            and resource.parallel_capacity > len(slots)
        ):
            issue(
                "inspection_parallel_exceeds_slots",
                "Parallel inspection capacity exceeds currently defined slots.",
                key,
                "parallel_capacity",
                "warning",
            )

    occupied_ports = set()
    for port in design.ports:
        key = port.port_id
        if not inside(port.cell):
            issue("port_out_of_bounds", "Port is outside the grid.", key, "cell")
        if port.cell in blocked or port.cell in solids:
            issue("port_not_walkable", "Port must be on a walkable cell.", key, "cell")
        if port.cell in occupied_ports:
            issue("port_overlap", "Two ports occupy the same cell.", key, "cell")
        occupied_ports.add(port.cell)
        if not port.bindings:
            issue(
                "empty_port",
                "Port has no target bindings yet.",
                key,
                "bindings",
                "warning",
            )
        targets = set()
        for binding in port.bindings:
            target = binding.target
            if target in targets:
                issue(
                    "duplicate_port_target",
                    "Target appears more than once on this port.",
                    key,
                    "bindings",
                )
            targets.add(target)
            resource = resource_index.get(
                (_TARGET_RESOURCE_TYPES[type(target)], target_owner_id(target))
            )
            if _target_cell(resource, target) is None:
                issue(
                    "invalid_port_target",
                    "Port target does not exist or has incompatible storage mode.",
                    key,
                    "bindings",
                )
            allowed = (
                {"charge"}
                if isinstance(target, ChargerTarget)
                else {"drop_off"}
                if isinstance(target, ScrapBinTarget)
                else {"pickup", "drop_off"}
            )
            if not set(binding.operations) <= allowed:
                issue(
                    "invalid_port_operation",
                    f"Target permits only {', '.join(sorted(allowed))}.",
                    key,
                    "bindings",
                )

    agv_cells = set()
    for agv in design.agvs:
        if not inside(agv.initial_cell):
            issue(
                "agv_out_of_bounds",
                "AGV initial cell is outside the grid.",
                agv.agv_id,
                "initial_cell",
            )
        if agv.initial_cell in blocked or agv.initial_cell in solids:
            issue(
                "agv_not_walkable",
                "AGV initial cell must be walkable.",
                agv.agv_id,
                "initial_cell",
            )
        if agv.initial_cell in agv_cells:
            issue(
                "agv_overlap",
                "Two AGVs share the same initial cell.",
                agv.agv_id,
                "initial_cell",
            )
        agv_cells.add(agv.initial_cell)
    return tuple(issues)
