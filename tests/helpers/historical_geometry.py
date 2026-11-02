"""Test-only typed worlds for legacy editor features and frozen trace evidence.

These construct resources and access relationships directly; they are not factory
files, catalog entries, or serialized copies of retired YAML presets.
"""

from dataclasses import replace

from smartsom.domain.factory_design import (
    AGVDesign,
    BatteryDesign,
    BufferDesign,
    BufferSlotTarget,
    BufferTarget,
    Cell,
    ChargerDesign,
    ChargerTarget,
    FactoryDesign,
    Footprint,
    GridDesign,
    InspectionSlotTarget,
    InspectionStationDesign,
    MachineDesign,
    PoolStorage,
    PortBinding,
    PortDesign,
    ScrapBinDesign,
    ScrapBinTarget,
    SlotDesign,
    SlotStorage,
)
from smartsom.domain.quality import QualityMode

BOTH = ("pickup", "drop_off")
NORMAL = (QualityMode("normal", "1", "0"),)
MODES = (
    QualityMode("slow", "1.2", ".01"),
    QualityMode("normal", "1", ".018"),
    QualityMode("fast", ".8", ".03"),
)


def slots(width, height, capacity=1):
    return tuple(
        SlotDesign(f"slot_{y * width + x + 1:03d}", Cell(x, y), capacity)
        for y in range(height)
        for x in range(width)
    )


def editor_factory(number):
    """Construct bounded editing cases with multi-slot stations and charging."""
    if number == 7:
        return slot_editor_factory()
    if number not in (1, 2, 3):
        raise ValueError(number)
    width, height = (42, 12) if number == 2 else (12, 8 if number == 3 else 12)
    size = 1 if number == 3 else 2
    columns = 2 if number == 1 else 4
    xs = (2, 8) if number == 1 else (8, 16, 24, 32) if number == 2 else (1, 4, 7, 10)
    catalog = tuple(f"operation_{i}" for i in range(1, 9 if number == 2 else 5))
    machines, buffers, ports = [], [], []

    def port(name, x, y, targets, operations=BOTH):
        ports.append(
            PortDesign(
                "pending",
                name,
                Cell(x, y),
                bindings=tuple(PortBinding(target, operations) for target in targets),
            )
        )

    io_width, io_height = (2, 4) if number == 2 else (1, 2)
    io_y = (height - io_height) // 2
    for i, (name, role, x) in enumerate(
        (("Input", "system_input", 0), ("Output", "system_output", width - io_width)), 1
    ):
        owner = f"buffer_{i:03d}"
        buffers.append(
            BufferDesign(
                owner,
                name,
                Footprint(x, io_y, io_width, io_height),
                role,
                PoolStorage(None),
            )
        )
        dock = io_width if i == 1 else width - io_width - 1
        for j, dy in enumerate((0, io_height - 1) if number == 2 else range(io_height)):
            ops = BOTH if number == 3 else (("pickup",) if i == 1 else ("drop_off",))
            port(f"{name} port {j + 1}", dock, io_y + dy, (BufferTarget(owner),), ops)
    for row, y in enumerate((0, height - size)):
        for col, x in enumerate(xs):
            i = row * columns + col + 1
            owner = f"machine_{i:03d}"
            op = i if number in (1, 2) else (i - 1) % 4 + 1
            machines.append(
                MachineDesign(
                    owner,
                    f"Machine {i}",
                    Footprint(x, y, size, size),
                    NORMAL,
                    operation_types=(f"operation_{op}",),
                )
            )
            for side, dx in (("pre", -size), ("post", size)):
                identifier = len(buffers) + (2 if number == 2 else 1)
                buf = f"buffer_{identifier:03d}"
                values = slots(size, size, 4 if number == 3 else 1)
                buffers.append(
                    BufferDesign(
                        buf,
                        f"Machine {i} {side}-buffer",
                        Footprint(x + dx, y, size, size),
                        f"machine_{side}",
                        SlotStorage(values),
                        owner,
                    )
                )
                px = (
                    x + (0 if side == "pre" else 1)
                    if number == 1
                    else x + (-1 if side == "pre" else 2)
                    if number == 2
                    else x + dx
                )
                port(
                    f"Machine {i} {side} port",
                    px,
                    size if row == 0 else height - size - 1,
                    tuple(BufferSlotTarget(buf, s.slot_id) for s in values),
                )
    station_x = (4,) if number == 1 else (14, 24) if number == 2 else (3, 7)
    station_width = 2 if number == 3 else 4
    center_y = (height - 2) // 2
    stations = []
    for i, x in enumerate(station_x, 1):
        owner = f"inspection_station_{i:03d}"
        values = slots(station_width, 2)
        if number == 1:
            coordinates = (
                (0, 0),
                (1, 0),
                (0, 1),
                (1, 1),
                (2, 0),
                (2, 1),
                (3, 0),
                (3, 1),
            )
            values = tuple(
                SlotDesign(f"slot_{j:03d}", Cell(a, b), 1)
                for j, (a, b) in enumerate(coordinates, 1)
            )
        name = (
            "Inspection station"
            if i == 1 and number != 3
            else f"Inspection station {i}"
        )
        stations.append(
            InspectionStationDesign(
                owner, name, Footprint(x, center_y, station_width, 2), values, 2, "max"
            )
        )
        for dx in range(station_width):
            for side, py in (("upper", center_y - 1), ("lower", center_y + 2)):
                matching = [
                    s
                    for s in values
                    if s.local_cell.x == dx
                    and (number == 3 or s.local_cell.y == (side == "lower"))
                ]
                if number == 1:
                    name = f"Inspection port {int(matching[0].slot_id[-3:])}"
                elif number == 2:
                    name = f"Inspection{' 2' if i == 2 else ''} port {int(matching[0].slot_id[-3:])}"
                else:
                    name = f"Inspection {i} column {dx + 1} {side}"
                port(
                    name,
                    x + dx,
                    py,
                    tuple(InspectionSlotTarget(owner, s.slot_id) for s in matching),
                )
    scrap_x = (8, 3) if number == 1 else (20,) if number == 2 else (9, 2)
    scraps = []
    for i, x in enumerate(scrap_x, 1):
        owner = f"scrap_bin_{i:03d}"
        scraps.append(
            ScrapBinDesign(
                owner,
                "Scrap bin" if i == 1 else f"Scrap bin {i}",
                Footprint(x, center_y, 2 if number == 2 else 1, 2),
                None,
            )
        )
        for side, py in (("upper", center_y - 1), ("lower", center_y + 2)):
            for dx in range(2 if number == 2 else 1):
                if number == 1:
                    name = f"{'Scrap' if i == 1 else 'Scrap bin 2'} port {1 if side == 'upper' else 3}"
                elif number == 2:
                    name = f"Scrap port {dx + (1 if side == 'upper' else 3)}"
                else:
                    name = f"Scrap bin{' 2' if i == 2 else ''} {side}"
                port(name, x + dx, py, (ScrapBinTarget(owner),), ("drop_off",))
    chargers = []
    charger_cells = (
        ((3, 0, 180), (38, 0, 180), (3, 11, 0), (38, 11, 0))
        if number == 2
        else (
            (0, io_y - 1, 90),
            (0, io_y + 2, 90),
            (11, io_y - 1, 270),
            (11, io_y + 2, 270),
        )
    )
    for i, (x, y, rotation) in enumerate(charger_cells, 1):
        owner = f"charger_{i:03d}"
        chargers.append(
            ChargerDesign(
                owner, f"Charger {i}", Footprint(x, y, 1, 1, rotation), 1, "10"
            )
        )
        dx, dy = (0, 1 if y == 0 else -1) if number == 2 else (1 if x == 0 else -1, 0)
        name = f"Charger port {i}" if number == 1 else f"Charger {i} port"
        port(name, x + dx, y + dy, (ChargerTarget(owner),), ("charge",))
    ports.sort(key=lambda p: (p.cell.y, p.cell.x))
    # Template 1 predates row-major IDs; gesture tests depend on those IDs.
    identifiers = (
        (
            2,
            3,
            4,
            5,
            1,
            25,
            7,
            8,
            23,
            31,
            9,
            29,
            6,
            11,
            14,
            17,
            22,
            27,
            15,
            16,
            24,
            32,
            12,
            30,
            18,
            19,
            20,
            21,
        )
        if number == 1
        else range(1, len(ports) + 1)
    )
    ports = tuple(
        replace(p, port_id=f"port_{i:03d}")
        for i, p in zip(identifiers, ports, strict=True)
    )
    start_x = (3, 38) if number == 2 else (3, 8) if number == 1 else (2, 9)
    start_y = (1, 10) if number == 2 else (io_y - 1, io_y + 2)
    agvs = tuple(
        AGVDesign(
            f"agv_{i:03d}",
            f"AGV {i}",
            Cell(x, y),
            initial_heading=("north" if y == 1 else "south")
            if number == 2
            else ("east" if x == start_x[0] else "west"),
            battery=BatteryDesign("100", "100", "1", "0"),
        )
        for i, (y, x) in enumerate(((y, x) for y in start_y for x in start_x), 1)
    )
    return FactoryDesign(
        f"factory_{number:03d}",
        f"Template {number}",
        GridDesign(width, height),
        tuple(machines),
        tuple(buffers),
        tuple(stations),
        tuple(scraps),
        tuple(chargers),
        ports,
        agvs,
        operation_types=catalog,
    )


def slot_editor_factory():
    """Eight unit machines with slot buffers and automatic inspection disposal."""
    machines = []
    buffers = []
    ports = []

    def access(name, x, y, target, ops=BOTH):
        ports.append(
            PortDesign(
                "pending", name, Cell(x, y), bindings=(PortBinding(target, ops),)
            )
        )

    for i, (x, name, role, op) in enumerate(
        (
            (0, "Input", "system_input", "pickup"),
            (18, "Output", "system_output", "drop_off"),
        ),
        1,
    ):
        owner = f"buffer_{i:03d}"
        buffers.append(
            BufferDesign(owner, name, Footprint(x, 0, 1, 9), role, PoolStorage(None))
        )
        for j, y in enumerate((2, 6), 1):
            access(
                f"{name} port {j}", 1 if i == 1 else 17, y, BufferTarget(owner), (op,)
            )
    positions = [(x, y) for y in (1, 4, 7) for x in (4, 9, 14) if (x, y) != (9, 4)]
    for i, (x, y) in enumerate(positions, 1):
        owner = f"machine_{i:03d}"
        op = min(i, 9 - i)
        machines.append(
            MachineDesign(
                owner,
                f"Machine {i} · O{op}",
                Footprint(x, y, 1, 1),
                MODES,
                operation_types=(f"operation_{op}",),
            )
        )
        for side, dx in (("pre", -1), ("post", 1)):
            buf = f"buffer_{len(buffers) + 1:03d}"
            role = f"machine_{side}"
            buffers.append(
                BufferDesign(
                    buf,
                    f"{owner} {role}",
                    Footprint(x + dx, y, 1, 1),
                    role,
                    SlotStorage(slots(1, 1, 4)),
                    owner,
                )
            )
            for dy in (-1, 1):
                access(
                    f"{buf} access", x + dx, y + dy, BufferSlotTarget(buf, "slot_001")
                )
    stations = tuple(
        InspectionStationDesign(
            f"inspection_station_{i:03d}",
            f"Inspection {i}",
            Footprint(x, 4, 1, 1),
            slots(1, 1, 4),
            2,
            "max",
            "scrap_bin_001",
        )
        for i, x in enumerate((8, 10), 1)
    )
    for s in stations:
        for dy in (-1, 1):
            access(
                f"{s.inspection_station_id} access",
                s.footprint.x,
                4 + dy,
                InspectionSlotTarget(s.inspection_station_id, "slot_001"),
            )
    ports = tuple(
        replace(p, port_id=f"port_{i:03d}")
        for i, p in enumerate(sorted(ports, key=lambda p: (p.cell.y, p.cell.x)), 1)
    )
    agvs = tuple(
        AGVDesign(
            f"agv_{i:03d}",
            f"AGV {i}",
            Cell(x, y),
            initial_heading="east" if x < 9 else "west",
        )
        for i, (y, x) in enumerate(((y, x) for y in (2, 6) for x in (2, 7, 11, 16)), 1)
    )
    return FactoryDesign(
        "factory_007",
        "Template 7 · Small",
        GridDesign(19, 9),
        tuple(machines),
        tuple(buffers),
        stations,
        (ScrapBinDesign("scrap_bin_001", "Scrap bin 1", Footprint(9, 4, 1, 1), None),),
        ports=ports,
        agvs=agvs,
        operation_types=tuple(f"operation_{i}" for i in range(1, 5)),
    )


def historical_base_factory():
    """Original 24x11 world required by frozen cloud/server trace assertions."""
    machines = []
    vehicles = []
    ports = []
    buffers = [
        BufferDesign(
            "input", "Input", Footprint(0, 0, 1, 11), "system_input", PoolStorage(None)
        ),
        BufferDesign(
            "output",
            "Output",
            Footprint(23, 0, 1, 11),
            "system_output",
            PoolStorage(None),
        ),
    ]

    def port(owner, x, y, operation, suffix):
        ports.append(
            PortDesign(
                f"{owner}_{suffix}",
                f"{owner} {suffix}",
                Cell(x, y),
                bindings=(PortBinding(BufferTarget(owner), (operation,)),),
            )
        )

    for row, y in enumerate((3, 7)):
        port("input", 1, y, "pickup", str(row))
        port("output", 22, y, "drop_off", str(row))
        for col, x in enumerate((4, 9, 14, 19)):
            i = row * 4 + col
            owner = f"machine_{i:02d}"
            machines.append(
                MachineDesign(
                    owner,
                    owner,
                    Footprint(x, y, 1, 1),
                    MODES,
                    operation_types=("ABCD"[(col + row) % 4],),
                )
            )
            vehicles.append(AGVDesign(f"agv_{i:02d}", f"AGV {i}", Cell(x, y - 2)))
            for side, dx, op in (("pre", -1, "drop_off"), ("post", 1, "pickup")):
                buf = f"{owner}_{side}"
                buffers.append(
                    BufferDesign(
                        buf,
                        buf,
                        Footprint(x + dx, y, 1, 1),
                        f"machine_{side}",
                        PoolStorage(4),
                        owner,
                    )
                )
                port(buf, x + dx, y - 1, op, "up")
                port(buf, x + dx, y + 1, op, "down")
                port(buf, x + 2 * dx, y, op, "side")
    return FactoryDesign(
        "base_small",
        "Base small",
        GridDesign(24, 11),
        tuple(machines),
        tuple(buffers),
        ports=tuple(ports),
        agvs=tuple(vehicles),
        operation_types=tuple("ABCD"),
    )


def historical_playback_factory():
    """Historical rendering uses the compact multi-slot editor case."""
    return editor_factory(1)


def editor_factory_1():
    return editor_factory(1)


def editor_factory_2():
    return editor_factory(2)
