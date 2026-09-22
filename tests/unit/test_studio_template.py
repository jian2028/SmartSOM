"""Independent acceptance for the approved, framework-free built-in factory templates."""

import importlib.resources
import os
import subprocess
import sys
from collections import Counter, deque
from decimal import Decimal
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
RESOURCE_IDS = {
    "machines": "machine_id",
    "buffers": "buffer_id",
    "inspection_stations": "inspection_station_id",
    "scrap_bins": "scrap_bin_id",
    "chargers": "charger_id",
    "ports": "port_id",
    "agvs": "agv_id",
}
SOLID_GROUPS = tuple(key for key in RESOURCE_IDS if key not in {"ports", "agvs"})
BOTH = {"pickup", "drop_off"}


@pytest.fixture(scope="module")
def template():
    resource = importlib.resources.files("smartsom").joinpath(
        "studio", "templates", "template_002.yaml"
    )
    return yaml.safe_load(resource.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def factory(template):
    return template["factory"]


@pytest.fixture(scope="module")
def compact_factory():
    resource = importlib.resources.files("smartsom").joinpath(
        "studio", "templates", "template_001.yaml"
    )
    return yaml.safe_load(resource.read_text(encoding="utf-8"))["factory"]


def cell(value):
    return value["x"], value["y"]


def rectangle(x, y, width, height):
    return {(a, b) for a in range(x, x + width) for b in range(y, y + height)}


def occupied(resource):
    value = resource["footprint"]
    # Every non-square footprint in this fixed template has rotation zero.
    assert value["rotation"] == 0 or value["width"] == value["height"] == 1
    return rectangle(value["x"], value["y"], value["width"], value["height"])


def bound_ports(factory, target_key, target_id):
    return [
        port
        for port in factory["ports"]
        if any(
            binding["target"].get(target_key) == target_id
            for binding in port["bindings"]
        )
    ]


def assert_slots(slots, width, height):
    assert len(slots) == width * height
    assert {slot["slot_id"] for slot in slots} == {
        f"slot_{index:03d}" for index in range(1, width * height + 1)
    }
    assert {cell(slot["local_cell"]) for slot in slots} == rectangle(
        0, 0, width, height
    )
    assert all(slot["capacity"] == 1 for slot in slots)


def test_template_identity_counts_and_stable_row_major_port_ids(template, factory):
    assert template["schema"] == "smartsom.factory/v2"
    assert factory["factory_id"] == "factory_002"
    assert factory["name"] == "Template 2"
    assert factory["grid"] == {"width": 42, "height": 12, "blocked_cells": []}
    assert {key: len(factory[key]) for key in RESOURCE_IDS} == {
        "machines": 8,
        "buffers": 18,
        "inspection_stations": 2,
        "scrap_bins": 1,
        "chargers": 4,
        "ports": 44,
        "agvs": 4,
    }
    identifiers = [
        item[id_key]
        for group, id_key in RESOURCE_IDS.items()
        for item in factory[group]
    ]
    assert len(identifiers) == len(set(identifiers))
    ordered = sorted(
        factory["ports"], key=lambda item: (item["cell"]["y"], item["cell"]["x"])
    )
    assert [port["port_id"] for port in ordered] == [
        f"port_{index:03d}" for index in range(1, 45)
    ]
    assert all(
        set(port["allowed_headings"]) == {"north", "east", "south", "west"}
        for port in factory["ports"]
    )


def test_template_geometry_is_symmetric_clear_and_all_368_free_cells_connect(factory):
    footprints = [occupied(item) for group in SOLID_GROUPS for item in factory[group]]
    solid = set().union(*footprints)
    assert sum(map(len, footprints)) == len(solid) == 136
    expected = rectangle(0, 4, 2, 4) | rectangle(40, 4, 2, 4)
    expected |= rectangle(14, 5, 4, 2) | rectangle(24, 5, 4, 2)
    expected |= rectangle(20, 5, 2, 2)
    expected |= {(3, 0), (38, 0), (3, 11), (38, 11)}
    for y in (0, 10):
        for x in (6, 14, 22, 30):
            expected |= rectangle(x, y, 6, 2)
    assert solid == expected
    assert {(41 - x, y) for x, y in solid} == solid
    assert {(x, 11 - y) for x, y in solid} == solid

    ports = {cell(port["cell"]) for port in factory["ports"]}
    assert len(ports) == 44
    assert ports.isdisjoint(solid)
    assert {(41 - x, y) for x, y in ports} == ports
    assert {(x, 11 - y) for x, y in ports} == ports
    starts = {cell(agv["initial_cell"]) for agv in factory["agvs"]}
    assert len(starts) == 4 and starts <= ports

    traversable = rectangle(0, 0, 42, 12) - solid
    assert len(traversable) == 368
    assert ports <= traversable
    seen = {next(iter(starts))}
    pending = deque(seen)
    while pending:
        x, y = pending.popleft()
        for neighbor in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            if neighbor in traversable and neighbor not in seen:
                seen.add(neighbor)
                pending.append(neighbor)
    assert seen == traversable


def test_each_machine_has_two_four_slot_buffers_and_explicit_two_way_ports(factory):
    machines = {item["machine_id"]: item for item in factory["machines"]}
    buffers = {item["buffer_id"]: item for item in factory["buffers"]}
    for number in range(1, 9):
        machine_id = f"machine_{number:03d}"
        start_x = (6, 14, 22, 30)[(number - 1) % 4]
        y, port_y = (0, 2) if number <= 4 else (10, 9)
        machine = machines[machine_id]
        assert occupied(machine) == rectangle(start_x + 2, y, 2, 2)
        assert len(machine["quality_modes"]) == 1
        normal = machine["quality_modes"][0]
        assert normal["quality_mode_id"] == "normal"
        assert Decimal(str(normal["time_scale"])) == 1
        assert Decimal(str(normal["error_rate"])) == 0
        for offset, role, x, port_x in (
            (0, "machine_pre", start_x, start_x + 1),
            (1, "machine_post", start_x + 4, start_x + 4),
        ):
            buffer_id = f"buffer_{4 + 2 * (number - 1) + offset:03d}"
            buffer = buffers[buffer_id]
            assert buffer["role"] == role and buffer["machine_id"] == machine_id
            assert occupied(buffer) == rectangle(x, y, 2, 2)
            assert buffer["storage"]["mode"] == "slots"
            assert_slots(buffer["storage"]["slots"], 2, 2)
            matched = bound_ports(factory, "buffer_id", buffer_id)
            assert len(matched) == 1
            assert cell(matched[0]["cell"]) == (port_x, port_y)
            bindings = matched[0]["bindings"]
            assert len(bindings) == 4
            assert {b["target"]["slot_id"] for b in bindings} == {
                slot["slot_id"] for slot in buffer["storage"]["slots"]
            }
            assert all(b["target"]["kind"] == "buffer_slot" for b in bindings)
            assert all(set(b["operations"]) == BOTH for b in bindings)


@pytest.mark.parametrize(
    "number,role,x,port_x,operation",
    [(1, "system_input", 0, 2, "pickup"), (2, "system_output", 40, 39, "drop_off")],
)
def test_input_and_output_have_two_pool_ports(
    factory, number, role, x, port_x, operation
):
    buffer_id = f"buffer_{number:03d}"
    buffer = next(b for b in factory["buffers"] if b["buffer_id"] == buffer_id)
    assert buffer["role"] == role and buffer["machine_id"] is None
    assert occupied(buffer) == rectangle(x, 4, 2, 4)
    assert buffer["storage"] == {"mode": "pool", "capacity": None}
    ports = bound_ports(factory, "buffer_id", buffer_id)
    assert {cell(port["cell"]) for port in ports} == {(port_x, 4), (port_x, 7)}
    assert len(ports) == 2
    for port in ports:
        assert port["bindings"] == [
            {
                "target": {"kind": "buffer", "buffer_id": buffer_id},
                "operations": [operation],
            }
        ]


def test_both_inspection_stations_bind_the_nearest_owned_slot(factory):
    assert all(b["buffer_id"] != "buffer_003" for b in factory["buffers"])
    for number, x in ((1, 14), (2, 24)):
        owner_key = "inspection_station_id"
        owner_id = f"inspection_station_{number:03d}"
        item = next(
            s for s in factory["inspection_stations"] if s[owner_key] == owner_id
        )
        slots = item["slots"]
        assert item["inspection_ticks"] == 2
        assert item["parallel_capacity"] == "max"
        assert occupied(item) == rectangle(x, 5, 4, 2)
        assert_slots(slots, 4, 2)
        positions = {
            slot["slot_id"]: (x + slot["local_cell"]["x"], 5 + slot["local_cell"]["y"])
            for slot in slots
        }
        ports = bound_ports(factory, owner_key, owner_id)
        assert len(ports) == 8
        assert {cell(port["cell"]) for port in ports} == {
            (px, py) for px in range(x, x + 4) for py in (4, 7)
        }
        targets = Counter()
        for port in ports:
            assert len(port["bindings"]) == 1
            binding = port["bindings"][0]
            assert binding["target"]["kind"] == "inspection_slot"
            assert set(binding["operations"]) == BOTH
            slot_id = binding["target"]["slot_id"]
            targets[slot_id] += 1
            sx, sy = positions[slot_id]
            px, py = cell(port["cell"])
            assert abs(sx - px) + abs(sy - py) == 1
        assert targets == Counter({slot["slot_id"]: 1 for slot in slots})


def test_scrap_is_drop_only_and_each_charger_has_one_fully_charged_agv(factory):
    scrap = factory["scrap_bins"][0]
    assert scrap["scrap_bin_id"] == "scrap_bin_001"
    assert occupied(scrap) == rectangle(20, 5, 2, 2)
    assert scrap["capacity"] is None
    scrap_ports = bound_ports(factory, "scrap_bin_id", "scrap_bin_001")
    assert {cell(port["cell"]) for port in scrap_ports} == {
        (20, 4),
        (21, 4),
        (20, 7),
        (21, 7),
    }
    assert all(
        port["bindings"]
        == [
            {
                "target": {"kind": "scrap_bin", "scrap_bin_id": "scrap_bin_001"},
                "operations": ["drop_off"],
            }
        ]
        for port in scrap_ports
    )
    agvs = {item["agv_id"]: item for item in factory["agvs"]}
    for index, (x, y, port_y, heading, rotation) in enumerate(
        [
            (3, 0, 1, "north", 180),
            (38, 0, 1, "north", 180),
            (3, 11, 10, "south", 0),
            (38, 11, 10, "south", 0),
        ],
        1,
    ):
        charger_id = f"charger_{index:03d}"
        charger = next(c for c in factory["chargers"] if c["charger_id"] == charger_id)
        assert occupied(charger) == {(x, y)}
        assert charger["footprint"]["rotation"] == rotation
        assert charger["agv_capacity"] == 1
        assert Decimal(str(charger["charge_energy_per_tick"])) == 10
        ports = bound_ports(factory, "charger_id", charger_id)
        assert len(ports) == 1 and cell(ports[0]["cell"]) == (x, port_y)
        assert ports[0]["bindings"] == [
            {
                "target": {"kind": "charger", "charger_id": charger_id},
                "operations": ["charge"],
            }
        ]
        agv = agvs[f"agv_{index:03d}"]
        assert cell(agv["initial_cell"]) == (x, port_y)
        assert agv["initial_heading"] == heading
        assert agv["job_capacity"] == agv["move_cells_per_tick"] == 1
        assert {key: Decimal(str(value)) for key, value in agv["battery"].items()} == {
            "energy_capacity": Decimal(100),
            "initial_energy": Decimal(100),
            "move_energy_per_cell": Decimal(1),
            "idle_energy_per_tick": Decimal(0),
        }


def test_compact_template_geometry_and_side_chargers(compact_factory):
    factory = compact_factory
    assert factory["factory_id"] == "factory_001"
    assert factory["name"] == "Template 1"
    assert factory["grid"] == {"width": 12, "height": 12, "blocked_cells": []}
    assert {key: len(factory[key]) for key in RESOURCE_IDS} == {
        "machines": 4,
        "buffers": 10,
        "inspection_stations": 1,
        "scrap_bins": 2,
        "chargers": 4,
        "ports": 28,
        "agvs": 4,
    }
    assert [m["name"] for m in factory["machines"]] == [
        f"Machine {n}" for n in range(1, 5)
    ]
    assert [m["operation_types"] for m in factory["machines"]] == [
        [f"operation_{n}"] for n in range(1, 5)
    ]
    rectangles = [occupied(r) for group in SOLID_GROUPS for r in factory[group]]
    solid = set().union(*rectangles)
    expected = rectangle(0, 4, 1, 4) | rectangle(11, 4, 1, 4)
    expected |= rectangle(3, 5, 6, 2)
    central = set().union(
        *(occupied(r) for r in factory["inspection_stations"] + factory["scrap_bins"])
    )
    assert central == rectangle(3, 5, 6, 2)
    for x, y in ((0, 0), (6, 0), (0, 10), (6, 10)):
        expected |= rectangle(x, y, 6, 2)
    assert solid == expected and sum(map(len, rectangles)) == len(solid) == 68
    assert {(11 - x, y) for x, y in solid} == solid
    assert {(x, 11 - y) for x, y in solid} == solid
    free = rectangle(0, 0, 12, 12) - solid
    positions = {cell(p["cell"]) for p in factory["ports"]}
    assert len(positions) == 28 and positions <= free
    assert {(11 - x, y) for x, y in positions} == positions
    assert {(x, 11 - y) for x, y in positions} == positions
    assert {p["port_id"] for p in factory["ports"]} == {
        f"port_{n:03d}" for n in range(1, 33) if n not in {10, 13, 26, 28}
    }
    assert factory["ports"] == sorted(
        factory["ports"], key=lambda p: cell(p["cell"])[::-1]
    )
    starts = {cell(a["initial_cell"]) for a in factory["agvs"]}
    assert starts == {(3, 4), (8, 4), (3, 7), (8, 7)} and starts <= free
    assert {(11 - x, y) for x, y in starts} == starts
    assert {(x, 11 - y) for x, y in starts} == starts
    seen, pending = {(1, 4)}, deque([(1, 4)])
    while pending:
        x, y = pending.popleft()
        for neighbor in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            if neighbor in free and neighbor not in seen:
                seen.add(neighbor)
                pending.append(neighbor)
    assert seen == free and len(free) == 76
    for n, x, y, px in ((1, 0, 4, 1), (2, 0, 7, 1), (3, 11, 4, 10), (4, 11, 7, 10)):
        charger = factory["chargers"][n - 1]
        assert occupied(charger) == {(x, y)}
        buffer_cells = set().union(*(occupied(b) for b in factory["buffers"]))
        assert (x, 5 if y == 4 else 6) in buffer_cells
        assert charger["agv_capacity"] == 1
        assert Decimal(charger["charge_energy_per_tick"]) == 10
        matched = bound_ports(factory, "charger_id", f"charger_{n:03d}")
        assert len(matched) == 1 and cell(matched[0]["cell"]) == (px, y)
        assert matched[0]["bindings"][0]["operations"] == ["charge"]


def test_compact_template_machine_buffers_and_terminal_bindings(compact_factory):
    factory = compact_factory
    for number, (x, y) in enumerate(((0, 0), (6, 0), (0, 10), (6, 10)), 1):
        machine_id = f"machine_{number:03d}"
        machine = next(m for m in factory["machines"] if m["machine_id"] == machine_id)
        assert occupied(machine) == rectangle(x + 2, y, 2, 2)
        buffers = [b for b in factory["buffers"] if b["machine_id"] == machine_id]
        assert {b["role"] for b in buffers} == {"machine_pre", "machine_post"}
        for buffer in buffers:
            pre = buffer["role"] == "machine_pre"
            assert occupied(buffer) == rectangle(x + (0 if pre else 4), y, 2, 2)
            assert_slots(buffer["storage"]["slots"], 2, 2)
            matched = bound_ports(factory, "buffer_id", buffer["buffer_id"])
            assert len(matched) == 1
            assert cell(matched[0]["cell"]) == (
                x + (2 if pre else 3),
                2 if y == 0 else 9,
            )
            assert {b["target"]["slot_id"] for b in matched[0]["bindings"]} == {
                s["slot_id"] for s in buffer["storage"]["slots"]
            }
            assert all(set(b["operations"]) == BOTH for b in matched[0]["bindings"])
    for number, role, x, px, operation in (
        (1, "system_input", 0, 1, "pickup"),
        (2, "system_output", 11, 10, "drop_off"),
    ):
        buffer = next(b for b in factory["buffers"] if b["role"] == role)
        assert occupied(buffer) == rectangle(x, 5, 1, 2)
        assert buffer["storage"] == {"mode": "pool", "capacity": None}
        matched = bound_ports(factory, "buffer_id", f"buffer_{number:03d}")
        assert {cell(p["cell"]) for p in matched} == {(px, 5), (px, 6)}
        assert all(p["bindings"][0]["operations"] == [operation] for p in matched)
    station = factory["inspection_stations"][0]
    assert occupied(station) == rectangle(4, 5, 4, 2)
    assert_slots(station["slots"], 4, 2)
    assert station["inspection_ticks"] == 2 and station["parallel_capacity"] == "max"
    matched = bound_ports(factory, "inspection_station_id", "inspection_station_001")
    assert {cell(p["cell"]) for p in matched} == {
        (x, y) for x in (4, 5, 6, 7) for y in (4, 7)
    }
    for p in matched:
        b = p["bindings"][0]
        s = next(s for s in station["slots"] if s["slot_id"] == b["target"]["slot_id"])
        assert b["target"]["kind"] == "inspection_slot"
        assert cell(p["cell"]) == (
            4 + s["local_cell"]["x"],
            4 if s["local_cell"]["y"] == 0 else 7,
        )
        assert set(b["operations"]) == BOTH
    for bin_id, x in (("scrap_bin_001", 8), ("scrap_bin_002", 3)):
        scrap = next(b for b in factory["scrap_bins"] if b["scrap_bin_id"] == bin_id)
        assert occupied(scrap) == rectangle(x, 5, 1, 2)
        assert scrap["capacity"] is None
        matched = bound_ports(factory, "scrap_bin_id", bin_id)
        assert len(matched) == 2
        assert {cell(p["cell"]) for p in matched} == {(x, 4), (x, 7)}
        assert all(p["bindings"][0]["operations"] == ["drop_off"] for p in matched)


@pytest.mark.parametrize("number", [1, 2, 3, 4, 5, 6])
def test_packaged_template_loads_and_validates_without_qt_or_learning_frameworks(
    number,
):
    code = """
import importlib.abc
import importlib.resources
import sys
number = int(sys.argv[1])

class RejectFrameworks(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {
            'PySide6', 'PyQt6', 'numpy', 'gymnasium', 'ray', 'pettingzoo', 'torch'
        }:
            raise AssertionError('unexpected optional import: ' + fullname)

sys.meta_path.insert(0, RejectFrameworks())
from smartsom.config.factory_design import load_factory_design
from smartsom.domain.factory_design import validate_factory_design

resource = importlib.resources.files('smartsom').joinpath(
    'studio', 'templates', f'template_{number:03d}.yaml'
)
with importlib.resources.as_file(resource) as path:
    design, digest = load_factory_design(path)
assert design.factory_id == f'factory_{number:03d}'
assert len(digest) == 64
issues = validate_factory_design(design)
assert not issues, issues
"""
    subprocess.run(
        [sys.executable, "-c", code, str(number)],
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        check=True,
        capture_output=True,
        text=True,
    )


def test_template_3_compact_geometry_capacity_capabilities_and_access():
    from smartsom.config.codec import primitive
    from smartsom.domain.factory_design import validate_factory_design
    from smartsom.studio.templates import load_template_file

    envelope = load_template_file(3)
    design = envelope.factory
    f = primitive(design)
    assert not validate_factory_design(design)
    assert (design.grid.width, design.grid.height) == (12, 8)
    assert design.operation_types == tuple(f"operation_{n}" for n in range(1, 5))
    assert envelope.authoring.operation_catalog_mode == "manual"
    assert Counter(t for m in design.machines for t in m.operation_types) == {
        f"operation_{n}": 2 for n in range(1, 5)
    }
    assert [(m.footprint.x, m.footprint.y) for m in design.machines] == [
        (x, y) for y in (0, 7) for x in (1, 4, 7, 10)
    ]
    for m in design.machines:
        assert (m.footprint.width, m.footprint.height) == (1, 1)
        owned = [b for b in design.buffers if b.machine_id == m.machine_id]
        assert {b.role for b in owned} == {"machine_pre", "machine_post"}
        for b in owned:
            assert (b.footprint.width, b.footprint.height) == (1, 1)
            assert b.footprint.y == m.footprint.y
            assert b.footprint.x == m.footprint.x + (
                -1 if b.role == "machine_pre" else 1
            )
            assert b.storage.mode == "slots" and len(b.storage.slots) == 1
            assert b.storage.slots[0].capacity == 4
            bindings = [
                (p, t)
                for p in design.ports
                for t in p.bindings
                if getattr(t.target, "buffer_id", None) == b.buffer_id
            ]
            assert len(bindings) == 1
            p, binding = bindings[0]
            assert (p.cell.x, p.cell.y) == (
                b.footprint.x,
                1 if b.footprint.y == 0 else 6,
            )
            assert set(binding.operations) == BOTH
            assert binding.target.slot_id == b.storage.slots[0].slot_id
    assert [
        (s.footprint.x, s.footprint.y, s.footprint.width, s.footprint.height)
        for s in design.inspection_stations
    ] == [(3, 3, 2, 2), (7, 3, 2, 2)]
    for station in design.inspection_stations:
        assert len(station.slots) == 4
        for slot in station.slots:
            assert slot.capacity == 1
            accesses = [
                p.cell
                for p in design.ports
                for b in p.bindings
                if getattr(b.target, "inspection_station_id", None)
                == station.inspection_station_id
                and b.target.slot_id == slot.slot_id
            ]
            assert {(c.x, c.y) for c in accesses} == {
                (station.footprint.x + slot.local_cell.x, y) for y in (2, 5)
            }
    assert (
        len(design.agvs) == 4 and len(design.chargers) == 4 and len(design.ports) == 36
    )
    assert {key: len(f[key]) for key in RESOURCE_IDS} == dict(
        machines=8,
        buffers=18,
        inspection_stations=2,
        scrap_bins=2,
        chargers=4,
        ports=36,
        agvs=4,
    )
    solid = set().union(*(occupied(r) for group in SOLID_GROUPS for r in f[group]))
    assert solid == {(11 - x, y) for x, y in solid} == {(x, 7 - y) for x, y in solid}
    free = {(x, y) for x in range(12) for y in range(8)} - solid
    assert {(x, y) for x in (5, 6) for y in range(1, 7)} <= free
    assert {(x, y) for x in range(1, 11) for y in (1, 2, 5, 6)} <= free
    reached, queue = set(), deque([next(iter(free))])
    while queue:
        point = queue.popleft()
        if point not in free or point in reached:
            continue
        reached.add(point)
        x, y = point
        queue.extend(((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)))
    assert reached == free
    port_cells = {(p.cell.x, p.cell.y) for p in design.ports}
    assert port_cells <= free
    assert (
        port_cells
        == {(11 - x, y) for x, y in port_cells}
        == {(x, 7 - y) for x, y in port_cells}
    )
    starts = {(a.initial_cell.x, a.initial_cell.y) for a in design.agvs}
    assert starts == {(2, 2), (9, 2), (2, 5), (9, 5)} <= free
