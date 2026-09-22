"""Geometry, access and multi-entry contracts for the three scale templates."""

from collections import Counter, deque

import pytest

from smartsom.config.factory_design import load_factory_design_file, save_factory_design
from smartsom.domain.factory_design import occupied_cells, validate_factory_design
from smartsom.domain.production import Demand, ProductionScenario, ProductionStep
from smartsom.engine.production import ProductionSimulator
from smartsom.studio.templates import load_template_file


@pytest.fixture(params=[4, 5, 6])
def factory(request):
    return load_template_file(request.param).factory


def test_scale_geometry_symmetry_and_all_service_points_are_connected(factory):
    assert not validate_factory_design(factory)
    width, height = factory.grid.width, factory.grid.height
    n, machines, groups, chargers = {
        "factory_004": (3, 8, 1, 4),
        "factory_005": (5, 20, 4, 7),
        "factory_006": (7, 40, 8, 7),
    }[factory.factory_id]
    assert (width, height) == (5 * n + 4, 3 * n + 2)
    assert len(factory.machines) == len(factory.agvs) == machines
    assert len(factory.inspection_stations) == 2 * groups
    assert len(factory.scrap_bins) == groups
    assert len(factory.chargers) == chargers
    assert Counter(b.role for b in factory.buffers) == {
        "system_input": n,
        "system_output": n,
        "machine_pre": machines,
        "machine_post": machines,
    }
    inspections = {(s.footprint.x, s.footprint.y) for s in factory.inspection_stations}
    assert inspections == {
        (b.footprint.x + dx, b.footprint.y)
        for b in factory.scrap_bins
        for dx in (-1, 1)
    }
    if n > 3:
        assert {(width // 2 + dx, height // 2) for dx in (-1, 0, 1)} <= {
            (c.footprint.x, c.footprint.y) for c in factory.chargers
        }
    solids = [
        {(c.x, c.y) for c in occupied_cells(r.footprint)}
        for group in (
            factory.machines,
            factory.buffers,
            factory.inspection_stations,
            factory.scrap_bins,
            factory.chargers,
        )
        for r in group
    ]
    occupied = set().union(*solids)
    assert sum(map(len, solids)) == len(occupied)
    free = {(x, y) for x in range(width) for y in range(height)} - occupied
    aisle_columns = {x for c in range(n + 1) for x in (1 + 5 * c, 2 + 5 * c)}
    assert {(x, y) for x in aisle_columns for y in range(height)} <= free
    assert {
        (x, y) for x in range(1, width - 1) for y in range(height) if y % 3 != 2
    } <= free
    pending = deque([next(iter(free))])
    reached = set()
    while pending:
        x, y = pending.popleft()
        if (x, y) in reached or (x, y) not in free:
            continue
        reached.add((x, y))
        pending.extend(((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)))
    assert reached == free
    ports = {(p.cell.x, p.cell.y) for p in factory.ports}
    starts = {(a.initial_cell.x, a.initial_cell.y) for a in factory.agvs}
    assert len(starts) == len(factory.agvs)
    assert ports <= free and starts <= free and not starts & ports
    for cells in (occupied, ports, starts, inspections):
        assert cells == {(width - 1 - x, y) for x, y in cells}
        assert cells == {(x, height - 1 - y) for x, y in cells}
    headings = {
        (a.initial_cell.x, a.initial_cell.y): a.initial_heading for a in factory.agvs
    }
    for (x, y), heading in headings.items():
        assert headings[width - 1 - x, y] == {"east": "west", "west": "east"}[heading]
        assert headings[x, height - 1 - y] == heading
    for group in (
        factory.machines,
        factory.inspection_stations,
        factory.scrap_bins,
        factory.chargers,
    ):
        assert all((r.footprint.width, r.footprint.height) == (1, 1) for r in group)
    assert all(
        len(s.slots) == 1 and s.slots[0].capacity == 1
        for s in factory.inspection_stations
    )


def test_scale_buffers_have_the_approved_capacity_and_access(factory):
    for buffer in factory.buffers:
        ports = [
            p
            for p in factory.ports
            if any(
                getattr(b.target, "buffer_id", None) == buffer.buffer_id
                for b in p.bindings
            )
        ]
        assert ports
        if buffer.machine_id:
            assert (buffer.footprint.width, buffer.footprint.height) == (1, 1)
            assert len(buffer.storage.slots) == 1
            assert buffer.storage.slots[0].capacity == 4
            assert {(p.cell.x, p.cell.y) for p in ports} == {
                (buffer.footprint.x, buffer.footprint.y + dy) for dy in (-1, 1)
            }
            assert all(
                set(b.operations) == {"pickup", "drop_off"}
                and b.target.slot_id == buffer.storage.slots[0].slot_id
                for p in ports
                for b in p.bindings
            )
        else:
            expected = "pickup" if buffer.role == "system_input" else "drop_off"
            assert all(b.operations == (expected,) for p in ports for b in p.bindings)
            assert (buffer.footprint.width, buffer.footprint.height) == (1, 3)
            offset = 1 if buffer.role == "system_input" else -1
            assert [(p.cell.x, p.cell.y) for p in ports] == [
                (buffer.footprint.x + offset, buffer.footprint.y + 1)
            ]


def test_scale_capabilities_and_yaml_round_trip(factory, tmp_path):
    assert len(factory.operation_types) == 10
    assert all(len(m.operation_types) == 2 for m in factory.machines)
    assert {t for m in factory.machines for t in m.operation_types} == set(
        factory.operation_types
    )
    path = tmp_path / "factory.yaml"
    save_factory_design(path, factory)
    assert load_factory_design_file(path)[0].factory == factory


def test_multi_entry_demands_arrive_at_their_own_facility(factory):
    inputs = [b.buffer_id for b in factory.buffers if b.role == "system_input"]
    demands = tuple(
        Demand(f"d{i}", (ProductionStep("op", "operation_1", 1),), input_id=bid)
        for i, bid in enumerate(inputs)
    )
    sim = ProductionSimulator(ProductionScenario(factory, demands))
    for demand in demands:
        job = f"{demand.demand_id}/attempt/1"
        assert sim.storage[demand.input_id]["pool"] == [job]
        assert sim.jobs[job]["location"] == demand.input_id
    for buffer in factory.buffers:
        if buffer.role == "system_output":
            assert sim.storage[buffer.buffer_id]["pool"] == []


@pytest.mark.parametrize(
    "input_id,message",
    [
        (None, "input_id required"),
        ("buffer_002", "must identify a system input"),
        ("missing", "must identify a system input"),
    ],
)
def test_multi_entry_rejects_ambiguous_or_invalid_demand_entry(
    factory, input_id, message
):
    demand = Demand("d", (ProductionStep("op", "operation_1", 1),), input_id=input_id)
    with pytest.raises(ValueError, match=message):
        ProductionSimulator(ProductionScenario(factory, (demand,)))
