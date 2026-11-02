"""Static geometry and physical eligibility of the four canonical candidates."""

from collections import deque

import pytest

from smartsom.config.base_diagnostic import validate_base_factory
from smartsom.config.map_catalog import load_map
from smartsom.domain.factory_design import occupied_cells, validate_factory_design


@pytest.mark.parametrize(
    "name,shape,count",
    [
        ("small-factory", (22, 6), 8),
        ("medium-factory", (27, 12), 20),
        ("small-wide-factory", (24, 6), 8),
        ("medium-wide-factory", (29, 19), 20),
    ],
)
def test_geometry_ports_and_capacities(name, shape, count):
    factory = load_map(name).factory
    assert (factory.grid.width, factory.grid.height) == shape
    assert len(factory.machines) == len(factory.agvs) == count
    assert not validate_factory_design(factory)
    validate_base_factory(factory)
    assert set(factory.operation_types) == set("ABCD")
    assert all(
        (m.footprint.width, m.footprint.height) == (1, 1) for m in factory.machines
    )
    assert all(b.storage.capacity == 4 for b in factory.buffers if b.machine_id)
    solids = {
        tuple((c.x, c.y))
        for r in (*factory.machines, *factory.buffers)
        for c in occupied_cells(r.footprint)
    }
    free = {(x, y) for y in range(shape[1]) for x in range(shape[0])} - solids
    pending = deque([next(iter(free))])
    reached = set()
    while pending:
        cell = pending.popleft()
        if cell in reached or cell not in free:
            continue
        reached.add(cell)
        x, y = cell
        pending.extend(((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)))
    assert reached == free
    assert {(p.cell.x, p.cell.y) for p in factory.ports} <= reached
    assert {(a.initial_cell.x, a.initial_cell.y) for a in factory.agvs} <= reached
