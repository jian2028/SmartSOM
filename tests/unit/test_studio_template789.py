"""Contracts for the 8/16/32-machine, continuous-I/O template family."""

from collections import Counter, deque
from dataclasses import replace
from decimal import Decimal

import pytest

from smartsom.config.factory_design import load_factory_design_file, save_factory_design
from smartsom.domain.factory_design import occupied_cells, validate_factory_design
from smartsom.studio.templates import load_template_file


@pytest.mark.parametrize(
    "number,width,height,count,agv_count,rows",
    [
        (7, 19, 9, 8, 8, [2, 6]),
        (8, 19, 18, 16, 16, [2, 6, 11, 15]),
        (9, 34, 18, 32, 32, [1, 3, 5, 7, 10, 12, 14, 16]),
        (10, 19, 9, 8, 10, [2, 6]),
        (11, 19, 18, 16, 20, [2, 6, 11, 15]),
        (12, 34, 18, 32, 40, [1, 3, 5, 7, 10, 12, 14, 16]),
    ],
)
def test_scale_template_contract(
    number, width, height, count, agv_count, rows, tmp_path
):
    factory = load_template_file(number).factory
    assert not validate_factory_design(factory)
    assert (factory.grid.width, factory.grid.height) == (width, height)
    assert len(factory.machines) == count
    assert len(factory.agvs) == agv_count
    assert not factory.chargers
    assert len(factory.scrap_bins) == count // 8
    assert all(b.capacity is None for b in factory.scrap_bins)
    assert not any(
        getattr(b.target, "scrap_bin_id", None)
        for p in factory.ports
        for b in p.bindings
    )
    assert len(factory.inspection_stations) == count // 4
    assert Counter(op for m in factory.machines for op in m.operation_types) == {
        f"operation_{i}": count // 4 for i in range(1, 5)
    }
    for m in factory.machines:
        assert len(m.operation_types) == 1
        assert (m.footprint.width, m.footprint.height) == (1, 1)
        assert [
            (q.quality_mode_id, q.time_scale, q.error_rate) for q in m.quality_modes
        ] == [
            ("slow", Decimal("1.2"), Decimal("0.01")),
            ("normal", Decimal("1"), Decimal("0.018")),
            ("fast", Decimal("0.8"), Decimal("0.03")),
        ]
    assert all(a.battery is None and a.job_capacity == 1 for a in factory.agvs)
    assert Counter(b.role for b in factory.buffers) == {
        "system_input": 1,
        "system_output": 1,
        "machine_pre": count,
        "machine_post": count,
    }
    for b in factory.buffers:
        ports = [
            p
            for p in factory.ports
            if any(
                getattr(binding.target, "buffer_id", None) == b.buffer_id
                for binding in p.bindings
            )
        ]
        if b.machine_id:
            assert (b.footprint.width, b.footprint.height) == (1, 1)
            assert len(b.storage.slots) == 1 and b.storage.slots[0].capacity == 4
            assert {(p.cell.x, p.cell.y) for p in ports} == {
                (b.footprint.x, b.footprint.y + dy) for dy in (-1, 1)
            }
            assert all(
                binding.target.slot_id == b.storage.slots[0].slot_id
                and set(binding.operations) == {"pickup", "drop_off"}
                for p in ports
                for binding in p.bindings
            )
        else:
            assert b.storage.capacity is None
            assert (b.footprint.y, b.footprint.width, b.footprint.height) == (
                0,
                1,
                height,
            )
            x = 1 if b.role == "system_input" else width - 2
            assert {(p.cell.x, p.cell.y) for p in ports} == {(x, y) for y in rows}
            operation = "pickup" if b.role == "system_input" else "drop_off"
            assert all(
                binding.operations == (operation,)
                for p in ports
                for binding in p.bindings
            )
    for s in factory.inspection_stations:
        assert (s.footprint.width, s.footprint.height) == (1, 1)
        assert s.inspection_ticks == 2 and s.parallel_capacity == "max"
        assert {(v.local_cell.x, v.local_cell.y, v.capacity) for v in s.slots} == {
            (0, 0, 4)
        }
        ports = [
            p
            for p in factory.ports
            if any(
                getattr(b.target, "inspection_station_id", None)
                == s.inspection_station_id
                for b in p.bindings
            )
        ]
        assert {(p.cell.x, p.cell.y) for p in ports} == {
            (s.footprint.x, s.footprint.y + dy) for dy in (-1, 1)
        }
        assert all(
            {b.target.slot_id for b in p.bindings} == {v.slot_id for v in s.slots}
            for p in ports
        )
    resources = (
        *factory.machines,
        *factory.buffers,
        *factory.inspection_stations,
        *factory.scrap_bins,
    )
    solids = [{(c.x, c.y) for c in occupied_cells(r.footprint)} for r in resources]
    occupied = set().union(*solids)
    assert len(occupied) == sum(map(len, solids))
    free = {(x, y) for x in range(width) for y in range(height)} - occupied
    ports = {(p.cell.x, p.cell.y) for p in factory.ports}
    starts = {(a.initial_cell.x, a.initial_cell.y) for a in factory.agvs}
    assert len(ports) == len(factory.ports) and len(starts) == agv_count
    assert ports <= free and starts <= free and not ports & starts
    for cells in (occupied, ports, starts):
        assert cells == {(width - 1 - x, y) for x, y in cells}
        assert cells == {(x, height - 1 - y) for x, y in cells}
    pending, reached = deque([next(iter(free))]), set()
    while pending:
        x, y = pending.popleft()
        if (x, y) in reached or (x, y) not in free:
            continue
        reached.add((x, y))
        pending.extend(((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)))
    assert reached == free
    path = tmp_path / "roundtrip.yaml"
    save_factory_design(path, factory)
    assert load_factory_design_file(path)[0].factory == factory


@pytest.mark.parametrize("original,variant", [(7, 10), (8, 11), (9, 12)])
def test_extra_agv_templates_preserve_original_map(original, variant):
    source = load_template_file(original)
    copy = load_template_file(variant)
    assert copy.authoring == source.authoring
    assert copy.reliability == source.reliability
    count = len(source.factory.agvs)
    assert copy.factory.agvs[:count] == source.factory.agvs
    assert (
        replace(
            copy.factory,
            factory_id=source.factory.factory_id,
            name=source.factory.name,
            agvs=source.factory.agvs,
        )
        == source.factory
    )
    reference = source.factory.agvs[0]
    for agv in copy.factory.agvs[count:]:
        assert (
            replace(
                agv,
                agv_id=reference.agv_id,
                name=reference.name,
                initial_cell=reference.initial_cell,
                initial_heading=reference.initial_heading,
            )
            == reference
        )
