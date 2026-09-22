from dataclasses import FrozenInstanceError, replace
from decimal import Decimal, localcontext

import pytest

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
    MachineTarget,
    PoolStorage,
    PortBinding,
    PortDesign,
    ScrapBinDesign,
    ScrapBinTarget,
    SlotDesign,
    SlotStorage,
    entity_id,
    iter_resources,
    occupied_cells,
    target_cell,
    target_owner_id,
    validate_factory_design,
    world_cell,
)
from smartsom.domain.quality import QualityMode


def machine(key="M", x=2, y=2):
    return MachineDesign(key, "加工设备", Footprint(x, y, 2, 2))


def design(**kwargs):
    return FactoryDesign("F", "工厂", GridDesign(22, 10), **kwargs)


def codes(data, severity=None):
    return {
        issue.code
        for issue in validate_factory_design(data)
        if severity is None or issue.severity == severity
    }


def populated_design():
    slot = SlotDesign("s", Cell(1, 0))
    return design(
        machines=(replace(machine(), operation_types=("operation_1",)),),
        operation_types=("operation_1",),
        buffers=(
            BufferDesign(
                "P", "前置池", Footprint(5, 2, 2, 2), "machine_pre", PoolStorage(4), "M"
            ),
            BufferDesign(
                "B", "存储格", Footprint(8, 2, 2, 2), storage=SlotStorage((slot,))
            ),
        ),
        inspection_stations=(
            InspectionStationDesign("I", "检验区", Footprint(11, 2, 2, 2), (slot,)),
        ),
        scrap_bins=(ScrapBinDesign("S", "废品接收站", Footprint(14, 2, 2, 2)),),
        chargers=(ChargerDesign("C", "充电点", Footprint(17, 2, 2, 2)),),
        ports=(
            PortDesign(
                "port",
                "多目标交互点",
                Cell(2, 5),
                bindings=(
                    PortBinding(MachineTarget("M")),
                    PortBinding(BufferTarget("P")),
                    PortBinding(BufferSlotTarget("B", "s")),
                    PortBinding(InspectionSlotTarget("I", "s")),
                    PortBinding(ScrapBinTarget("S"), ("drop_off",)),
                    PortBinding(ChargerTarget("C"), ("charge",)),
                ),
            ),
        ),
        agvs=(AGVDesign("A", "AGV", Cell(2, 5)),),
    )


def test_empty_design_and_complete_design_are_valid():
    assert validate_factory_design(design()) == ()
    assert validate_factory_design(populated_design()) == ()
    assert codes(design(machines=(machine(),))) == {"unspecified_machine_capability"}


def test_machine_categories_are_shared_capabilities_not_resource_ids():
    categories = ["operation_1", "operation_2", "milling"]
    first = replace(machine("M1"), operation_types=categories)
    second = replace(machine("M2", x=6), operation_types=("operation_1",))
    categories.clear()
    assert first.operation_types == ("operation_1", "operation_2", "milling")
    assert (
        validate_factory_design(
            design(
                machines=(first, second),
                operation_types=("operation_1", "operation_2", "milling"),
            )
        )
        == ()
    )
    assert machine().operation_types == ()


@pytest.mark.parametrize(
    "categories", [("operation_1", "operation_1"), ("bad id",), (1,), "operation_1"]
)
def test_machine_rejects_invalid_processing_categories(categories):
    with pytest.raises(ValueError):
        replace(machine(), operation_types=categories)


def test_collections_are_detached_and_models_are_frozen():
    cells = [Cell(0, 0)]
    machines = [machine()]
    original = FactoryDesign("F", "Factory", GridDesign(20, 10, cells), machines)
    cells.clear()
    machines.clear()
    assert original.grid.blocked_cells == (Cell(0, 0),)
    assert original.machines == (machine(),)
    with pytest.raises(FrozenInstanceError):
        original.name = "changed"
    with pytest.raises(FrozenInstanceError):
        original.machines[0].footprint.width = 99


def test_current_rotated_geometry_is_not_rotated_twice():
    footprint = Footprint(7, 3, 2, 3, rotation=90)
    assert occupied_cells(footprint) == (
        Cell(7, 3),
        Cell(8, 3),
        Cell(7, 4),
        Cell(8, 4),
        Cell(7, 5),
        Cell(8, 5),
    )
    assert world_cell(footprint, Cell(1, 2)) == Cell(8, 5)
    buffer = BufferDesign(
        "B", "Buffer", footprint, storage=SlotStorage((SlotDesign("s", Cell(1, 2)),))
    )
    data = design(buffers=(buffer,))
    assert target_cell(data, BufferSlotTarget("B", "s")) == Cell(8, 5)
    assert codes(data) == set()


def test_resource_helpers_include_ports_and_agvs_and_resolve_typed_targets():
    data = populated_design()
    assert tuple(entity_id(r) for r in iter_resources(data)) == (
        "M",
        "P",
        "B",
        "I",
        "S",
        "C",
        "port",
        "A",
    )
    cases = (
        (MachineTarget("M"), "M", Cell(3, 3)),
        (BufferTarget("P"), "P", Cell(6, 3)),
        (BufferSlotTarget("B", "s"), "B", Cell(9, 2)),
        (InspectionSlotTarget("I", "s"), "I", Cell(12, 2)),
        (ScrapBinTarget("S"), "S", Cell(15, 3)),
        (ChargerTarget("C"), "C", Cell(18, 3)),
    )
    for target, owner, cell in cases:
        assert target_owner_id(target) == owner
        assert target_cell(data, target) == cell
    for target in (
        MachineTarget("unknown"),
        BufferTarget("B"),
        BufferSlotTarget("P", "s"),
        BufferSlotTarget("B", "missing"),
    ):
        assert target_cell(data, target) is None


@pytest.mark.parametrize(
    "identifier",
    ["", "  ", "9abc", "中文", "équipement", "A.B", "A/B", "A B", True, None, 1],
)
def test_design_identifiers_are_strict_ascii(identifier):
    with pytest.raises(ValueError):
        MachineDesign(identifier, "Machine", Footprint(0, 0, 1, 1))


@pytest.mark.parametrize("identifier", ["a", "Machine_01", "agv-02", "Z9"])
def test_design_identifiers_accept_documented_characters(identifier):
    assert MachineTarget(identifier).machine_id == identifier


@pytest.mark.parametrize(
    "make",
    [
        lambda: Cell(True, 0),
        lambda: Cell(1.5, 0),
        lambda: GridDesign(0, 3),
        lambda: GridDesign(3, False),
        lambda: GridDesign(3, 3, ("0,0",)),
        lambda: Footprint(0, 0, -1, 1),
        lambda: Footprint(0, 0, 1, 1, 45),
        lambda: Footprint(0, 0, 1, 1, False),
        lambda: PoolStorage(-1),
        lambda: PoolStorage(True),
        lambda: PoolStorage(1, mode="slots"),
        lambda: SlotStorage((), mode="pool"),
        lambda: SlotDesign("s", (0, 0)),
        lambda: SlotDesign("s", Cell(0, 0), 0),
        lambda: MachineTarget("M", kind="buffer"),
        lambda: MachineDesign("M", "", Footprint(0, 0, 1, 1)),
        lambda: MachineDesign("M", "M", Footprint(0, 0, 1, 1), ()),
        lambda: MachineDesign(
            "M",
            "M",
            Footprint(0, 0, 1, 1),
            (QualityMode("q", 1, 0), QualityMode("q", 2, 0.1)),
        ),
        lambda: BufferDesign("B", "B", Footprint(0, 0, 1, 1), role="pre"),
        lambda: InspectionStationDesign(
            "I", "I", Footprint(0, 0, 1, 1), inspection_ticks=0
        ),
        lambda: InspectionStationDesign(
            "I", "I", Footprint(0, 0, 1, 1), parallel_capacity=0
        ),
        lambda: InspectionStationDesign(
            "I", "I", Footprint(0, 0, 1, 1), (SlotDesign("s", Cell(0, 0), 2),)
        ),
        lambda: ChargerDesign("C", "C", Footprint(0, 0, 1, 1), agv_capacity=0),
        lambda: ChargerDesign(
            "C", "C", Footprint(0, 0, 1, 1), charge_energy_per_tick=0
        ),
        lambda: AGVDesign("A", "A", Cell(0, 0), job_capacity=0),
        lambda: AGVDesign("A", "A", Cell(0, 0), move_cells_per_tick=0.5),
        lambda: AGVDesign("A", "A", Cell(0, 0), initial_heading="up"),
        lambda: AGVDesign("A", "A", Cell(0, 0), battery={}),
        lambda: PortDesign("P", "P", Cell(0, 0), allowed_headings=()),
        lambda: PortDesign("P", "P", Cell(0, 0), allowed_headings=("east", "east")),
        lambda: PortBinding(MachineTarget("M"), ()),
        lambda: PortBinding(MachineTarget("M"), ("fly",)),
        lambda: PortBinding(MachineTarget("M"), ("pickup", "pickup")),
        lambda: design(machines=("M",)),
    ],
)
def test_malformed_local_fields_fail_before_a_design_can_be_created(make):
    with pytest.raises(ValueError):
        make()


@pytest.mark.parametrize("value", ["NaN", "Infinity", "-Infinity", True, "bad", -1])
def test_battery_rejects_nonfinite_or_negative_energy(value):
    with pytest.raises(ValueError):
        BatteryDesign(initial_energy=value)


def test_battery_decimal_normalization_and_bounds_are_exact():
    with localcontext() as context:
        context.prec = 2
        battery = BatteryDesign("123.45000", "123.450", "0.0012300", "-0.00")
    assert battery.energy_capacity.as_tuple() == Decimal("123.45").as_tuple()
    assert battery.move_energy_per_cell.as_tuple() == Decimal("0.00123").as_tuple()
    assert battery.idle_energy_per_tick.as_tuple() == Decimal(0).as_tuple()
    with pytest.raises(ValueError, match="exceeds"):
        BatteryDesign(energy_capacity=10, initial_energy=11)
    assert AGVDesign("A", "A", Cell(0, 0)).battery is None
    assert ScrapBinDesign("S", "S", Footprint(0, 0, 1, 1)).capacity is None
    assert PoolStorage().capacity == 4


def test_footprint_obstacles_ports_and_initial_agv_occupancy():
    data = replace(
        design(
            machines=(machine(), machine("N", 3, 3), machine("Outside", 21, 9)),
            ports=(PortDesign("P", "P", Cell(2, 2)), PortDesign("Q", "Q", Cell(2, 2))),
            agvs=(AGVDesign("A", "A", Cell(2, 2)), AGVDesign("B", "B", Cell(2, 2))),
        ),
        grid=GridDesign(22, 10, (Cell(2, 2), Cell(2, 2), Cell(22, 10))),
    )
    assert codes(data, "error") == {
        "resource_overlap",
        "footprint_out_of_bounds",
        "blocked_overlap",
        "port_not_walkable",
        "port_overlap",
        "agv_not_walkable",
        "agv_overlap",
        "blocked_out_of_bounds",
        "duplicate_blocked_cell",
    }
    assert codes(
        design(
            ports=(PortDesign("P", "P", Cell(-1, 0)),),
            agvs=(AGVDesign("A", "A", Cell(22, 0)),),
        ),
        "error",
    ) == {"port_out_of_bounds", "agv_out_of_bounds"}


def test_global_resource_identity_includes_ports_and_agvs_but_slots_are_local():
    data = populated_design()
    assert "duplicate_entity_id" not in codes(data)
    assert "duplicate_entity_id" in codes(
        replace(data, agvs=(AGVDesign("M", "A", Cell(0, 0)),))
    )
    assert "duplicate_entity_id" in codes(
        replace(data, ports=(PortDesign("M", "P", Cell(0, 0)),))
    )


def test_buffer_roles_ownership_allow_multiple_system_facilities():
    buffers = (
        BufferDesign("B1", "B1", Footprint(0, 0, 1, 1), "machine_pre"),
        BufferDesign("B2", "B2", Footprint(1, 0, 1, 1), "machine_pre", machine_id="M"),
        BufferDesign("B3", "B3", Footprint(2, 0, 1, 1), "machine_pre", machine_id="M"),
        BufferDesign(
            "B4", "B4", Footprint(3, 0, 1, 1), "machine_post", machine_id="unknown"
        ),
        BufferDesign("B5", "B5", Footprint(4, 0, 1, 1), "storage", machine_id="M"),
        BufferDesign("B6", "B6", Footprint(5, 0, 1, 1), "system_input"),
        BufferDesign("B7", "B7", Footprint(6, 0, 1, 1), "system_input"),
        BufferDesign("B8", "B8", Footprint(7, 0, 1, 1), "system_output"),
        BufferDesign("B9", "B9", Footprint(8, 0, 1, 1), "system_output"),
    )
    data = design(machines=(machine(),), buffers=buffers)
    assert codes(data, "error") == {
        "duplicate_machine_buffer",
        "unknown_machine",
        "unexpected_machine_owner",
    }
    assert codes(data, "warning") == {
        "orphan_machine_buffer",
        "unspecified_machine_capability",
    }


def test_slot_geometry_duplicates_and_incomplete_inspection_drafts():
    slots = (
        SlotDesign("s", Cell(0, 0)),
        SlotDesign("s", Cell(0, 0)),
        SlotDesign("outside", Cell(2, 0)),
    )
    buffer = BufferDesign("B", "B", Footprint(0, 0, 2, 2), storage=SlotStorage(slots))
    inspection = InspectionStationDesign(
        "I", "I", Footprint(3, 0, 2, 2), parallel_capacity=2
    )
    data = design(buffers=(buffer,), inspection_stations=(inspection,))
    assert codes(data, "error") == {
        "duplicate_slot_id",
        "slot_overlap",
        "slot_out_of_bounds",
    }
    assert codes(data, "warning") == {
        "empty_slots",
        "inspection_parallel_exceeds_slots",
    }
    assert "inspection_parallel_exceeds_slots" not in codes(
        replace(
            data, inspection_stations=(replace(inspection, parallel_capacity="max"),)
        )
    )


def test_bindings_enforce_type_and_operation_without_inventing_adjacency():
    data = populated_design()
    original = data.ports[0]
    invalid = replace(
        original,
        bindings=(
            PortBinding(ScrapBinTarget("S"), ("pickup",)),
            PortBinding(ChargerTarget("C"), ("drop_off",)),
            PortBinding(BufferTarget("B")),
            PortBinding(InspectionSlotTarget("I", "missing")),
            PortBinding(MachineTarget("M")),
            PortBinding(MachineTarget("M"), ("pickup",)),
        ),
    )
    issues = validate_factory_design(replace(data, ports=(invalid,)))
    assert {issue.code for issue in issues} == {
        "invalid_port_operation",
        "invalid_port_target",
        "duplicate_port_target",
    }
    assert all(
        issue.entity_id == "port" and issue.field == "bindings" for issue in issues
    )
    assert validate_factory_design(data) == ()  # Distant typed targets are deliberate.


def test_draft_warnings_do_not_promote_to_errors():
    data = design(
        buffers=(
            BufferDesign(
                "B", "B", Footprint(0, 0, 2, 2), "machine_post", SlotStorage()
            ),
        ),
        ports=(PortDesign("P", "P", Cell(3, 0)),),
    )
    assert codes(data, "error") == set()
    assert codes(data, "warning") == {
        "orphan_machine_buffer",
        "empty_slots",
        "empty_port",
    }
