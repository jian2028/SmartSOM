"""Hand-calculated physical transitions, independent of playback projections."""

from dataclasses import replace

import pytest

from smartsom.domain.factory_design import (
    AGVDesign,
    BufferDesign,
    BufferTarget,
    Cell,
    FactoryDesign,
    Footprint,
    GridDesign,
    MachineDesign,
    MachineTarget,
    PoolStorage,
    PortBinding,
    PortDesign,
)
from smartsom.domain.production import (
    Demand,
    JointCommand,
    MachineCommand,
    Outage,
    ProductionScenario,
    ProductionStep,
)
from smartsom.engine.production import ProductionSimulator


def small_scenario(**kwargs):
    factory = FactoryDesign(
        "small",
        "Hand calculated",
        GridDesign(3, 2),
        operation_types=("drill",),
        machines=(
            MachineDesign(
                "machine", "Machine", Footprint(1, 1, 1, 1), operation_types=("drill",)
            ),
        ),
        buffers=(
            BufferDesign(
                "input",
                "Input",
                Footprint(0, 1, 1, 1),
                "system_input",
                PoolStorage(None),
            ),
            BufferDesign(
                "output",
                "Output",
                Footprint(2, 1, 1, 1),
                "system_output",
                PoolStorage(None),
            ),
        ),
        ports=(
            PortDesign(
                "in_port",
                "Input",
                Cell(0, 0),
                bindings=(PortBinding(BufferTarget("input"), ("pickup",)),),
            ),
            PortDesign(
                "machine_port",
                "Machine",
                Cell(1, 0),
                bindings=(PortBinding(MachineTarget("machine")),),
            ),
            PortDesign(
                "out_port",
                "Output",
                Cell(2, 0),
                bindings=(PortBinding(BufferTarget("output"), ("drop_off",)),),
            ),
        ),
        agvs=(AGVDesign("agv", "AGV", Cell(0, 0)),),
    )
    return ProductionScenario(
        factory, (Demand("demand", (ProductionStep("op", "drill", 2),)),), **kwargs
    )


def act(sim, action="WAIT", machine=None):
    return sim.step(
        JointCommand(
            agvs=(("agv", action),),
            machines=() if machine is None else (("machine", machine),),
        )
    )


def loaded_machine(sim):
    act(sim, "INTERACT")
    act(sim, "RIGHT")
    act(sim, "INTERACT")
    return sim.machine_state["machine"]["job"]


@pytest.mark.parametrize("machine", ["missing", "machine"])
def test_duration_override_cannot_grant_capability(machine):
    scenario = small_scenario()
    factory = scenario.factory
    if machine == "machine":
        factory = replace(
            factory,
            machines=(replace(factory.machines[0], operation_types=()),),
        )
    scenario = replace(
        scenario,
        factory=factory,
        demands=(Demand("demand", (ProductionStep("op", "drill", 2, {machine: 3}),)),),
    )
    with pytest.raises(ValueError, match="unknown or incapable"):
        ProductionSimulator(scenario)


def test_machine_duration_snapshot_roundtrip_is_immutable_and_portable():
    from smartsom.config.codec import primitive
    from smartsom.config.scenario_snapshot import scenario_from_snapshot

    table = {"machine": 5}
    step = ProductionStep("op", "drill", 2, table)
    table["machine"] = 999
    scenario = replace(small_scenario(), demands=(Demand("demand", (step,)),))
    payload = primitive(scenario)
    assert payload["demands"][0]["steps"][0]["machine_nominal_ticks"] == {"machine": 5}
    assert scenario_from_snapshot(payload) == scenario


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_machine_duration_rejects_invalid_ticks(value):
    with pytest.raises(ValueError, match="positive integers"):
        ProductionStep("op", "drill", 2, {"machine": value})


def quality_scenario(error_rate="0", **kwargs):
    from decimal import Decimal

    from smartsom.domain.factory_design import (
        InspectionSlotTarget,
        InspectionStationDesign,
        ScrapBinDesign,
        ScrapBinTarget,
        SlotDesign,
    )
    from smartsom.domain.quality import QualityMode

    case = small_scenario(**kwargs)
    factory = replace(
        case.factory,
        factory_id="quality",
        grid=GridDesign(5, 2),
        machines=(
            replace(
                case.factory.machines[0],
                quality_modes=(
                    QualityMode("normal", Decimal(1), Decimal(error_rate)),
                    QualityMode("fast", Decimal("0.5"), Decimal("0.5")),
                ),
            ),
        ),
        buffers=(
            case.factory.buffers[0],
            replace(case.factory.buffers[1], footprint=Footprint(4, 1, 1, 1)),
        ),
        inspection_stations=(
            InspectionStationDesign(
                "inspection",
                "Inspection",
                Footprint(2, 1, 1, 1),
                slots=(SlotDesign("slot", Cell(0, 0)),),
            ),
        ),
        scrap_bins=(ScrapBinDesign("scrap", "Scrap", Footprint(3, 1, 1, 1)),),
        ports=case.factory.ports[:2]
        + (
            PortDesign(
                "inspection_port",
                "Inspection",
                Cell(2, 0),
                bindings=(PortBinding(InspectionSlotTarget("inspection", "slot")),),
            ),
            PortDesign(
                "scrap_port",
                "Scrap",
                Cell(3, 0),
                bindings=(PortBinding(ScrapBinTarget("scrap"), ("drop_off",)),),
            ),
            replace(case.factory.ports[2], cell=Cell(4, 0)),
        ),
    )
    return replace(case, factory=factory)


def inspected_job(sim):
    job = loaded_machine(sim)
    act(sim, machine=MachineCommand(job, "normal"))
    act(sim)
    act(sim, "INTERACT")
    act(sim, "RIGHT")
    act(sim, "INTERACT")
    assert sim.tick == 8 and sim.jobs[job]["location"] == "inspection"
    return job


@pytest.mark.parametrize(
    "sample",
    [
        ("missing", "op", "machine", 2),
        ("demand", "missing", "machine", 2),
        ("demand", "op", "missing", 2),
    ],
)
def test_fixed_processing_samples_reject_invalid_references(sample):
    from smartsom.domain.production import ProcessingSample

    with pytest.raises(ValueError, match="sample references"):
        ProductionSimulator(
            small_scenario(processing_samples=(ProcessingSample(*sample),))
        )


@pytest.mark.parametrize(
    "field", ["demands", "outages", "processing_samples", "quality_samples"]
)
def test_direct_scenario_rejects_untyped_rows(field):
    with pytest.raises(ValueError, match="must contain"):
        replace(small_scenario(), **{field: ({"unknown": "row"},)})


def test_direct_demand_and_outage_reject_invalid_types():
    with pytest.raises(ValueError, match="ProductionStep"):
        Demand("demand", ({"operation_id": "op"},))
    with pytest.raises(ValueError, match="input_id"):
        replace(small_scenario().demands[0], input_id=True)
    with pytest.raises(ValueError, match="machine_id"):
        Outage(True, 1, 2)


def automatic_inspection_case(count=5, associated=True):
    from smartsom.domain.factory_design import SlotDesign

    case = quality_scenario()
    station = replace(
        case.factory.inspection_stations[0],
        slots=(SlotDesign("slot", Cell(0, 0), 4),),
        auto_disposal_bin_id="scrap" if associated else None,
    )
    return replace(
        case,
        factory=replace(case.factory, inspection_stations=(station,)),
        demands=tuple(
            replace(case.demands[0], demand_id=f"d{i}") for i in range(count)
        ),
        mode="dynamic",
    )


def carry_to_inspection(sim, job, defective=False):
    # Arrange a completed operation and an AGV at the station port. Admission,
    # timers, pickup masks and disposal are exercised through physical steps.
    sim._remove(job)
    sim.jobs[job].update(defective=defective, step=1, location="agv", slot=None)
    sim.agvs["agv"].update(job=job, cell=[2, 0])
    return act(sim, "INTERACT")


@pytest.mark.parametrize(
    "change,code",
    [
        ({"auto_disposal_bin_id": "missing"}, "invalid_disposal_bin"),
        ({"footprint": Footprint(0, 0, 1, 1)}, "nonadjacent_disposal_bin"),
    ],
)
def test_disposal_link_validation(change, code):
    from smartsom.domain.factory_design import validate_factory_design

    factory = automatic_inspection_case().factory
    factory = replace(
        factory,
        inspection_stations=(replace(factory.inspection_stations[0], **change),),
    )
    assert code in {i.code for i in validate_factory_design(factory)}
