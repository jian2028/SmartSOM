"""Capabilities select eligible machines; per-machine nominal ticks select duration."""

from dataclasses import replace

import pytest
from test_production_runtime import act, small_scenario

from smartsom.domain.factory_design import (
    Cell,
    Footprint,
    GridDesign,
    MachineDesign,
    MachineTarget,
    PortBinding,
    PortDesign,
)
from smartsom.domain.production import (
    Demand,
    JointCommand,
    MachineCommand,
    ProductionStep,
)
from smartsom.engine.production import ProductionSimulator


def flexible_grid():
    case = small_scenario()
    f = case.factory
    return replace(
        case,
        factory=replace(
            f,
            grid=GridDesign(4, 2),
            machines=(
                *f.machines,
                MachineDesign(
                    "other", "Other", Footprint(3, 1, 1, 1), operation_types=("drill",)
                ),
            ),
            ports=(
                *f.ports,
                PortDesign(
                    "other_port",
                    "Other",
                    Cell(3, 0),
                    bindings=(PortBinding(MachineTarget("other")),),
                ),
            ),
        ),
        demands=(Demand("demand", (ProductionStep("op", "drill", 1, {"other": 5}),)),),
    )


@pytest.mark.parametrize("machine,end", [("machine", 7), ("other", 13)])
def test_eligible_machine_changes_duration_and_explicit_transport_timetable(
    machine, end
):
    sim = ProductionSimulator(flexible_grid())
    job = "demand/attempt/1"
    assert sim.decision()["jobs"][job]["machine_nominal_ticks"] == {
        "machine": 1,
        "other": 5,
    }
    act(sim, "INTERACT")
    for _ in range(1 if machine == "machine" else 3):
        act(sim, "RIGHT")
    act(sim, "INTERACT")
    ticks = 1 if machine == "machine" else 5
    row = sim.step(JointCommand(machines=((machine, MachineCommand(job, "normal")),)))
    assert sim.machine_state[machine]["nominal"] == ticks
    for _ in range(ticks - 1):
        row = act(sim)
    assert any(e["kind"] == "processing_completed" for e in row["events"])
    act(sim, "INTERACT")
    act(sim, "RIGHT" if machine == "machine" else "LEFT")
    act(sim, "INTERACT")
    assert sim.tick == end and sim.completed == {"demand"}


def test_equal_capabilities_do_not_grant_machine_access_to_remote_jobs():
    sim = ProductionSimulator(flexible_grid())
    act(sim, "INTERACT")
    act(sim, "RIGHT")
    act(sim, "INTERACT")
    assert sim.machine_choices("machine") == [("demand/attempt/1", "normal")]
    assert sim.machine_choices("other") == []
    row = sim.step(
        JointCommand(
            machines=(("other", MachineCommand("demand/attempt/1", "normal")),)
        )
    )
    assert row["rejections"] == {"machine:other": "invalid"}
    assert sim.jobs["demand/attempt/1"]["location"] == "machine"
