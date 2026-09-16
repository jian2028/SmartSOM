"""Simultaneous resource claims use public semantics and atomic grid arbitration."""

from dataclasses import replace

import pytest
from test_production_runtime import small_scenario

from smartsom.domain.factory_design import (
    AGVDesign,
    BufferDesign,
    BufferTarget,
    Cell,
    Footprint,
    GridDesign,
    PoolStorage,
    PortBinding,
    PortDesign,
)
from smartsom.domain.production import JointCommand
from smartsom.engine.production import ProductionSimulator


def shared_ports(capacity=None, *, jobs=2):
    case = small_scenario(mode="dynamic", tick_limit=20)
    f = case.factory
    factory = replace(
        f,
        grid=GridDesign(4, 3),
        buffers=(
            *f.buffers,
            BufferDesign(
                "store",
                "Store",
                Footprint(3, 1, 1, 1),
                "storage",
                PoolStorage(capacity),
            ),
        ),
        ports=(
            *f.ports,
            PortDesign(
                "in2",
                "Input 2",
                Cell(0, 2),
                bindings=(PortBinding(BufferTarget("input"), ("pickup",)),),
            ),
            PortDesign(
                "store1",
                "Store 1",
                Cell(3, 0),
                bindings=(PortBinding(BufferTarget("store")),),
            ),
            PortDesign(
                "store2",
                "Store 2",
                Cell(3, 2),
                bindings=(PortBinding(BufferTarget("store")),),
            ),
        ),
        agvs=(*f.agvs, AGVDesign("other", "Other", Cell(0, 2))),
    )
    return replace(
        case,
        factory=factory,
        demands=tuple(
            replace(case.demands[0], demand_id=f"job{i}") for i in range(jobs)
        ),
    )


def test_two_pickup_claimants_are_both_rejected_without_resampling():
    sim = ProductionSimulator(shared_ports(jobs=1))
    before = sim.snapshot()
    row = sim.step(JointCommand(agvs=(("agv", "INTERACT"), ("other", "INTERACT"))))
    assert row["rejections"] == {"agv:agv": "conflict", "agv:other": "conflict"}
    assert sim.snapshot()["storage"] == before["storage"]
    assert all(a["job"] is None for a in sim.agvs.values())
    assert sim.tick == 1


@pytest.mark.parametrize("capacity", [0, 1, 2, None])
def test_joint_drop_capacity_counts_actual_free_space_without_reservations(capacity):
    sim = ProductionSimulator(shared_ports(capacity))
    sim.step(JointCommand(agvs=(("agv", "INTERACT"),)))
    sim.step(JointCommand(agvs=(("other", "INTERACT"),)))
    assert sim.agvs["agv"]["job"] != sim.agvs["other"]["job"]
    for _ in range(3):
        sim.step(JointCommand(agvs=(("agv", "RIGHT"), ("other", "RIGHT"))))
    row = sim.step(JointCommand(agvs=(("agv", "INTERACT"), ("other", "INTERACT"))))
    if capacity in (0, 1):
        reason = "invalid" if capacity == 0 else "conflict"
        assert row["rejections"] == {"agv:agv": reason, "agv:other": reason}
        assert sim.storage["store"]["pool"] == []
        assert all(a["job"] is not None for a in sim.agvs.values())
    else:
        assert row["rejections"] == {}
        assert set(sim.storage["store"]["pool"]) == {"job0/attempt/1", "job1/attempt/1"}
        assert all(a["job"] is None for a in sim.agvs.values())
    sim._check()


def test_stale_mover_proposal_cannot_reinterpret_a_completed_pickup():
    sim = ProductionSimulator(shared_ports())
    command = JointCommand(agvs=(("agv", "INTERACT"),))
    sim.step(command)
    held = sim.agvs["agv"]["job"]
    row = sim.step(command)
    assert row["rejections"] == {"agv:agv": "invalid"}
    assert sim.agvs["agv"]["job"] == held
    assert sim.jobs[held]["location"] == "agv"
