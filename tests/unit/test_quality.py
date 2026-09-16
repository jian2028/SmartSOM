"""Independent item-10 hand outcomes and composition through the public engine."""

import itertools
from dataclasses import replace
from decimal import Decimal

import pytest

from smartsom.config.codec import primitive


@pytest.mark.parametrize(
    "agv,ja,mb,upt,buffers", tuple(itertools.product((False, True), repeat=5))
)
def test_all_module_combinations(agv, ja, mb, upt, buffers):
    from test_production_runtime import quality_scenario

    from smartsom.algorithms.production import GreedyProductionPolicy
    from smartsom.domain.factory_design import (
        BufferDesign,
        BufferTarget,
        Cell,
        Footprint,
        GridDesign,
        PoolStorage,
        PortBinding,
        PortDesign,
    )
    from smartsom.domain.production import Outage, QualitySample
    from smartsom.engine.production import ProductionSimulator
    from smartsom.trace.production import ExecutionAudit

    case = quality_scenario("0.5", mode="dynamic" if ja else "static", tick_limit=100)
    case = replace(
        case,
        demands=(replace(case.demands[0], release_at=2, reveal_at=1),)
        if ja
        else case.demands,
        outages=(Outage("machine", 4, 7),) if mb else (),
        processing_low=Decimal("0.8") if upt else Decimal(1),
        processing_high=Decimal("1.2") if upt else Decimal(1),
        quality_samples=(
            QualitySample("demand", "op", 0),
            QualitySample("demand", "op", 2**53 - 1, attempt=2),
        ),
    )
    if buffers:
        f = case.factory
        pre = BufferDesign(
            "pre",
            "Pre",
            Footprint(5, 1, 1, 1),
            "machine_pre",
            PoolStorage(1),
            machine_id="machine",
        )
        case = replace(
            case,
            factory=replace(
                f,
                grid=GridDesign(6, 2),
                buffers=(*f.buffers, pre),
                ports=(
                    *f.ports,
                    PortDesign(
                        "pre_port",
                        "Pre",
                        Cell(5, 0),
                        bindings=(PortBinding(BufferTarget("pre")),),
                    ),
                ),
            ),
        )
    if not agv:
        with pytest.raises(ValueError, match="at least one AGV"):
            ProductionSimulator(replace(case, factory=replace(case.factory, agvs=())))
        return
    sim, policy = (
        ProductionSimulator(case),
        GreedyProductionPolicy(case.factory, quality_mode="normal"),
    )
    checker = ExecutionAudit(case, sim.snapshot())
    observations = []
    while not sim.done:
        rankings = policy.rank(sim.decision())
        view = sim.decision(rankings)
        observations.append(primitive(view))
        assert all("defective" not in job for job in view["jobs"].values())
        row = sim.step(policy.act(view))
        checker.append(row)
    assert sim.status == "completed" and sim.completed == {"demand"}
    assert sim.jobs["demand/attempt/1"]["quality"] == "FAIL"
    assert sim.jobs["demand/attempt/2"]["quality"] == "PASS"
    assert checker.result()["status"] == "passed"
    assert observations[0]["tick"] == 0 and observations[-1]["tick"] == sim.tick - 1


@pytest.mark.parametrize(
    "rate,draw,defective",
    [
        ("0", 0, False),
        ("1", 2**53 - 1, True),
        ("0.5", 2**52 - 1, True),
        ("0.5", 2**52, False),
        ("0.5", 2**52 + 1, False),
    ],
)
def test_quality_probability_uses_strict_exact_boundary_and_hides_latent_result(
    rate, draw, defective
):
    from test_production_runtime import act, loaded_machine, quality_scenario

    from smartsom.domain.production import MachineCommand, QualitySample
    from smartsom.engine.production import ProductionSimulator

    case = replace(
        quality_scenario(rate), quality_samples=(QualitySample("demand", "op", draw),)
    )
    sim = ProductionSimulator(case)
    job = loaded_machine(sim)
    act(sim, machine=MachineCommand(job, "normal"))
    act(sim)
    assert sim.jobs[job]["defective"] is defective
    public = sim.decision()["jobs"][job]
    assert "defective" not in public and public["quality"] == "UNKNOWN"


def test_machine_cannot_change_mode_while_processing_or_paused():
    from test_production_runtime import act, loaded_machine, quality_scenario

    from smartsom.domain.production import MachineCommand, Outage
    from smartsom.engine.production import ProductionSimulator

    sim = ProductionSimulator(quality_scenario(outages=(Outage("machine", 4, 6),)))
    job = loaded_machine(sim)
    act(sim, machine=MachineCommand(job, "normal"))
    assert sim.machine_state["machine"]["down"]
    for _ in range(2):
        row = act(sim, machine=MachineCommand(job, "fast"))
        assert row["rejections"] == {"machine:machine": "invalid"}
        assert sim.machine_state["machine"]["mode"] == "normal"
        assert sim.machine_state["machine"]["remaining"] == 1
    row = act(sim, machine=MachineCommand(job, "fast"))
    assert row["rejections"] == {"machine:machine": "invalid"}
    assert sim.machine_state["machine"]["status"] == "BLOCKED"
    assert sim.machine_state["machine"]["elapsed"] == 2
