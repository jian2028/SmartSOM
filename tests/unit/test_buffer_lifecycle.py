"""Buffer lifecycle is physical occupancy, without hidden reservations or queues."""

from dataclasses import replace

from test_production_runtime import act, small_scenario

from smartsom.domain.production import JointCommand
from smartsom.engine.production import ProductionSimulator


def test_finite_input_refills_after_real_pickup_at_the_next_boundary():
    case = small_scenario()
    factory = replace(
        case.factory,
        buffers=tuple(
            replace(b, storage=replace(b.storage, capacity=1))
            if b.role == "system_input"
            else b
            for b in case.factory.buffers
        ),
    )
    case = replace(
        case,
        factory=factory,
        demands=tuple(replace(case.demands[0], demand_id=f"d{i}") for i in range(3)),
    )
    sim = ProductionSimulator(case)
    assert sim.storage["input"]["pool"] == ["d0/attempt/1"]
    assert sim.queue == ["d1/attempt/1", "d2/attempt/1"]
    act(sim, "INTERACT")
    assert sim.agvs["agv"]["job"] == "d0/attempt/1"
    assert sim.storage["input"]["pool"] == ["d1/attempt/1"]
    assert sim.queue == ["d2/attempt/1"]
    assert sim.snapshot(public=True)["queue"] == []
    assert "d2/attempt/1" not in sim.decision()["jobs"]


def test_missing_pre_post_creates_no_implicit_storage():
    sim = ProductionSimulator(small_scenario())
    assert not sim.pre and not sim.post
    assert set(sim.storage) == {"input", "output"}
    act(sim, "INTERACT")
    act(sim, "RIGHT")
    act(sim, "INTERACT")
    assert sim.machine_state["machine"]["status"] == "READY"
    assert sim.jobs["demand/attempt/1"]["location"] == "machine"
    # A ready unprocessed part cannot be extracted as a completed operation.
    row = act(sim, "INTERACT")
    assert row["rejections"] == {"agv:agv": "invalid"}
    assert sim.machine_state["machine"]["job"] == "demand/attempt/1"


def test_zero_capacity_input_is_unadmitted_work_not_an_unlimited_buffer():
    case = small_scenario(tick_limit=4)
    case = replace(
        case,
        factory=replace(
            case.factory,
            buffers=tuple(
                replace(b, storage=replace(b.storage, capacity=0))
                if b.role == "system_input"
                else b
                for b in case.factory.buffers
            ),
        ),
    )
    sim = ProductionSimulator(case)
    while not sim.done:
        sim.step(JointCommand())
    assert sim.status == "truncated" and not sim.completed
    assert sim.storage["input"]["pool"] == [] and sim.queue == ["demand/attempt/1"]
