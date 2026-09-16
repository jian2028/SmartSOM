"""Independent arithmetic and conservation across composed grid facilities."""

from dataclasses import replace
from itertools import product

import pytest
from test_production_runtime import quality_scenario, small_scenario

from smartsom.algorithms.production import GreedyProductionPolicy
from smartsom.domain.production import Outage, ProcessingSample
from smartsom.engine.production import ProductionSimulator


@pytest.mark.parametrize(
    "arrivals,outage,processing,finite,inspection",
    list(product((False, True), repeat=5)),
)
def test_composed_physics_against_duration_ownership_and_grid_arithmetic(
    arrivals, outage, processing, finite, inspection
):
    case = quality_scenario() if inspection else small_scenario()
    demands = tuple(
        replace(
            case.demands[0],
            demand_id=f"job{i}",
            release_at=2 * i if arrivals else 0,
            reveal_at=i if arrivals else 0,
        )
        for i in range(3)
    )
    factory = replace(
        case.factory,
        buffers=tuple(
            replace(b, storage=replace(b.storage, capacity=1))
            if finite and b.role == "system_input"
            else b
            for b in case.factory.buffers
        ),
    )
    case = replace(
        case,
        factory=factory,
        demands=demands,
        tick_limit=200,
        mode="dynamic" if arrivals else "static",
        outages=(Outage("machine", 4, 7),) if outage else (),
        processing_samples=tuple(
            ProcessingSample(d.demand_id, "op", "machine", 3) for d in demands
        )
        if processing
        else (),
    )
    sim = ProductionSimulator(case)
    policy = GreedyProductionPolicy(factory, rule="spt", quality_mode="normal")
    releases = {d.demand_id: d.release_at for d in demands}
    started, finished, delivered = {}, {}, set()
    previous = sim.snapshot()
    while not sim.done:
        rankings = sim.prepare_rankings(policy.rank(sim.decision()))
        row = sim.step(policy.act(sim.decision(rankings)))
        state = row["state"]
        assert row["tick"] == previous["tick"] + 1
        for key, vehicle in state["agvs"].items():
            old = previous["agvs"][key]
            assert sum(abs(a - b) for a, b in zip(vehicle["cell"], old["cell"])) <= 1
        # Recompute physical ownership independently from the core's invariant.
        held = []
        for owner, slots in state["storage"].items():
            for slot, jobs in slots.items():
                held.extend(jobs)
                if finite and owner == "input":
                    assert len(jobs) <= 1
                for job in jobs:
                    assert state["jobs"][job]["location"] == owner
                    assert state["jobs"][job]["slot"] == slot
        for resources in (state["agvs"], state["machines"]):
            held.extend(v["job"] for v in resources.values() if v["job"] is not None)
        assert len(held) == len(set(held))
        assert set(state["completed"]) >= delivered
        delivered = set(state["completed"])
        for event in row["events"]:
            if "job" in event:
                demand = event["job"].split("/attempt/")[0]
                assert event["tick"] >= releases[demand]
            if event["kind"] == "processing_started":
                assert event["job"] not in started
                started[event["job"]] = event["tick"]
            if event["kind"] == "processing_completed":
                job = event["job"]
                assert job not in finished
                end, start = event["tick"], started[job]
                actual_work = sum(
                    not any(o.start <= tick < o.end for o in case.outages)
                    for tick in range(start, end)
                )
                assert actual_work == (3 if processing else 2)
                finished[job] = end
        previous = state
    assert sim.status == "completed" and sim.completed == set(releases)
    assert len(started) == len(finished) == 3
    intervals = sorted((started[job], end) for job, end in finished.items())
    assert all(a[1] <= b[0] for a, b in zip(intervals, intervals[1:]))
