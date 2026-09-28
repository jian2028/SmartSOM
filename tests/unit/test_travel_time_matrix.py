"""Hand timing, capacity and semantic replay of matrix transport."""

from collections import Counter
from dataclasses import replace
from decimal import Decimal
from pathlib import Path

import pytest

from smartsom import api
from smartsom.algorithms.production_composition import (
    BoundaryCoordinator,
    replay_boundary,
)
from smartsom.algorithms.production_rules import RulePolicy
from smartsom.config.factory_design import load_factory_design_file
from smartsom.config.travel_time import (
    MatrixFile,
    TransportSettings,
    materialize_matrix,
)
from smartsom.domain.production import Demand, MachineCommand, ProductionStep
from smartsom.engine.production import ProductionSimulator
from smartsom.experiments.composable_study import heterogeneity

ROOT = Path(__file__).resolve().parents[2]


def scenario(mode="zero", jobs=1):
    p = api.prepare(
        api.load_config(ROOT / f"configs/test/runs/small_rules_{mode}.yaml"),
        training=False,
    )
    demands = tuple(
        Demand(f"hand_{i}", (ProductionStep("op", "operation_1", 1),), due_at=100)
        for i in range(jobs)
    )
    factory = replace(
        p.scenario.factory,
        machines=tuple(
            replace(
                m,
                quality_modes=tuple(
                    replace(q, error_rate=Decimal(0)) for q in m.quality_modes
                ),
            )
            for m in p.scenario.factory.machines
        ),
    )
    return replace(p.scenario, demands=demands, factory=factory, tick_limit=200)


def driver(sim):
    names = {
        "machine": "normal_first",
        "buffer": "edd",
        "dispatcher": "nearest",
        "mover": "automatic_travel",
    }
    return BoundaryCoordinator(
        sim,
        {k: RulePolicy(k, v) for k, v in names.items()},
        {k: {"default": k} for k in names},
    )


def test_auto_all_pairs_equal_factory_shortest_paths():
    sim = ProductionSimulator(scenario("auto"), contract="v3")
    points = {p: (x, y) for p, x, y in sim.scenario.transport_matrix.points}
    for a, b, ticks in sim.scenario.transport_matrix.times:
        result = sim.protocol.distance(points[a], points[b])
        assert ticks == (None if result == float("inf") else result)


def test_manual_asymmetric_overrides_missing_and_unknown_entries():
    factory = scenario().factory
    first, second = (p.port_id for p in factory.ports[:2])
    spec = MatrixFile.model_validate(
        {
            "schema": "smartsom.travel-time-matrix/v1",
            "source": "manual",
            "default_ticks": 0,
            "overrides": {first: {second: 3}, second: {first: 7}},
        }
    )
    matrix = materialize_matrix(
        factory, TransportSettings(mode="travel_time_matrix", resolved=spec)
    )
    edges = {(a, b): v for a, b, v in matrix.times}
    assert edges[(first, second)] == 3 and edges[(second, first)] == 7
    with pytest.raises(ValueError, match="missing travel"):
        materialize_matrix(
            factory,
            TransportSettings(
                mode="travel_time_matrix",
                resolved=spec.model_copy(update={"default_ticks": None}),
            ),
        )
    with pytest.raises(ValueError, match="unknown"):
        materialize_matrix(
            factory,
            TransportSettings(
                mode="travel_time_matrix",
                resolved=spec.model_copy(
                    update={"overrides": {"missing": {second: 0}}}
                ),
            ),
        )
    with pytest.raises(ValueError, match="nonnegative|negative"):
        MatrixFile.model_validate_json(
            '{"schema":"smartsom.travel-time-matrix/v1","source":"manual","overrides":{"a":{"b":-1}}}'
        )


def test_zero_still_requires_services_and_inspection_and_replays():
    original = scenario()
    sim = ProductionSimulator(original, contract="v3")
    replay = ProductionSimulator(original, contract="v3")
    controller = driver(sim)
    first = controller.tick()
    assert len(first["actions"]["matching"]) == 1
    assert (
        first["boundary_state"]["agvs"][first["actions"]["matching"][0][0]]["service"][
            "remaining"
        ]
        == 1
    )
    replay_boundary(replay, first)
    while not sim.done:
        row = controller.tick()
        replay_boundary(replay, row)
        assert row["actions"]["movers"] == ()
        assert not any(d["role"] == "mover" for d in row["decisions"])
    assert sim.tick == 9 and len(sim.completed) == 1
    assert sim.metrics["pickup_services"] == sim.metrics["drop_services"] == 3
    assert sim.metrics["matrix_travel_ticks"] == 0


def test_positive_trip_locks_dispatch_until_exact_arrival():
    original = scenario()
    matrix = original.transport_matrix
    matrix = replace(
        matrix, times=tuple((a, b, 0 if a == b else 4) for a, b, _ in matrix.times)
    )
    sim = ProductionSimulator(replace(original, transport_matrix=matrix), contract="v3")
    controller = driver(sim)
    first = controller.tick()
    vehicle = next(k for k, v in sim.agvs.items() if v["travel"])
    assert sim.agvs[vehicle]["travel"]["departed_at"] == 0
    assert sim.agvs[vehicle]["travel"]["arrival_tick"] == 4
    assert first["actions"]["matching"] == ()
    for _ in range(3):
        row = controller.tick()
        assert not any(
            d["role"] == "dispatcher" and d["owner"] == vehicle
            for d in row["decisions"]
        )
        assert not sim.agvs[vehicle]["job"]
    assert sim.tick == 4 and sim.agvs[vehicle]["travel"] is None
    controller.tick()
    assert sim.agvs[vehicle]["job"] is not None


def test_same_port_queue_serves_one_and_retains_other_reservation():
    sim = ProductionSimulator(scenario(jobs=2), contract="v3")
    protocol = sim.protocol
    requests = protocol.begin()
    owner = next(o for o in protocol.sources if sim.roles.get(o) == "system_input")
    port = protocol.ports_for(owner, "pickup")[0]
    from smartsom.domain.production_decisions import DispatchTarget

    vehicles = list(sim.agvs)[:2]
    requests = protocol.accept_proposals(
        {}, dict.fromkeys(vehicles, DispatchTarget(owner, port.port_id))
    )
    assert len(requests) == 1 and requests[0].count == 1
    served = protocol.service_vehicles[owner][0]
    job = requests[0].candidates[0].action
    assert protocol.prepare_services({owner: (job,)}, ((served, job),)) == ()
    row = protocol.commit({})
    assert sum(v["job"] is not None for v in row["state"]["agvs"].values()) == 1
    waiting = next(v for v in vehicles if v != served)
    assert sim.agvs[waiting]["reservation"] == owner
    assert sim.agvs[served]["reservation"] is None


def test_full_destination_waits_without_reserving_slot():
    sim = ProductionSimulator(scenario(jobs=6), contract="v3")
    vehicle = next(iter(sim.agvs))
    jobs = list(sim.jobs)
    job = jobs.pop()
    sim._remove(job)
    sim.jobs[job].update(location=vehicle, slot=None)
    sim.agvs[vehicle]["job"] = job
    owner = sim.protocol.destination_owners(job)[0]
    machine = next(m for m, p in sim.pre.items() if p == owner)
    slot = next(iter(sim.capacity[owner]))
    assert sim.capacity[owner][slot] == 4
    for j in jobs[:4]:
        sim._remove(j)
        sim._place(j, owner, slot)
    processing = jobs[4]
    sim._remove(processing)
    sim.jobs[processing].update(location=machine, slot=None)
    sim.machine_state[machine].update(
        job=processing, status="PROCESSING", remaining=100, nominal=100, mode="normal"
    )
    controller = driver(sim)
    row = controller.tick()
    assert sim.agvs[vehicle]["job"] == job and sim.agvs[vehicle]["service"] is None
    assert sim.metrics["destination_waiting"] == 1
    assert not any(
        e["kind"] == "drop_started" and e["agv"] == vehicle for e in row["events"]
    )
    freed = sim.storage[owner][slot][0]
    sim._remove(freed)
    sim._place(freed, sim._input(sim.demands[sim.jobs[freed]["demand"]]), "pool")
    controller.tick()
    assert sim.agvs[vehicle]["job"] is None and sim.jobs[job]["location"] == owner


def test_g1_nominal_invariants_and_exact_single_rounding():
    base, _ = load_factory_design_file(ROOT / "configs/test/factories/small.yaml")
    h1, _ = load_factory_design_file(ROOT / "configs/test/factories/small_h1.yaml")
    h = heterogeneity(h1.factory, base.factory)
    assert h["speed"] == pytest.approx(0.1)
    assert h["quality"] == pytest.approx(0.248746859276655)
    assert h["H"] == pytest.approx(0.1895718861)
    sim = ProductionSimulator(replace(scenario(), factory=h1.factory), contract="v3")
    machine = next(
        m.machine_id
        for m in h1.factory.machines
        if m.processing_rate_multiplier == Decimal("1.1")
        and "operation_1" in m.operation_types
    )
    job = next(iter(sim.jobs))
    sim.demands[sim.jobs[job]["demand"]] = replace(
        sim.demands[sim.jobs[job]["demand"]],
        steps=(ProductionStep("op", "operation_1", 30),),
    )
    sim._start(machine, MachineCommand(job, "normal"))
    assert (
        sim.machine_state[machine]["nominal"]
        == sim.machine_state[machine]["remaining"]
        == 28
    )
    assert sim.protocol.drop_slot(sim.pre[machine], Counter()) is not None


def test_matrix_rejects_wrong_points_and_stranded_pickup_route():
    original = scenario()
    matrix = original.transport_matrix
    changed = replace(matrix, points=tuple((p, x + 1, y) for p, x, y in matrix.points))
    with pytest.raises(ValueError, match="points differ"):
        ProductionSimulator(replace(original, transport_matrix=changed), contract="v3")
    sim = ProductionSimulator(original, contract="v3")
    source = next(o for o in sim.protocol.sources if sim.roles.get(o) == "system_input")
    port = sim.protocol.ports_for(source, "pickup")[0].port_id
    changed = replace(
        matrix,
        times=tuple(
            (a, b, None if a == port and b != a else value)
            for a, b, value in matrix.times
        ),
    )
    with pytest.raises(ValueError, match="next leg"):
        ProductionSimulator(replace(original, transport_matrix=changed), contract="v3")


def test_legacy_defaults_omit_only_semantically_absent_additions():
    from smartsom.config.codec import digest, primitive

    original = scenario()
    grid = replace(original, transport_matrix=None, processing_rounding="half_up")
    encoded = primitive(grid)
    assert "transport_matrix" not in encoded and "processing_rounding" not in encoded
    assert all(
        "processing_rate_multiplier" not in m for m in encoded["factory"]["machines"]
    )
    assert digest(grid) != digest(replace(grid, processing_rounding="ceil"))
    assert digest(grid) != digest(original)
    machine = grid.factory.machines[0]
    faster = replace(machine, processing_rate_multiplier=Decimal("1.1"))
    assert primitive(faster)["processing_rate_multiplier"] == "1.1"
    assert digest(machine) != digest(faster)
