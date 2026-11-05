"""Independent physical/accounting invariants of the new opt-in Base contract."""

import math
import random
from dataclasses import replace
from pathlib import Path

import pytest

from smartsom.config.base_diagnostic import (
    BaseWorkload,
    generate_scenario,
)
from smartsom.config.base_layout import base_factory
from smartsom.config.factory_design import load_factory_design
from smartsom.domain.base_decisions import FEATURES
from smartsom.domain.production import (
    Demand,
    JointCommand,
    Outage,
    ProductionScenario,
    ProductionStep,
)
from smartsom.engine.production import ProductionSimulator
from smartsom.experiments.reservation_runtime import episode
from smartsom.policies.rule import BaseRule
from smartsom.traffic.rule_mover import SharedMover
from tests.helpers.algorithm import algorithm_options, with_protocol
from tests.helpers.historical_geometry import historical_base_factory


def options():
    return algorithm_options(schema="smartsom.base-algorithm/v1", method="rule")


def workload(n=8):
    return BaseWorkload(
        schema="smartsom.base-workload/v1",
        orders=n,
        initial_orders=n // 2,
        mean_interarrival=4,
        base_ticks=dict.fromkeys("ABCD", 2),
    )


def sim_with(steps=("A", "B"), n=4, outages=()):
    demands = tuple(
        Demand(
            f"job_{i}",
            tuple(ProductionStep(f"op_{k}", kind, 2) for k, kind in enumerate(steps)),
            due_at=5,
        )
        for i in range(n)
    )
    return ProductionSimulator(
        ProductionScenario(
            # These service scenarios use explicit historical dock coordinates.
            historical_base_factory(),
            demands,
            tick_limit=1000,
            outages=outages,
            processing_rounding="ceil",
        ),
        contract="base",
    )


def place(sim, job, owner):
    sim._remove(job)
    if owner in sim.agvs:
        sim.agvs[owner]["job"] = job
        sim.jobs[job].update(location=owner, slot=None, since=sim.tick)
    else:
        sim._place(job, owner, "pool")


def decide(sim, choose=None):
    generator = sim.protocol.decisions()
    requests = []
    policy = BaseRule()
    try:
        request = next(generator)
        while True:
            requests.append(request)
            request = generator.send(
                choose(request) if choose else policy.choose(request)
            )
    except StopIteration:
        return requests


def wait(sim):
    return sim.step(
        JointCommand(agvs=tuple((v, "WAIT") for v in sim.agvs)), compact=True
    )


@pytest.mark.parametrize(
    "scale,count,shape", [("small", 8, (22, 6)), ("medium", 20, (27, 12))]
)
def test_layout_reachability_and_capacities(scale, count, shape):
    f = base_factory(scale)
    path = (
        Path(__file__).resolve().parents[2]
        / "src/smartsom/config/maps"
        / f"{scale}-factory.yaml"
    )
    assert load_factory_design(path)[0] == f
    scenario = generate_scenario(f, workload(), 17, "normal", 2000)
    sim = ProductionSimulator(scenario, contract="base")
    assert len(f.machines) == len(f.agvs) == count
    assert (f.grid.width, f.grid.height) == shape
    assert len({tuple(a["cell"]) for a in sim.agvs.values()}) == count
    reachable = sim.protocol.routes.distances(
        (tuple(next(iter(sim.agvs.values()))["cell"]),)
    )
    for m in sim.machines:
        assert sim.capacity[sim.pre[m]] == sim.capacity[sim.post[m]] == {"pool": 4}
        assert sim.protocol.quota(sim.pre[m]) == 9
        for b in (sim.pre[m], sim.post[m]):
            assert len(sim.protocol.routes.ports[b]) == 3
            assert all(c in reachable for c in sim.protocol.routes.ports[b])


@pytest.mark.parametrize("scale,columns,rows", [("small", 4, 2), ("medium", 5, 4)])
def test_compact_unit_corridors_and_distinct_boundary_ports(scale, columns, rows):
    f = base_factory(scale)
    from smartsom.domain.factory_design import validate_factory_design

    assert not [e for e in validate_factory_design(f) if e.severity == "error"]
    footprints = {b.buffer_id: b.footprint for b in f.buffers}
    machines = [m.footprint for m in f.machines]
    for r in range(rows):
        row = machines[r * columns : (r + 1) * columns]
        # Count free cells between the complete Pre/Machine/Post units.
        for left, right in zip(row, row[1:]):
            assert right.x - 1 - (left.x + 1) - 1 == 2
        assert row[0].x - 1 - (footprints["input"].x + 1) == 1
        assert footprints["output"].x - (row[-1].x + 2) == 1
        if r:
            assert row[0].y - machines[(r - 1) * columns].y - 1 == 2
    assert machines[0].y == 1
    assert f.grid.height - machines[-1].y - 1 == 1
    assert len({(p.cell.x, p.cell.y) for p in f.ports}) == len(f.ports)
    for r in range(rows):
        for owner, adjacent in (
            ("input", f"machine_{r * columns:02d}_pre"),
            ("output", f"machine_{(r + 1) * columns - 1:02d}_post"),
        ):
            dock = next(p for p in f.ports if p.port_id == f"{owner}_{r}")
            assert {b.target.buffer_id for b in dock.bindings} == {owner}
            side = next(p for p in f.ports if p.port_id == f"{adjacent}_side")
            assert dock.cell.x == side.cell.x
            assert side.cell.y - dock.cell.y == 1


@pytest.mark.parametrize("scale,width,height", [("small", 22, 6), ("medium", 27, 12)])
def test_compact_base_and_social_observations_use_actual_geometry(scale, width, height):
    from smartsom.domain.base_decisions import BaseCandidate

    sim = ProductionSimulator(
        generate_scenario(base_factory(scale), workload(), 17, "normal", 2000),
        contract="base",
    )
    p = sim.protocol
    # Initial AGVs at (3, 0) and (8, 0); the nearest Input port is (1, 0).
    p.accept("agv_01", "input")
    base = p.vehicle_vector("agv_01", "input")
    assert base[0] == pytest.approx(7 / p.distance_scale)
    assert base[16:18] == pytest.approx((8 / width, 0 / height))
    # The bottom-right vehicle tests nonzero y and both map dimensions.
    last = f"agv_{len(sim.agvs) - 1:02d}"
    assert p.vehicle_vector(last)[16:18] == pytest.approx(
        ((width - 4) / width, (height - 3) / height)
    )
    candidate = BaseCandidate("input", ("input",), p.vehicle_vector("agv_00", "input"))
    plain = p.make_request(
        "dispatcher", "agv_00", "pickup", (candidate,), destinations=("input",)
    )
    p.social_arm = "dispatcher-pickup"
    social = p.make_request(
        "dispatcher", "agv_00", "pickup", (candidate,), destinations=("input",)
    )
    assert replace(social, social=None) == plain
    assert social.social[0].links[0].peers[0][2] == pytest.approx(7 / p.distance_scale)
    # Distinct boundary docks remain one cell apart along the same corridor.
    sim.agvs["agv_01"]["cell"] = [1, 0]
    assert p.vehicle_vector("agv_01", "input")[0] == 0
    assert p.vehicle_vector("agv_01", "machine_00_pre")[0] == pytest.approx(
        1 / p.distance_scale
    )


def test_source_reservation_is_not_job_binding_and_loading_is_one_tick():
    sim = sim_with()
    p = sim.protocol
    p.accept("agv_00", "input")
    assert sim.agvs["agv_00"]["job"] is None
    assert p.quota("input") == 3
    sim.agvs["agv_00"]["cell"] = [1, 3]
    decide(sim)
    service = sim.agvs["agv_00"]["service"]
    job = service["job"]
    assert sim.jobs[job]["location"] == "input"
    assert job not in p.ready("input")
    assert "agv_00" in p.commitments("input")
    wait(sim)
    assert sim.tick == 1 and sim.agvs["agv_00"]["job"] == job
    assert "agv_00" not in p.commitments("input")


def test_empty_buffer_observes_incoming_jobs_direct_downstream():
    sim = sim_with(steps=("A", "A", "B"), n=1)
    p = sim.protocol
    job = p.items("input")[0]
    place(sim, job, "agv_00")
    request = p.make_request(
        "pre", "machine_00_pre", "accept", (), applicants=("agv_00",)
    )
    assert request.sets[0] == ()  # Incoming cargo is not local inventory.
    assert len(request.sets[3]) == 9  # This station plus two compatible B stations.
    assert request.sets[1][0][18:22] == (0, 1, 0, 0)
    assert all(row[21:25] == (0, 1, 0, 0) for row in request.sets[3][3:])
    machine = p.make_request("machine", "machine_00", "process", ())
    assert len(machine.sets[3]) == 3  # Machine still has no remote summaries.


def test_site_pair_features_distinguish_vehicle_interaction_positions():
    sim = sim_with(n=2)
    p = sim.protocol
    for v, cell in (("agv_00", [1, 3]), ("agv_01", [1, 7])):
        p.accept(v, "input")
        sim.agvs[v]["cell"] = cell
    request = next(
        r for r in decide(sim) if r.group == "input" and r.stage == "service"
    )
    job = request.candidates[0].action[0]
    pairs = [c for c in request.candidates if c.action[0] == job]
    assert len(pairs) == 2
    assert pairs[0].features[20:22] != pairs[1].features[20:22]


def test_stop_rejects_only_remaining_applicants_and_preserves_accepted():
    sim = sim_with(n=2)
    p = sim.protocol
    jobs = p.items("input")
    for job in jobs:
        place(sim, job, "machine_00_post")
    accepted = []

    def choose(r):
        if r.stage == "accept" and r.group == "post":
            if not accepted:
                action = r.candidates[0].action
                accepted.append(action[0])
                return action
            return ("STOP",)
        return r.candidates[0].action

    requests = decide(sim, choose)
    assert p.commitments("machine_00_post") == tuple(accepted)
    assert p.metrics["stop"] == 1
    assert any(v for v in p.rejected.values())
    assert all(
        not (r.group == "input" and any(c.action == ("STOP",) for c in r.candidates))
        for r in requests
    )


def test_unload_locks_actual_pre_space_not_station_quota():
    sim = sim_with(n=6)
    p = sim.protocol
    jobs = list(p.items("input"))
    for job in jobs[:3]:
        place(sim, job, "machine_00_pre")
    for v, job, cell in zip(
        ("agv_00", "agv_01", "agv_02"), jobs[3:], ((3, 2), (3, 4), (2, 3)), strict=True
    ):
        place(sim, job, v)
        sim.agvs[v]["cell"] = list(cell)
        p.accept(v, "machine_00_pre")
    # One local start frees a second real slot; three arrivals cannot all unload.
    decide(sim)
    services = [a["service"] for a in sim.agvs.values() if a["service"]]
    assert len(services) == 2
    assert len(p.items("machine_00_pre")) == 2
    assert len(p.commitments("machine_00_pre")) == 3
    wait(sim)
    assert len(p.items("machine_00_pre")) == 4
    assert len(p.commitments("machine_00_pre")) == 1


def test_fault_at_unload_completion_cancels_without_transferring():
    sim = sim_with(n=1, outages=(Outage("machine_00", 1, 5),))
    p = sim.protocol
    job = p.items("input")[0]
    place(sim, job, "agv_00")
    sim.agvs["agv_00"]["cell"] = [3, 2]
    p.accept("agv_00", "machine_00_pre")
    decide(sim)
    assert sim.agvs["agv_00"]["service"]
    wait(sim)
    assert sim.agvs["agv_00"]["job"] == job
    assert sim.agvs["agv_00"]["target"] is None
    assert not p.items("machine_00_pre")
    assert p.metrics["cancellations"] == 1


def test_completion_at_fault_and_same_machine_continuation():
    sim = sim_with(steps=("A", "A", "B"), n=1, outages=(Outage("machine_00", 2, 5),))
    job = sim.protocol.items("input")[0]
    place(sim, job, "machine_00_pre")
    decide(sim)
    wait(sim)
    wait(sim)
    assert sim.jobs[job]["step"] == 1
    assert sim.jobs[job]["location"] == "machine_00"
    assert sim.machine_choices("machine_00") == []
    assert sim.machine_state["machine_00"]["status"] == "READY"
    while sim.tick < 5:
        wait(sim)
    assert {j for j, _ in sim.machine_choices("machine_00")} == {job}
    decide(sim)
    assert sim.machine_state["machine_00"]["mode"] == "normal"
    assert sim.protocol.metrics["machine_choices"] == 2


def test_observations_do_not_use_hidden_future_or_defects():
    sim = sim_with(n=2)
    p = sim.protocol
    first = decide(sim)
    critic = p.critic_features()
    for row in sim.jobs.values():
        row["defective"] = not row["defective"]
    sim.scenario = replace(sim.scenario, outages=(Outage("machine_00", 100, 999),))
    assert p.critic_features() == critic
    for r in first:
        assert len(r.context) == FEATURES
        assert all(len(c.features) == FEATURES for c in r.candidates)
        assert len(r.sets) == 4


def test_truncated_distribution_and_extension_preserve_realization():
    w, f = workload(), base_factory("small")
    rng = random.Random(7)
    distribution = w.fault_intervals["high"]
    samples = [distribution.draw(rng) for _ in range(20000)]
    assert min(samples) >= 15 and max(samples) <= 450
    assert abs(sum(samples) / len(samples) - 140.1718) < 3
    a = generate_scenario(f, w, 17, "high", 200)
    b = generate_scenario(f, w, 17, "high", 2000)
    assert a.demands == b.demands
    assert a.outages == tuple(o for o in b.outages if o.start < a.tick_limit)
    assert all(4 <= len(d.steps) <= 5 for d in a.demands)
    assert a.tick_limit == max(d.release_at for d in a.demands) + 200


def test_shared_mover_displaces_idle_vehicles_without_collision():
    sim = sim_with()
    mover = SharedMover(sim.protocol.routes)
    vehicles = sim.protocol.traffic_view()
    vehicles["agv_00"].update(cell=(20, 4), owner="output", ready=True, due=-100)
    vehicles["agv_01"].update(cell=(21, 4), rush=True)
    vehicles["agv_02"].update(cell=(22, 4))
    initial = vehicles["agv_00"]["cell"]
    from smartsom.domain.production import MOVES

    for t in range(20):
        actions = mover.moves(vehicles, t)
        for v, action in actions.items():
            x, y = vehicles[v]["cell"]
            dx, dy = MOVES.get(action, (0, 0))
            vehicles[v]["cell"] = (x + dx, y + dy)
        assert len({v["cell"] for v in vehicles.values()}) == len(vehicles)
        if vehicles["agv_00"]["cell"] in sim.protocol.routes.ports["output"]:
            break
    assert vehicles["agv_00"]["cell"] != initial
    assert vehicles["agv_00"]["cell"] in sim.protocol.routes.ports["output"]


def test_final_failure_ends_once_and_idle_jump_cannot_cross_event():
    sim = sim_with(n=1)
    p = sim.protocol
    job = p.items("input")[0]
    place(sim, job, "agv_00")
    sim.jobs[job].update(step=2, defective=True)
    sim.agvs["agv_00"]["cell"] = [22, 3]
    p.accept("agv_00", "output")
    decide(sim)
    with pytest.raises(ValueError, match="cross a required physical event"):
        sim.step(JointCommand(agvs=tuple((v, "WAIT") for v in sim.agvs)), elapsed=2)
    outcome = wait(sim)
    assert sim.status == "completed" and sim.tick == 1
    assert outcome["reward"] == -1000
    assert sim.jobs[job]["quality"] == "FAIL"
    assert len(sim.completed) == 1 and len(sim.jobs) == 1
    assert sim.privileged_reward_components()["failed"] == 1


def test_sparse_execution_trace_reconstructs_integer_positions():
    import io
    import json

    from smartsom.trace.base import positions_at

    factory = base_factory("small")
    trace = io.StringIO()
    report, _, _ = episode(
        factory, workload(), options(), BaseRule(), 17, "normal", 2000, trace=trace
    )
    records = [json.loads(line) for line in trace.getvalue().splitlines()]
    initial = positions_at(factory, records, 0)
    final = positions_at(factory, records, report["tick"])
    assert set(initial) == set(final) and len(set(final.values())) == 8
    for row in records:
        if row["end"] - row["start"] > 1:
            assert positions_at(factory, records, row["start"]) == positions_at(
                factory, records, row["end"] - 1
            )
    with pytest.raises(ValueError, match="not recorded"):
        positions_at(factory, records, report["tick"] + 1)


@pytest.mark.parametrize("condition", ["normal", "high"])
def test_full_episode_ledger_and_random_mode_choices(condition):
    opts = options()
    report, rewards, _ = episode(
        base_factory("small"),
        workload(),
        opts,
        BaseRule("random", 3),
        42,
        condition,
        2000,
    )
    assert report["completed"] == 8
    expected = -(
        report["mean_tardiness"]
        + opts.quality_penalty * (1 - report["final_passing_rate"])
    )
    assert math.isclose(sum(rewards), expected, abs_tol=1e-7)
    assert report["protocol"]["physical_ticks"] == len(rewards) == report["tick"]


@pytest.mark.parametrize("version", ["native-v2"])
def test_dispatcher_observation_version_preserves_following_operation(version):
    from smartsom.domain.base_observation import VEHICLE

    # Current B followed by A makes the overwritten A bit observable.
    sim = sim_with(steps=("B", "A"), n=1)
    protocol = sim.protocol
    protocol.configure(options())
    place(sim, protocol.items("input")[0], "agv_00")
    protocol.rejected["agv_00"] = {"machine_00_pre", "machine_01_pre"}
    vehicle = protocol.vehicle_vector("agv_00")
    request = protocol.make_request("dispatcher", "agv_00", "delivery", ())
    assert vehicle[VEHICLE.following_operation] == (1, 0, 0, 0)
    assert request.context[VEHICLE.following_operation] == (1, 0, 0, 0)
    assert request.context[22] == 0.1
    # Detached numeric observations never expose source state for mutation.
    assert isinstance(request.context, tuple)
    assert sim.jobs[sim.agvs["agv_00"]["job"]]["step"] == 0


def test_empty_wait_is_opt_in_and_consumes_one_physical_tick():
    sim = sim_with(n=1)
    protocol = sim.protocol
    protocol.configure(with_protocol(options(), allow_empty_wait=True))
    requests = decide(
        sim,
        lambda r: ("EMPTY_WAIT",) if r.group == "dispatcher" else BaseRule().choose(r),
    )
    dispatch = [r for r in requests if r.group == "dispatcher"]
    assert len(dispatch) == len({r.owner for r in dispatch}) == len(sim.agvs)
    assert all(r.stage == "pickup" for r in dispatch)
    assert all(a["target"] is None and a["job"] is None for a in sim.agvs.values())
    protocol.advance({v: "WAIT" for v in sim.agvs})
    assert sim.tick == 1
    assert any(r.group == "dispatcher" for r in decide(sim))


def test_stop_can_be_disabled_without_changing_service_permissions():
    sim = sim_with(n=1)
    sim.protocol.configure(with_protocol(options(), allow_stop=False))
    requests = decide(sim)
    assert all(c.action != ("STOP",) for r in requests for c in r.candidates)
    assert all(c.action != ("EMPTY_WAIT",) for r in requests for c in r.candidates)


def test_static_route_owner_queries_follow_public_mutations():
    from smartsom.engine.base_routes import StaticRoutes

    factory = base_factory("small")
    routes = StaticRoutes(factory)
    fresh = StaticRoutes(factory)
    owners = list(routes.ports)
    origin, destination = owners[:2]
    cell = routes.ports[origin][0]

    def expected():
        return routes.distances(routes.ports[destination]).get(
            tuple(cell), float("inf")
        )

    assert routes.to_owner(cell, destination) == expected()
    original = fresh.to_owner(cell, destination)
    goals = tuple(sorted(set(routes.ports[destination])))
    routes.cache[goals][tuple(cell)] = 12345
    assert routes.to_owner(cell, destination) == 12345
    assert routes.between(origin, destination) == min(
        routes.distances(routes.ports[destination]).get(tuple(c), float("inf"))
        for c in routes.ports[origin]
    )
    assert fresh.to_owner(cell, destination) == original
    routes.cache.clear()
    assert routes.to_owner(cell, destination) == original
    routes.cache = {}
    assert routes.to_owner(cell, destination) == original
    routes.ports[destination][:] = [cell]
    assert routes.to_owner(cell, destination) == expected() == 0
    routes.ports[destination] = [routes.cells[-1]]
    assert routes.to_owner(cell, destination) == expected()
    assert fresh.to_owner(cell, destination) == original
    routes.ports[origin] = []
    with pytest.raises(ValueError, match="empty"):
        routes.between(origin, "missing-owner")
    with pytest.raises(KeyError):
        routes.to_owner(cell, "missing-owner")


def test_static_route_queries_all_cells_and_owner_pairs():
    from smartsom.engine.base_routes import StaticRoutes

    routes = StaticRoutes(base_factory("small"))
    for owner, ports in routes.ports.items():
        expected = routes.distances(ports)
        for cell in routes.cells:
            assert routes.to_owner(cell, owner) == expected.get(cell, float("inf"))
        for origin, cells in routes.ports.items():
            assert routes.between(origin, owner) == min(
                expected.get(tuple(c), float("inf")) for c in cells
            )


def _nominal_enumerated(factory, steps, routes):
    from itertools import product

    pre = {
        b.machine_id: b.buffer_id for b in factory.buffers if b.role == "machine_pre"
    }
    post = {
        b.machine_id: b.buffer_id for b in factory.buffers if b.role == "machine_post"
    }
    source = next(b.buffer_id for b in factory.buffers if b.role == "system_input")
    output = next(b.buffer_id for b in factory.buffers if b.role == "system_output")
    candidates = [
        [m for m in factory.machines if step.operation_type in m.operation_types]
        for step in steps
    ]
    costs = []
    for path in product(*candidates):
        previous, previous_type, cost = None, None, 0.0
        for step, machine in zip(steps, path, strict=True):
            if previous_type == step.operation_type and previous != machine.machine_id:
                break
            processing = max(
                1,
                math.ceil(
                    float(step.work_ticks_on(machine.machine_id))
                    / float(machine.processing_rate_multiplier)
                ),
            )
            travel = (
                0
                if previous == machine.machine_id
                else routes.between(
                    source if previous is None else post[previous],
                    pre[machine.machine_id],
                )
                + 2
            )
            cost = cost + travel + processing
            previous, previous_type = machine.machine_id, step.operation_type
        else:
            costs.append(cost + routes.between(post[previous], output) + 2)
    if not costs:
        raise ValueError("no feasible operation chain")
    return min(costs)


def test_nominal_layout_preserves_reference_and_changed_factory():
    from smartsom.config.base_diagnostic import nominal_baseline
    from smartsom.engine.base_routes import StaticRoutes

    factory = base_factory("small")
    changed = replace(
        factory,
        machines=tuple(
            replace(m, processing_rate_multiplier="1.25") for m in factory.machines
        ),
    )
    for current in (factory, changed, factory):
        routes = StaticRoutes(current)
        for kinds in (("A", "B", "C", "D"), ("A", "A", "B"), ("D",)):
            steps = tuple(
                ProductionStep(str(i), kind, 17, reference_ticks="17.125")
                for i, kind in enumerate(kinds)
            )
            assert nominal_baseline(current, steps, routes) == _nominal_enumerated(
                current, steps, routes
            )
        with pytest.raises(ValueError, match="no feasible"):
            nominal_baseline(current, (ProductionStep("bad", "Z", 1),), routes)


def test_scenario_nominal_reuse_keeps_seed_config_and_worlds_detached():
    from smartsom.config.base_diagnostic import nominal_baseline
    from smartsom.engine.base_routes import StaticRoutes

    factory, jobs = base_factory("small"), workload()
    before = random.getstate()
    original = generate_scenario(factory, jobs, 73, "normal", 100)
    varied = generate_scenario(
        factory, jobs.model_copy(update={"orders": 9}), 74, "high", 200
    )
    again = generate_scenario(factory, jobs, 73, "normal", 100)
    assert original == again and original is not again
    assert original.demands is not again.demands
    assert varied != original
    assert random.getstate() == before
    # Evaluating unrelated nominal work cannot alter a later world's outcomes.
    routes = StaticRoutes(factory)
    for demand in original.demands:
        assert nominal_baseline(factory, demand.steps, routes) == _nominal_enumerated(
            factory, demand.steps, routes
        )
    assert generate_scenario(factory, jobs, 73, "normal", 100) == original


def test_declared_work_cache_preserves_exact_overrides_and_replacement():
    from fractions import Fraction

    from smartsom.domain.production import _declared_work

    step = ProductionStep("operation", "A", 3, (("m1", 7),), "22/7")
    _declared_work.cache_clear()
    for _ in range(3):
        assert step.work_ticks_on("m1") == Fraction(7)
        assert step.work_ticks_on("m2") == Fraction(22, 7)
    assert _declared_work.cache_info().hits == 4
    changed = replace(step, reference_ticks="23/7", machine_nominal_ticks=(("m1", 8),))
    assert changed.work_ticks_on("m1") == 8
    assert changed.work_ticks_on("m2") == Fraction(23, 7)
    # Public frozen-object bypass must not return the preceding declaration.
    object.__setattr__(step, "reference_ticks", "1/3")
    assert step.work_ticks_on("m2") == Fraction(1, 3)
    assert step.work_ticks_on("m1") == 7


def test_dispatcher_projection_reuse_is_scoped_to_one_request():
    sim = sim_with(n=4)
    protocol = sim.protocol
    original = protocol.make_request
    requests = []

    def checked(*args, **kwargs):
        reused = original(*args, **kwargs)
        uncached = original(
            *args, **{k: v for k, v in kwargs.items() if k != "station_rows"}
        )
        assert reused == uncached
        if kwargs.get("station_rows") is not None:
            requests.append(reused)
        return reused

    protocol.make_request = checked
    mover = SharedMover(protocol.routes)
    for _ in range(80):
        decide(sim)
        if sim.done:
            break
        protocol.advance(mover.moves(protocol.traffic_view(), sim.tick))
    assert len(requests) > 1
    assert len({r.tick for r in requests}) > 1


def test_parking_distance_lookup_is_per_vehicle_and_preserves_ties():
    from types import SimpleNamespace

    calls = []
    cells = ((0, 0), (1, 0), (2, 0))

    def distances(goals):
        calls.append(goals)
        return {cell: abs(cell[0] - goals[0][0]) for cell in cells}

    routes = SimpleNamespace(
        cells=cells,
        ports={},
        neighbors={
            cells[0]: (cells[1],),
            cells[1]: (cells[0], cells[2]),
            cells[2]: (cells[1],),
        },
        distances=distances,
    )
    mover = SharedMover(routes)
    vehicles = {
        "v": {
            "cell": cells[1],
            "rush": False,
            "ready": False,
            "due": 0,
            "service": False,
            "owner": None,
        }
    }
    assert mover.moves(vehicles, 0) == {"v": "WAIT"}
    # One map to score every parking cell; one to reserve the selected goal.
    assert len(calls) == 2


def test_dispatcher_summary_reuse_matches_live_facts_before_each_action():
    sim = sim_with(n=6)
    protocol = sim.protocol
    original = protocol.make_request
    seen = set()

    def checked(group, owner, stage, candidates, **kwargs):
        if group == "dispatcher":
            for candidate in candidates:
                target = candidate.action[0]
                if target == "EMPTY_WAIT":
                    continue
                machine = protocol.station.get(target)
                features = candidate.features
                if sim.agvs[owner]["job"] is None:
                    assert features[2:14] == protocol.summary(protocol.ready(target))
                if machine:
                    state = sim.machine_state[machine]
                    assert features[15:27] == protocol.summary(
                        protocol.items(sim.pre[machine])
                    )
                    jobs = (state["job"],) if state["job"] else ()
                    assert features[27:39] == protocol.summary(jobs)
                    assert (
                        features[42]
                        == sum(
                            protocol.nominal(j, machine)
                            for j in protocol.items(sim.pre[machine])
                        )
                        / protocol.time_scale
                    )
                seen.add((stage, bool(machine)))
        return original(group, owner, stage, candidates, **kwargs)

    protocol.make_request = checked
    mover = SharedMover(protocol.routes)
    while not sim.done:
        decide(sim)
        protocol.advance(mover.moves(protocol.traffic_view(), sim.tick))
    assert ("pickup", False) in seen and ("pickup", True) in seen
    assert ("delivery", True) in seen and ("delivery", False) in seen


def test_parking_owner_map_reuse_matches_public_queries_and_mutations():
    from types import SimpleNamespace

    from smartsom.engine.base_routes import StaticRoutes

    routes = StaticRoutes(base_factory("small"))
    # A public-route proxy exercises the original per-cell query path.
    reference_routes = SimpleNamespace(
        cells=routes.cells,
        ports=routes.ports,
        neighbors=routes.neighbors,
        distances=routes.distances,
        to_owner=routes.to_owner,
    )
    actual, reference = SharedMover(routes), SharedMover(reference_routes)
    owner = next(iter(routes.ports))
    vehicles = {
        str(i): {
            "cell": cell,
            "rush": False,
            "ready": False,
            "due": i,
            "service": False,
            "owner": owner,
        }
        for i, cell in enumerate(routes.cells[:5])
    }
    moves = {
        "LEFT": (-1, 0),
        "RIGHT": (1, 0),
        "UP": (0, -1),
        "DOWN": (0, 1),
        "WAIT": (0, 0),
    }
    for tick in range(8):
        if tick == 2:
            routes.ports[owner][:] = [routes.cells[-1]]
        elif tick == 4:
            routes.cache.clear()
        elif tick == 6:
            routes._distances_to_owner(owner)[routes.cells[0]] = -100
        actions = actual.moves(vehicles, tick)
        assert actions == reference.moves(vehicles, tick)
        assert actual.get_state() == reference.get_state()
        for vehicle, action in actions.items():
            x, y = vehicles[vehicle]["cell"]
            dx, dy = moves[action]
            vehicles[vehicle]["cell"] = (x + dx, y + dy)
    # An instance override remains observable even on the native routes class.
    calls = []

    def custom(cell, target):
        calls.append((cell, target))
        return 0

    routes.to_owner = custom
    actual.moves(vehicles, 8)
    assert len(calls) > len(vehicles)


def test_local_facts_match_uncached_requests_services_and_physics():
    from contextlib import nullcontext

    actual, reference = sim_with(n=8), sim_with(n=8)
    actual.protocol.social_arm = reference.protocol.social_arm = "all"
    reference.protocol.local_facts = nullcontext
    movers = [SharedMover(s.protocol.routes) for s in (actual, reference)]
    requests = 0
    while not actual.done:
        left, right = decide(actual), decide(reference)
        assert left == right
        requests += len(left)
        assert actual.protocol._local_facts is None
        assert actual.protocol.metrics == reference.protocol.metrics
        results = [
            s.protocol.advance(m.moves(s.protocol.traffic_view(), s.tick))
            for s, m in zip((actual, reference), movers, strict=True)
        ]
        assert results[0] == results[1]
        assert actual.snapshot(public=False) == reference.snapshot(public=False)
    assert reference.done and requests > 10


def test_local_facts_end_on_exception_and_before_observation_callback():
    from types import SimpleNamespace

    sim = sim_with()
    protocol = sim.protocol
    with pytest.raises(RuntimeError):
        with protocol.local_facts():
            assert protocol._local_facts is not None
            raise RuntimeError("abort read phase")
    assert protocol._local_facts is None
    observed = []

    class Observation:
        def applies(self, request):
            observed.append(getattr(protocol, "_local_facts", None))
            return False

    protocol.observation_transform = Observation()
    decide(sim)
    assert observed and all(x is None for x in observed)
    owner = next(iter(sim.storage))
    # External/central reservation readers are not captured by a native cache.
    protocol.central = SimpleNamespace(destinations_reserved=lambda target: ())
    with protocol.local_facts():
        assert protocol._local_facts is None
        assert protocol.items(owner) == tuple(
            j for rows in sim.storage[owner].values() for j in rows
        )


def test_parking_rank_reuse_preserves_live_maps_and_restore():
    from types import SimpleNamespace

    from smartsom.engine.base_routes import StaticRoutes

    routes = StaticRoutes(base_factory("small"))
    reference_routes = SimpleNamespace(
        cells=routes.cells,
        ports=routes.ports,
        neighbors=routes.neighbors,
        distances=routes.distances,
        to_owner=routes.to_owner,
    )
    actual, reference = SharedMover(routes), SharedMover(reference_routes)
    owner = next(iter(routes.ports))
    cell = next(
        c
        for c in routes.cells
        if c not in {p for ps in routes.ports.values() for p in ps}
    )
    vehicles = {
        "v": dict(cell=cell, rush=False, ready=False, due=0, service=False, owner=owner)
    }
    for tick in range(12):
        if tick == 3:
            routes.distances((cell,))[routes.cells[-1]] = -200
        if tick == 5:
            routes._distances_to_owner(owner)[routes.cells[-1]] = -300
        if tick == 7:
            routes.cache.clear()
        if tick == 9:
            routes.ports[owner] = [routes.cells[-1]]
        assert actual.moves(vehicles, tick) == reference.moves(vehicles, tick)
        assert actual.get_state() == reference.get_state()
    assert actual._parking_orders
    restored = SharedMover(routes)
    restored.set_state(actual.get_state())
    assert restored.moves(vehicles, 12) == actual.moves(vehicles, 12)


def test_parking_rank_detects_inplace_road_cell_changes():
    from types import SimpleNamespace

    from smartsom.engine.base_routes import StaticRoutes

    routes = StaticRoutes(base_factory("small"))
    port_cells = {p for ps in routes.ports.values() for p in ps}
    cell = next(c for c in routes.cells if c not in port_cells)
    routes.cells = [c for c in routes.cells if c != cell]
    proxy = SimpleNamespace(
        cells=routes.cells,
        ports=routes.ports,
        neighbors=routes.neighbors,
        distances=routes.distances,
        to_owner=routes.to_owner,
    )
    actual, reference = SharedMover(routes), SharedMover(proxy)
    vehicles = {
        "v": dict(cell=cell, rush=False, ready=False, due=0, service=False, owner=None)
    }
    assert actual.moves(vehicles, 0) == reference.moves(vehicles, 0)
    routes.cells.append(cell)
    assert reference.moves(vehicles, 1) == {"v": "WAIT"}
    assert actual.moves(vehicles, 1) == {"v": "WAIT"}


def test_read_phase_vectors_refresh_after_same_tick_mutation():
    sim = sim_with()
    p = sim.protocol
    job = next(iter(sim.jobs))
    with p.local_facts():
        before = p.job_vector(job)
        summary = p.summary((job, job))
        assert summary[0] == 2 / p.count_scale
        assert p.summary(iter((job, job))) == summary
    sim.jobs[job]["risk"] = before[5] + 0.5
    with p.local_facts():
        assert p.job_vector(job)[5] == before[5] + 0.5
        assert p.job_vector(job) != before


def test_standard_work_refreshes_machine_and_operation_declarations():
    from decimal import Decimal

    sim = sim_with(steps=("A",), n=1)
    p = sim.protocol
    job = next(iter(sim.jobs))
    demand_id = sim.jobs[job]["demand"]
    op = ProductionStep("fractional", "A", 7, reference_ticks="13/2")
    sim.demands[demand_id] = replace(sim.demands[demand_id], steps=(op,))
    for m, spec in tuple(sim.machines.items()):
        sim.machines[m] = replace(spec, processing_rate_multiplier=Decimal("2"))
    assert p.remaining_work(job) == 4.0
    sim.tick += 1
    for m, spec in tuple(sim.machines.items()):
        sim.machines[m] = replace(spec, processing_rate_multiplier=Decimal("3"))
    assert p.remaining_work(job) == 3.0
    sim.tick += 1
    sim.demands[demand_id] = replace(
        sim.demands[demand_id], steps=(replace(op, reference_ticks="19/2"),)
    )
    assert p.remaining_work(job) == 4.0


def test_standard_work_keeps_custom_operation_calls():
    from fractions import Fraction

    calls = []

    class CustomStep(ProductionStep):
        def work_ticks_on(self, machine_id):
            calls.append(machine_id)
            return Fraction(5)

    sim = sim_with(steps=("A",), n=1)
    job = next(iter(sim.jobs))
    demand_id = sim.jobs[job]["demand"]
    sim.demands[demand_id] = replace(
        sim.demands[demand_id], steps=(CustomStep("custom", "A", 1),)
    )
    assert sim.protocol.remaining_work(job) == 5.0
    first = tuple(calls)
    assert first
    sim.tick += 1
    assert sim.protocol.remaining_work(job) == 5.0
    assert tuple(calls) == first + first


def test_candidate_read_facts_refresh_services_and_nominal_at_same_tick():
    from decimal import Decimal

    sim = sim_with(steps=("A",), n=1)
    p = sim.protocol
    job = next(iter(sim.jobs))
    owner = sim.jobs[job]["location"]
    machine = next(m for m, spec in sim.machines.items() if "A" in spec.operation_types)
    with p.local_facts():
        assert job in p.ready(owner)
        assert p.nominal(job, machine) == 2
    vehicle = next(iter(sim.agvs.values()))
    vehicle["service"] = {"kind": "pickup", "job": job}
    sim.machines[machine] = replace(
        sim.machines[machine], processing_rate_multiplier=Decimal("2")
    )
    with p.local_facts():
        assert job not in p.ready(owner)
        assert p.nominal(job, machine) == 1
    vehicle["service"] = None
    with p.local_facts():
        assert job in p.ready(owner)


@pytest.mark.parametrize(
    "fault,message",
    [
        ("overflow", "storage overflow"),
        ("location", "storage identity mismatch"),
        ("duplicate_owner", "job has multiple physical owners"),
        (
            "duplicate_demand",
            "duplicate active attempt or completed demand still active",
        ),
        ("lost", "lost demand"),
        ("qualified", "qualified shipment has no shipped identity"),
    ],
)
def test_invariant_checks_preserve_failure_boundaries(fault, message):
    sim = sim_with(n=1)
    job = next(iter(sim.jobs))
    row = sim.jobs[job]
    inventory = sim.storage[row["location"]][row["slot"]]
    if fault == "overflow":
        owner = next(iter(sim.pre.values()))
        slot = next(iter(sim.storage[owner]))
        sim.storage[owner][slot] = [job] * (sim.capacity[owner][slot] + 1)
        # Overflow must be reported before the deliberately wrong job owner.
    elif fault == "location":
        row["location"] = "wrong-owner"
    elif fault == "duplicate_owner":
        sim.queue.append(job)
    elif fault == "duplicate_demand":
        sim.jobs["duplicate"] = dict(row)
        inventory.append("duplicate")
    elif fault == "lost":
        inventory.remove(job)
    else:
        sim._qualified_shipments.add("unshipped")
    with pytest.raises(AssertionError, match=message):
        sim._check()


@pytest.mark.parametrize("replacement", [False, True])
def test_nominal_read_cache_preserves_custom_work_calls(monkeypatch, replacement):
    from fractions import Fraction

    calls = []

    def custom_work(self, machine):
        calls.append(machine)
        return Fraction(1 + len(calls))

    class CustomStep(ProductionStep):
        work_ticks_on = custom_work

    sim = sim_with(steps=("A",), n=1)
    job = next(iter(sim.jobs))
    demand = sim.jobs[job]["demand"]
    machine = next(m for m, spec in sim.machines.items() if "A" in spec.operation_types)
    if replacement:
        monkeypatch.setattr(ProductionStep, "work_ticks_on", custom_work)
    else:
        sim.demands[demand] = replace(
            sim.demands[demand], steps=(CustomStep("custom", "A", 1),)
        )
    with sim.protocol.local_facts():
        assert sim.protocol.nominal(job, machine) == 2
        assert sim.protocol.nominal(job, machine) == 3
    assert calls == [machine, machine]


@pytest.mark.parametrize("kind", [None, "pickup", "drop_off"])
def test_service_membership_refreshes_between_read_scopes(kind):
    sim = sim_with(steps=("A",), n=1)
    p = sim.protocol
    job = next(iter(sim.jobs))
    owner = sim.jobs[job]["location"]
    vehicle = next(iter(sim.agvs.values()))
    for current in (None, kind, None):
        vehicle["service"] = (
            {"kind": current, "job": job} if current is not None else None
        )
        # The public unscoped path is the reference for each new service state.
        expected_ready = p.ready(owner)
        expected_vector = p.job_vector(job)
        with p.local_facts():
            assert p.ready(owner) == expected_ready
            assert p.job_vector(job) == expected_vector
            assert p.ready(owner) == expected_ready
            assert p.job_vector(job) == expected_vector
        assert (job in expected_ready) == (current != "pickup")


def test_nominal_reuse_refreshes_mode_operation_and_work_method(monkeypatch):
    from fractions import Fraction

    sim = sim_with(steps=("A",), n=1)
    p = sim.protocol
    job = next(iter(sim.jobs))
    demand = sim.jobs[job]["demand"]
    machine = next(m for m, spec in sim.machines.items() if "A" in spec.operation_types)
    assert p.nominal(job, machine) == 2
    resource = sim.machines[machine]
    sim.machines[machine] = replace(
        resource,
        quality_modes=tuple(
            replace(mode, time_scale="1.5")
            if mode.quality_mode_id == "normal"
            else mode
            for mode in resource.quality_modes
        ),
    )
    assert p.nominal(job, machine) == 3
    sim.demands[demand] = replace(
        sim.demands[demand],
        steps=(ProductionStep("replacement", "A", 7, reference_ticks="19/3"),),
    )
    assert p.nominal(job, machine) == 10
    assert p.nominal(job, machine) == 10
    calls = []

    def work(self, machine_id):
        calls.append(machine_id)
        return Fraction(2 + len(calls))

    monkeypatch.setattr(ProductionStep, "work_ticks_on", work)
    assert p.nominal(job, machine) == 5
    assert p.nominal(job, machine) == 6
    assert calls == [machine, machine]


def test_work_callback_service_mutation_is_visible_to_next_job_vector():
    from fractions import Fraction

    sim = sim_with(steps=("A",), n=2)
    p = sim.protocol
    first, second = tuple(sim.jobs)
    vehicle = next(iter(sim.agvs.values()))
    calls = []

    class ServiceStartingStep(ProductionStep):
        def work_ticks_on(self, machine_id):
            calls.append(machine_id)
            vehicle["service"] = {"kind": "pickup", "job": second}
            return Fraction(2)

    demand = sim.jobs[first]["demand"]
    sim.demands[demand] = replace(
        sim.demands[demand], steps=(ServiceStartingStep("custom", "A", 2),)
    )
    sim.tick += 1
    with p.local_facts():
        p.job_vector(first)
        after_callback = p.job_vector(second)
    assert calls
    assert vehicle["service"]["job"] == second
    assert after_callback == p.job_vector(second)
