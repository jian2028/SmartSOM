"""Independent physical/accounting invariants of the new opt-in Base contract."""

import math
import random
from dataclasses import replace

import pytest

from smartsom.algorithms.base_rules import BaseRule, SharedMover
from smartsom.config.base_diagnostic import (
    BaseAlgorithm,
    BaseWorkload,
    generate_scenario,
)
from smartsom.config.base_layout import base_factory
from smartsom.domain.base_decisions import FEATURES
from smartsom.domain.production import (
    Demand,
    JointCommand,
    Outage,
    ProductionScenario,
    ProductionStep,
)
from smartsom.engine.production import ProductionSimulator
from smartsom.experiments.base_diagnostic import episode


def options():
    return BaseAlgorithm(schema="smartsom.base-algorithm/v1", method="rule")


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
            base_factory("small"),
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
    "scale,count,shape", [("small", 8, (24, 11)), ("target", 20, (29, 19))]
)
def test_layout_reachability_and_capacities(scale, count, shape):
    f = base_factory(scale)
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


@pytest.mark.parametrize("version", ["native-v1", "native-v2"])
def test_dispatcher_observation_version_preserves_following_operation(version):
    from smartsom.domain.base_observation import VEHICLE

    # Current B followed by A makes the overwritten A bit observable.
    sim = sim_with(steps=("B", "A"), n=1)
    protocol = sim.protocol
    protocol.configure(options().model_copy(update={"observation_schema": version}))
    place(sim, protocol.items("input")[0], "agv_00")
    protocol.rejected["agv_00"] = {"machine_00_pre", "machine_01_pre"}
    vehicle = protocol.vehicle_vector("agv_00")
    request = protocol.make_request("dispatcher", "agv_00", "delivery", ())
    assert vehicle[VEHICLE.following_operation] == (1, 0, 0, 0)
    if version == "native-v1":
        assert request.context[18:22] == (0.1, 0, 0, 0)
        assert request.context[22] == 0
    else:
        assert request.context[VEHICLE.following_operation] == (1, 0, 0, 0)
        assert request.context[22] == 0.1
    # Detached numeric observations never expose source state for mutation.
    assert isinstance(request.context, tuple)
    assert sim.jobs[sim.agvs["agv_00"]["job"]]["step"] == 0


def test_empty_wait_is_opt_in_and_consumes_one_physical_tick():
    sim = sim_with(n=1)
    protocol = sim.protocol
    protocol.configure(options().model_copy(update={"allow_empty_wait": True}))
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
    sim.protocol.configure(options().model_copy(update={"allow_stop": False}))
    requests = decide(sim)
    assert all(c.action != ("STOP",) for r in requests for c in r.candidates)
    assert all(c.action != ("EMPTY_WAIT",) for r in requests for c in r.candidates)
