"""Current credit contracts, ported from the frozen cloud/server handoff."""

from dataclasses import replace

import pytest

from smartsom.domain.base_decisions import BaseCandidate
from smartsom.domain.production import (
    Demand,
    JointCommand,
    MachineCommand,
    Outage,
    ProductionScenario,
    ProductionStep,
)
from smartsom.engine.production import ProductionSimulator
from smartsom.engine.social_observation import project_social
from smartsom.policies.rule import BaseRule
from smartsom.traffic.rule_mover import SharedMover
from tests.helpers.algorithm import algorithm_options
from tests.helpers.historical_geometry import historical_base_factory


def fixture(*, n=6, steps=("A", "B"), outages=(), arm="base"):
    sim = ProductionSimulator(
        ProductionScenario(
            historical_base_factory(),
            tuple(
                Demand(
                    f"job_{i}",
                    tuple(
                        ProductionStep(f"op_{k}", kind, 2)
                        for k, kind in enumerate(steps)
                    ),
                    due_at=50,
                )
                for i in range(n)
            ),
            tick_limit=400,
            outages=outages,
            processing_rounding="ceil",
        ),
        contract="base",
    )
    sim.protocol.configure(
        algorithm_options(
            schema="smartsom.base-algorithm/v1",
            method="rule",
            allow_stop=False,
            allow_empty_wait=False,
            social_arm=arm,
        )
    )
    sim.protocol.social_arm = arm
    return sim


def place(sim, job, owner):
    sim._remove(job)
    if owner in sim.agvs:
        sim.agvs[owner]["job"] = job
        sim.jobs[job].update(location=owner, slot=None, since=sim.tick)
    else:
        sim._place(job, owner, "pool")


def step(sim):
    return sim.step(
        JointCommand(agvs=tuple((v, "WAIT") for v in sim.agvs)), compact=True
    )


def decide(sim):
    gen, policy, seen = sim.protocol.decisions(), BaseRule(), []
    try:
        request = next(gen)
        while True:
            seen.append(request)
            request = gen.send(policy.choose(request))
    except StopIteration:
        return seen


def test_pre_jobs_never_create_post_credit():
    sim = fixture()
    p = sim.protocol
    for j in p.items("input")[:3]:
        place(sim, j, "machine_00_pre")
    assert p.quota("machine_00_post") == 0
    assert p.idle_since == {v: 0 for v in sim.agvs}
    assert p.acceptance_vector("agv_00", "machine_00_pre") == p.vehicle_vector(
        "agv_00", "machine_00_pre"
    )


def test_four_post_plus_processing_allows_five_without_binding_job_or_pre():
    sim = fixture()
    p, machine, post = sim.protocol, "machine_00", "machine_00_post"
    jobs = p.items("input")
    for j in jobs[:4]:
        place(sim, j, post)
    place(sim, jobs[4], sim.pre[machine])
    sim._start(machine, MachineCommand(jobs[4], "normal"))
    assert p.post_credit_components(post) == (4, 1, 0, 5)
    assert p.source_supply(post) == 5
    for v in tuple(sim.agvs)[:5]:
        p.accept(v, post)
        assert sim.agvs[v]["job"] is None
        assert sim.agvs[v]["service"] is None
        assert not p.commitments(sim.pre[machine])
    assert p.quota(post) == 0
    p.check_reservations()
    step(sim)
    step(sim)
    assert sim.machine_state[machine]["status"] == "BLOCKED"
    assert p.post_credit_components(post) == (4, 1, 5, 0)
    sim.machine_state[machine]["down"] = True
    assert p.post_credit_components(post) == (4, 1, 5, 0)


def test_loading_lock_and_actual_transfer_release_supply_and_commitment_together():
    sim = fixture(n=1)
    p, post = sim.protocol, "machine_00_post"
    job = p.items("input")[0]
    place(sim, job, post)
    p.accept("agv_00", post)
    p.start_service("agv_00", "pickup", job, post, "pool")
    assert p.ready(post) == ()
    assert p.post_credit_components(post) == (1, 0, 1, 0)
    assert p.source_supply(post) == 1
    p.check_reservations()
    step(sim)
    assert p.post_credit_components(post) == (0, 0, 0, 0)
    assert p.source_supply(post) == 0
    assert sim.agvs["agv_00"]["job"] == job
    assert not p.commitments("machine_02_pre")


def test_physical_and_machine_alias_cannot_duplicate_one_credit():
    sim = fixture(n=1)
    p, post = sim.protocol, "machine_00_post"
    job = p.items("input")[0]
    place(sim, job, post)
    # Deliberately overlapping references at an atomic-transfer boundary.
    sim.machine_state["machine_00"].update(job=job, status="BLOCKED")
    assert p.post_credit_components(post) == (1, 0, 0, 1)


@pytest.mark.parametrize("steps", [("A", "B"), ("A", "A", "B")])
def test_started_token_survives_completion_local_ready_pause_and_post_transfer(steps):
    sim = fixture(n=1, steps=steps, outages=(Outage("machine_00", 2, 5),))
    p, m, post = sim.protocol, "machine_00", "machine_00_post"
    job = p.items("input")[0]
    place(sim, job, sim.pre[m])
    sim._start(m, MachineCommand(job, "normal"))
    p.accept("agv_00", post)
    for _ in range(2):
        step(sim)
        p.check_reservations()
        assert sum(p.post_credit_components(post)[:2]) == 1
    if steps[1] == "A":
        assert sim.machine_state[m]["status"] == "READY"
        assert p.post_credit_components(post) == (0, 1, 1, 0)
        decide(sim)  # Existing down/empty-POST eligibility cancels the pickup.
        assert any(e["kind"] == "reservation_cancelled" for e in sim.events)
        assert p.post_credit_components(post) == (0, 1, 0, 1)
        assert p.idle_since["agv_00"] == sim.tick
    else:
        assert p.post_credit_components(post) == (1, 0, 1, 0)


def test_idle_age_only_uses_physical_ticks_and_does_not_change_vehicle_vector():
    sim = fixture()
    p = sim.protocol
    sim.tick = 7
    sim.agvs["agv_00"]["cell"][0] += 1  # Unassigned yield does not reset idle.
    old = p.vehicle_vector("agv_00", "input")
    for _ in range(3):
        features = p.acceptance_vector("agv_00", "input")
        assert features[22] == 0 and features[23] == 7 / p.time_scale
        assert features[:23] == old[:23] and features[24:] == old[24:]
    p.accept("agv_00", "input")
    assert "agv_00" not in p.idle_since
    sim.tick = 10
    p.cancel("agv_00", "test_cancel")
    assert p.idle_since["agv_00"] == 10
    assert p.acceptance_vector("agv_00", "input")[23] == 0
    sim.tick = sim.scenario.tick_limit
    p.sync_idle_clocks()
    assert p.idle_since == {}


def test_unload_completion_starts_idle_but_loaded_cancellation_does_not():
    sim = fixture(n=2)
    p = sim.protocol
    job = p.items("input")[0]
    place(sim, job, "agv_00")
    p.accept("agv_00", "machine_00_pre")
    p.cancel("agv_00", "target_breakdown")
    assert "agv_00" not in p.idle_since
    p.accept("agv_00", "machine_00_pre")
    p.start_service("agv_00", "drop_off", job, "machine_00_pre", "pool")
    step(sim)
    assert p.idle_since["agv_00"] == 1
    assert sim.agvs["agv_00"]["job"] is None


def test_pair_dynamic_fields_filter_down_full_unreachable_and_terminal(monkeypatch):
    sim = fixture(n=2)
    p = sim.protocol
    job = p.items("input")[0]
    feasible = [m for m, spec in sim.machines.items() if "A" in spec.operation_types]
    assert len(feasible) == 2
    f = p.pickup_pair_vector(job, "agv_00", "input")
    assert f[22:26] == (2 / p.count_scale, 0, 1, 0)
    sim.machine_state[feasible[0]]["down"] = True
    other = p.items("input")[1]
    place(sim, other, sim.pre[feasible[1]])
    f = p.pickup_pair_vector(job, "agv_00", "input")
    assert f[22:26] == (1 / p.count_scale, 2 / p.time_scale, 1, 0)
    quota = p.quota
    monkeypatch.setattr(
        p, "quota", lambda owner: 0 if owner == sim.pre[feasible[1]] else quota(owner)
    )
    assert p.pickup_pair_vector(job, "agv_00", "input")[22:26] == (0, 0, 0, 0)
    monkeypatch.setattr(p, "quota", quota)
    original = p.routes.to_owner
    monkeypatch.setattr(
        p.routes,
        "to_owner",
        lambda cell, owner: (
            float("inf") if owner == sim.pre[feasible[1]] else original(cell, owner)
        ),
    )
    assert p.pickup_pair_vector(job, "agv_00", "input")[22:26] == (0, 0, 0, 0)
    sim.jobs[job]["step"] = 2
    assert p.pickup_pair_vector(job, "agv_00", "input")[22:26] == (0, 0, 0, 1)


def test_feature_extensions_are_public_preserve_social_and_do_not_read_future():
    sim = fixture(n=2, arm="all")
    p, post = sim.protocol, "machine_00_post"
    job = p.items("input")[0]
    place(sim, job, post)
    p.accept("agv_00", post)
    f = p.pickup_pair_vector(job, "agv_00", post)
    c = BaseCandidate(f"load:{job}:agv_00", (job, "agv_00"), f)
    request = p.make_request("post", post, "service", (c,))
    assert request.context[28:32] == (1 / p.count_scale, 0, 1 / p.count_scale, 0)
    assert request.social == project_social(p, request)
    critic = p.critic_features()
    sim.jobs[job]["defective"] = not sim.jobs[job]["defective"]
    sim.scenario = replace(sim.scenario, outages=(Outage("machine_00", 100, 200),))
    assert p.pickup_pair_vector(job, "agv_00", post) == f
    assert p.critic_features() == critic


def test_unexpected_credit_deficit_stops_admission_and_reports_without_cancellation():
    sim = fixture(n=1)
    p, post = sim.protocol, "machine_00_post"
    job = p.items("input")[0]
    place(sim, job, post)
    p.accept("agv_00", post)
    place(sim, job, "input")  # Deliberate invalid external loss of promised stock.
    assert p.quota(post) == 0
    assert p.quota(post) == 0
    assert p.metrics["pickup_credit_deficits"] == 1
    assert p.metrics["cancellations"] == 0
    with pytest.raises(AssertionError, match="pickup_credit_deficit"):
        p.check_reservations()
    assert p.commitments(post) == ("agv_00",)


@pytest.mark.parametrize("arm", ["base", "all"])
def test_bounded_native_engineering_execution_with_new_invariants(arm):
    sim = fixture(n=4, arm=arm)
    mover = SharedMover(sim.protocol.routes)
    stages = set()
    while not sim.done:
        for r in decide(sim):
            stages.add((r.group, r.stage))
            assert all(c.identity not in ("STOP", "EMPTY_WAIT") for c in r.candidates)
        sim.protocol.advance(mover.moves(sim.protocol.traffic_view(), sim.tick))
    assert sim.status == "completed"
    assert not sim.protocol.metrics["pickup_credit_deficits"]
    assert not sim.protocol.idle_since
    assert ("post", "service") in stages and ("input", "service") in stages


@pytest.mark.parametrize(
    "group,owner", [("input", "input"), ("post", "machine_00_post")]
)
def test_accept_opportunity_counts_one_batch_not_each_candidate_or_sequential_pick(
    group, owner
):
    sim = fixture(n=2)
    p = sim.protocol
    if group == "post":
        for job in p.items("input"):
            place(sim, job, owner)
    requests = decide(sim)
    accept = [r for r in requests if (r.group, r.stage) == (group, "accept")]
    prefix = f"accept_{group}_"
    assert p.metrics[prefix + "rounds"] == 1
    assert p.metrics[prefix + "round_waiting_total"] == 8
    assert p.metrics[prefix + "round_quota_total"] == 2
    assert p.metrics[prefix + "oversubscribed_rounds"] == 1
    assert p.metrics[prefix + "decisions"] == len(accept) == 2
    assert p.metrics[prefix + "decision_real_candidates_total"] == 8 + 7
    assert p.metrics[prefix + "decisions_with_multiple_candidates"] == 2
    assert p.metrics[prefix + "idle_all_zero_rounds"] == 1
    assert p.metrics[prefix + "idle_all_equal_rounds"] == 1
    assert p.metrics[prefix + "idle_distinct_rounds"] == 0


def test_automatic_input_admission_has_round_but_no_accept_decision():
    sim = fixture(n=8)
    decide(sim)
    metrics = sim.protocol.metrics
    assert metrics["accept_input_rounds"] == 1
    assert metrics["accept_input_round_waiting_total"] == 8
    assert metrics["accept_input_round_quota_total"] == 8
    assert metrics["accept_input_oversubscribed_rounds"] == 0
    assert metrics["accept_input_decisions"] == 0


def test_idle_opportunity_categories_have_explicit_round_denominator():
    sim = fixture()
    p = sim.protocol
    waiting = ("agv_00", "agv_01")
    p.record_accept_opportunity("input", "input", waiting)
    sim.tick = 5
    p.record_accept_opportunity("input", "input", waiting)
    p.idle_since["agv_01"] = 4
    clocks = dict(p.idle_since)
    p.record_accept_opportunity("input", "input", waiting)
    assert p.idle_since == clocks
    assert p.metrics["accept_input_rounds"] == 3
    assert p.metrics["accept_input_idle_all_zero_rounds"] == 1
    assert p.metrics["accept_input_idle_all_equal_rounds"] == 2
    assert p.metrics["accept_input_idle_distinct_rounds"] == 1
    # PRE has no empty-unassigned clock interpretation; Output has no actor.
    p.record_accept_opportunity("pre", "machine_00_pre", waiting)
    assert p.metrics["accept_pre_rounds"] == 1
    assert not any(key.startswith("accept_pre_idle") for key in p.metrics)
    before = dict(p.metrics)
    p.record_accept_opportunity(None, "output", waiting)
    assert dict(p.metrics) == before


def test_passive_accept_metrics_do_not_change_requests_or_physical_execution(
    monkeypatch,
):
    observed = fixture(n=4, arm="all")
    control = fixture(n=4, arm="all")
    monkeypatch.setattr(
        control.protocol, "record_accept_opportunity", lambda *a, **kw: None
    )
    runs = []
    for sim in (observed, control):
        mover, requests = SharedMover(sim.protocol.routes), []
        while not sim.done:
            requests.extend(decide(sim))
            sim.protocol.advance(mover.moves(sim.protocol.traffic_view(), sim.tick))
        runs.append(
            (
                requests,
                sim.tick,
                sim.jobs,
                sim.agvs,
                sim.machine_state,
                sim.total_reward,
            )
        )
    assert runs[0] == runs[1]
    assert observed.protocol.metrics["accept_input_rounds"] > 0
    assert not any(k.startswith("accept_") for k in control.protocol.metrics)
