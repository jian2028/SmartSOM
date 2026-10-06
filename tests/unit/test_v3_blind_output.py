"""Blind V3 shipment identities, privileged quality and public noninterference."""

from copy import deepcopy
from dataclasses import replace

import pytest
from test_travel_time_matrix import driver, scenario
from test_v3_dispatch_liveness import loaded

from smartsom.algorithms.production_composition import replay_boundary
from smartsom.domain.travel_time import physical_contract, validate_model_contract
from smartsom.engine.production import ProductionSimulator
from smartsom.experiments.composable import episode_metrics
from smartsom.trace.performance import TaskPerformance


def ready_output(*, defective=False, jobs=1, mode="finite"):
    case = scenario("zero", jobs=jobs)
    case = replace(case, mode=mode)
    sim = ProductionSimulator(case, contract="v3")
    vehicle, job = next(iter(sim.agvs)), next(iter(sim.jobs))
    sim.jobs[job].update(step=1, quality="UNKNOWN", defective=defective)
    output = next(o for o, role in sim.roles.items() if role == "system_output")
    loaded(sim, vehicle, job, output)
    return sim, vehicle, job


def test_hidden_good_and_bad_shipments_have_identical_public_trajectory():
    good, _, job = ready_output()
    bad = deepcopy(good)
    bad.jobs[job]["defective"] = True
    assert good.snapshot() == bad.snapshot()
    assert good.protocol.public_view() == bad.protocol.public_view()
    replay = deepcopy(good)
    good_driver, bad_driver = driver(good), driver(bad)
    good_record, bad_record = good_driver.tick(), bad_driver.tick()
    assert good_record == bad_record
    assert good_driver.records == bad_driver.records
    assert good.protocol.public_view() == bad.protocol.public_view()
    assert good.done and bad.done
    assert good.status == bad.status == "completed"
    assert good.shipped == bad.shipped == {good.jobs[job]["demand"]}
    assert good.shipment_times == bad.shipment_times
    assert good.jobs[job]["quality"] == bad.jobs[job]["quality"] == "UNKNOWN"
    assert good.privileged_output_quality()["passing_rate"] == 1
    assert bad.privileged_output_quality()["passing_rate"] == 0
    assert not any(e["kind"] == "quality_revealed" for e in good_record["events"])
    assert not any(e["kind"] == "attempt_queued" for e in bad_record["events"])
    assert all("defective" not in row for row in bad_record["state"]["jobs"].values())
    assert (
        not {"passed", "output_rejected", "good_shipped", "bad_shipped"}
        & bad.snapshot()["metrics"].keys()
    )
    replay_boundary(replay, good_record)
    assert replay.snapshot() == good.snapshot()
    good._check()
    bad._check()


def test_bad_shipment_closes_original_identity_and_privileged_metrics():
    sim, _, job = ready_output(defective=True)
    assert sim.privileged_output_quality()["passing_rate"] is None
    driver(sim).tick()
    demand = sim.jobs[job]["demand"]
    assert sim.attempts[demand] == 1
    assert not sim.queue
    assert len(sim.jobs) == 1
    assert sim.jobs[job]["location"] in sim.storage
    metrics = episode_metrics(sim)
    assert metrics["delivered"] == metrics["output_submitted"] == 1
    assert metrics["output_qualified"] == 0 and metrics["output_bad_shipped"] == 1
    assert metrics["passing_rate"] == 0
    assert "output_rejected" not in metrics
    assert metrics["fixed_job_makespan"] == sim.shipment_times[demand]
    assert metrics["unfinished_demands"] == []


def test_early_known_fail_scrap_replaces_same_demand_with_original_due():
    sim, vehicle, job = ready_output(defective=True)
    demand = sim.jobs[job]["demand"]
    sim.jobs[job]["quality"] = "FAIL"
    scrap = next(iter(sim.scrap))
    assert not sim._admit(
        job, next(o for o, role in sim.roles.items() if role == "system_output")
    )
    original_due = sim.demands[demand].due_at
    sim._transfer(vehicle, ("drop", job, scrap, "sink"))
    assert not sim.shipped and not sim.shipment_times
    assert sim.metrics["pre_output_scrap"] == 1
    assert sim.metrics["submitted"] == 0
    assert sim.attempts[demand] == 2
    replacement = sim.jobs[sim.queue[-1]]
    assert replacement["demand"] == demand
    assert sim.demands[replacement["demand"]].due_at == original_due
    assert sim.privileged_output_quality()["passing_rate"] is None
    sim._check()


def test_public_performance_uses_shipment_time_and_never_oracle_quality():
    sim, _, job = ready_output(defective=True)
    demand = sim.jobs[job]["demand"]
    sim.demands[demand] = replace(sim.demands[demand], due_at=0)
    performance = TaskPerformance(due={demand: 0})
    performance.observe(0, sim.snapshot())
    driver(sim).tick()
    performance.observe(1, sim.snapshot())
    totals = performance.cumulative(1)
    assert totals["shipped"] == 1
    assert totals["throughput"] == 1
    assert totals["qualified"] is None and totals["passing_rate"] is None
    assert totals["total_tardiness"] == sim.shipment_times[demand]
    assert performance.window(1)["passing_rate"] is None


def test_dynamic_window_does_not_end_early_or_change_cap():
    sim, _, _ = ready_output(defective=True, mode="dynamic")
    sim.scenario = replace(sim.scenario, tick_limit=4096)
    driver(sim).tick()
    assert len(sim.shipped) == 1 and not sim.done
    assert sim.scenario.tick_limit == 4096


def test_old_output_contract_is_rejected():
    from smartsom.domain.production_decisions import (
        ACTION_CONTRACT,
        OBSERVATION_CONTRACT,
    )

    contract = physical_contract(scenario())
    assert contract["output_semantics"] == "blind-shipment/v1"
    metadata = dict(
        action_contract=ACTION_CONTRACT,
        observation_contract=OBSERVATION_CONTRACT,
        physical_contract=contract,
    )
    validate_model_contract(metadata)
    for value in (None, "output-inspection/v1"):
        old = deepcopy(metadata)
        if value is None:
            del old["physical_contract"]["output_semantics"]
        else:
            old["physical_contract"]["output_semantics"] = value
        with pytest.raises(ValueError, match="incompatible"):
            validate_model_contract(old)


def test_actor_and_critic_encoded_inputs_are_oracle_independent():
    np = pytest.importorskip("numpy")
    pytest.importorskip("torch")
    from smartsom.learning.production_models import PublicEncoder

    good, vehicle, job = ready_output()
    bad = deepcopy(good)
    bad.jobs[job]["defective"] = True
    encoder = PublicEncoder(
        good.factory, {"time_scale": 100.0, "count_scale": 128.0}, role="dispatcher"
    )
    for advance in (False, True):
        if advance:
            driver(good).tick()
            driver(bad).tick()
        a = encoder.empty("dispatcher", vehicle, good.protocol.public_view())
        b = encoder.empty("dispatcher", vehicle, bad.protocol.public_view())
        assert a.keys() == b.keys()
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])


def test_duplicate_shipment_rejects_without_reset_or_mutation():
    sim, vehicle, job = ready_output()
    driver(sim).tick()
    before = deepcopy(sim.snapshot())
    quality = sim.privileged_output_quality()
    times = deepcopy(sim.shipment_times)
    output = sim.jobs[job]["location"]
    with pytest.raises(ValueError, match="already shipped"):
        sim._transfer(vehicle, ("drop", job, output, sim.jobs[job]["slot"]))
    assert sim.snapshot() == before
    assert sim.shipment_times == times
    assert sim.privileged_output_quality() == quality


def test_dynamic_window_complete_is_not_complete_shipping_makespan():
    sim, _, _ = ready_output(defective=True, jobs=2, mode="dynamic")
    sim.scenario = replace(sim.scenario, tick_limit=1)
    driver(sim).tick()
    metrics = episode_metrics(sim)
    assert sim.status == "completed" and sim.done
    assert metrics["delivered"] == 1
    assert metrics["makespan"] is None and metrics["fixed_job_makespan"] is None
    assert not metrics["fulfillment_complete"] and not metrics["makespan_complete"]


def test_processing_completion_does_not_publish_latent_defect(monkeypatch):
    from decimal import Decimal

    case = scenario()
    factory = replace(
        case.factory,
        machines=tuple(
            replace(
                m,
                quality_modes=tuple(
                    replace(q, error_rate=Decimal("0.5")) for q in m.quality_modes
                ),
            )
            for m in case.factory.machines
        ),
    )
    good = ProductionSimulator(replace(case, factory=factory), contract="v3")
    job, machine = next(iter(good.jobs)), next(iter(good.machines))
    good._remove(job)
    good.jobs[job].update(location=machine, slot=None, step=0, quality="UNKNOWN")
    good.machine_state[machine].update(
        job=job,
        status="PROCESSING",
        remaining=1,
        elapsed=0,
        nominal=1,
        mode=good.machines[machine].quality_modes[0].quality_mode_id,
    )
    good.events = []
    bad = deepcopy(good)
    monkeypatch.setattr(good, "_draw", lambda *args: 1)
    monkeypatch.setattr(bad, "_draw", lambda *args: 0)
    good._advance()
    bad._advance()
    assert not good.jobs[job]["defective"] and bad.jobs[job]["defective"]
    assert good.events == bad.events
    assert all("defect" not in event for event in good.events)
    assert good.snapshot() == bad.snapshot()
    assert good.protocol.public_view() == bad.protocol.public_view()


def test_empty_demand_case_has_zero_completed_makespan():
    sim = ProductionSimulator(replace(scenario("zero"), demands=()), contract="v3")
    row = episode_metrics(sim)
    assert row["fulfillment_complete"]
    assert row["makespan"] == 0
    assert row["passing_rate"] is None
