"""Independent counting and paired-analysis examples, no learned dependency."""

import copy
import json
from dataclasses import replace
from io import StringIO

import pytest

from smartsom.config.base_diagnostic import BaseWorkload
from smartsom.config.base_layout import base_factory
from smartsom.domain.production import (
    Demand,
    Outage,
    ProductionScenario,
    ProductionStep,
)
from smartsom.engine.production import ProductionSimulator
from smartsom.experiments import reservation_runtime as runtime
from smartsom.policies.rule import BaseRule
from smartsom.telemetry.research_comparison import (
    LearningTarget,
    RecoveryProtocol,
    acceleration,
    fixed_window,
    recovery,
    robustness,
    time_to_target,
)
from smartsom.telemetry.research_metrics import SCHEMA, ResearchMetrics, aggregate
from tests.helpers.algorithm import algorithm_options


def run(level, condition="normal", *, limit=200):
    options = algorithm_options(schema="smartsom.base-algorithm/v1", method="rule")
    work = BaseWorkload(
        schema="smartsom.base-workload/v1",
        orders=4,
        initial_orders=2,
        mean_interarrival=8,
        base_ticks=dict.fromkeys("ABCD", 2),
    )
    policy = BaseRule("rule", 51)
    trace = StringIO()
    row, rewards, critic = runtime.episode(
        base_factory("small"),
        work,
        options,
        policy,
        71,
        condition,
        300,
        max_ticks=limit,
        metrics_level=level,
        trace=trace,
    )
    return row, rewards, critic, trace.getvalue(), policy.rng.getstate()


@pytest.mark.parametrize("condition", ["normal", "low", "medium", "high"])
def test_telemetry_preserves_actions_events_rewards_and_rng(condition):
    off = run("off", condition)
    for level in ("summary", "detailed"):
        on = run(level, condition)
        assert on[1:] == off[1:]
        assert {
            k: v
            for k, v in on[0].items()
            if k not in ("wall_seconds", "research_metrics")
        } == {k: v for k, v in off[0].items() if k != "wall_seconds"}
        report = on[0]["research_metrics"]
        assert report["schema"] == SCHEMA
        assert report["summary"]["distribution_count"] == on[0]["completed"]
        if level == "summary":
            assert "orders" not in report and "series" not in report
        else:
            assert (
                sum(sum(x.values()) for x in report["machines"].values())
                == 8 * on[0]["tick"]
            )
            assert {p["tick"] for p in report["series"]} >= set(
                range(0, on[0]["tick"] + 1, 50)
            )
            assert all(
                sum(v.get(k, 0) for k in ("loaded", "empty")) == on[0]["tick"]
                for v in report["vehicles"].values()
            )
            json.dumps(report, allow_nan=False)


def test_initial_fault_and_sparse_interval_accounting():
    factory = base_factory("small")
    machine = factory.machines[0].machine_id
    scenario = ProductionScenario(
        factory,
        (Demand("j", (ProductionStep("op", "A", 2),), release_at=100, due_at=120),),
        tick_limit=200,
        mode="finite",
        outages=(Outage(machine, 0, 60),),
        processing_rounding="ceil",
    )
    sim = ProductionSimulator(scenario, contract="base")
    options = algorithm_options(schema="smartsom.base-algorithm/v1", method="rule")
    sim.protocol.configure(options)
    collector = ResearchMetrics(sim, detailed=True, interval=10)
    assert any(e["kind"] == "breakdown" and e["tick"] == 0 for e in collector.events)
    collector.before()
    result = sim.protocol.advance({v: "WAIT" for v in sim.agvs})
    assert sim.tick > 1
    collector.after(result)
    report = collector.finish(runtime.metrics(sim, options, "high", 0, 0, 0))
    assert report["machines"][machine]["down"] == sim.tick
    assert all(
        s["completed"] == 0 and s["future_unreleased"] == 1 for s in report["series"]
    )
    assert report["orders"][0]["status"] == "censored"


def test_censored_costs_and_zero_denominators_are_not_successes():
    row = dict(
        total_orders=10, completed=0, failed=0, tick=0, mean_tardiness=0, joint_cost=0
    )
    out = aggregate(row)
    assert out["joint_cost"] is None and out["mean_tardiness"] is None
    assert out["conditional_passing_rate"] is None
    assert out["qualified_throughput_per_1000_ticks"] is None
    assert out["qualified_coverage"] == 0 and out["cost_status"] == "censored"
    assert aggregate({})["status"] == "not_recorded"
    with pytest.raises(ValueError):
        aggregate({**row, "failed": 1})


def detail(completions=(10, 20, 30, 40, 90), *, good=True):
    return {
        "evaluation_identity": {
            "checkpoint_sha256": "model",
            "method": "ppo",
            "learning": True,
            "selection": "sample",
            "policy_seed": 42,
            "source_sha256": "source",
            "runtime_sha256": "runtime",
            "algorithm_sha256": "options",
        },
        "research_metrics": {
            "schema": SCHEMA,
            "level": "detailed",
            "start_tick": 0,
            "end_tick": 100,
            "fingerprints": {"exogenous": "same"},
            "orders": [{"completion_tick": t, "final_pass": good} for t in completions],
            "outage_realization": [{"machine_id": "m", "start": 0, "end": 10}],
            "events": [
                {"kind": "breakdown", "tick": 0, "machine": "m"},
                {"kind": "repair", "tick": 10, "machine": "m"},
            ],
        },
    }


def test_common_window_excludes_start_includes_end_and_quality():
    nominal, disturbed = detail(), detail((10, 25, 40, 90), good=False)
    assert fixed_window(nominal, 10, 30)["completed"] == 2
    out = robustness(nominal, disturbed, 10, 40)
    assert out["throughput_ratio"] == pytest.approx(2 / 3)
    assert out["qualified_throughput_ratio"] == 0
    assert robustness(nominal, disturbed, 40, 50)["status"] == "not_applicable"
    assert fixed_window(nominal, 0, 101)["status"] == "censored"
    assert fixed_window({}, 0, 10)["status"] == "not_recorded"
    disturbed["evaluation_identity"]["policy_seed"] = 43
    with pytest.raises(ValueError, match="policy"):
        robustness(nominal, disturbed, 10, 40)


def point(budget, cost, *, quality=0.9, drained=True):
    return dict(
        budget=budget,
        cost=cost,
        passing_rate=quality,
        all_drained=drained,
        model_sha256=str(budget),
        train_seed=1,
        decoder="sample",
        panel_sha256="panel",
    )


def test_learning_requires_quality_drain_persistence_and_independent_model():
    target = LearningTarget("v1", "rule:hash", 100, 0.9, 2, "physical_ticks")
    rows = [point(0, 150), point(50, 90, drained=False), point(100, 90), point(150, 80)]
    out = time_to_target(rows, target)
    assert out["first_qualifying_interval"] == [50, 100]
    assert out["confirmation_budget"] == 150
    assert time_to_target(rows[:-1], target)["status"] == "censored"
    rows[-1]["model_sha256"] = "100"
    assert time_to_target(rows, target)["status"] == "censored"
    assert acceleration(out, out)["ratio"] == 1
    assert acceleration(out, time_to_target(rows, target))["ratio"] is None
    assert target.sha256 != replace(target, consecutive=3).sha256
    with pytest.raises(ValueError):
        time_to_target([point(100, 90), point(50, 90)], target)
    with pytest.raises(ValueError):
        time_to_target([point(100, 90), {**point(150, 80), "train_seed": 2}], target)
    with pytest.raises(ValueError):
        replace(target, maximum_cost=float("nan"))


def test_recovery_nonoverlapping_windows_censor_and_zero_reference():
    p = RecoveryProtocol("v1", "independent-rule-calibration", 10, 2)
    out = recovery(
        detail(), detail(), machine_id="m", fault_tick=0, repair_tick=10, protocol=p
    )
    assert out["status"] == "threshold_confirmed"
    assert out["confirmation_tick"] == 30
    assert out["repair_to_confirmation_ticks"] == 20
    assert out["adaptability"] == 0.05
    assert (
        recovery(
            detail(),
            detail((90,)),
            machine_id="m",
            fault_tick=0,
            repair_tick=10,
            protocol=p,
        )["status"]
        == "not_applicable"
    )
    disturbed = detail()
    disturbed["research_metrics"]["events"].append(
        {"kind": "breakdown", "tick": 25, "machine": "m"}
    )
    assert (
        recovery(
            detail(),
            disturbed,
            machine_id="m",
            fault_tick=0,
            repair_tick=10,
            protocol=p,
        )["reason"]
        == "new_fault"
    )
    assert (
        recovery(
            detail((20, 30)),
            detail((20, 30)),
            machine_id="m",
            fault_tick=0,
            repair_tick=10,
            protocol=p,
        )["reason"]
        == "demand_exhausted"
    )
    broken = copy.deepcopy(detail())
    broken["research_metrics"]["fingerprints"]["exogenous"] = "different"
    with pytest.raises(ValueError):
        recovery(
            detail(), broken, machine_id="m", fault_tick=0, repair_tick=10, protocol=p
        )


def test_gain_sign_and_curve_area_cannot_drop_censored_panels():
    from smartsom.telemetry.research_comparison import (
        compare_cost_curves,
        relative_gain,
    )

    gain = relative_gain(100, 80, direction="lower")
    assert gain["raw_sig"] == -0.2 and gain["improvement"] == 0.2
    assert relative_gain(0, 2, direction="higher")["status"] == "not_applicable"
    a, b = [point(0, 100), point(10, 80)], [point(0, 100), point(10, 60)]
    result = compare_cost_curves(a, b, budget_unit="physical_ticks")
    assert result["base_auc"] == 900 and result["social_auc"] == 800
    assert result["social_mean_cost"] == 80
    b[1]["all_drained"] = False
    assert (
        compare_cost_curves(a, b, budget_unit="physical_ticks")["status"] == "censored"
    )
    with pytest.raises(ValueError):
        compare_cost_curves(a, [b[0], point(20, 60)], budget_unit="physical_ticks")


def test_recovery_requires_same_machine_and_censors_simultaneous_faults():
    p = RecoveryProtocol("v1", "rule-reference", 10, 1)
    a, b = detail(), detail()
    b["research_metrics"]["events"][1]["machine"] = "other"
    assert (
        recovery(a, b, machine_id="m", fault_tick=0, repair_tick=10, protocol=p)[
            "reason"
        ]
        == "event_boundary_missing"
    )
    b = detail()
    b["research_metrics"]["outage_realization"].append(
        {"machine_id": "other", "start": 0, "end": 10}
    )
    assert (
        recovery(a, b, machine_id="m", fault_tick=0, repair_tick=10, protocol=p)[
            "reason"
        ]
        == "overlapping_fault"
    )


def test_learning_gaps_failures_and_repeated_models_remain_visible():
    target = LearningTarget("v1", "rule:hash", 100, 0.9, 2, "episodes")
    rows = [point(0, 200), point(100, None), point(200, 80), point(300, 70)]
    result = time_to_target(rows, target)
    assert result["first_qualifying_interval"] == [0, 200]
    assert result["evidence_gaps"] is True
    rows = [
        {**point(10, 90), "model_sha256": "A"},
        {**point(20, None, drained=False), "model_sha256": "A"},
        point(30, 80),
    ]
    assert time_to_target(rows, target)["status"] == "censored"
    rows = [
        {**point(0, 200), "model_sha256": "A"},
        point(100, 80),
        {**point(200, 200), "model_sha256": "A"},
        point(300, 70),
    ]
    assert time_to_target(rows, target)["status"] == "censored"
    rows = [
        {**point(100, None), "model_sha256": "A"},
        {**point(200, 80), "model_sha256": "A"},
        point(300, 70),
    ]
    assert time_to_target(rows, target)["status"] == "reached"
    with pytest.raises(ValueError, match="cost"):
        time_to_target([point(0, float("nan"))], target)
    empty = time_to_target([], target)
    full = time_to_target(rows, target)
    assert acceleration(empty, empty)["status"] == "not_recorded"
    assert acceleration(full, empty)["ratio"] is None
    assert acceleration(empty, full)["status"] == "not_recorded"
    with pytest.raises(ValueError, match="protocol"):
        acceleration(empty, time_to_target([], replace(target, version="v2")))


def test_pairing_checks_runtime_identity_and_missing_provenance():
    a, b = detail(), detail()
    b["evaluation_identity"]["runtime_sha256"] = "different-runtime"
    with pytest.raises(ValueError, match="paired"):
        robustness(a, b, 0, 40)
    del b["evaluation_identity"]["runtime_sha256"]
    assert robustness(a, b, 0, 40)["status"] == "not_recorded"
