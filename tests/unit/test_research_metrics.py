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


def test_new_common_window_allows_only_known_drained_tail():
    row = detail((10, 20))
    row.update(completed=2, total_orders=2)
    assert fixed_window(row, 0, 200)["throughput"] == 0.01
    row["wall_clock_fragment"] = True
    assert fixed_window(row, 0, 200)["status"] == "censored"


def test_hierarchical_weights_and_sig_are_not_mean_percentages():
    from smartsom.telemetry.evaluation_summary import hierarchical, paired_sig

    def row(world, repeat, cost):
        return dict(
            total_orders=10,
            completed=10,
            failed=0,
            tick=100,
            joint_cost=cost,
            mean_tardiness=cost,
            condition="normal",
            scenario_seed=world,
            action_seed=repeat,
            selection="sample",
            evaluation_identity={
                "split": "validation",
                "world_sha256": str(world),
                "source_sha256": "source",
                "runtime_sha256": "runtime",
                "training_progress": {"train_seed": 1, "episodes": 10},
                "sl_comparison": {
                    "status": "supported",
                    "arm": "base",
                    "non_sl_sha256": "common",
                },
            },
        )

    base = [row(1, 0, 1), row(2, 0, 100)]
    social = [row(1, 0, 0), row(2, 0, 90)]
    for r in social:
        r["evaluation_identity"]["sl_comparison"]["arm"] = "all"
    assert paired_sig(base, social)["sig"] == pytest.approx(11 / 101)
    repeated = [row(1, 0, 1), row(1, 1, 1), row(2, 0, 100)]
    assert hierarchical(repeated)["metrics"]["joint_cost"] == 50.5
    repeated[-1]["joint_cost"] = None
    assert hierarchical(repeated)["metrics"]["joint_cost"] is None


def test_platform_recovery_first_threshold_and_confirmation():
    from smartsom.config.evaluation_protocol import RecoveryCandidate
    from smartsom.telemetry.recovery_protocol import recovery_evidence

    candidate = RecoveryCandidate(
        name="demo",
        load="normal",
        orders=100,
        initial_orders=10,
        mean_interarrival=1,
        horizon_ticks=100,
        window_ticks=10,
        baseline_windows=3,
        minimum_window_deliveries=1,
        maximum_cv=0,
        maximum_relative_trend=0,
        faults=[dict(machine_id="m", start_tick=30, duration_ticks=10)],
    )
    nominal = detail(tuple(range(5, 100, 10)))
    disturbed = detail((5, 15, 25, 45, 55, 65, 75, 85, 95))
    for row in (nominal, disturbed):
        row.update(tick=100, last_arrival=110)
    evidence = recovery_evidence(nominal, disturbed, candidate, candidate.faults[0])
    assert evidence["status"] == "recovered"
    assert evidence["first_qualifying_tick"] == 50
    assert evidence["confirmation_tick"] == 70
    assert evidence["adaptability"] == pytest.approx(1 / 20)
    nominal["last_arrival"] = 99
    assert (
        recovery_evidence(nominal, disturbed, candidate, candidate.faults[0])["status"]
        == "insufficient_supply"
    )


def test_partial_episode_preserves_evidence_without_faking_completion():
    import time

    options = algorithm_options(schema="smartsom.base-algorithm/v1", method="rule")
    work = BaseWorkload(
        schema="smartsom.base-workload/v1",
        orders=4,
        initial_orders=2,
        mean_interarrival=8,
    )
    row, _, _ = runtime.episode(
        base_factory("small"),
        work,
        options,
        BaseRule("rule", 51),
        71,
        "normal",
        300,
        wall_deadline=time.monotonic() - 1,
        partial_on_timeout=True,
        common_window_end=500,
    )
    assert row["wall_clock_fragment"]
    assert row["joint_cost"] is None
    assert row["common_window"]["status"] == "censored"


def test_recovery_candidate_scan_real_entry_and_receipts(tmp_path):
    from pathlib import Path

    from smartsom import api

    candidate = dict(
        name="engineering",
        load="normal",
        orders=4,
        initial_orders=2,
        mean_interarrival=50,
        horizon_ticks=100,
        window_ticks=10,
        baseline_windows=3,
        minimum_window_deliveries=1,
        maximum_cv=0.1,
        maximum_relative_trend=0.1,
        faults=[dict(machine_id="machine_00", start_tick=40, duration_ticks=10)],
    )
    config = (
        Path(__file__).resolve().parents[2]
        / "experiments/examples/small/default/experiment.yaml"
    )
    from smartsom.config.reservation import compile_reservation

    inputs = tmp_path / "standalone-inputs"
    inputs.mkdir()
    config = compile_reservation(config).materialize(inputs)
    root = tmp_path / "scan"
    result = api.run_experiment(
        config,
        task="calibrate",
        sets=[
            f'experiment.run.output="{root}"',
            "experiment.run.methods=[rule]",
            "experiment.run.seconds=30",
            "experiment.tracking.mode=disabled",
            "experiment.evaluation.recovery.action=calibrate",
            f"experiment.evaluation.recovery.candidates={json.dumps([candidate])}",
        ],
    )
    assert result["status"] == "completed"
    report = json.loads((root / "recovery-calibration.json").read_text())
    assert report["accepted"] == {"engineering": False}
    assert len(report["results"]) == 1
    receipts = list((root / "case-commits").glob("case-*.json"))
    assert len(receipts) == 1
    row = json.loads((root / "cases.jsonl").read_text())
    assert row["tick"] == 100
    assert row["research_metrics"]["entity_series"][-1]["tick"] == 100


def test_frozen_recovery_scan_is_consumed_by_real_controlled_evaluation(tmp_path):
    from pathlib import Path

    from smartsom import api

    candidate = dict(
        name="engineering",
        load="normal",
        orders=64,
        initial_orders=32,
        mean_interarrival=100,
        horizon_ticks=1400,
        window_ticks=200,
        baseline_windows=3,
        minimum_window_deliveries=1,
        maximum_cv=100,
        maximum_relative_trend=100,
        faults=[dict(machine_id="machine_00", start_tick=600, duration_ticks=10)],
    )
    config = (
        Path(__file__).resolve().parents[2]
        / "experiments/examples/small/default/experiment.yaml"
    )
    from smartsom.config.reservation import compile_reservation

    inputs = tmp_path / "standalone-inputs"
    inputs.mkdir()
    config = compile_reservation(config).materialize(inputs)
    shared = [
        "experiment.run.methods=[rule]",
        "experiment.run.seconds=30",
        "experiment.tracking.mode=disabled",
        "workload.base_ticks={A: 1, B: 1, C: 1, D: 1}",
        "experiment.validation_cases=4",
        f"experiment.evaluation.recovery.candidates={json.dumps([candidate])}",
    ]
    scan, evaluated = tmp_path / "scan", tmp_path / "evaluate"
    api.run_experiment(
        config,
        task="calibrate",
        sets=shared
        + [
            f'experiment.run.output="{scan}"',
            "experiment.evaluation.recovery.action=calibrate",
        ],
    )
    report = json.loads((scan / "recovery-calibration.json").read_text())
    assert report["accepted"]["engineering"], report["results"]
    api.run_experiment(
        config,
        task="evaluate",
        sets=shared
        + [
            f'experiment.run.output="{evaluated}"',
            "experiment.evaluation.recovery.action=evaluate",
            "experiment.evaluation.recovery.selected=engineering",
            f'experiment.evaluation.recovery.calibration="{scan / "recovery-calibration.json"}"',
        ],
    )
    rows = [
        json.loads(line)
        for line in (evaluated / "cases.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 2
    assert rows[0]["research_metrics"]["outage_realization"] == []
    assert rows[1]["research_metrics"]["outage_realization"] == [
        dict(machine_id="machine_00", start=600, end=610)
    ]
    assert (
        rows[0]["research_metrics"]["fingerprints"]["orders"]
        == rows[1]["research_metrics"]["fingerprints"]["orders"]
    )
    result = json.loads((evaluated / "recovery-summary.json").read_text())
    assert len(result["results"]) == 1
    assert result["results"][0]["status"] in {
        "recovered",
        "not_recovered",
        "no_observed_degradation",
    }


def test_checkpoint_copy_keeps_only_matching_evidence(tmp_path):
    import hashlib
    import json

    import pytest

    from smartsom.experiments.reservation_validation import copy_checkpoint

    source, model, best = (
        tmp_path / name for name in ("last.pt", "model.pt", "best.pt")
    )
    source.write_bytes(b"committed state")
    evidence = {
        "schema": "smartsom.checkpoint-evidence/v1",
        "checkpoint_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "train_seed": 7,
    }
    source.with_suffix(".pt.json").write_text(json.dumps(evidence))
    copy_checkpoint(source, model)
    copy_checkpoint(model, best)
    assert json.loads(best.with_suffix(".pt.json").read_text()) == evidence
    source.write_bytes(b"different state")
    with pytest.raises(ValueError, match="sidecar"):
        copy_checkpoint(source, model)
    assert model.read_bytes() == b"committed state"
    source.with_suffix(".pt.json").unlink()
    copy_checkpoint(source, model)
    assert not model.with_suffix(".pt.json").exists()


def test_evaluation_progress_emits_committed_case_count(tmp_path, monkeypatch):
    import json

    from smartsom.experiments.reservation_runtime import publish_progress
    from smartsom.telemetry import runtime

    emitted = []
    monkeypatch.setattr(
        runtime, "emit", lambda root, event, **kwargs: emitted.append((event, kwargs))
    )
    publish_progress(
        tmp_path,
        {
            "phase": "evaluation",
            "status": "completed",
            "completed_cases": 4,
            "total_cases": 4,
            "tick": 123,
        },
    )
    row, options = emitted[0]
    assert row["completed"] == 4 and options["total"] == 4
    assert options["unit"] == "cases" and options["final"] is True
    assert json.loads((tmp_path / "progress.json").read_text())["completed_cases"] == 4
