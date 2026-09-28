"""Paired pools, due-time units and bounded MD V construction."""

import json
from dataclasses import replace
from fractions import Fraction

import pytest

from smartsom.config.codec import canonical_json, primitive
from smartsom.config.production import WorkloadFile
from smartsom.config.workload_v3 import WorkloadV3, load_workload_v3
from smartsom.domain.production import Demand, ProductionStep
from smartsom.workloads.job_content import job_distance
from smartsom.workloads.workload_v3 import materialize_workload


def recipe(**overrides):
    data = {
        "schema": "smartsom.workload/v3",
        "templates": [
            {"id": "A", "route": ["a"], "times": [3], "count": 5},
            {"id": "B", "route": ["b", "a"], "times": [2, 4], "count": 5},
            {
                "id": "C",
                "route": ["a", "c", "b"],
                "times": [2, 3, 5],
                "count": 5,
                "novel": True,
            },
        ],
        "arrivals": {"interval_seconds": 2, "initial_jobs": 1, "window_jobs": 3},
        "volatility": {"level": "mid", "swap_proposals": 12},
        "due": {"rush_probability": 0.4},
    }
    data.update(overrides)
    return WorkloadV3.model_validate_json(json.dumps(data))


def test_pairing_shared_pool_history_classes_allowances_slots_and_window_rush():
    config = recipe()
    outputs = {
        name: materialize_workload(
            config.model_copy(
                update={
                    "volatility": config.volatility.model_copy(update={"level": name})
                }
            ),
            33,
        )
        for name in ("low", "mid", "high")
    }
    values = [outputs[name].provenance["V"] for name in ("low", "mid", "high")]
    assert values[0] < values[1] < values[2]
    reference = outputs["mid"].provenance
    original = {job["id"]: job for job in reference["pool"]}
    slots = None
    for output in outputs.values():
        assert output.provenance["pool_sha256"] == reference["pool_sha256"]
        assert (
            output.provenance["initial_history_samples"]
            == reference["initial_history_samples"]
        )
        assert (
            output.provenance["rush_counts_by_window"]
            == reference["rush_counts_by_window"]
        )
        assert {d.demand_id for d in output.workload.demands} == set(original)
        arrivals = [d.release_at for d in output.workload.demands]
        assert slots is None or slots == arrivals
        slots = arrivals
        for demand in output.workload.demands:
            job = original[demand.demand_id]
            assert demand.rush == job["rush"]
            assert demand.due_at - demand.release_at == job["allowance_ticks"]
            assert demand.reveal_at == demand.release_at
            assert demand.priority == 1
            assert 1 <= len(demand.steps) <= 5
            assert all(
                a.operation_type != b.operation_type
                for a, b in zip(demand.steps, demand.steps[1:])
            )
        rush_counts = [
            sum(d.rush for d in output.workload.demands[i : i + 3])
            for i in range(0, len(original), 3)
        ]
        assert rush_counts == reference["rush_counts_by_window"]
        assert {job["base_type"] for job in reference["initial_history_samples"]} == {
            "A",
            "B",
        }


def test_separate_streams_and_datasets_no_policy_or_environment_input():
    config = recipe()
    a = materialize_workload(config, 33)
    assert canonical_json(a) == canonical_json(materialize_workload(config, 33))
    b = materialize_workload(config, 34)
    c = materialize_workload(config, 33, "validation/000")
    assert a.provenance["pool_sha256"] != b.provenance["pool_sha256"]
    assert a.provenance["pool_sha256"] != c.provenance["pool_sha256"]
    assert len(set(a.provenance["random_streams"].values())) == 6
    # Frozen cache is never handed out by reference.
    a.provenance["pool"][0]["rush"] = "corrupt"
    assert type(materialize_workload(config, 33).provenance["pool"][0]["rush"]) is bool


def test_due_function_rounds_once_after_rush_ratio_and_public_rush_roundtrip():
    config = recipe(
        due={
            "base_seconds": 1.4,
            "reference_work_factor": 0.5,
            "operation_seconds": 0.2,
            "rush_probability": 1,
            "rush_ratio": 0.7,
        }
    )
    output = materialize_workload(config, 33)
    jobs = {j["id"]: j for j in output.provenance["pool"]}
    for demand in output.workload.demands:
        normal = Fraction(jobs[demand.demand_id]["normal_allowance_seconds"])
        expected = -(
            -(normal * Fraction(7, 10)).numerator
            // (normal * Fraction(7, 10)).denominator
        )
        assert demand.due_at - demand.release_at == expected
        assert demand.rush
    assert (
        WorkloadFile.model_validate_json(canonical_json(output.workload))
        == output.workload
    )


def test_additive_rush_defaults_keep_legacy_snapshot_bytes():
    demand = Demand("old", (ProductionStep("op", "a", 1),))
    assert "rush" not in primitive(demand)
    assert primitive(replace(demand, rush=True))["rush"] is True
    with pytest.raises(ValueError, match="rush"):
        replace(demand, rush=1)


def test_templates_quotas_personalization_and_fixed_bound():
    config = recipe()
    data = materialize_workload(config, 33).provenance
    counts = {}
    for job in data["pool"]:
        key = job["base_type"], job["category"]
        counts[key] = counts.get(key, 0) + 1
    for template in config.templates:
        assert [
            counts[(template.id, category)]
            for category in ("standard", "parameter", "insertion", "mixed")
        ] == [1, 2, 1, 1]
    bound = config.occurrence_bound
    scale = Fraction(data["reference_unit_seconds"])
    assert all(
        Fraction(t) * scale <= bound for job in data["pool"] for _, t in job["spec"]
    )
    assert Fraction(data["occurrence_bound_seconds"]) == bound
    assert job_distance(
        (("a", 3), ("b", 2), ("c", 6)), (("a", 3), ("b", 5), ("c", 6)), 10
    ) == Fraction(3, 100)


def test_md_default_horizon_derives_fixed_support_not_sample_maximum():
    config = recipe(
        templates=[
            {
                "id": "A",
                "route": ["a", "b", "c", "d"],
                "times": [20, 30, 40, 30],
                "count": 200,
            }
        ],
        segments=10,
        arrivals={"interval_seconds": 2, "initial_jobs": 10, "window_jobs": 40},
        occurrence_bound_seconds=52,
    )
    assert config.occurrence_bound == 52
    assert config.total_jobs == 2000 and config.default_tick_limit == 4928


def test_strict_inputs_and_diagnosed_no_valid_v_levels(tmp_path):
    config = recipe()
    path = tmp_path / "workload.yaml"
    path.write_text(canonical_json(config))
    assert load_workload_v3(path) == config
    with pytest.raises(ValueError, match="extra"):
        recipe(algorithm="ppo")
    with pytest.raises(ValueError, match="support bound"):
        recipe(occurrence_bound_seconds=1)
    with pytest.raises(ValueError, match="whole arrival windows"):
        recipe(arrivals={"window_jobs": 4, "initial_jobs": 1})
    with pytest.raises(ValueError, match="future|less than or equal"):
        recipe(arrivals={"window_jobs": 3, "initial_jobs": 1, "notice_seconds": 1})
    impossible = recipe(
        templates=[{"id": "A", "route": ["a"], "times": [3], "count": 5}],
        arrivals={"initial_jobs": 1, "window_jobs": 5},
        personalization={"category_weights": [1, 0, 0, 0]},
        volatility={"level": "low", "swap_proposals": 2},
    )
    with pytest.raises(ValueError, match="no Job or rush label was redrawn"):
        materialize_workload(impossible, 33)


def test_fractional_reference_work_roundtrip_preserves_single_physical_rounding():
    from test_production_runtime import act, loaded_machine, small_scenario

    from smartsom.config.production import scenario_from_snapshot
    from smartsom.domain.production import MachineCommand
    from smartsom.domain.quality import QualityMode
    from smartsom.engine.production import ProductionSimulator

    source = small_scenario(processing_rounding="ceil")
    mode = QualityMode("fast", "0.75", "0")
    factory = replace(
        source.factory,
        machines=(replace(source.factory.machines[0], quality_modes=(mode,)),),
    )
    step = ProductionStep("op", "drill", 2, reference_ticks="11/10")
    scenario = replace(
        source, factory=factory, demands=(Demand("demand", (step,), rush=True),)
    )
    assert scenario_from_snapshot(primitive(scenario)) == scenario
    sim = ProductionSimulator(scenario)
    job = loaded_machine(sim)
    assert sim.decision()["jobs"][job]["rush"] is True
    act(sim, machine=MachineCommand(job, "fast"))
    started = next(
        event for event in sim.events if event["kind"] == "processing_started"
    )
    assert started["actual_ticks"] == 1  # ceil(1.1 × 0.75), not ceil(ceil(1.1) × 0.75).
    assert "reference_ticks" not in primitive(ProductionStep("old", "drill", 2))
    assert step.work_ticks_on("machine") == Fraction(11, 10)
    with pytest.raises(ValueError, match="positive rational"):
        ProductionStep("op", "drill", 2, reference_ticks="0/3")


def test_shared_segments_are_preserved_by_every_v_swap():
    config = recipe(segments=2, volatility={"level": "mid", "swap_proposals": 6})
    output = materialize_workload(config, 33)
    jobs = {job["id"]: job for job in output.provenance["pool"]}
    for row in output.provenance["levels"].values():
        for index, identity in enumerate(row["order"]):
            assert jobs[identity]["segment"] == index // 15


def test_initial_batch_slots_and_reference_resolution_are_not_resampled():
    config = recipe(
        arrivals={
            "interval_seconds": 2,
            "initial_jobs": 3,
            "window_jobs": 3,
            "tick_seconds": 0.3,
        }
    )
    output = materialize_workload(config, 33)
    assert [d.release_at for d in output.workload.demands[:4]] == [0, 0, 0, 20]
    scale = Fraction(output.provenance["reference_unit_seconds"])
    original = {j["id"]: j for j in output.provenance["pool"]}
    for demand in output.workload.demands:
        for step, (_, units) in zip(
            demand.steps, original[demand.demand_id]["spec"], strict=True
        ):
            assert step.work_ticks_on("any") == units * scale / Fraction(3, 10)


def test_physical_arrival_rounding_cannot_change_v_window():
    with pytest.raises(ValueError, match="rounding would change"):
        recipe(
            arrivals={
                "interval_seconds": 1,
                "initial_jobs": 1,
                "window_jobs": 3,
                "tick_seconds": 3,
            }
        )
    with pytest.raises(ValueError, match="align with physical ticks"):
        recipe(
            arrivals={
                "interval_seconds": 1,
                "initial_jobs": 1,
                "window_jobs": 3,
                "tick_seconds": 2,
            }
        )


def test_future_job_rush_not_public_before_external_arrival():
    from test_production_runtime import act, small_scenario

    from smartsom.engine.production import ProductionSimulator

    scenario = replace(
        small_scenario(mode="dynamic"),
        demands=(
            Demand(
                "future", (ProductionStep("op", "drill", 2),), release_at=2, rush=True
            ),
        ),
    )
    sim = ProductionSimulator(scenario)
    assert sim.decision()["jobs"] == {} and sim.decision()["announced"] == []
    act(sim)
    assert sim.decision()["jobs"] == {}
    act(sim)
    assert next(iter(sim.decision()["jobs"].values()))["rush"] is True
    old_announced = replace(
        scenario,
        demands=(
            Demand(
                "future", (ProductionStep("op", "drill", 2),), release_at=2, reveal_at=0
            ),
        ),
    )
    row = ProductionSimulator(old_announced).decision()["announced"][0]
    assert "rush" not in row and "reference_ticks" not in row["steps"][0]
