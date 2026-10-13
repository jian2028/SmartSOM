"""Paired pools, due-time units and bounded MD V construction."""

import json
import math
import random
from dataclasses import replace
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

import pytest

from smartsom.config.codec import canonical_json, digest, primitive
from smartsom.config.factory_design import load_factory_design_file
from smartsom.config.production import WorkloadFile, named_seed
from smartsom.config.workload_v3 import WorkloadV3, load_workload_v3
from smartsom.domain.factory_design import PoolStorage
from smartsom.domain.production import Demand, ProductionStep
from smartsom.workloads.job_content import job_distance
from smartsom.workloads.workload_v3 import (
    apply_workload_timing,
    materialize_workload,
    nominal_work_ticks,
)


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


@pytest.mark.parametrize(
    "seed,split,expected",
    [
        (
            33,
            "train",
            "d9624a67357e76e7e7ff19e53f4382f3305e5b8c551103a1aa177127971ef45a",
        ),
        (
            34,
            "train",
            "5e1d55ce2889b79d67887e018b94b1a07df6dcc434fe1b30ec08652fbaf66643",
        ),
        (
            33,
            "validation/000",
            "dcc3e31243dfebc47f549d087849f1192994227e9f9e49949caf77724109d462",
        ),
    ],
)
def test_legacy_generator_snapshot_and_rng_bytes(seed, split, expected):
    # Captured with the generator at integrated main 8d2cab8, before timing changes.
    assert (
        digest(recipe())
        == "73d528f2d8bc2f8dc8f345098ffee089f63e3ff3e8f393535beac59b0c40f6ad"
    )
    assert digest(materialize_workload(recipe(), seed, split)) == expected


def timing_factory(capacity=10):
    source = (
        Path(__file__).resolve().parents[2] / "configs/test/factories/factory_hand.yaml"
    )
    factory = load_factory_design_file(source)[0].factory
    return replace(
        factory,
        buffers=tuple(
            replace(b, storage=PoolStorage(capacity)) if b.role == "system_input" else b
            for b in factory.buffers
        ),
    )


def timing_recipe(initial=8, interval=None, factor=None, count=128):
    demands = tuple(
        Demand(
            f"{'common' if i < 96 else 'novel'}-{i:03d}",
            (
                ProductionStep("op1", "operation_1", 3),
                ProductionStep("op2", "operation_1", 7),
            ),
        )
        for i in range(count)
    )
    return WorkloadV3.model_validate_json(
        canonical_json(
            {
                "schema": "smartsom.workload/v3",
                "demands": demands,
                "mode": "finite",
                "tick_limit": 4096,
                "arrivals": {
                    "schedule": "continuous",
                    "initial_jobs": initial,
                    "interval_seconds": interval
                    or {"distribution": "exponential", "mean": 30},
                },
                "due": {
                    "nominal_work_factor": factor
                    or {"distribution": "uniform", "low": 1, "high": 3}
                },
            }
        )
    )


@pytest.mark.parametrize(
    "field,spec",
    [
        ("initial_jobs", -1),
        ("initial_jobs", True),
        ("initial_jobs", {"capacity_ratio": -0.1}),
        ("initial_jobs", {"capacity_ratio": 1.01}),
        ("interval_seconds", {"distribution": "exponential", "mean": 0}),
        ("interval_seconds", {"distribution": "exponential", "mean": -2}),
        ("interval_seconds", {"distribution": "exponential", "mean": float("inf")}),
        ("interval_seconds", {"distribution": "uniform", "low": 2, "high": 1}),
        ("interval_seconds", {"distribution": "uniform", "low": 0, "high": 1}),
        ("interval_seconds", {"distribution": "constant", "value": 0}),
        ("interval_seconds", {"distribution": "poisson", "mean": 30}),
        (
            "interval_seconds",
            {"distribution": "constant", "value": 1, "expression": "x"},
        ),
        ("nominal_work_factor", {"distribution": "constant", "value": 0}),
        ("nominal_work_factor", {"distribution": "uniform", "low": 0, "high": 3}),
        (
            "nominal_work_factor",
            {"distribution": "uniform", "low": float("nan"), "high": 3},
        ),
    ],
)
def test_timing_distribution_rejects_invalid_values(field, spec):
    data = primitive(timing_recipe())
    data["due" if field == "nominal_work_factor" else "arrivals"][field] = spec
    with pytest.raises(ValueError):
        WorkloadV3.model_validate_json(canonical_json(data))


@pytest.mark.parametrize(
    "capacity,initial,expected",
    [
        (10, 0, 0),
        (10, 1, 1),
        (10, {"capacity_ratio": 0.8}, 8),
        (20, {"capacity_ratio": 0.8}, 16),
        (3, {"capacity_ratio": 0.8}, 2),
        (10, {"capacity_ratio": 0}, 0),
        (10, {"capacity_ratio": 1}, 10),
        (None, 1, 1),
        (10, {"distribution": "constant", "value": 8}, 8),
        (10, {"distribution": "uniform", "low": 1.9, "high": 1.9}, 1),
    ],
)
def test_initial_counts_capacity_ratio_rounding_and_conservation(
    capacity, initial, expected
):
    config = timing_recipe(initial)
    rows, evidence = apply_workload_timing(
        config, config.demands, timing_factory(capacity), 7, "train"
    )
    assert sum(d.release_at == 0 for d in rows) == evidence["initial_jobs"] == expected
    assert len(rows) == len({d.demand_id for d in rows}) == 128
    assert [d.steps for d in rows] == [d.steps for d in config.demands]
    assert evidence["demands_sha256"] == digest(rows)
    assert all(d.reveal_at == d.release_at for d in rows)


@pytest.mark.parametrize(
    "capacity,initial,count,message",
    [
        (None, {"capacity_ratio": 0.8}, 128, "finite input capacity"),
        (10, 11, 128, "input capacity"),
        (20, 2, 1, "planned Job count"),
    ],
)
def test_initial_capacity_errors_are_not_silently_clamped(
    capacity, initial, count, message
):
    config = timing_recipe(initial, count=count)
    with pytest.raises(ValueError, match=message):
        apply_workload_timing(
            config, config.demands, timing_factory(capacity), 7, "train"
        )


def test_continuous_intervals_accumulate_before_ceil_and_never_fit_horizon():
    config = timing_recipe(0, interval={"distribution": "exponential", "mean": 0.01})
    rows, evidence = apply_workload_timing(
        config, config.demands, timing_factory(), 7, "train"
    )
    rng = random.Random(evidence["random_streams"]["arrival"])
    seconds = Fraction(0)
    for demand, exact in zip(
        rows, evidence["arrival_slots_seconds_exact"], strict=True
    ):
        seconds += Fraction(str(rng.expovariate(1) * 0.01))
        assert Fraction(exact) == seconds
        assert demand.release_at == math.ceil(seconds)
    assert len({d.release_at for d in rows}) < len(rows)
    slow = timing_recipe(8, interval={"distribution": "constant", "value": 60})
    rows, evidence = apply_workload_timing(
        slow, slow.demands, timing_factory(), 7, "train"
    )
    assert len(rows) == 128 and rows[-1].release_at == 120 * 60 > 4096
    assert evidence["demands"][-1]["release_at"] == rows[-1].release_at


def test_nominal_due_uses_real_route_eligible_machines_overrides_rates_and_ceil():
    factory = timing_factory()
    machine = factory.machines[0]
    factory = replace(
        factory,
        machines=(
            replace(machine, machine_id="fast", processing_rate_multiplier=Decimal(2)),
            replace(machine, machine_id="slow", processing_rate_multiplier=Decimal(1)),
        ),
    )
    demand = Demand(
        "job",
        (
            ProductionStep("op1", "operation_1", 2, reference_ticks="7/3"),
            ProductionStep("op2", "operation_1", 10, (("slow", 3),)),
        ),
    )
    # ceil(7/3 / 2), ceil(7/3) -> 2,3; ceil(10/2), override3 -> 5,3.
    assert nominal_work_ticks(demand, factory) == Fraction(13, 2)
    config = timing_recipe(0)
    rows, evidence = apply_workload_timing(config, (demand,), factory, 7, "train")
    draw = evidence["due_draws"][0]
    assert 1 <= draw["alpha"] <= 3
    assert draw["nominal_work_ticks"] == "13/2"
    assert rows[0].due_at == rows[0].release_at + math.ceil(
        Fraction(str(draw["alpha"])) * Fraction(13, 2)
    )


def test_independent_timing_domains_are_repeatable_and_do_not_change_other_draws():
    config = timing_recipe()
    factory = timing_factory()
    original, evidence = apply_workload_timing(
        config, config.demands, factory, 7, "train"
    )
    assert (original, evidence) == apply_workload_timing(
        config, config.demands, factory, 7, "train"
    )
    changed_due = timing_recipe(factor=2)
    due_rows, _ = apply_workload_timing(
        changed_due, config.demands, factory, 7, "train"
    )
    assert [d.release_at for d in original] == [d.release_at for d in due_rows]
    changed_arrival = timing_recipe(interval=2)
    _, arrival_evidence = apply_workload_timing(
        changed_arrival, config.demands, factory, 7, "train"
    )
    assert arrival_evidence["due_draws"] == evidence["due_draws"]
    assert len(set(evidence["random_streams"].values())) == 3
    assert evidence["random_streams"]["due"] == named_seed(7, "workload-v3/train/due")
    assert (
        evidence
        != apply_workload_timing(config, config.demands, factory, 8, "train")[1]
    )
    assert (
        evidence
        != apply_workload_timing(config, config.demands, factory, 7, "validation")[1]
    )


@pytest.mark.parametrize("capacity,common,novel", [(10, 6, 2), (20, 12, 4)])
def test_explicit_initial_ids_preserve_96_common_32_novel_and_initial_mix(
    capacity, common, novel
):
    config = timing_recipe({"capacity_ratio": 0.8})
    data = primitive(config)
    ids = [d.demand_id for d in config.demands[:common]] + [
        d.demand_id for d in config.demands[96 : 96 + novel]
    ]
    data["arrivals"]["initial_job_ids"] = ids
    config = WorkloadV3.model_validate_json(canonical_json(data))
    rows, evidence = apply_workload_timing(
        config, config.demands, timing_factory(capacity), 7, "train"
    )
    assert evidence["initial_job_ids"] == ids
    assert sum(d.demand_id.startswith("common") for d in rows) == 96
    assert sum(d.demand_id.startswith("novel") for d in rows) == 32
    assert [d.demand_id for d in rows if d.release_at == 0] == ids


def test_templates_keep_content_history_ordering_and_legacy_due_when_only_arrivals_change():
    factory = timing_factory(2)
    factory = replace(
        factory,
        machines=(replace(factory.machines[0], operation_types=("a", "b", "c")),),
    )
    original = materialize_workload(recipe(), 33)
    changed = recipe(
        tick_limit=4096,
        arrivals={
            "initial_jobs": 1,
            "window_jobs": 3,
            "interval_seconds": {"distribution": "exponential", "mean": 30},
        },
    )
    output = materialize_workload(changed, 33, factory=factory)
    for field in (
        "pool_sha256",
        "pool",
        "initial_history",
        "levels",
        "category_counts",
    ):
        assert output.provenance[field] == original.provenance[field]
    assert output.provenance["window_basis"] == "job-index"
    assert "window_seconds" not in output.provenance
    for new, old in zip(
        output.workload.demands, original.workload.demands, strict=True
    ):
        assert new.steps == old.steps
        assert new.due_at - new.release_at == old.due_at - old.release_at
