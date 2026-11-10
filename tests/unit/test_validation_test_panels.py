"""Stable validation/test domains and backward-readable frozen inventories."""

import json
from pathlib import Path

import pytest
import yaml

from smartsom.config.base_diagnostic import BaseExperiment, load_inputs
from smartsom.config.evaluation_cases import FixedTestPanel, experiment_panels
from smartsom.config.reservation import compile_reservation

ROOT = Path(__file__).resolve().parents[2]
BASE = dict(
    schema="smartsom.resolved-experiment/v1", factory="f", workload="w", algorithm="a"
)


def test_count_means_per_condition_with_stable_prefix_and_round_trip():
    small = BaseExperiment.model_validate(
        BASE | {"validation": {"seeds": 3}, "test": {"seeds": 3}}
    )
    large = BaseExperiment.model_validate(
        BASE | {"validation": {"seeds": 20}, "test": {"seeds": 20}}
    )
    assert small.validation_cases == small.test_cases == 12
    assert large.validation_cases == large.test_cases == 80
    assert small.validation.seeds == (10001, 10002, 10003)
    assert large.validation.seeds[:3] == small.validation.seeds
    assert {c.seed for c in small.test.cases} == {20001, 20002, 20003}
    assert [c.model_dump() for c in large.test.cases[:12]] == [
        c.model_dump() for c in small.test.cases
    ]
    assert {c.action_seeds for c in small.test.cases} == {(0,)}
    data = large.model_dump(mode="json", by_alias=True)
    assert "periodic_evaluation" not in data
    assert (
        BaseExperiment.model_validate(data).model_dump(mode="json", by_alias=True)
        == data
    )


@pytest.mark.parametrize(
    "seeds", [0, -1, True, False, [], [1, 1], [-1], [False], [1.5], "20", None]
)
@pytest.mark.parametrize("split", ["validation", "test"])
def test_invalid_seed_inputs_fail_closed(seeds, split):
    with pytest.raises(ValueError):
        BaseExperiment.model_validate(BASE | {split: {"seeds": seeds}})


def test_explicit_seeds_and_policy_seeds_are_distinct():
    ex = BaseExperiment.model_validate(
        BASE
        | {
            "validation": {"seeds": [39, 7], "sample_action_seeds": 2},
            "test": {"seeds": [28, 6], "action_seeds": [8, 9]},
        }
    )
    assert ex.validation.seeds == (39, 7)
    assert [c.seed for c in ex.test.cases] == [28] * 4 + [6] * 4
    assert {c.action_seeds for c in ex.test.cases} == {(8, 9)}


@pytest.mark.parametrize(
    "extra",
    [
        {"validation": {"seeds": [7]}, "test": {"seeds": [7]}},
        {"validation": {}, "periodic_evaluation": {}},
        {"validation_cases": 4, "periodic_evaluation_cases": 4},
        {"validation": {"seeds": 2}, "validation_cases": 4},
        {"test": {"seeds": 2}, "test_cases": 4},
        {"test": {"seeds": 2, "cases": []}},
        {"validation": {"seeds": 2}, "case_sets": {"validation": "old.json"}},
    ],
)
def test_conflicts_and_overlap_rejected(extra):
    with pytest.raises(ValueError):
        BaseExperiment.model_validate(BASE | extra)


def test_old_count_and_serialization_are_retained():
    old = BaseExperiment.model_validate(
        BASE
        | {"periodic_evaluation": {"milestones": [50]}, "periodic_evaluation_cases": 12}
    )
    assert old.validation_split == "periodic_evaluation"
    assert old.test_directory == "evaluation"
    assert old.validation_cases == 12 and old.validation.seeds is None
    data = old.model_dump(mode="json", by_alias=True)
    assert "validation" not in data and "validation_cases" not in data
    assert "seeds" not in data["periodic_evaluation"]
    assert set(data["replay"]) == {"train", "periodic_evaluation", "evaluation"}
    assert (
        BaseExperiment.model_validate(data).model_dump(mode="json", by_alias=True)
        == data
    )


def test_existing_explicit_test_inventory_is_not_replaced():
    panel = FixedTestPanel.model_validate(
        {
            "conditions": ["normal"],
            "cases_per_condition": 1,
            "cases": [
                {
                    "id": "historical",
                    "condition": "normal",
                    "seed": 771,
                    "action_seeds": [491],
                }
            ],
        }
    )
    ex = BaseExperiment.model_validate(BASE | {"test": panel.model_dump(mode="json")})
    assert ex.test.cases[0].id == "historical"
    assert ex.test.cases[0].seed == 771
    assert ex.test.cases[0].action_seeds == (491,)
    assert ex.test_cases == 1


def test_compile_freezes_both_panels_and_detects_tampering(tmp_path):
    plan = compile_reservation(
        ROOT / "experiments/v3/small/rule-normal/experiment.yaml",
        sets=["experiment.validation.seeds=1", "experiment.test.seeds=1"],
    )
    path = plan.materialize(tmp_path)
    ex, factory, workload, _, _ = load_inputs(path)
    assert ex.validation_split == "validation"
    assert ex.validation_cases == ex.test_cases == 4
    assert ex.validation.seeds is None  # Frozen references, no second mutable source.
    assert ex.test is None
    panels = experiment_panels(ex, tmp_path, factory, workload)
    assert {c["seed"] for c in panels["validation"]} == {10001}
    assert {c["seed"] for c in panels["test"]} == {20001}
    assert set(ex.case_set_sha256) == {"validation", "test"}
    frozen = tmp_path / "cases/validation.json"
    data = json.loads(frozen.read_text())
    data["cases"][0]["seed"] += 1
    frozen.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="digest differs"):
        compile_reservation(path)


def test_v3_training_recipes_preserve_confirmed_panel_protocol():
    paths = [
        path
        for scale in ("small", "medium", "large")
        for path in (ROOT / "experiments/v3" / scale).glob("*/experiment.yaml")
    ]
    assert len(paths) == 27
    for path in paths:
        data = yaml.safe_load(path.read_text())
        assert data["validation"]["seeds"] == data["test"]["seeds"] == 50
        assert data["concurrency"]["evaluation_processes"] == 20
        assert "periodic_evaluation" not in data
        assert data["tracking"]["mode"] == "online"
        if data["run"]["task"] == "train":
            assert data["train_seeds"] == [101, 102, 103, 104, 105]
            assert data["validation"]["selection"] == "argmax_delivery_cost"
        else:
            assert data["train_seeds"] == [101]


@pytest.mark.parametrize(
    "scale,count,initial,candidates", [("small", 300, 40, 2), ("medium", 750, 100, 5)]
)
def test_v3_workload_balances_templates_and_preserves_arrival_realization(
    scale, count, initial, candidates
):
    from collections import Counter
    from math import ceil

    from smartsom.config.base_diagnostic import generate_scenario

    path = ROOT / f"experiments/v3/{scale}/rule-normal/experiment.yaml"
    ex, factory, workload, _, _ = load_inputs(path)
    source = yaml.safe_load((ROOT / f"configs/scenarios/v3/{scale}.yaml").read_text())
    families = source["jobs"]["routes"]["families"]
    assert Counter(map(len, families)) == {3: 8, 4: 8, 5: 8, 6: 8}
    assert len({tuple(route) for route in families}) == 32
    for length in (3, 4, 5, 6):
        routes = [route for route in families if len(route) == length]
        assert all(len(set(route)) == length for route in routes)
        for column in zip(*routes, strict=True):
            assert set(column) == set("ABCDEFGH")
    machines = source["factory"]["machines"]["items"]
    assert all(len(set(m["operations"])) == 2 for m in machines)
    assert Counter(op for m in machines for op in m["operations"]) == dict.fromkeys(
        "ABCDEFGH", candidates
    )
    assert workload.orders == count and workload.initial_orders == initial
    assert workload.optional_repeat_probability == 0
    world = generate_scenario(factory, workload, 101, "normal", ex.drain_ticks)
    old_path = ROOT / f"experiments/v2/{scale}/justin-101/experiment.yaml"
    old_ex, old_factory, old_workload, _, _ = load_inputs(old_path)
    old_world = generate_scenario(
        old_factory, old_workload, 101, "normal", old_ex.drain_ticks
    )
    assert [d.release_at for d in world.demands] == [
        d.release_at for d in old_world.demands
    ]
    for demand in world.demands:
        types = [step.operation_type for step in demand.steps]
        assert types in families and len(set(types)) == len(types)
        for step in demand.steps:
            work = float(step.reference_ticks)
            base = workload.base_ticks[step.operation_type]
            assert 0.8 * base <= work <= 1.2 * base
            assert type(step.nominal_ticks) is int and step.nominal_ticks == ceil(work)


@pytest.mark.parametrize("scale", ["small", "medium"])
def test_v3_calibration_is_separate_from_training_panels(scale):
    calibration, *_ = load_inputs(
        ROOT / f"experiments/v3/calibration/{scale}/experiment.yaml"
    )
    training, *_ = load_inputs(
        ROOT / f"experiments/v3/{scale}/base-shared/experiment.yaml"
    )
    seeds = {c.seed for c in calibration.test.cases}
    assert seeds == set(range(71001, 71009))
    assert len(calibration.test.cases) == 32
    assert not seeds.intersection(training.validation.seeds)
    assert not seeds.intersection(c.seed for c in training.test.cases)


@pytest.mark.parametrize("difference", [None, "count", "validation_hash", "test_hash"])
def test_dashboard_keeps_different_frozen_panels_in_separate_cohorts(
    tmp_path, monkeypatch, difference
):
    from smartsom.telemetry import wandb_dashboard

    records = []
    for index in range(2):
        root = tmp_path / str(index)
        (root / "inputs").mkdir(parents=True)
        document = {
            "validation_cases": 400,
            "test_cases": 400,
            "case_set_sha256": {"validation": "same-val", "test": "same-test"},
        }
        if index and difference == "count":
            document["validation_cases"] = 80
        elif index and difference:
            document["case_set_sha256"][difference.removesuffix("_hash")] = "other"
        (root / "inputs/experiment.yaml").write_text(yaml.safe_dump(document))
        records.append(
            {
                "root": str(root),
                "status": {"entity": "test", "project": "test"},
                "manifest": {
                    "inputs": {"factory": "same", "workload": "same"},
                    "source": {"implementation_sha256": "same"},
                },
                "events": [{"phase": "validation", "selection": "argmax"}],
            }
        )
    monkeypatch.setattr(wandb_dashboard, "records", lambda root: records)
    groups = wandb_dashboard.plan(tmp_path)
    assert sorted(map(len, groups.values())) == ([2] if difference is None else [1, 1])


@pytest.mark.parametrize("mode", ["first", "all", "off"])
def test_legacy_replay_fallback_and_explicit_override(mode):
    from smartsom.config.base_diagnostic import ReplaySettings

    old = ReplaySettings(periodic_evaluation=mode, evaluation=mode)
    assert old.for_split("periodic_evaluation") == mode
    assert old.for_split("validation") == mode
    assert old.for_split("test") == mode
    assert (
        ReplaySettings(periodic_evaluation="all", validation="off").for_split(
            "validation"
        )
        == "off"
    )


def test_validation_inventory_name_is_unambiguous():
    ex = BaseExperiment.model_validate(BASE | {"case_sets": {"validation": "v.json"}})
    assert ex.validation_split == "validation"
    with pytest.raises(ValueError, match="inventory key"):
        BaseExperiment.model_validate(
            BASE | {"validation": {}, "case_sets": {"periodic_evaluation": "v.json"}}
        )


@pytest.mark.parametrize("new", [False, True])
def test_final_receipt_and_report_paths_follow_frozen_inputs(tmp_path, new):
    from smartsom.experiments.incremental_batch import pending_evaluation
    from smartsom.experiments.research_report import evaluation_panels

    child = tmp_path / "trial"
    (child / "inputs").mkdir(parents=True)
    ex = BaseExperiment.model_validate(
        BASE | {"final_evaluation": True} | ({"validation": {}} if new else {})
    )
    (child / "inputs/experiment.yaml").write_text(
        yaml.safe_dump(ex.model_dump(mode="json", by_alias=True))
    )
    row = {"task": "train", "child": "trial"}
    assert pending_evaluation(tmp_path, row)
    directory = child / ("test" if new else "evaluation")
    directory.mkdir()
    (directory / "completed.json").write_text("{}")
    assert not pending_evaluation(tmp_path, row)
    (directory / "episode-00000050/argmax").mkdir(parents=True)
    assert evaluation_panels(child) == [
        (50, "argmax", directory / "episode-00000050/argmax")
    ]


def test_validation_phase_meter_preserves_time_and_case_work():
    from smartsom.telemetry.estimates import PhaseMeter

    results = []
    for phase in ("validation", "periodic_evaluation"):
        meter = PhaseMeter(minimum=0)
        for now, tick in ((1, 10), (11, 110)):
            meter.observe(
                {
                    "stage": phase,
                    "status": "running",
                    "values": {phase + "_tick": tick, phase + "_finished": 0},
                },
                now=now,
                wall_time=now,
            )
        results.append((dict(meter.seconds), dict(meter.work)))
    assert results[0] == results[1]
    assert results[0][0] == {"periodic_evaluation": 10}


def test_validation_recovery_cannot_silently_replace_authored_seeds():
    recovery = {
        "recovery": {
            "action": "calibrate",
            "candidates": [
                {
                    "name": "normal",
                    "load": "normal",
                    "orders": 100,
                    "initial_orders": 10,
                    "mean_interarrival": 2,
                    "horizon_ticks": 1000,
                    "window_ticks": 10,
                    "baseline_windows": 3,
                    "minimum_window_deliveries": 1,
                    "maximum_cv": 1,
                    "maximum_relative_trend": 1,
                    "faults": [
                        {"machine_id": "m1", "start_tick": 100, "duration_ticks": 100}
                    ],
                }
            ],
        }
    }
    # Prove the existing recovery configuration itself is valid before adding seeds.
    BaseExperiment.model_validate(BASE | {"evaluation": recovery})
    for panel in (
        {"validation": {"seeds": [1234]}},
        {"case_sets": {"validation": "v.json"}},
    ):
        with pytest.raises(
            ValueError, match="validation panels do not support controlled recovery"
        ):
            BaseExperiment.model_validate(BASE | panel | {"evaluation": recovery})


def test_inferred_canonical_names_survive_compilation(tmp_path):
    source = ROOT / "experiments/v3/small/rule-normal/experiment.yaml"
    data = yaml.safe_load(source.read_text())
    for name in ("scenario", "algorithm"):
        data[name] = str((source.parent / data[name]).resolve())
    data.pop("validation")
    data["test"] = {"seeds": 1}
    data["run"].pop("split", None)
    path = tmp_path / "recipe.yaml"
    path.write_text(yaml.safe_dump(data))
    mixed = {
        **data,
        "periodic_evaluation": {"milestones": [], "modes": ["argmax"]},
        "periodic_evaluation_cases": 12,
    }
    path.write_text(yaml.safe_dump(mixed))
    mixed_plan = compile_reservation(path)
    assert mixed_plan.experiment.validation_split == "validation"
    assert mixed_plan.experiment.validation_cases == 12
    path.write_text(yaml.safe_dump(data))
    plan = compile_reservation(path)
    assert plan.experiment.validation_split == "validation"
    assert plan.experiment.test_directory == "test"
    frozen = tmp_path / "frozen"
    frozen.mkdir()
    plan.materialize(frozen)
    assert load_inputs(frozen / "experiment.yaml")[0].test_directory == "test"

    # Freeze validation-only shorthand, remove redundant explicit schedule, and recompile.
    data.pop("test")
    data["validation"] = {"seeds": 1}
    path.write_text(yaml.safe_dump(data))
    other = tmp_path / "validation-only"
    other.mkdir()
    compile_reservation(path).materialize(other)
    frozen_path = other / "experiment.yaml"
    raw = yaml.safe_load(frozen_path.read_text())
    raw.pop("validation")
    raw.pop("validation_cases")
    frozen_path.write_text(yaml.safe_dump(raw))
    assert compile_reservation(frozen_path).experiment.validation_split == "validation"


@pytest.mark.parametrize("phase", ["validation", "test"])
def test_display_retains_canonical_case_events(tmp_path, phase):
    from smartsom.telemetry.runtime import DisplayOptions, RuntimeDisplay

    view = RuntimeDisplay(DisplayOptions(progress="off"), quiet=True)
    view.update(
        "case",
        {
            "stage": phase,
            "status": "running",
            "data_seed": 7,
            phase + "_tick": 100,
            phase + "_finished": 1,
            phase + "_requested": 4,
        },
    )
    values = view.tasks["case"]["values"]
    assert values["data_seed"] == 7
    assert values[phase + "_tick"] == 100
    assert values[phase + "_finished"] == 1
    assert values[phase + "_requested"] == 4


def test_canonical_validation_rounds_have_independent_case_counters():
    from smartsom.telemetry.estimates import PhaseMeter

    meter = PhaseMeter(minimum=0)
    for now, node, tick in ((1, 50, 100), (2, 50, 200), (3, 100, 10), (4, 100, 20)):
        meter.observe(
            {
                "stage": "validation",
                "status": "running",
                "values": {
                    "validation_round": node,
                    "validation_tick": tick,
                    "validation_finished": 0,
                },
            },
            now=now,
            wall_time=now,
        )
    assert meter.reason is None
    assert meter.work["periodic_evaluation"] == 110
