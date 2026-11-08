"""Scientific identity and canonical study expansion without tracking or training."""

import json

import pytest
import yaml

from smartsom.config.base_diagnostic import (
    BaseExperiment,
    generate_scenario,
    load_inputs,
)
from smartsom.config.codec import digest, primitive
from smartsom.config.reservation import compile_reservation
from smartsom.experiments.launch import compile_launch
from tests.unit.test_algorithm_components import author, recipe


@pytest.mark.parametrize(
    "seeds, expected", [(1, (101,)), (3, (101, 102, 103)), ([42, 123], (42, 123))]
)
def test_training_seed_shortcut(seeds, expected):
    ex = BaseExperiment(
        schema="smartsom.resolved-experiment/v1",
        factory="a",
        workload="b",
        algorithm="c",
        train_seeds=seeds,
    )
    assert ex.train_seeds == expected


@pytest.mark.parametrize(
    "seeds", [True, 0, -1, [], [101, 101], [True], [1.2], [-1], "3"]
)
def test_invalid_training_seeds(seeds):
    with pytest.raises(ValueError):
        BaseExperiment(
            schema="smartsom.resolved-experiment/v1",
            factory="a",
            workload="b",
            algorithm="c",
            train_seeds=seeds,
        )


def test_exact_nodes_require_compatible_budget():
    base = dict(
        schema="smartsom.resolved-experiment/v1",
        factory="a",
        workload="b",
        algorithm="c",
        train_ticks=40,
        periodic_evaluation={"unit": "physical_ticks", "milestones": [0, 17, 40]},
    )
    assert BaseExperiment(**base).periodic_evaluation.milestones == (0, 17, 40)
    for changes in (
        {"train_ticks": 39},
        {"train_episodes": 2},
        {"budget_boundary": "complete_episode"},
    ):
        with pytest.raises(ValueError):
            BaseExperiment(**{**base, **changes})


def case_file(tmp_path, split, factory, workload, seed):
    case = {
        "id": f"case-{seed}",
        "condition": "normal",
        "seed": seed,
        "action_seeds": [5, 7],
        "scenario": primitive(generate_scenario(factory, workload, seed, "normal", 30)),
    }
    file = tmp_path / f"{split}.json"
    file.write_text(
        json.dumps(
            {
                "schema": "smartsom.fixed-cases/v1",
                "split": split,
                "factory_sha256": digest(primitive(factory)),
                "workload_sha256": digest(workload.model_dump(mode="json")),
                "cases": [case],
            }
        )
    )
    return file


def test_case_import_freezes_bytes_identity_and_separates_test(tmp_path):
    path = author(tmp_path, recipe("rule"))
    ex, factory, workload, _, _ = load_inputs(path)
    for split, seed in (("periodic_evaluation", 11), ("test", 12)):
        case_file(tmp_path, split, factory, workload, seed)
    doc = yaml.safe_load(path.read_text())
    doc["case_sets"] = {
        "periodic_evaluation": "periodic_evaluation.json",
        "test": "test.json",
    }
    path.write_text(yaml.safe_dump(doc))
    plan = compile_reservation(path)
    assert plan.experiment.periodic_evaluation_cases == plan.experiment.test_cases == 1
    assert len(plan.experiment.case_set_sha256) == 2
    frozen = tmp_path / "frozen"
    frozen.mkdir()
    plan.materialize(frozen)
    (tmp_path / "periodic_evaluation.json").unlink()
    assert (
        json.loads((frozen / "cases/periodic_evaluation.json").read_text())["cases"][0][
            "seed"
        ]
        == 11
    )
    case_file(tmp_path, "periodic_evaluation", factory, workload, 12)
    with pytest.raises(ValueError, match="same physical world"):
        compile_reservation(path)


def test_research_directory_expands_seeds_and_ignores_auxiliary_yaml(tmp_path):
    paths = []
    for variant in ("shared", "independent"):
        folder = tmp_path / "experiments/net/01-test" / variant
        folder.mkdir(parents=True)
        data = recipe("ppo")
        if variant == "independent":
            data["production"]["policies"]["ppo"]["parameter_sharing"] = {
                "roles": {"machine": "independent"}
            }
        path = author(folder, data)
        document = yaml.safe_load(path.read_text())
        document.update(train_seeds=2, drain_frozen=True, run={"task": "train"})
        path.write_text(yaml.safe_dump(document))
        paths.append(path)
    study = paths[0].parent.parent
    (study / "common").mkdir()
    (study / "common/experiment.yaml").write_text("not an entry")
    plan = compile_launch(study)
    assert len(plan.files) == 4
    assert plan.output_root == tmp_path / "runs/net/01-test"
    assert {row["child"] for row in plan.files} == {
        f"trials/{v}/seed-{s}" for v in ("shared", "independent") for s in (101, 102)
    }
    assert len({row["scientific_sha256"] for row in plan.files}) == 4
    assert not plan.output_root.exists()
    single = compile_launch(paths[0])
    directory = compile_launch(paths[0].parent)
    assert [r["child"] for r in single.files] == [r["child"] for r in directory.files]
    assert len(single.files) == 2
    explicit = compile_launch(paths[0], seed=42)
    assert len(explicit.files) == 1
    assert explicit.files[0]["compiled"].experiment.run.seed == 42


def test_original_map_studio_roundtrip_preserves_science(tmp_path):
    from pathlib import Path

    from smartsom.config.factory_design import (
        load_factory_design_file,
        save_factory_design_file,
    )

    source = (
        Path(__file__).resolve().parents[2]
        / "src/smartsom/config/maps/medium-wide-factory.yaml"
    )
    original, _ = load_factory_design_file(source)
    target = tmp_path / "edited.yaml"
    save_factory_design_file(target, original)
    restored, _ = load_factory_design_file(target)
    assert restored == original


def test_report_uses_evaluated_model_and_marks_incomplete_panel(tmp_path):
    from smartsom.experiments.research_report import write_reports

    trial = tmp_path / "trials/shared/seed-101"
    evaluation = trial / "evaluation/argmax"
    (evaluation / "case-commits").mkdir(parents=True)
    (trial / "best.pt").write_bytes(b"best")
    (trial / "last.pt").write_bytes(b"last")
    (evaluation / "manifest.json").write_text(
        json.dumps({"checkpoint": "best.pt", "checkpoint_sha256": "selected-model"})
    )
    (evaluation / "case-commits/plan.json").write_text(json.dumps({"cases": [1, 2]}))
    write_reports(
        tmp_path, {"files": [{"study": ".", "child": "trials/shared/seed-101"}]}
    )
    import csv

    rows = list(csv.DictReader((tmp_path / "report/metrics.csv").open()))
    argmax = next(row for row in rows if row["decoder"] == "argmax")
    assert argmax["checkpoint"].endswith("best.pt")
    assert argmax["checkpoint_sha256"] == "selected-model"
    assert argmax["evaluation_complete"] == "False"
    assert argmax["planned_cases"] == "2"


def test_report_keeps_prespecified_checkpoint_panels_separate(tmp_path):
    import csv

    from smartsom.experiments.research_report import write_reports

    trial = tmp_path / "trials/base/seed-4101"
    trial.mkdir(parents=True)
    (trial / "authoring.json").write_text(
        json.dumps(
            {
                "evaluation": {
                    "checkpoints": [
                        {"episode": n, "modes": ["argmax"]} for n in (500, 750, 1000)
                    ]
                }
            }
        )
    )
    for episode, cost in ((500, 2.0), (1000, 8.0)):
        panel = trial / "evaluation" / f"episode-{episode:08d}" / "argmax"
        (panel / "case-commits").mkdir(parents=True)
        checkpoint = trial / "checkpoints" / f"episode-{episode:08d}.pt"
        checkpoint.parent.mkdir(exist_ok=True)
        checkpoint.write_bytes(str(episode).encode())
        (panel / "manifest.json").write_text(
            json.dumps({"checkpoint": str(checkpoint)})
        )
        (panel / "case-commits/plan.json").write_text(json.dumps({"cases": [1]}))
        case = dict(
            total_orders=1,
            completed=1,
            failed=0,
            tick=10,
            mean_tardiness=cost,
            joint_cost=cost,
            method="dqn",
            scenario_seed=7,
            condition="normal",
            action_seed=0,
        )
        (panel / "cases.jsonl").write_text(json.dumps(case) + "\n")
    write_reports(
        tmp_path, {"files": [{"study": ".", "child": "trials/base/seed-4101"}]}
    )
    rows = list(csv.DictReader((tmp_path / "report/metrics.csv").open()))
    assert [r["checkpoint_episode"] for r in rows] == ["500", "750", "1000"]
    assert [r["actual_cases"] for r in rows] == ["1", "0", "1"]
    assert [r["evaluation_complete"] for r in rows] == ["True", "False", "True"]
    assert [r["joint_cost"] for r in rows] == ["2.0", "", "8.0"]
    assert rows[-1]["checkpoint"].endswith("checkpoints/episode-00001000.pt")
    summary = read_json = json.loads((tmp_path / "report/summary.json").read_text())
    costs = [r for r in summary["statistics"] if r["metric"] == "joint_cost"]
    assert [r["checkpoint_episode"] for r in costs] == [500, 750, 1000]
    assert read_json["schema"] == "smartsom.research-report/v1"


def test_imported_validation_cannot_reuse_generated_test_world(tmp_path):
    from smartsom.config.base_diagnostic import seed_for

    path = author(tmp_path, recipe("rule"))
    ex, factory, workload, _, _ = load_inputs(path)
    file = case_file(
        tmp_path, "periodic_evaluation", factory, workload, seed_for(90000, 0) % (2**31)
    )
    data = json.loads(file.read_text())
    data["cases"][0]["scenario"] = primitive(
        generate_scenario(
            factory, workload, data["cases"][0]["seed"], "normal", ex.drain_ticks
        )
    )
    file.write_text(json.dumps(data))
    document = yaml.safe_load(path.read_text())
    document["case_sets"] = {"periodic_evaluation": "periodic_evaluation.json"}
    path.write_text(yaml.safe_dump(document))
    with pytest.raises(ValueError, match="same physical world"):
        compile_reservation(path)


@pytest.mark.parametrize(
    "field,value",
    [
        ("training_conditions", ("high",)),
        ("training_world_domain", "another-world"),
        ("training_world_prefix", ("another-prefix",)),
        ("training_seed_modulus", None),
    ],
)
def test_training_world_changes_are_not_social_only(tmp_path, field, value):
    from smartsom.telemetry.evaluation_summary import sl_comparison_identity

    path = author(tmp_path, recipe("ppo"))
    ex, factory, workload, algorithm, paths = load_inputs(path)
    before = sl_comparison_identity(algorithm, ex, factory=factory, workload=workload)
    after = sl_comparison_identity(
        algorithm,
        ex.model_copy(update={field: value}),
        factory=factory,
        workload=workload,
    )
    assert before["non_sl_sha256"] != after["non_sl_sha256"]


def inline_panel(factory, workload, drain=30):
    return {
        "conditions": ["normal", "low", "medium", "high"],
        "cases_per_condition": 1,
        "cases": [
            {
                "id": f"test-{condition}",
                "condition": condition,
                "seed": 71 + i,
                "action_seeds": [123456789123456789 + i],
                "scenario_sha256": digest(
                    primitive(
                        generate_scenario(factory, workload, 71 + i, condition, drain)
                    )
                ),
            }
            for i, condition in enumerate(("normal", "low", "medium", "high"))
        ],
    }


def test_inline_panel_freezes_relocates_and_recompiles_without_authored_case_files(
    tmp_path,
):
    from smartsom.config.evaluation_cases import experiment_panels

    path = author(tmp_path, recipe())
    _, factory, workload, _, _ = load_inputs(path)
    document = yaml.safe_load(path.read_text())
    document.pop("test_cases", None)
    document.update(test=inline_panel(factory, workload), drain_ticks=30)
    path.write_text(yaml.safe_dump(document))
    ex, factory, workload, _, _ = load_inputs(path)
    assert ex.test_cases == 4 and not ex.case_sets
    expected = experiment_panels(ex, path.parent, factory, workload)["test"]
    plan = compile_reservation(path)
    assert plan.experiment.test is None
    assert plan.experiment.case_set_sha256["test"] == digest(expected)
    frozen = tmp_path / "relocated"
    frozen.mkdir()
    frozen_path = plan.materialize(frozen)
    path.unlink()
    restored, f, w, _, _ = load_inputs(frozen_path)
    assert experiment_panels(restored, frozen, f, w)["test"] == expected
    recompiled = compile_reservation(frozen_path)
    assert recompiled.experiment.case_set_sha256 == plan.experiment.case_set_sha256
    # Checksum enforcement also applies without condition_scenarios.
    inventory = frozen / restored.case_sets["test"]
    data = json.loads(inventory.read_text())
    data["cases"][0]["action_seeds"] = [42]
    inventory.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="inventory digest"):
        compile_reservation(frozen_path)


@pytest.mark.parametrize(
    "change",
    ["count", "duplicate", "boolean_seed", "boolean_action", "condition", "both"],
)
def test_inline_panel_rejects_invalid_or_ambiguous_definition(tmp_path, change):
    path = author(tmp_path, recipe())
    _, factory, workload, _, _ = load_inputs(path)
    document = yaml.safe_load(path.read_text())
    document.pop("test_cases", None)
    document.update(test=inline_panel(factory, workload), drain_ticks=30)
    panel = document["test"]
    if change == "count":
        panel["cases_per_condition"] = 2
    elif change == "duplicate":
        panel["cases"][1]["id"] = panel["cases"][0]["id"]
    elif change == "boolean_seed":
        panel["cases"][0]["seed"] = True
    elif change == "boolean_action":
        panel["cases"][0]["action_seeds"] = [True]
    elif change == "condition":
        panel["cases"][0]["condition"] = "mid"
    else:
        document["case_sets"] = {"test": "not-read.json"}
    path.write_text(yaml.safe_dump(document))
    with pytest.raises(ValueError):
        compile_reservation(path)


def test_inline_case_world_pin_rejects_changed_scenario_or_drain(tmp_path):
    path = author(tmp_path, recipe())
    _, factory, workload, _, _ = load_inputs(path)
    document = yaml.safe_load(path.read_text())
    document.pop("test_cases", None)
    document.update(test=inline_panel(factory, workload), drain_ticks=30)
    path.write_text(yaml.safe_dump(document))
    for override in ("experiment.drain_ticks=31", "scenario.jobs.orders.total=129"):
        with pytest.raises(ValueError, match="frozen historical world"):
            compile_reservation(path, sets=[override])


@pytest.mark.parametrize("calibrated", [False, True])
def test_inline_completion_panel_uses_effective_protection_horizon(
    tmp_path, calibrated
):
    from smartsom.config.evaluation_cases import experiment_panels

    path = author(tmp_path, recipe())
    _, f, w, _, _ = load_inputs(path)
    document = yaml.safe_load(path.read_text())
    document.pop("test_cases", None)
    panel = inline_panel(f, w)
    for case in panel["cases"]:
        case.pop("scenario_sha256")  # New worlds, no old-world pins to preserve.
    document.update(
        test=panel,
        drain_ticks=30,
        evaluation={
            "mode": "completion",
            "completion_drain_multiplier": 2,
        },
    )
    if calibrated:
        (tmp_path / "calibration.json").write_text(json.dumps({"drain_ticks": 41}))
        document.setdefault("run", {})["calibration"] = "calibration.json"
    path.write_text(yaml.safe_dump(document))
    plan = compile_reservation(path)
    inventory = json.loads(dict(plan.components)["cases/test.json"])["cases"]
    actual_drain = 82 if calibrated else 60
    for case in inventory:
        assert case["scenario"] == primitive(
            generate_scenario(
                f,
                w,
                case["seed"],
                case["condition"],
                actual_drain,
            )
        )
    ex, f, w, _, _ = load_inputs(path)
    assert (
        experiment_panels(ex, tmp_path, f, w, drain_ticks=actual_drain)["test"]
        == inventory
    )
    if not calibrated:
        assert experiment_panels(ex, tmp_path, f, w)["test"] == inventory


def test_inline_test_cannot_be_silently_ignored_by_controlled_recovery(tmp_path):
    path = author(tmp_path, recipe())
    _, f, w, _, _ = load_inputs(path)
    document = yaml.safe_load(path.read_text())
    document.pop("test_cases", None)
    document.update(
        test=inline_panel(f, w),
        evaluation={
            "recovery": {
                "action": "calibrate",
                "candidates": [
                    {
                        "name": "candidate",
                        "load": "normal",
                        "orders": 4,
                        "initial_orders": 2,
                        "mean_interarrival": 2,
                        "horizon_ticks": 100,
                        "window_ticks": 2,
                        "baseline_windows": 3,
                        "minimum_window_deliveries": 1,
                        "maximum_cv": 1,
                        "maximum_relative_trend": 1,
                        "faults": [
                            {
                                "machine_id": "machine_00",
                                "start_tick": 10,
                                "duration_ticks": 2,
                            }
                        ],
                    }
                ],
            }
        },
    )
    path.write_text(yaml.safe_dump(document))
    with pytest.raises(
        ValueError, match="inline test panels do not support controlled recovery"
    ):
        compile_reservation(path)


def test_v2_learning_recipes_share_twelve_case_validation_and_bounded_storage():
    from collections import Counter
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    count = 0
    for scale in ("small", "medium", "large"):
        for seed in (101, 102, 103):
            reference = None
            for method in (
                "base-shared",
                "isl-shared",
                "base-independent",
                "isl-independent",
                "morris",
            ):
                path = root / f"experiments/v2/{scale}/{method}-{seed}/experiment.yaml"
                document = yaml.safe_load(path.read_text())
                assert document["checkpoint_retention"] == "best_last"
                assert document["checkpoint_every_episodes"] == 50
                assert document["periodic_evaluation_cases"] == 12
                assert document["periodic_evaluation"]["milestones"] == list(
                    range(50, 1001, 50)
                )
                assert (
                    document["periodic_evaluation"]["selection"]
                    == "argmax_complete_cost"
                )
                assert document["periodic_evaluation"]["modes"] == ["argmax"]
                assert Counter(
                    c["condition"] for c in document["test"]["cases"]
                ) == dict.fromkeys(("normal", "low", "medium", "high"), 20)
                document.pop("algorithm")
                assert reference is None or document == reference
                reference = document
                count += 1
    assert count == 45


def test_bounded_retention_rejects_unretained_evaluation_nodes():
    base = dict(
        schema="smartsom.resolved-experiment/v1",
        factory="a",
        workload="b",
        algorithm="c",
        train_episodes=100,
        final_evaluation=True,
        checkpoint_retention="best_last",
        periodic_evaluation={
            "milestones": [50, 100],
            "modes": ["argmax"],
            "selection": "argmax_complete_cost",
        },
    )
    assert (
        BaseExperiment(
            **base, evaluation={"checkpoints": [{"episode": 100}]}
        ).checkpoint_retention
        == "best_last"
    )
    with pytest.raises(ValueError, match="final prescribed"):
        BaseExperiment(**base, evaluation={"checkpoints": [{"episode": 50}]})
    with pytest.raises(ValueError, match="requires periodic_evaluation best"):
        BaseExperiment(**{**base, "periodic_evaluation": {}})
