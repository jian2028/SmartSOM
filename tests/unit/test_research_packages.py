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
        schema="smartsom.base-experiment/v1",
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
            schema="smartsom.base-experiment/v1",
            factory="a",
            workload="b",
            algorithm="c",
            train_seeds=seeds,
        )


def test_exact_nodes_require_compatible_budget():
    base = dict(
        schema="smartsom.base-experiment/v1",
        factory="a",
        workload="b",
        algorithm="c",
        train_ticks=40,
        validation={"unit": "physical_ticks", "milestones": [0, 17, 40]},
    )
    assert BaseExperiment(**base).validation.milestones == (0, 17, 40)
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
    for split, seed in (("validation", 11), ("test", 12)):
        case_file(tmp_path, split, factory, workload, seed)
    doc = yaml.safe_load(path.read_text())
    doc["case_sets"] = {"validation": "validation.json", "test": "test.json"}
    path.write_text(yaml.safe_dump(doc))
    plan = compile_reservation(path)
    assert plan.experiment.validation_cases == plan.experiment.test_cases == 1
    assert len(plan.experiment.case_set_sha256) == 2
    frozen = tmp_path / "frozen"
    frozen.mkdir()
    plan.materialize(frozen)
    (tmp_path / "validation.json").unlink()
    assert (
        json.loads((frozen / "cases/validation.json").read_text())["cases"][0]["seed"]
        == 11
    )
    case_file(tmp_path, "validation", factory, workload, 12)
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
        / "configs/factories/original-20-machine.yaml"
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


def test_imported_validation_cannot_reuse_generated_test_world(tmp_path):
    from smartsom.config.base_diagnostic import seed_for

    path = author(tmp_path, recipe("rule"))
    ex, factory, workload, _, _ = load_inputs(path)
    file = case_file(
        tmp_path, "validation", factory, workload, seed_for(90000, 0) % (2**31)
    )
    data = json.loads(file.read_text())
    data["cases"][0]["scenario"] = primitive(
        generate_scenario(
            factory, workload, data["cases"][0]["seed"], "normal", ex.drain_ticks
        )
    )
    file.write_text(json.dumps(data))
    document = yaml.safe_load(path.read_text())
    document["case_sets"] = {"validation": "validation.json"}
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
    ex, _, _, algorithm, paths = load_inputs(path)
    before = sl_comparison_identity(algorithm, ex, paths)
    after = sl_comparison_identity(
        algorithm, ex.model_copy(update={field: value}), paths
    )
    assert before["non_sl_sha256"] != after["non_sl_sha256"]
