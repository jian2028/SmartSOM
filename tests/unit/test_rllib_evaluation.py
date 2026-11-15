"""Fixed-panel scientific aggregation, independent of optional frameworks."""

import pytest

from smartsom.learning.rllib_evaluation import summarize_panel


def panel(mode="sample"):
    cases = [
        {"case_id": "a", "mode": mode},
        {"case_id": "b", "mode": mode},
        {"case_id": "c", "mode": mode},
    ]
    rows = [
        {
            **cases[0],
            "joint_cost": 2,
            "makespan": 10,
            "mean_tardiness": 20,
            "final_passing_rate": 1,
        },
        {
            **cases[1],
            "joint_cost": 4,
            "makespan": 30,
            "mean_tardiness": 60,
            "final_passing_rate": 0.5,
        },
        {
            **cases[2],
            "joint_cost": None,
            "makespan": None,
            "mean_tardiness": None,
            "final_passing_rate": None,
        },
    ]
    return cases, rows


@pytest.mark.parametrize("mode", ["sample", "argmax"])
def test_complete_default_panel_selects_all_prescribed_cases(mode):
    cases, rows = panel(mode)
    cases, rows = cases[:2], rows[:2]
    result = summarize_panel(cases, list(reversed(rows)))
    assert result == {
        "cases": rows,
        "complete_panel": True,
        "selection_mode": mode,
        "joint_cost": 3,
        "makespan": 20,
        "mean_tardiness": 40,
        "final_passing_rate": 0.75,
    }


def test_incomplete_default_panel_cannot_select_completed_subset():
    cases, rows = panel()
    rows[1]["joint_cost"] = None
    result = summarize_panel(cases, rows)
    assert result["cases"] == rows
    assert not result["complete_panel"]
    assert all(
        result[key] is None
        for key in (
            "joint_cost",
            "makespan",
            "mean_tardiness",
            "final_passing_rate",
        )
    )


@pytest.mark.parametrize("problem", ["missing", "duplicate", "mode"])
def test_panel_rejects_changed_inventory(problem):
    cases, rows = panel()
    if problem == "missing":
        rows.pop()
    elif problem == "duplicate":
        rows[1] = rows[0]
    else:
        rows[1]["mode"] = "argmax"
    with pytest.raises((ValueError, RuntimeError)):
        summarize_panel(cases, rows)


@pytest.mark.parametrize("mode,replicates", [("sample", 2), ("argmax", 1)])
def test_compiled_panels_preserve_worlds_seeds_and_split(mode, replicates):
    from pathlib import Path

    from smartsom.config.base_diagnostic import generate_scenario, load_inputs, seed_for
    from smartsom.experiments.rllib_panels import compile_panels

    path = (
        Path(__file__).resolve().parents[2]
        / "experiments/examples/small/default/experiment.yaml"
    )
    ex, factory, workload, _, _ = load_inputs(path)
    ex = ex.model_copy(
        update={
            "case_sets": {},
            "test": None,
            "periodic_evaluation_cases": 4,
            "test_cases": 4,
            "evaluation": ex.evaluation.model_copy(update={"mode": "limited"}),
            "periodic_evaluation": ex.periodic_evaluation.model_copy(
                update={"sample_action_seeds": 2}
            ),
        }
    )
    panels = compile_panels(ex, path.parent, factory, workload, drain=17, mode=mode)
    for split, namespace in (("periodic_evaluation", 80000), ("test", 90000)):
        rows = panels[split]
        assert len(rows) == 4 * replicates
        assert len({row["case_id"] for row in rows}) == len(rows)
        for index, condition in enumerate(("normal", "low", "medium", "high")):
            chosen = rows[index * replicates : (index + 1) * replicates]
            expected_seed = seed_for(namespace, 0) % (2**31)
            assert [row["action_seed_identity"] for row in chosen] == list(
                range(replicates)
            )
            assert [row["action_seed"] for row in chosen] == [
                seed_for(expected_seed, replicate) % (2**31)
                for replicate in range(replicates)
            ]
            expected = generate_scenario(
                *ex.inputs_for(condition, factory, workload),
                expected_seed,
                condition,
                17,
            )
            assert all(row["scenario"] == expected for row in chosen)
            assert all(row["mode"] == mode for row in chosen)
    assert {r["world_seed"] for r in panels["test"]}.isdisjoint(
        r["world_seed"] for r in panels["periodic_evaluation"]
    )


def test_compiled_imported_panel_preserves_all_explicit_sample_seeds(tmp_path):
    import json
    from pathlib import Path

    from smartsom.config.base_diagnostic import generate_scenario, load_inputs
    from smartsom.config.codec import digest, primitive
    from smartsom.config.evaluation_cases import condition_input_identity, load_cases
    from smartsom.experiments.rllib_panels import compile_panels

    path = (
        Path(__file__).resolve().parents[2]
        / "experiments/examples/small/default/experiment.yaml"
    )
    ex, factory, workload, _, _ = load_inputs(path)
    scenario = generate_scenario(
        *ex.inputs_for("normal", factory, workload), 736, "normal", 17
    )
    inventory = {
        "schema": "smartsom.fixed-cases/v1",
        "split": "periodic_evaluation",
        "factory_sha256": digest(primitive(factory)),
        "workload_sha256": digest(workload.model_dump(mode="json")),
        "condition_inputs_sha256": condition_input_identity(ex._condition_inputs),
        "cases": [
            {
                "id": "physical-world/736",
                "condition": "normal",
                "seed": 736,
                "action_seeds": [3, 11, 29],
                "scenario": primitive(scenario),
            }
        ],
    }
    (tmp_path / "cases.json").write_text(json.dumps(inventory))
    validated = load_cases(
        tmp_path / "cases.json",
        split="periodic_evaluation",
        factory=factory,
        workload=workload,
        condition_inputs=ex._condition_inputs,
    )
    ex = ex.model_copy(
        update={
            "case_sets": {"periodic_evaluation": "cases.json"},
            "case_set_sha256": {"periodic_evaluation": digest(validated)},
            "periodic_evaluation_cases": 1,
            "test_cases": 4,
            "test": None,
            "evaluation": ex.evaluation.model_copy(update={"mode": "limited"}),
        }
    )
    rows = compile_panels(ex, tmp_path, factory, workload, drain=17, mode="sample")[
        "periodic_evaluation"
    ]
    assert [r["action_seed"] for r in rows] == [3, 11, 29]
    assert all(r["scenario"] == scenario for r in rows)
    assert all(r["world_id"] == "physical-world/736" for r in rows)
    changed = ex.model_copy(
        update={"case_set_sha256": {"periodic_evaluation": "0" * 64}}
    )
    with pytest.raises(ValueError, match="digest"):
        compile_panels(changed, tmp_path, factory, workload, drain=17, mode="sample")
