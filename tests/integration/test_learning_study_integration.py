"""Small real spawn/backend checks for the learning-study orchestration."""

import importlib.util
import os

import pytest

from smartsom.config.experiment import apply_overrides, load_preset
from smartsom.experiments.catalog import read_json
from smartsom.experiments.learning_study import run_learning_batch, search


def tiny_recipe(tmp_path):
    if importlib.util.find_spec("sb3_contrib") is None:
        if os.environ.get("SMARTSOM_REQUIRE_LEARNING") == "1":
            pytest.fail("required learning-sb3 dependency is missing")
        pytest.skip("optional learning-sb3 dependency is missing")
    return apply_overrides(
        load_preset("sb3_micro"),
        [
            ("output.root", str(tmp_path / "runs")),
            ("output.name", "study-smoke"),
            ("logging.tensorboard", False),
            ("logging.progress", "off"),
            ("logging.verbose", 0),
            ("training.total_steps", 32),
            ("training.steps_per_update", 16),
            ("algorithm.batch_size", 8),
            ("algorithm.n_epochs", 1),
            ("algorithm.hidden_sizes", [16, 16]),
            ("validation.enabled", False),
            ("checkpointing.every_updates", 2),
        ],
    )


def test_real_batch_has_independent_children_and_reuses_completed_evidence(tmp_path):
    first = tiny_recipe(tmp_path)
    second = apply_overrides(first, [("seed", 102)])
    result = run_learning_batch([first, second], max_concurrent=2)
    assert result.completed == 2
    assert result.failed == result.pending == 0
    children = [entry["runs"][0]["run_dir"] for entry in result.entries]
    assert all("attempt-000" in child for child in children)
    assert {entry["runs"][0]["seed"] for entry in result.entries} == {101, 102}
    before = {
        str(p): p.read_bytes() for p in (result.run_dir / "trials").rglob("*.json")
    }
    progress_before = read_json(result.run_dir / "logs/progress.json")
    restored = run_learning_batch(resume=result.run_dir)
    assert restored.completed == 2
    progress_after = read_json(result.run_dir / "logs/progress.json")
    assert [row["values"] for row in progress_after["tasks"]] == [
        row["values"] for row in progress_before["tasks"]
    ]
    assert all(row["values"]["ppo_updates"] == 2 for row in progress_after["tasks"])
    assert {
        str(p): p.read_bytes() for p in (result.run_dir / "trials").rglob("*.json")
    } == before


def test_real_grid_search_scores_only_audited_complete_validation(tmp_path):
    config = apply_overrides(
        tiny_recipe(tmp_path),
        [
            ("validation.enabled", True),
            ("validation.every_updates", 2),
            ("validation.replications", 1),
            ("search.objective", "makespan"),
            ("search.failure_policy", "all_complete"),
            (
                "search.space",
                {
                    "algorithm.learning_rate": {
                        "type": "categorical",
                        "choices": [0.0003],
                    }
                },
            ),
        ],
    )
    result = search(config)
    assert result.pending == 0
    assert len(result.entries) == 1
    entry = result.entries[0]
    assert entry["status"] in {"completed", "ineligible"}
    row = entry["runs"][0]
    assert row["training_audit"]["status"] == "passed"
    assert row["checkpoint_sha256"]
    if entry["status"] == "ineligible":
        assert entry["value"] is None
        assert "best_trial" not in read_json(result.run_dir / "summary.json")
    else:
        assert row["objective"]["completed"] == row["objective"]["episodes"]
        assert entry["value"] is not None


def test_real_optuna_search_persists_candidate_and_terminal_status(tmp_path):
    import os

    from smartsom.experiments.search import require_optuna

    optuna = (
        require_optuna()
        if os.environ.get("SMARTSOM_REQUIRE_SEARCH") == "1"
        else pytest.importorskip("optuna")
    )
    config = apply_overrides(
        tiny_recipe(tmp_path),
        [
            ("validation.enabled", True),
            ("validation.every_updates", 2),
            ("validation.replications", 1),
            ("search.method", "optuna"),
            ("search.trials", 1),
            ("search.objective", "makespan"),
            ("search.failure_policy", "all_complete"),
            (
                "search.space",
                {
                    "algorithm.learning_rate": {
                        "type": "float",
                        "low": 0.0001,
                        "high": 0.0004,
                    }
                },
            ),
        ],
    )
    result = search(config)
    assert result.pending == 0
    entry = result.entries[0]
    assert entry["status"] in {"completed", "ineligible"}
    database = optuna.load_study(
        study_name="smartsom-search",
        storage=f"sqlite:///{result.run_dir / 'optuna.sqlite3'}",
    )
    assert len(database.trials) == 1
    assert database.trials[0].state.name == (
        "COMPLETE" if entry["status"] == "completed" else "FAIL"
    )
    restored = search(resume=result.run_dir)
    assert restored.entries == result.entries
    assert len(database.trials) == 1


def test_unstarted_batch_runs_from_imported_snapshot_after_authoring_disappears(
    tmp_path,
):
    import shutil

    from smartsom.config.experiment import PRESET_ROOT, from_legacy
    from smartsom.experiments import learning_study as study
    from smartsom.experiments.packaging import export_experiment, import_bundle

    sources = tmp_path / "authoring"
    shutil.copytree(PRESET_ROOT, sources)
    authored = from_legacy(sources / "configs/runs/learning_sb3.yaml")
    recipe = apply_overrides(
        tiny_recipe(tmp_path),
        [
            ("scenario", authored.scenario),
            ("algorithm.source", authored.algorithm.source),
        ],
    )
    trial = study._trial((recipe,), 0)
    plan = {
        "schema": study.PLAN_SCHEMA,
        "kind": "learning_batch",
        "max_concurrent": 1,
        "trials": [trial],
    }
    root, record = study._allocate(plan, recipe.output.root, "frozen-batch")
    record["trial_budget"] = 1
    study._atomic(root / "run.json", record)
    archive = export_experiment(root, tmp_path / "frozen.zip")
    old_bytes = (root / "trials" / trial["id"] / "snapshot-000.json").read_bytes()
    shutil.rmtree(sources)
    root.rename(tmp_path / "old-batch")
    imported = import_bundle(archive, tmp_path / "imported")
    result = run_learning_batch(resume=imported)
    assert result.completed == 1
    assert result.failed == 0
    assert (
        imported / "trials" / trial["id"] / "snapshot-000.json"
    ).read_bytes() == old_bytes
    assert result.entries[0]["runs"][0]["environment_steps"] == 32


def test_unsuggested_optuna_trial_uses_imported_seed_templates_without_authoring(
    tmp_path,
    monkeypatch,
):
    import os
    import shutil

    from smartsom.config.experiment import PRESET_ROOT, from_legacy
    from smartsom.experiments import learning_study as study
    from smartsom.experiments.packaging import export_experiment, import_bundle
    from smartsom.experiments.search import require_optuna

    require_optuna() if os.environ.get(
        "SMARTSOM_REQUIRE_SEARCH"
    ) == "1" else pytest.importorskip("optuna")
    sources = tmp_path / "authoring"
    shutil.copytree(PRESET_ROOT, sources)
    authored = from_legacy(sources / "configs/runs/learning_sb3.yaml")
    config = apply_overrides(
        tiny_recipe(tmp_path),
        [
            ("scenario", authored.scenario),
            ("algorithm.source", authored.algorithm.source),
            ("validation.enabled", True),
            ("validation.every_updates", 2),
            ("validation.replications", 1),
            ("search.method", "optuna"),
            ("search.trials", 1),
            ("search.objective", "makespan"),
            ("search.failure_policy", "all_complete"),
            (
                "search.space",
                {
                    "algorithm.learning_rate": {
                        "type": "float",
                        "low": 0.0001,
                        "high": 0.0004,
                    }
                },
            ),
        ],
    )
    with monkeypatch.context() as patch:
        patch.setattr(study, "_execute", lambda root, **_: root)
        root = search(config)
    assert read_json(root / "run.json")["trials"] == []
    original_plan = (root / "config/plan.json").read_bytes()
    archive = export_experiment(root, tmp_path / "search.zip")
    shutil.rmtree(sources)
    root.rename(tmp_path / "old-search")
    imported = import_bundle(archive, tmp_path / "imported-search")
    result = search(resume=imported)
    assert result.pending == 0
    assert len(result.entries) == 1
    assert result.entries[0]["status"] in {"completed", "ineligible"}
    assert result.entries[0]["runs"][0]["environment_steps"] == 32
    assert (imported / "config/plan.json").read_bytes() == original_plan
