from pathlib import Path
from types import SimpleNamespace

import pytest

from smartsom.config.codec import ConfigurationError, canonical_json, primitive
from smartsom.config.experiment import apply_overrides, load_preset
from smartsom.experiments import learning_study as study
from smartsom.experiments.evidence import write_json
from smartsom.experiments.search import candidates, trial_configs, validation_score


@pytest.fixture
def recipe(tmp_path):
    config = load_preset("sb3_micro")
    return apply_overrides(
        config,
        [
            ("output.root", str(tmp_path / "runs")),
            ("logging.tensorboard", False),
            ("training.total_steps", 128),
            ("training.steps_per_update", 32),
            ("algorithm.batch_size", 16),
            ("algorithm.n_epochs", 1),
            ("validation.every_updates", 4),
            ("validation.replications", 1),
            (
                "search.space",
                {
                    "algorithm.learning_rate": {
                        "type": "categorical",
                        "choices": [0.0001, 0.0003],
                    }
                },
            ),
            ("search.objective", "makespan"),
            ("search.failure_policy", "all_complete"),
        ],
    )


def validation(value=12, *, complete=True, update=4):
    return {
        "ppo_updates": update,
        "episodes": 1,
        "completed": int(complete),
        "successful_inputs": ["fixed-world"] if complete else [],
        "results": [
            {
                "input_id": "fixed-world",
                "reason": "completed" if complete else "deadlock",
                "makespan": value if complete else None,
            }
        ],
        "metrics": {"makespan": value if complete else None},
        "weights": {"policy": "final"},
    }


def test_grid_and_random_proposals_are_frozen_with_independent_rng(recipe):
    import random

    assert candidates(recipe) == (
        {"algorithm.learning_rate": 0.0001},
        {"algorithm.learning_rate": 0.0003},
    )
    random.seed(123)
    state = random.getstate()
    random_recipe = apply_overrides(
        recipe, [("search.method", "random"), ("search.trials", 8)]
    )
    assert candidates(random_recipe) == candidates(random_recipe)
    assert random.getstate() == state
    seeds = apply_overrides(recipe, [("search.seeds", [7, 8])])
    assert [c.seed for c in trial_configs(seeds, candidates(seeds)[0])] == [7, 8]


@pytest.mark.parametrize(
    "overrides,match",
    [
        ([("search.failure_policy", None)], "explicit"),
        (
            [
                (
                    "search.space",
                    {
                        "algorithm.source": {
                            "type": "categorical",
                            "choices": ["different.yaml"],
                        }
                    },
                )
            ],
            "algorithm.source is fixed",
        ),
        (
            [
                (
                    "search.space",
                    {"validation.seed": {"type": "categorical", "choices": [404]}},
                )
            ],
            "algorithm or training",
        ),
        ([("validation.enabled", False)], "validation.enabled"),
        ([("checkpointing.save_last", False)], "checkpointing.save_last"),
        ([("validation.every_updates", 3)], "final update"),
        ([("search.trials", 1)], "Cartesian"),
        ([("search.pruning", True)], "optuna"),
    ],
)
def test_invalid_search_fails_before_output_allocation(recipe, overrides, match):
    changed = apply_overrides(recipe, overrides)
    with pytest.raises(ConfigurationError, match=match):
        study.search(changed)
    assert not Path(changed.output.root).exists()


def test_completion_first_does_not_create_search_scores_for_partial_cases():
    assert validation_score(validation(), "makespan") == 12
    assert validation_score(validation(complete=False), "makespan") is None
    invalid = validation()
    invalid["metrics"]["makespan"] = 1
    with pytest.raises(ValueError, match="aggregate"):
        validation_score(invalid, "makespan")


def test_adaptive_templates_preserve_seed_world_and_reject_missing_coverage(recipe):
    from smartsom.config.codec import digest

    config = apply_overrides(recipe, [("search.seeds", [101, 102])])
    frozen = [study._freeze(item) for item in trial_configs(config, {})]
    plan = {"base": primitive(config), "templates": frozen}
    templates = study._templates(plan)
    trial = study._trial(
        trial_configs(config, {"algorithm.learning_rate": 0.0002}),
        0,
        templates=templates,
    )
    assert [row["config"]["seed"] for row in trial["configs"]] == [101, 102]
    for row, template in zip(trial["configs"], frozen, strict=True):
        for field in ("scenario_json", "workload_json", "workload_source_json"):
            assert (
                row["resolved_training"][field] == template["resolved_training"][field]
            )
        assert row["scientific_sha256"] != template["scientific_sha256"]
        assert row["config"]["algorithm"]["learning_rate"] == 0.0002
    with pytest.raises(ConfigurationError, match="cover"):
        study._templates({**plan, "templates": frozen[:1]})
    frozen[0]["resolved_training"]["algorithm_seed"] = 999
    frozen[0]["config_sha256"] = digest(frozen[0]["config"])
    with pytest.raises(ConfigurationError, match="scientific identity"):
        study._templates(plan)


def allocate(recipe, monkeypatch, *, search=False):
    monkeypatch.setattr(study, "execution_identity", lambda: {"fixed": True})
    trial = study._trial((recipe,), 0)
    plan = {
        "schema": study.PLAN_SCHEMA,
        "kind": "search" if search else "learning_batch",
        "max_concurrent": 1,
        "trials": [trial],
    }
    if search:
        plan["base"] = primitive(recipe)
    root, record = study._allocate(plan, recipe.output.root, "test")
    record["trial_budget"] = 1
    study._atomic(root / "run.json", record)
    directory = root / "trials" / trial["id"]
    attempt = directory / "attempt-000"
    attempt.mkdir()
    write_json(attempt / "attempt.json", {})
    return root, record, trial, directory, attempt


def as_config(value):
    return (
        type(load_preset("sb3_micro")).model_validate_json(value.config_json)
        if hasattr(value, "config_json")
        else value
    )


def fake_train(config, *, on_progress=None):
    config = as_config(config)
    child = Path(config.output.root) / "child"
    training = child / "evidence/training"
    checkpoint = training / "checkpoint"
    checkpoint.mkdir(parents=True)
    (child / "config").mkdir()
    write_json(child / "config/experiment.json", primitive(config))
    write_json(
        child / "run.json",
        {
            "schema": "smartsom.experiment/v2",
            "id": "child",
            "name": "child",
            "kind": "training",
            "status": "completed",
            "paths": {"training": "evidence/training"},
        },
    )
    write_json(
        training / "summary.json", {"environment_steps": 128, "learner_updates": 4}
    )
    write_json(
        checkpoint / "checkpoint.json",
        {
            "provider": "sb3.maskable_ppo",
            "final_weights_sha256": "final",
            "environment_steps": 128,
        },
    )
    write_json(training / "validation-000004.json", validation())
    if on_progress:
        on_progress({"stage": "validation", "ppo_updates": 4, "report": validation()})
    return SimpleNamespace(
        run_dir=child,
        training_dir=training,
        last_checkpoint=checkpoint,
        environment_steps=128,
        learner_updates=4,
        status="completed",
    )


def test_training_worker_reuses_verified_replicates_and_rejects_drift(
    recipe, tmp_path, monkeypatch
):
    from smartsom import api

    _, _, trial, directory, attempt = allocate(recipe, monkeypatch)
    calls = []
    monkeypatch.setattr(
        api,
        "train_prepared",
        lambda config, **kw: (
            calls.append(as_config(config).seed) or fake_train(config, **kw)
        ),
    )
    result = study._run_trial(directory, attempt, trial, {"fixed": True})
    assert result["status"] == "completed"
    second = study._run_trial(directory, attempt, trial, {"fixed": True})
    assert second == result
    assert calls == [101]
    child = directory / result["runs"][0]["run_dir"]
    (child / "evidence/training/summary.json").write_text("{}")
    with pytest.raises(ConfigurationError, match="evidence changed"):
        study._run_trial(directory, attempt, trial, {"fixed": True})


def test_source_and_frozen_plan_drift_refuse_resume(recipe, monkeypatch):
    root, _, _, directory, _ = allocate(recipe, monkeypatch)
    assert study._load(root)[1]["schema"] == study.PLAN_SCHEMA
    monkeypatch.setattr(study, "execution_identity", lambda: {"fixed": False})
    with pytest.raises(ConfigurationError, match="source or dependency"):
        study._load(root)
    monkeypatch.setattr(study, "execution_identity", lambda: {"fixed": True})
    (directory / "plan.json").write_text("{}")
    with pytest.raises(ConfigurationError, match="trial plan digest"):
        study._load(root)


def test_search_worker_requires_training_audit_and_matching_final_weights(
    recipe, monkeypatch
):
    from smartsom import api
    from smartsom.experiments import training_audit

    _, _, trial, directory, attempt = allocate(recipe, monkeypatch, search=True)
    monkeypatch.setattr(api, "train_prepared", fake_train)
    monkeypatch.setattr(
        training_audit,
        "audit_training",
        lambda _: {"status": "passed", "budget_completed": True},
    )
    result = study._run_trial(directory, attempt, trial, {"fixed": True}, recipe.search)
    assert result["value"] == 12
    assert result["runs"][0]["training_audit"]["status"] == "passed"
    assert result["runs"][0]["checkpoint_sha256"]
    assert result["runs"][0]["objective"]["successful_inputs"] == ["fixed-world"]


def test_audit_failure_does_not_become_a_scored_trial(recipe, monkeypatch):
    from smartsom import api
    from smartsom.experiments import training_audit

    _, _, trial, directory, attempt = allocate(recipe, monkeypatch, search=True)
    monkeypatch.setattr(api, "train_prepared", fake_train)
    monkeypatch.setattr(
        training_audit, "audit_training", lambda _: {"status": "failed"}
    )
    with pytest.raises(ValueError, match="audit did not pass"):
        study._run_trial(directory, attempt, trial, {"fixed": True}, recipe.search)


def test_frozen_inputs_checked_before_worker_launches_training(recipe, monkeypatch):
    from smartsom import api

    _, _, trial, directory, attempt = allocate(recipe, monkeypatch)
    trial["configs"][0]["scientific_sha256"] = "different"
    monkeypatch.setattr(
        api,
        "train_prepared",
        lambda *_args, **_kwargs: pytest.fail("training was launched"),
    )
    with pytest.raises(ConfigurationError, match="inputs differ"):
        study._run_trial(directory, attempt, trial, {"fixed": True})


def test_completed_child_after_coordinator_loss_is_read_without_training_again(
    recipe, monkeypatch
):
    from smartsom import api

    _, _, trial, directory, attempt = allocate(recipe, monkeypatch)
    child_config = type(recipe).model_validate_json(canonical_json(primitive(recipe)))
    child_config.output.root = str(attempt / "replication-000")
    fake_train(child_config)
    monkeypatch.setattr(
        api,
        "train_prepared",
        lambda *_args, **_kwargs: pytest.fail("duplicated completed training"),
    )
    result = study._run_trial(directory, attempt, trial, {"fixed": True})
    assert result["status"] == "completed"


def test_optional_optuna_failure_is_explicit_and_precedes_allocation(
    recipe, monkeypatch
):
    from smartsom.experiments import search as module

    def missing(_):
        raise ImportError("not installed")

    monkeypatch.setattr(module.importlib, "import_module", missing)
    config = apply_overrides(
        recipe, [("search.method", "optuna"), ("search.trials", 2)]
    )
    with pytest.raises(RuntimeError, match="extra search"):
        study.search(config)
    assert not Path(recipe.output.root).exists()


def test_no_checkpoint_interruption_keeps_completed_seed_and_starts_new_attempt(
    recipe, monkeypatch
):
    import queue

    from smartsom import api
    from smartsom.experiments import training_audit

    config = apply_overrides(recipe, [("search.seeds", [101, 102])])
    monkeypatch.setattr(study, "execution_identity", lambda: {"fixed": True})
    trial = study._trial(trial_configs(config, candidates(config)[0]), 0)
    plan = {
        "schema": study.PLAN_SCHEMA,
        "kind": "search",
        "max_concurrent": 1,
        "base": primitive(config),
        "trials": [trial],
    }
    root, record = study._allocate(plan, config.output.root, "resume-test")
    record.update(trial_budget=1, direction="min")
    study._atomic(root / "run.json", record)
    directory = root / "trials" / trial["id"]
    old = directory / "attempt-000"
    old.mkdir()
    write_json(old / "attempt.json", {})
    calls = []

    def interrupted(config, **kwargs):
        config = as_config(config)
        calls.append(config.seed)
        if config.seed == 102:
            path = Path(config.output.root) / "partial"
            path.mkdir(parents=True)
            write_json(
                path / "run.json",
                {
                    "schema": "smartsom.experiment/v2",
                    "status": "interrupted",
                    "paths": {},
                },
            )
            raise KeyboardInterrupt
        return fake_train(config, **kwargs)

    monkeypatch.setattr(api, "train_prepared", interrupted)
    monkeypatch.setattr(
        training_audit, "audit_training", lambda _: {"status": "passed"}
    )
    with pytest.raises(KeyboardInterrupt):
        study._run_trial(directory, old, trial, {"fixed": True}, config.search)
    write_json(
        old / "result.json",
        {
            "status": "interrupted",
            "runs": list(study._saved_runs(old).values()),
            "value": None,
        },
    )
    assert not study._can_resume(old)

    def resumed(config, **kwargs):
        config = as_config(config)
        calls.append(config.seed)
        return fake_train(config, **kwargs)

    class LocalQueue(queue.Queue):
        def close(self):
            pass

    class LocalProcess:
        def __init__(self, *, target, args):
            self.args = args
            self.exitcode = 0

        def start(self):
            directory, attempt, trial, identity, _search, _optuna, messages = self.args
            result = study._run_trial(
                Path(directory), Path(attempt), trial, identity, config.search
            )
            study._atomic(Path(attempt) / "result.json", result)

        def is_alive(self):
            return False

        def join(self):
            pass

    monkeypatch.setattr(api, "train_prepared", resumed)
    monkeypatch.setattr(
        study.multiprocessing,
        "get_context",
        lambda _: SimpleNamespace(Process=LocalProcess, Queue=LocalQueue),
    )
    result = study.search(resume=root)
    assert result.completed == 1
    assert calls == [101, 102, 102]
    assert [row["run_dir"].split("/")[0] for row in result.entries[0]["runs"]] == [
        "attempt-000",
        "attempt-001",
    ]
    assert (old / "result.json").exists()
    assert (directory / "attempt-001/attempt.json").exists()


def test_completed_trial_cannot_omit_a_replication(recipe, monkeypatch):
    _, _, _, directory, attempt = allocate(recipe, monkeypatch)
    write_json(
        attempt / "result.json", {"status": "completed", "runs": [], "value": None}
    )
    with pytest.raises(ConfigurationError, match="replication coverage"):
        study._state(directory)
