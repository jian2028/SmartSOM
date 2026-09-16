"""Scientific selection and recovery preflight without constructing learners."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from smartsom.config import resolve_training_run
from smartsom.config.codec import digest
from smartsom.config.experiment import ExperimentConfig
from smartsom.config.study import study_roots
from smartsom.experiments.production_training import (
    identity,
    request_interrupt,
    retain_checkpoints,
    train_prepared,
    verify_checkpoint,
)
from smartsom.experiments.references import protect_model_reference
from smartsom.experiments.training import apply_training_controls
from smartsom.experiments.training_controls import TrainingControls, ValidationControls
from smartsom.experiments.training_validation import select_best, validation_inputs
from smartsom.learning.checkpoint import file_hash

ROOT = Path(__file__).resolve().parents[2]


def report(successful, makespan=10):
    return {
        "episodes": 5,
        "completed": len(successful),
        "successful_inputs": sorted(successful),
        "metrics": {"makespan": makespan, "return": -makespan, "passing_rate": 1.0},
    }


@pytest.mark.parametrize(
    "kwargs",
    [
        {"checkpoint_every_updates": 0},
        {"keep_last": True},
        {"resume_from": "a", "initialize_from": "b"},
        {"save_best": 1},
        {"stop_after_updates": -1},
        {"validation": {}},
        {"device": "mps"},
        {"num_envs": 0},
        {"sampling_processes": 2, "num_envs": 1},
        {"numerical_threads": 0},
    ],
)
def test_controls_reject_ambiguous_or_unsupported_choices(kwargs):
    with pytest.raises((TypeError, ValueError)):
        TrainingControls(**kwargs)


def test_selection_prioritizes_completion_and_never_compares_changed_survivors():
    c = ValidationControls()
    assert select_best(report([]), None, c) == (False, "no_completed_episodes")
    assert select_best(report(["a"], 100), None, c)[0]
    assert select_best(report(["a", "b"], 100), report(["a"], 1), c)[0]
    assert not select_best(report(["a"], 1), report(["a", "b"], 100), c)[0]
    assert select_best(report(["b"], 1), report(["a"], 100), c) == (
        False,
        "different_successful_inputs",
    )
    assert select_best(report(["a"], 9), report(["a"], 10), c)[0]
    assert not select_best(report(["a"], 10), report(["a"], 10), c)[0]
    assert not select_best(
        report(["a"], 9), report(["a"], 10), replace(c, min_delta=1)
    )[0]


def test_strict_and_custom_selection_require_explicit_failure_policy():
    assert not select_best(
        report(["a"]), None, ValidationControls(best_mode="all_complete")
    )[0]
    assert select_best(
        report(list("abcde")), None, ValidationControls(best_mode="all_complete")
    )[0]
    with pytest.raises(ValueError, match="explicit failure policy"):
        ValidationControls(best_mode="custom")
    c = ValidationControls(
        best_mode="custom",
        metric="return",
        direction="max",
        failure_policy="successful_only",
    )
    assert select_best(report(["a"], 1), report(["a"], 2), c)[0]
    assert not select_best(report(["b"], 1), report(["a"], 2), c)[0]


def test_fixed_validation_worlds_use_study_seeds_without_training_episode_rehash(
    monkeypatch,
):
    monkeypatch.setattr("smartsom.learning.checkpoint.require_backend", lambda _: {})
    resolved = resolve_training_run(ROOT / "configs/runs/learning_sb3.yaml")
    c = ValidationControls()
    values = validation_inputs(resolved, c)
    assert len(values) == 5
    assert len({item["input_id"] for item in values}) == 5
    for rep, item in enumerate(values):
        world = study_roots(303, "learning", rep, "")[0]
        assert item["world_seed"] == world
        assert [(d.demand_id, d.steps) for d in item["episode"].demands] == [
            (d.demand_id, d.steps) for d in resolved.resolved.scenario.demands
        ]
        assert item["input_id"] == digest([world, rep, item["episode"]])
    assert values == validation_inputs(resolved, c)


def seal(directory, **values):
    members = {
        str(p.relative_to(directory)): file_hash(p)
        for p in directory.rglob("*")
        if p.is_file() and p.name != "update.json"
    }
    (directory / "update.json").write_text(json.dumps({**values, "files": members}))


def test_checkpoint_members_are_verified_and_report_references_do_not_invalidate(
    tmp_path,
):
    checkpoint = tmp_path / "checkpoints/update-000001"
    checkpoint.mkdir(parents=True)
    (checkpoint / "state.bin").write_bytes(b"trusted fixture, not a model")
    seal(checkpoint, status="complete")
    assert protect_model_reference(checkpoint, "formal-report.json")
    assert verify_checkpoint(checkpoint)["status"] == "complete"
    (checkpoint / "state.bin").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="digests"):
        verify_checkpoint(checkpoint)


def test_retention_keeps_latest_best_and_registered_report_references(tmp_path):
    paths = []
    for update in range(6):
        directory = tmp_path / f"checkpoints/update-{update:06d}"
        directory.mkdir(parents=True)
        seal(directory)
        paths.append(directory)
    protect_model_reference(paths[1], "evaluation.json")
    state = {
        "best": str(paths[0].relative_to(tmp_path)),
        "last": str(paths[-1].relative_to(tmp_path)),
    }
    retain_checkpoints(tmp_path, state, 2)
    assert [p.exists() for p in paths] == [True, True, False, False, True, True]


def test_first_interrupt_requests_boundary_and_second_interrupt_is_immediate():
    state = {"interrupted": False}
    request_interrupt(state)
    assert state["interrupted"]
    with pytest.raises(KeyboardInterrupt):
        request_interrupt(state)


@pytest.mark.parametrize(
    "field",
    [
        "source_commit",
        "source",
        "recipe",
        "dependencies",
        "python",
        "platform",
        "runtime",
        "validation_controls",
        "checkpoint_controls",
    ],
)
def test_resume_rejects_changed_identity_before_loading_framework_state(
    tmp_path, field
):
    prepared = resolve_training_run(ROOT / "configs/runs/learning_sb3.yaml")
    prepared = apply_training_controls(prepared, TrainingControls(validation=None))
    config = ExperimentConfig.model_validate_json(prepared.config_json)
    changed = identity(prepared, config)
    changed[field] = "changed"
    seal(tmp_path, identity=changed, steps=32, updates=1)
    with pytest.raises(ValueError, match="identical source, frozen inputs and runtime"):
        train_prepared(prepared, resume_from=tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["update.json"]


def test_parallel_quota_and_missing_cuda_fail_before_creating_attempt(
    monkeypatch, tmp_path
):
    import sys
    from types import SimpleNamespace

    from smartsom.config.experiment import load_config, prepare

    config = load_config(ROOT / "configs/runs/learning_sb3.yaml")
    config.output.root = str(tmp_path / "runs")
    prepared = prepare(config)
    with pytest.raises(ValueError, match="divide evenly"):
        apply_training_controls(prepared, TrainingControls(num_envs=3))
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)),
    )
    with pytest.raises(ValueError, match="CUDA was requested but is unavailable"):
        train_prepared(
            apply_training_controls(prepared, TrainingControls(device="cuda"))
        )
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize("tamper", ["duplicate", "quota"])
def test_stream_audit_rejects_duplicate_identity_and_unequal_quota(tmp_path, tamper):
    pytest.importorskip("gymnasium")

    from smartsom.config.codec import primitive
    from smartsom.config.experiment import load_config, prepare
    from smartsom.experiments.production_training import ProductionEvidence
    from smartsom.experiments.training_audit import audit_training
    from smartsom.learning.production_sampling import GridStream, ProductionSamplingSpec

    config = load_config(ROOT / "configs/runs/learning_sb3.yaml")
    config.runtime.num_envs = 2
    prepared = prepare(config)
    recipe = prepared.resolved
    rows = []
    for stream in range(2):
        env = GridStream(ProductionSamplingSpec(recipe, config.seed), stream, 2)
        try:
            env.reset()
            for _ in range(1 if tamper == "quota" and stream == 1 else 2):
                env.step(
                    next(
                        i for i, valid in enumerate(env.adapter.action_masks()) if valid
                    )
                )
            rows.append(ProductionEvidence.row(env.snapshot()))
        finally:
            env.env.close()
    if tamper == "duplicate":
        rows.append(rows[0])
    (tmp_path / "config").mkdir()
    (tmp_path / "config/grid_recipe.json").write_text(json.dumps(primitive(recipe)))
    (tmp_path / "run.json").write_text(json.dumps({"status": "interrupted"}))
    checkpoint = tmp_path / "checkpoints/update-000001"
    checkpoint.mkdir(parents=True)
    (checkpoint / "recipe.json").write_text(
        json.dumps({"config": primitive(config), "recipe": primitive(recipe)})
    )
    (checkpoint / "episodes.jsonl").write_text("")
    (checkpoint / "active_episodes.json").write_text(json.dumps(rows))
    seal(checkpoint, steps=sum(len(r["steps"]) for r in rows), updates=1)
    with pytest.raises(ValueError, match="duplicate|sampling quota"):
        audit_training(checkpoint)
