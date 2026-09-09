"""Scientific selection and recovery preflight without constructing learners."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from smartsom.config import resolve_training_run
from smartsom.config.codec import digest
from smartsom.config.study import study_roots
from smartsom.experiments.training_controls import TrainingControls, ValidationControls
from smartsom.experiments.training_lifecycle import (
    SCHEMA,
    TrainingLifecycle,
    _members,
    inspect_resume_checkpoint,
    protect_checkpoint,
)
from smartsom.experiments.training_validation import select_best, validation_inputs

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
        {"device": "cuda"},
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
        assert item["episode"].workload is resolved.base.workload
        assert item["input_id"] == digest([world, rep, item["episode"]])
    assert values == validation_inputs(resolved, c)


def test_checkpoint_members_are_verified_and_report_references_do_not_invalidate(
    tmp_path,
):
    (tmp_path / "state.bin").write_bytes(b"trusted fixture, not a model")
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {"schema": SCHEMA, "status": "complete", "files": _members(tmp_path)}
        )
    )
    protect_checkpoint(tmp_path, "formal-report.json")
    assert inspect_resume_checkpoint(tmp_path)["status"] == "complete"
    (tmp_path / "state.bin").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="digests"):
        inspect_resume_checkpoint(tmp_path)


def test_retention_keeps_latest_best_and_registered_report_references(tmp_path):
    lifecycle = object.__new__(TrainingLifecycle)
    lifecycle.root = tmp_path
    lifecycle.controls = TrainingControls(keep_last=2)
    paths = []
    for update in range(6):
        directory = tmp_path / f"update-{update:06d}"
        directory.mkdir()
        (directory / "manifest.json").write_text("{}")
        paths.append(directory)
    lifecycle.best_checkpoint, lifecycle.last_checkpoint = paths[0], paths[-1]
    (paths[1] / "references").mkdir()
    lifecycle._retain()
    assert [p.exists() for p in paths] == [True, True, False, False, True, True]


def test_first_interrupt_requests_boundary_and_second_interrupt_is_immediate():
    lifecycle = object.__new__(TrainingLifecycle)
    lifecycle.interrupts, lifecycle.stop_requested = 0, False
    lifecycle._interrupt()
    assert lifecycle.stop_requested
    with pytest.raises(KeyboardInterrupt):
        lifecycle._interrupt()


@pytest.mark.parametrize(
    "field",
    [
        "source_commit",
        "source_sha256",
        "recipe_sha256",
        "dependencies",
        "device",
        "validation",
    ],
)
def test_resume_rejects_changed_identity_before_loading_framework_state(
    tmp_path, monkeypatch, field
):
    monkeypatch.setattr("smartsom.learning.checkpoint.require_backend", lambda _: {})
    resolved = resolve_training_run(ROOT / "configs/runs/learning_sb3.yaml")
    controls = TrainingControls(validation=None)
    identity = dict(TrainingLifecycle(resolved, controls).identity)
    identity[field] = "changed"
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "schema": SCHEMA,
                "status": "complete",
                "files": {},
                "identity": identity,
                "environment_steps": 32,
                "ppo_updates": 1,
            }
        )
    )
    with pytest.raises(
        ValueError, match="identical source, recipe, dependencies and device"
    ):
        TrainingLifecycle(resolved, replace(controls, resume_from=tmp_path))
