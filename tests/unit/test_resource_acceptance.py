"""Acceptance eligibility is enforced without starting a learner or simulator."""

import copy
import importlib
import json
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest
import yaml
from test_resource_training import bundle

from smartsom.config import resolve_study, resolve_training_run
from smartsom.domain.machine_events import MachineOutagePlan
from smartsom.domain.processing_times import ProcessingTime, ProcessingTimePlan
from smartsom.experiments.evidence import write_json

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def acceptance(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    monkeypatch.setattr("smartsom.learning.checkpoint.require_backend", lambda p: {})
    return importlib.import_module("validation.resource_acceptance")


@pytest.fixture
def study(acceptance, tmp_path):
    run, _ = bundle(tmp_path)
    algorithm = tmp_path / "resource.json"
    write_json(algorithm, run.algorithm)
    spec = yaml.safe_load(
        (ROOT / "configs/studies/resource_evaluation.yaml").read_text()
    )
    spec["cases"][0]["scenario"] = str(ROOT / "configs/scenarios/learning_micro.yaml")
    spec["algorithms"][0]["config"] = str(ROOT / "configs/algorithms/spt.yaml")
    spec["algorithms"][1]["config"] = str(algorithm)
    spec["output_root"] = str(tmp_path / "output")
    path = tmp_path / "study.yaml"
    path.write_text(yaml.safe_dump(spec))
    return resolve_study(path)


def test_authorized_recipe_is_frozen_independently_of_output_and_weights(
    acceptance, study
):
    training = resolve_training_run(ROOT / "configs/runs/learning_marl.yaml")
    acceptance.require_training_recipe(training)
    acceptance.require_training_recipe(
        replace(
            training,
            run=training.run.model_copy(update={"output_root": "/different-output"}),
        )
    )
    acceptance.require_evaluation_recipe(study)
    rows = []
    for entry in study.entries:
        run = entry.resolved
        if entry.algorithm_id == "Resource-PPO":
            run = replace(
                run,
                algorithm=run.algorithm.model_copy(
                    update={
                        "algorithm": run.algorithm.algorithm.model_copy(
                            update={
                                "checkpoint": "/different-model",
                                "checkpoint_sha256": "a" * 64,
                            }
                        )
                    }
                ),
            )
        rows.append(replace(entry, resolved=run))
    acceptance.require_evaluation_recipe(replace(study, entries=tuple(reversed(rows))))


@pytest.mark.parametrize("change", ["seed", "budget", "scale", "factory", "arrivals"])
def test_training_recipe_rejects_drift(acceptance, change):
    resolved = resolve_training_run(ROOT / "configs/runs/learning_marl.yaml")
    if change == "seed":
        resolved = replace(resolved, run=resolved.run.model_copy(update={"seed": 102}))
    elif change == "budget":
        resolved = replace(
            resolved,
            run=resolved.run.model_copy(
                update={
                    "budget": resolved.run.budget.model_copy(
                        update={"environment_steps": 8192}
                    )
                }
            ),
        )
    elif change == "scale":
        spec = resolved.algorithm.algorithm
        resolved = replace(
            resolved,
            algorithm=resolved.algorithm.model_copy(
                update={
                    "algorithm": spec.model_copy(
                        update={
                            "parameters": spec.parameters.model_copy(
                                update={"learner_reward_scale": 0.001}
                            )
                        }
                    )
                }
            ),
        )
    elif change == "factory":
        resolved = replace(
            resolved,
            base=replace(
                resolved.base,
                factory=replace(
                    resolved.base.factory,
                    buffers=tuple(
                        replace(b, pre_capacity=1)
                        for b in resolved.base.factory.buffers
                    ),
                ),
            ),
        )
    else:
        resolved = replace(resolved, base=replace(resolved.base, arrivals=None))
    with pytest.raises(ValueError, match="frozen resource"):
        acceptance.require_training_recipe(resolved)


@pytest.mark.parametrize(
    "field,value",
    [
        ("arrivals", None),
        ("machine_events", MachineOutagePlan(())),
        ("processing_times", "changed"),
        ("buffers_enabled", False),
        ("holding_buffer_enabled", False),
    ],
)
def test_evaluation_recipe_covers_all_inputs_and_switches(
    acceptance, study, field, value
):
    entry = study.entries[0]
    if field == "processing_times":
        value = ProcessingTimePlan(
            tuple(
                ProcessingTime(
                    op.operation_id,
                    mode.processing_mode_id,
                    mode.nominal_ticks,
                    mode.nominal_ticks + 1,
                )
                for op in entry.resolved.workload.operations
                for mode in op.modes
            )
        )
    changed = replace(entry.resolved, **{field: value})
    assert (
        acceptance.run_identity(changed)["world_sha256"]
        != acceptance.run_identity(entry.resolved)["world_sha256"]
    )
    with pytest.raises(ValueError, match="evaluation differs"):
        acceptance.require_evaluation_recipe(
            replace(
                study, entries=(replace(entry, resolved=changed), *study.entries[1:])
            )
        )


def source(commit, **changes):
    return {"git": {"commit": commit, "dirty": False, "status": "", **changes}}


@pytest.mark.parametrize(
    "bad",
    [
        None,
        {},
        {"git": {}},
        source("b" * 40),
        source("a" * 40, dirty=True),
        source("a" * 40, status=" M file"),
        source("a" * 40, dirty=None),
    ],
)
def test_missing_dirty_or_mismatched_sources_are_rejected(acceptance, bad):
    with pytest.raises(ValueError, match="clean source"):
        acceptance.require_matching_source(
            bad, "a" * 40, "training/evaluation/final audit"
        )


def test_formal_source_accepts_clean_main_and_detached_ancestor_only(
    acceptance, tmp_path
):
    def git(*args):
        return subprocess.check_output(
            [
                "git",
                "-c",
                "user.name=Acceptance Fixture",
                "-c",
                "user.email=fixture@example.invalid",
                *args,
            ],
            cwd=tmp_path,
            text=True,
            stderr=subprocess.PIPE,
        ).strip()

    git("init", "-b", "main")
    git("commit", "--allow-empty", "-m", "fixture implementation")
    implementation = git("rev-parse", "HEAD")
    assert (
        acceptance.require_integrated_source(tmp_path, source(implementation))
        == implementation
    )
    git("commit", "--allow-empty", "-m", "fixture completion record")
    git("checkout", "--detach", implementation)
    assert (
        acceptance.require_integrated_source(tmp_path, source(implementation))
        == implementation
    )
    git("commit", "--allow-empty", "-m", "unintegrated fixture")
    other = git("rev-parse", "HEAD")
    with pytest.raises(ValueError, match="integrated into local main"):
        acceptance.require_integrated_source(tmp_path, source(other))
    with pytest.raises(ValueError, match="integrated into local main"):
        acceptance.require_integrated_source(tmp_path, source(implementation))


def report():
    checks = [
        "action_replay",
        "schedule_replay",
        "action_observations",
        "schedule_observations",
    ]
    return {
        "completed": 10,
        "failed": 0,
        "pending": 0,
        "evaluation": [
            {
                "algorithm": provider,
                "replication": replication,
                "world_sha256": str(replication),
                "status": "passed",
                "checks": checks.copy(),
                **(
                    {"joint_replay": {"status": "passed"}}
                    if provider == "rllib.resource_ppo"
                    else {}
                ),
            }
            for replication in range(5)
            for provider in ("builtin.spt", "rllib.resource_ppo")
        ],
    }


@pytest.mark.parametrize(
    "change",
    ["missing", "duplicate", "provider", "world", "audit", "checks", "joint", "counts"],
)
def test_completed_batch_cannot_hide_missing_pairs_or_failed_audits(acceptance, change):
    good = report()
    acceptance.require_evaluation_result(good)
    bad = copy.deepcopy(good)
    if change == "missing":
        bad["evaluation"].pop()
    elif change == "duplicate":
        bad["evaluation"][-1] = copy.deepcopy(bad["evaluation"][0])
    elif change == "provider":
        bad["evaluation"][1]["algorithm"] = "builtin.spt"
    elif change == "world":
        bad["evaluation"][0]["world_sha256"] = "different"
    elif change == "audit":
        bad["evaluation"][0]["status"] = "failed"
    elif change == "checks":
        bad["evaluation"][0]["checks"].remove("schedule_replay")
    elif change == "joint":
        bad["evaluation"][1]["joint_replay"] = {"status": "failed"}
    else:
        bad["pending"] = 1
    with pytest.raises(ValueError):
        acceptance.require_evaluation_result(bad)


def test_default_formal_rejection_is_retained_and_development_is_explicit(
    acceptance, tmp_path, monkeypatch
):
    module = importlib.import_module("validate_resource_learning")
    monkeypatch.setattr(
        module,
        "source_identity",
        lambda: {**source("a" * 40, dirty=True), "platform": "test"},
    )
    for development in (False, True):
        output = tmp_path / str(development)
        monkeypatch.setattr(
            "sys.argv",
            [
                "validate_resource_learning.py",
                "--training-dir",
                str(tmp_path / "missing"),
                "--output-dir",
                str(output),
                *(["--development"] if development else []),
            ],
        )
        assert module.main() == 1
        saved = json.loads((output / "report.json").read_text())
        assert saved["accepted_source_sha"] is None
        assert saved["evidence_kind"] == ("development" if development else "formal")
        assert ("clean source" in saved["error"]) is not development
