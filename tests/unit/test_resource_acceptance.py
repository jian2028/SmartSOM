"""Acceptance eligibility is enforced without starting a learner or simulator."""

import copy
import importlib
import json
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from smartsom.config import resolve_study, resolve_training_run
from smartsom.config.codec import canonical_json
from smartsom.config.experiment import ExperimentConfig
from smartsom.domain.production import Outage
from smartsom.experiments.evidence import write_json

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def acceptance(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    monkeypatch.setattr("smartsom.learning.checkpoint.require_backend", lambda p: {})
    return importlib.import_module("validation.resource_acceptance")


@pytest.fixture
def study(acceptance, tmp_path):
    # Planning fixture has no model weights; it never stands in for model inference.
    spec = yaml.safe_load(
        (ROOT / "configs/studies/resource_evaluation.yaml").read_text()
    )
    spec["cases"][0]["scenario"] = str(ROOT / "configs/scenarios/learning_micro.yaml")
    spec["algorithms"][0]["config"] = str(ROOT / "configs/algorithms/spt.yaml")
    spec["algorithms"][1]["config"] = str(
        ROOT / "configs/algorithms/rllib_resource_ppo.yaml"
    )
    spec["output_root"] = str(tmp_path / "output")
    path = tmp_path / "study.yaml"
    path.write_text(yaml.safe_dump(spec))
    return resolve_study(path)


def test_authorized_recipe_is_frozen_independently_of_output_and_weights(
    acceptance, study
):
    training = resolve_training_run(ROOT / "configs/runs/learning_marl.yaml")
    acceptance.require_training_recipe(training)
    config = ExperimentConfig.model_validate_json(training.config_json)
    config.output.root = "/different-output"
    acceptance.require_training_recipe(
        replace(training, config_json=canonical_json(config))
    )
    acceptance.require_evaluation_recipe(study)
    rows = []
    for entry in study.entries:
        prepared = entry.resolved
        if entry.algorithm_id == "Resource-PPO":
            recipe = prepared.resolved
            algorithm = recipe.algorithm.model_copy(
                update={"checkpoint": "/different-model"}
            )
            prepared = replace(
                prepared,
                resolved=replace(recipe, algorithm_json=canonical_json(algorithm)),
            )
        rows.append(replace(entry, resolved=prepared))
    acceptance.require_evaluation_recipe(replace(study, entries=tuple(reversed(rows))))


@pytest.mark.parametrize("change", ["seed", "budget", "scale", "factory", "arrivals"])
def test_training_recipe_rejects_drift(acceptance, change):
    prepared = resolve_training_run(ROOT / "configs/runs/learning_marl.yaml")
    recipe = prepared.resolved
    config = ExperimentConfig.model_validate_json(prepared.config_json)
    if change == "seed":
        config.seed = 102
        prepared = replace(prepared, config_json=canonical_json(config))
    elif change == "budget":
        config.training.total_steps = 8192
        prepared = replace(
            prepared,
            config_json=canonical_json(config),
            resolved=replace(recipe, training_json=canonical_json(config.training)),
        )
    elif change == "scale":
        prepared = replace(
            prepared,
            resolved=replace(
                recipe,
                algorithm_json=canonical_json(
                    recipe.algorithm.model_copy(update={"learner_reward_scale": 0.001})
                ),
            ),
        )
    else:
        scenario = recipe.scenario
        if change == "factory":
            factory = scenario.factory
            scenario = replace(
                scenario, factory=replace(factory, name="changed factory")
            )
        else:
            scenario = replace(
                scenario,
                demands=tuple(
                    replace(d, release_at=0, reveal_at=0) for d in scenario.demands
                ),
            )
        prepared = replace(
            prepared, resolved=replace(recipe, scenario_json=canonical_json(scenario))
        )
    with pytest.raises(ValueError, match="frozen resource"):
        acceptance.require_training_recipe(prepared)


@pytest.mark.parametrize(
    "field", ["arrivals", "outages", "processing", "buffers", "quality", "seed"]
)
def test_evaluation_recipe_covers_all_inputs_and_switches(acceptance, study, field):
    entry = study.entries[0]
    recipe = entry.resolved.resolved
    scenario = recipe.scenario
    if field == "arrivals":
        scenario = replace(
            scenario,
            demands=tuple(
                replace(d, release_at=0, reveal_at=0) for d in scenario.demands
            ),
        )
    elif field == "outages":
        scenario = replace(
            scenario, outages=(Outage(scenario.factory.machines[0].machine_id, 1, 2),)
        )
    elif field == "processing":
        scenario = replace(
            scenario,
            demands=(
                replace(
                    scenario.demands[0],
                    steps=tuple(
                        replace(step, nominal_ticks=step.nominal_ticks + 1)
                        for step in scenario.demands[0].steps
                    ),
                ),
                *scenario.demands[1:],
            ),
        )
    elif field == "buffers":
        scenario = replace(scenario, factory=replace(scenario.factory, buffers=()))
    elif field == "quality":
        scenario = replace(scenario, quality_probability_visibility="hidden")
    else:
        scenario = replace(scenario, seed=scenario.seed + 1)
    changed = replace(
        entry.resolved, resolved=replace(recipe, scenario_json=canonical_json(scenario))
    )
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
        "frozen_inputs",
        "state_hashes",
        "semantic_commands",
        "events",
        "qualified_demand_coverage",
        "observation_replay",
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
                    {"learning_replay": {"status": "passed"}}
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
        bad["evaluation"][0]["checks"].remove("semantic_commands")
    elif change == "joint":
        bad["evaluation"][1]["learning_replay"] = {"status": "failed"}
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


@pytest.mark.parametrize(
    "change",
    [
        "none",
        "budget",
        "provider",
        "role",
        "actor",
        "critic",
        "metadata_count",
        "metrics_count",
        "metric_gap",
        "nonfinite",
        "saturation",
    ],
)
def test_grid_training_requires_four_updated_actors_and_critics(
    acceptance, tmp_path, change
):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    audit = {
        "status": "passed",
        "budget_completed": True,
        "environment_steps": 4096,
        "ppo_updates": 16,
        "provider": "rllib.resource_ppo",
        "checkpoint": str(checkpoint),
    }
    metadata = {
        "modules": list(acceptance.ROLES),
        "environment_steps": 4096,
        "learner_updates": 16,
        "component_changes": {
            role: {"actor": True, "critic": True} for role in acceptance.ROLES
        },
    }
    metrics = [
        {
            "sampled_steps": step,
            "metrics": {
                f"{role}/{metric}": 0.001
                for role in acceptance.ROLES
                for metric in ("vf_loss", "vf_loss_unclipped")
            },
        }
        for step in range(256, 4097, 256)
    ]
    if change == "budget":
        audit["budget_completed"] = False
    elif change == "provider":
        audit["provider"] = "rllib.ppo"
    elif change == "role":
        metadata["modules"].pop()
    elif change in ("actor", "critic"):
        metadata["component_changes"]["quality_policy"][change] = False
    elif change == "metadata_count":
        metadata["environment_steps"] = 4095
    elif change == "metrics_count":
        metrics.pop()
    elif change == "metric_gap":
        metrics[0]["sampled_steps"] = 512
    elif change == "nonfinite":
        metrics[0]["metrics"]["buffer_policy/vf_loss"] = float("nan")
    elif change == "saturation":
        metrics[0]["metrics"]["buffer_policy/vf_loss_unclipped"] = 20.0
    write_json(checkpoint / "checkpoint.json", metadata)
    (checkpoint / "learner_metrics.jsonl").write_text(
        "\n".join(json.dumps(row) for row in metrics)
    )
    if change == "none":
        acceptance.require_training_result(audit, tmp_path)
    else:
        with pytest.raises(ValueError):
            acceptance.require_training_result(audit, tmp_path)
