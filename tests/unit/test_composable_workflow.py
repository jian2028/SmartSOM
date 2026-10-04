"""User entry points, classified authoring, evaluation evidence and process sampling."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from smartsom import api
from smartsom.config.codec import canonical_json
from smartsom.config.composable_authoring import migration_preview, scaffold
from smartsom.config.experiment import prepare
from smartsom.experiments.composable import TrainingSession, allocate, evaluate
from smartsom.trace.production import Playback, audit

ROOT = Path(__file__).resolve().parents[2]


def test_classified_scaffold_is_runnable_and_never_overwrites(tmp_path):
    result = scaffold(tmp_path / "configs", "trial")
    assert all("configs" in Path(path).parts for path in result["files"])
    config = api.load_config(tmp_path / "configs/runs/trial_train_machine_ppo.yaml")
    assert api.show_config(config)["groups"]["machine"]["training"]
    with pytest.raises(FileExistsError):
        scaffold(tmp_path / "configs", "trial")


def test_migration_never_converts_legacy_steps(tmp_path):
    with pytest.raises(ValueError, match="physical ticks"):
        migration_preview(
            ROOT / "configs/test/runs/template1_train_static.yaml",
            tmp_path / "configs/runs/new.yaml",
        )
    result = migration_preview(
        ROOT / "configs/test/runs/template1_train_static.yaml",
        tmp_path / "configs/runs/new.yaml",
        total_ticks=100,
        ticks_per_update=20,
    )
    assert result["configuration"]["training"]["total_ticks"] == 100
    assert not (tmp_path / "configs/runs/new.yaml").exists()


def test_rule_evaluation_trace_and_execution_audit(tmp_path):
    config = api.load_config(ROOT / "configs/test/runs/evaluate_all_rules.yaml")
    config.output.root = str(tmp_path)
    config.evaluation.replications = 1
    config.scenario_overrides["tick_limit"] = 12
    result = api.evaluate(config=config)
    assert result.engineering_failures == 0
    assert result.results[0]["truncated"]
    playback = Playback(result.run_dir / "evidence/case-0000")
    assert playback.last_tick == 12
    assert playback.manifest["action_contract"].endswith("/v3")
    assert playback.row(1)["decisions"][0]["candidates"]
    assert audit(result.run_dir)["checks"] == 1


@pytest.mark.parametrize("processes", [0, 2])
def test_budget_multiple_environments_and_sampling_processes(tmp_path, processes):
    pytest.importorskip("torch")
    pytest.importorskip("ray")
    config = api.load_config(ROOT / "configs/test/runs/train_all_ppo.yaml")
    config.training.total_ticks = 12
    config.training.ticks_per_update = 6
    config.runtime.num_envs = 2
    config.runtime.sampling_processes = processes
    config.validation.enabled = False
    config.output.root = str(tmp_path)
    prepared = prepare(config)
    parameters = json.loads(prepared.parameters_json)
    parameters.update(batch_size=4, n_epochs=1)
    prepared = replace(prepared, parameters_json=canonical_json(parameters))
    root, record, prepared = allocate(prepared, "training")
    session = TrainingSession(prepared, root, record)
    session.execute()
    assert session.ticks == sum(sim.tick for sim in session.sims) == 12
    assert {r["env"] for r in session.actions} == {0, 1}
    assert sum(session.optimizations.values()) > 0


def test_best_selection_validation_and_source_options(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("ray")
    config = api.load_config(ROOT / "configs/test/runs/train_all_ppo.yaml")
    config.training.total_ticks = 8
    config.training.ticks_per_update = 4
    config.validation.every_updates = 1
    config.validation.replications = 1
    config.evaluation.replications = 1
    config.scenario_overrides["tick_limit"] = 8
    config.output.root = str(tmp_path)
    prepared = prepare(config)
    params = json.loads(prepared.parameters_json)
    params.update(batch_size=4, n_epochs=1)
    prepared = replace(prepared, parameters_json=canonical_json(params))
    root, record, prepared = allocate(prepared, "training")
    session = TrainingSession(prepared, root, record)
    trained = session.execute()
    assert (root / "logs/validation-000001.json").exists()
    assert trained.best_checkpoint is None  # no completed validation episode yet
    assert (root / "checkpoints/recovery.json").exists()
    with pytest.raises(ValueError, match="best checkpoint"):
        evaluate(source=root, selection="best", output_root=tmp_path)
    result = evaluate(source=root, selection="last", output_root=tmp_path)
    assert result.engineering_failures == 0
    from smartsom.config.experiment_v3 import EvaluationOptionsV3

    changed = api.evaluate(
        root, EvaluationOptionsV3(seed=204, replications=2), output_root=tmp_path
    )
    assert len(changed.results) == 2
    assert changed.results[0]["seed"] != result.results[0]["seed"]


def test_cli_evaluation_preserves_frozen_options_unless_explicit():
    from smartsom.config.experiment_v3 import EvaluationOptionsV3
    from smartsom.experiments.cli import _evaluation_options, _parser

    defaults = EvaluationOptionsV3(
        seed=987, replications=7, checkpoint="best", deterministic=False, record=False
    )
    args = _parser().parse_args(["evaluate", "RUN"])
    assert _evaluation_options(args, defaults) == defaults
    args = _parser().parse_args(
        [
            "evaluate",
            "RUN",
            "--seed",
            "123",
            "--replications",
            "2",
            "--deterministic",
            "--checkpoint",
            "last",
        ]
    )
    selected = _evaluation_options(args, defaults)
    assert (selected.seed, selected.replications, selected.checkpoint) == (
        123,
        2,
        "last",
    )
    assert selected.deterministic and not selected.record
