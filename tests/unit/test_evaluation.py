"""Grid evaluation orchestration; stub inference is not model-training evidence.

Real save/load, parameter updates and learned completion live in the production
workflow integration tests. These tests inject a deterministic driver to isolate
pairing, failure retention, packaging and audit behavior from PPO convergence.
"""

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_extension_replay import SemanticDriver

from smartsom.config.codec import ConfigurationError, canonical_json, digest
from smartsom.config.production import ProductionRecipe
from smartsom.config.training import episode_root
from smartsom.experiments import evaluation, production, production_evaluation
from smartsom.experiments.audit import audit_run
from smartsom.experiments.evidence import write_json
from smartsom.experiments.packaging import model_locator
from smartsom.trace.production import seal_checkpoint


class OrchestrationDriver(SemanticDriver):
    """Explicit unit double using real encoding, not a trained policy."""

    def __init__(self, directory, scenario, **kwargs):
        from smartsom.config.production import AlgorithmConfig
        from smartsom.learning.episode import EpisodeLimits

        metadata = json.loads((Path(directory) / "checkpoint.json").read_text())
        algorithm = AlgorithmConfig.model_validate_json(
            json.dumps(metadata["algorithm"])
        )
        super().__init__(scenario, algorithm, EpisodeLimits())


@pytest.fixture
def saved(tmp_path, monkeypatch):
    pytest.importorskip("gymnasium")
    from smartsom.api import load_config, prepare

    config = load_config("configs/runs/sb3_production.yaml")
    config.validation.enabled = False
    prepared = prepare(config)
    source = tmp_path / "source"
    checkpoint = source / "checkpoint"
    checkpoint.mkdir(parents=True)
    write_json(
        checkpoint / "recipe.json", {"recipe": prepared.resolved, "config": config}
    )
    write_json(
        checkpoint / "checkpoint.json",
        {
            "schema": "smartsom.production-checkpoint/v1",
            "provider": prepared.resolved.algorithm.provider,
            "algorithm": prepared.resolved.algorithm,
        },
    )
    (checkpoint / "unit-driver.txt").write_text(
        "Unit orchestration fixture, not model weights."
    )
    seal_checkpoint(checkpoint)
    monkeypatch.setattr(
        "smartsom.learning.production.LearnedProductionDriver", OrchestrationDriver
    )
    return source, checkpoint, prepared.resolved


def opts(**kwargs):
    return SimpleNamespace(verbose=False, **kwargs)


def evaluate(saved, tmp_path, **kwargs):
    return evaluation.evaluate_checkpoint(
        saved[0], opts(**kwargs), output_root=tmp_path / "evaluations"
    )


def test_default_five_model_only_runs_and_all_replays(saved, tmp_path):
    result = evaluate(saved, tmp_path)
    assert result.status == "completed", result.results
    assert (result.completed, result.failed, result.engineering_failures) == (5, 0, 0)
    assert result.checkpoint == saved[1]
    assert len(result.results) == 5
    assert {row["algorithm_id"] for row in result.results} == {"model"}
    assert all(row["replay"]["status"] == "passed" for row in result.results)
    assert json.loads((result.run_dir / "summary.json").read_text())["completed"] == 5
    assert not (saved[0] / "run.json").exists()


def test_one_materialization_per_replication_and_paired_seeds(
    saved, tmp_path, monkeypatch
):
    calls = []
    original = ProductionRecipe.episode

    def episode(self, seed, **kwargs):
        calls.append(seed)
        return original(self, seed, **kwargs)

    monkeypatch.setattr(ProductionRecipe, "episode", episode)
    result = evaluate(saved, tmp_path, replications=2, baselines=("spt",))
    assert result.status == "completed", result.results
    assert len(calls) == 2
    for replication in range(2):
        paired = [row for row in result.results if row["replication"] == replication]
        assert len({row["world_sha256"] for row in paired}) == 1
        assert len({row["world_seed"] for row in paired}) == 1
        assert len({row["algorithm_seed"] for row in paired}) == 2
        manifests = [
            json.loads((result.run_dir / row["run_dir"] / "run.json").read_text())
            for row in paired
        ]
        assert manifests[0]["inputs"]["scenario"] == manifests[1]["inputs"]["scenario"]
        assert [m["algorithm_seed"] for m in manifests] == [
            r["algorithm_seed"] for r in paired
        ]


def test_seed_changes_do_not_hide_reused_scenario_or_recorded_overlap(saved, tmp_path):
    checkpoint, recipe = saved[1:]
    world_sha = digest(recipe.episode(episode_root(202, 0)))
    (checkpoint / "episodes.jsonl").write_text(
        canonical_json({"input_sha256": world_sha}) + "\n"
    )
    write_json(
        checkpoint / "validation.json", {"results": [{"world_sha256": world_sha}]}
    )
    seal_checkpoint(checkpoint)
    result = evaluate(saved, tmp_path, replications=2, seed=202)
    coverage = json.loads((result.run_dir / "summary.json").read_text())[
        "input_coverage"
    ]
    assert coverage["evaluation_inputs"] == coverage["evaluation_unique_worlds"] == 2
    assert coverage["training_history"]["overlapping_evaluation_worlds"] == [world_sha]
    assert coverage["training_history"]["all_training_samples_covered"] is False
    assert coverage["validation"]["worlds_disjoint"] is False
    assert coverage["cases"][0]["same_training_factory"]
    assert coverage["cases"][0]["same_training_workload"]


def test_missing_history_is_unavailable_not_disjoint(saved, tmp_path):
    result = evaluate(saved, tmp_path, replications=1)
    coverage = json.loads((result.run_dir / "run.json").read_text())["input_coverage"]
    assert coverage["training_history"] == {"status": "unavailable"}
    assert coverage["validation"] == {"status": "unavailable"}


def test_default_reconstruction_does_not_open_authoring_source(saved, tmp_path):
    path = saved[1] / "recipe.json"
    payload = json.loads(path.read_text())
    payload["config"]["scenario"] = "/missing/historical/scenario.yaml"
    write_json(path, payload)
    seal_checkpoint(saved[1])
    assert evaluate(saved, tmp_path, replications=1).status == "completed"


def test_explicit_scenario_without_snapshot(saved, tmp_path):
    (saved[1] / "recipe.json").unlink()
    seal_checkpoint(saved[1])
    result = evaluate(
        saved,
        tmp_path,
        replications=1,
        scenarios=("configs/scenarios/scenario_test.yaml",),
    )
    assert result.status == "completed", result.results
    assert result.results[0]["case_id"] != "training"
    assert (
        json.loads((result.run_dir / "run.json").read_text())["input_coverage"]["cases"]
        == []
    )


def test_checkpoint_selection_and_errors_retain_failed_output(saved, tmp_path):
    assert model_locator(saved[1]) == model_locator(saved[0]) == saved[1]
    with pytest.raises(ConfigurationError, match="best") as caught:
        evaluate(saved, tmp_path, checkpoint="best")
    record = json.loads((caught.value.run_dir / "run.json").read_text())
    assert record["status"] == "failed" and record["stage"] == "checkpoint_selection"
    assert record["results"] == []
    (saved[0] / "checkpoints").mkdir(exist_ok=True)
    write_json(saved[0] / "checkpoints/last.json", {"checkpoint": "../checkpoint"})
    assert model_locator(saved[0]) == saved[1]
    (saved[0] / "checkpoints").mkdir(exist_ok=True)
    write_json(saved[0] / "checkpoints/last.json", {"checkpoint": "last.json"})
    with pytest.raises(ValueError, match="cyclic"):
        model_locator(saved[0])


@pytest.mark.parametrize(
    "values",
    [
        {"replications": 0},
        {"seed": True},
        {"deterministic": 1},
        {"checkpoint": "latest"},
        {"baselines": "spt"},
        {"scenarios": ("",)},
        {"scenarios": ("  ",)},
        {"baselines": ("\t",)},
    ],
)
def test_bad_options_allocate_no_output(saved, tmp_path, values):
    with pytest.raises(ConfigurationError):
        evaluate(saved, tmp_path, **values)
    assert not (tmp_path / "evaluations").exists()


def test_stall_is_noncompletion_and_only_prefix_replay(saved, tmp_path, monkeypatch):
    monkeypatch.setattr(
        OrchestrationDriver,
        "step",
        lambda self: self.env.step(5 if self.env.role == "agv" else 0),
    )
    path = saved[1] / "recipe.json"
    payload = json.loads(path.read_text())
    world = json.loads(payload["recipe"]["scenario_json"])
    world["tick_limit"] = 10
    payload["recipe"]["scenario_json"] = canonical_json(world)
    write_json(path, payload)
    seal_checkpoint(saved[1])
    result = evaluate(saved, tmp_path, replications=1)
    assert result.status == "completed_with_failures", result.results
    assert (result.completed, result.failed, result.engineering_failures) == (0, 1, 0)
    row = result.results[0]
    assert row["status"] == "truncated" and row["makespan"] is None
    assert row["replay"]["status"] == "partial_verified"


def test_engineering_error_is_distinct(saved, tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("injected unavailable model")

    monkeypatch.setattr(production, "run", fail)
    result = evaluate(saved, tmp_path, replications=1)
    assert result.status == "failed" and result.engineering_failures == 1
    assert result.results[0]["makespan"] is None
    assert result.results[0]["replay"]["status"] == "unavailable_no_run_evidence"


def test_stochastic_flag_and_no_replay_are_explicit(saved, tmp_path, monkeypatch):
    original = OrchestrationDriver.__init__
    flags = []

    def init(self, *args, **kwargs):
        flags.append(kwargs["deterministic"])
        original(self, *args, **kwargs)

    monkeypatch.setattr(OrchestrationDriver, "__init__", init)
    result = evaluate(
        saved, tmp_path, replications=1, deterministic=False, full_replay=False
    )
    assert result.status == "completed", result.results
    assert flags == [False]
    assert result.results[0]["replay"] == {"status": "not_requested"}
    assert isinstance(result.results[0]["algorithm_seed"], int)


def test_corruption_never_passes_full_audit(saved, tmp_path):
    result = evaluate(saved, tmp_path, replications=1)
    directory = result.run_dir / result.results[0]["run_dir"]
    (directory / "trace.jsonl").write_text("{}\n")
    with pytest.raises(ValueError, match="invalid trace schema|digest mismatch"):
        audit_run(directory)


@pytest.mark.parametrize("alias", ["cp", "cp_sat", "pyjobshop.cp_sat"])
def test_ineligible_cp_retains_failed_preparation_without_execution(
    saved, tmp_path, alias
):
    with pytest.raises(
        ConfigurationError, match="no grid production adapter"
    ) as caught:
        evaluate(saved, tmp_path, baselines=(alias,))
    record = json.loads((caught.value.run_dir / "run.json").read_text())
    assert record["status"] == "failed" and record["stage"] == "input_preparation"
    assert not (caught.value.run_dir / "evidence/runs").exists()


def test_grid_template_audit_has_no_legacy_schedule_artifact(tmp_path):
    from smartsom.api import load_config, run
    from smartsom.config.authoring import create_template

    project = create_template("minimal_jsp", tmp_path / "static")
    config = load_config(project / "run.yaml")
    config.output.root = str(tmp_path / "runs")
    config.algorithm.source = str(
        Path("configs/algorithms/crossing_script.yaml").resolve()
    )
    result = run(config, verbose=False)
    assert not (result.run_dir / "execution_schedule.json").exists()
    assert audit_run(result.run_dir)["status"] == "passed"


def test_keyboard_interrupt_retains_row_and_stops_further_runs(
    saved, tmp_path, monkeypatch
):
    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt()

    monkeypatch.setattr(OrchestrationDriver, "next_tick", interrupt)
    with pytest.raises(KeyboardInterrupt) as caught:
        evaluate(saved, tmp_path)
    record = json.loads((caught.value.run_dir / "run.json").read_text())
    assert record["status"] == "interrupted" and record["pending"] == 4
    assert record["results"][0]["status"] == "interrupted"
    assert (
        caught.value.run_dir / record["results"][0]["run_dir"] / "trace.jsonl"
    ).exists()


def test_portable_model_and_evaluation_packages_use_copied_checkpoint(saved, tmp_path):
    from smartsom.experiments.packaging import (
        export_experiment,
        export_model,
        import_bundle,
    )

    archive = export_model(saved[1], tmp_path / "model.zip")
    imported = import_bundle(archive, tmp_path / "imported-model")
    selected = model_locator(imported)
    assert selected.is_relative_to(imported) and (selected / "recipe.json").is_file()
    result = evaluation.evaluate_checkpoint(
        imported, opts(replications=1), output_root=tmp_path / "evaluations"
    )
    assert result.status == "completed", result.results
    archive = export_experiment(result.run_dir, tmp_path / "evaluation.zip")
    imported_eval = import_bundle(archive, tmp_path / "imported-evaluation")
    selected = model_locator(imported_eval)
    assert selected.is_relative_to(imported_eval) and selected != result.checkpoint
    assert (selected / "recipe.json").is_file()
    assert audit_run(imported_eval / result.results[0]["run_dir"])["status"] == "passed"
    assert audit_run(imported_eval)["status"] == "passed"


def test_outer_audit_checks_coverage_duplicate_runs_and_result_identity(
    saved, tmp_path
):
    result = evaluate(saved, tmp_path, replications=2)
    outer = tmp_path / "outer"
    outer.mkdir()
    write_json(
        outer / "run.json",
        {
            "schema": "smartsom.experiment/v2",
            "paths": {"evaluation": str(result.run_dir)},
        },
    )
    assert audit_run(outer)["status"] == "passed"
    path = result.run_dir / "run.json"
    original = path.read_text()
    record = json.loads(original)
    record["results"][1]["run_dir"] = record["results"][0]["run_dir"]
    write_json(path, record)
    with pytest.raises(ValueError, match="duplicate evaluation child"):
        audit_run(outer)
    record = json.loads(original)
    record["results"][0]["makespan"] += 1
    write_json(path, record)
    with pytest.raises(ValueError, match="differs from child"):
        audit_run(outer)
    record["results"].pop()
    write_json(path, record)
    with pytest.raises(ValueError, match="coverage mismatch"):
        audit_run(outer)


@pytest.mark.parametrize("kind", ["model", "experiment"])
def test_zip_source_imports_durable_model_without_a_caller_import(
    saved, tmp_path, kind
):
    from smartsom.experiments.packaging import export_experiment, export_model

    archive = export_model(saved[1], tmp_path / "model.zip")
    if kind == "experiment":
        previous = evaluation.evaluate_checkpoint(
            archive, opts(replications=1), output_root=tmp_path / "previous"
        )
        archive = export_experiment(previous.run_dir, tmp_path / "experiment.zip")
        shutil.rmtree(previous.run_dir)
    archive_bytes = archive.read_bytes()
    shutil.rmtree(saved[1])
    result = evaluation.evaluate_checkpoint(
        archive, opts(replications=1), output_root=tmp_path / "evaluations"
    )
    assert result.status == "completed", result.results
    imported = result.run_dir / "evidence/imported-model"
    assert result.checkpoint.is_relative_to(imported)
    record = json.loads((result.run_dir / "run.json").read_text())
    assert record["stage"] == "finished"
    assert record["model_input"]["sha256"] == production_evaluation.file_hash(archive)
    assert record["paths"]["imported_model"] == "evidence/imported-model"
    assert Path(record["checkpoint"]["training_snapshot"]).is_relative_to(imported)
    assert archive.read_bytes() == archive_bytes
    assert all(
        Path(row["checkpoint"]["path"]).is_relative_to(imported)
        for row in record["results"]
    )


def test_missing_model_has_durable_failure_metadata_and_exception_directory(tmp_path):
    with pytest.raises(ConfigurationError, match="cannot read last") as caught:
        evaluation.evaluate_checkpoint(
            tmp_path / "missing", opts(), output_root=tmp_path / "outputs"
        )
    record = json.loads((caught.value.run_dir / "run.json").read_text())
    summary = json.loads((caught.value.run_dir / "summary.json").read_text())
    assert record["status"] == summary["status"] == "failed"
    assert record["stage"] == "checkpoint_selection" and record["checkpoint"] is None
    assert record["results"] == [] and record["error"]["type"] == "ConfigurationError"
    with pytest.raises(ValueError, match="no saved checkpoint"):
        model_locator(caught.value.run_dir)


def test_corrupt_zip_rejection_is_recorded_without_unverified_extraction(tmp_path):
    archive = tmp_path / "model.zip"
    archive.write_bytes(b"not a ZIP bundle")
    with pytest.raises(
        ConfigurationError, match="cannot import model bundle"
    ) as caught:
        evaluation.evaluate_checkpoint(
            archive, opts(), output_root=tmp_path / "outputs"
        )
    record = json.loads((caught.value.run_dir / "run.json").read_text())
    assert record["status"] == "failed" and record["stage"] == "model_import"
    assert not (caught.value.run_dir / "evidence/imported-model").exists()
    assert archive.read_bytes() == b"not a ZIP bundle"


def test_unreadable_training_snapshot_preserves_import_and_failure(saved, tmp_path):
    from smartsom.experiments.packaging import export_model

    (saved[1] / "recipe.json").write_text("{invalid JSON")
    seal_checkpoint(saved[1])
    archive = export_model(saved[1], tmp_path / "model.zip")
    with pytest.raises(ValueError) as caught:
        evaluation.evaluate_checkpoint(
            archive, opts(), output_root=tmp_path / "outputs"
        )
    root = caught.value.run_dir
    record = json.loads((root / "run.json").read_text())
    assert record["status"] == "failed" and record["stage"] == "input_preparation"
    assert (
        Path(record["checkpoint"]["path"]) / "recipe.json"
    ).read_text() == "{invalid JSON"
    assert not (root / "evidence/runs").exists()


def test_import_interruption_retains_verified_files_and_attaches_run_dir(
    saved, tmp_path, monkeypatch
):
    from smartsom.experiments.packaging import export_model

    archive = export_model(saved[1], tmp_path / "model.zip")
    original = production_evaluation.import_bundle
    interrupted = KeyboardInterrupt("stop after verified extraction")

    def stop(source, destination):
        original(source, destination)
        raise interrupted

    monkeypatch.setattr(production_evaluation, "import_bundle", stop)
    with pytest.raises(KeyboardInterrupt) as caught:
        evaluation.evaluate_checkpoint(
            archive, opts(), output_root=tmp_path / "outputs"
        )
    assert caught.value is interrupted
    root = interrupted.run_dir
    record = json.loads((root / "run.json").read_text())
    assert record["status"] == "interrupted" and record["stage"] == "model_import"
    assert list((root / "evidence/imported-model").rglob("checkpoint.json"))
    assert record["results"] == []


def test_failure_metadata_error_does_not_replace_original_exception(
    saved, tmp_path, monkeypatch
):
    original = production_evaluation.write_json
    failure = RuntimeError("input preparation failed")

    def write(path, value):
        if value.get("status") == "failed":
            raise OSError("metadata directory unavailable")
        return original(path, value)

    def prepare(*args):
        raise failure

    monkeypatch.setattr(production_evaluation, "write_json", write)
    monkeypatch.setattr(ProductionRecipe, "episode", prepare)
    with pytest.raises(RuntimeError) as caught:
        evaluate(saved, tmp_path)
    assert caught.value is failure and failure.run_dir.is_dir()
    assert "failure metadata could not be saved" in failure.__notes__[0]


def test_outer_json_selection_precedes_preserved_materialized_alias(saved, tmp_path):
    outer = tmp_path / "outer-current"
    old = outer / "checkpoints/last"
    new = outer / "evidence/training/current/checkpoint"
    shutil.copytree(saved[1], old)
    shutil.copytree(saved[1], new)
    write_json(
        outer / "run.json",
        {
            "schema": "smartsom.experiment/v2",
            "paths": {"training": "evidence/training/current"},
        },
    )
    original = (old / "checkpoint.json").read_bytes()
    assert model_locator(outer) == model_locator(outer / "run.json") == new
    assert (old / "checkpoint.json").read_bytes() == original


def test_unreadable_checkpoint_manifest_retains_selection_failure(saved, tmp_path):
    (saved[1] / "checkpoint.json").write_text("{invalid JSON")
    with pytest.raises(ConfigurationError, match="cannot read") as caught:
        evaluate(saved, tmp_path)
    record = json.loads((caught.value.run_dir / "run.json").read_text())
    assert record["status"] == "failed" and record["stage"] == "checkpoint_selection"
    assert record["results"] == []


def test_replay_interruption_retains_completed_physical_run_reference(
    saved, tmp_path, monkeypatch
):
    from smartsom.trace.production import ExecutionAudit

    def stop(*args):
        raise KeyboardInterrupt("replay interrupted")

    monkeypatch.setattr(ExecutionAudit, "result", stop)
    with pytest.raises(KeyboardInterrupt) as caught:
        evaluate(saved, tmp_path, replications=2)
    root = caught.value.run_dir
    record = json.loads((root / "run.json").read_text())
    assert record["status"] == "interrupted" and record["pending"] == 1
    row = record["results"][0]
    assert row["status"] == "interrupted" and row["makespan"] is None
    assert row["replay"]["status"] == "interrupted"
    assert (root / row["run_dir"] / "trace.jsonl").is_file()
