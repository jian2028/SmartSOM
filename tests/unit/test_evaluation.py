"""Independent evaluation uses paired scientific inputs and real run evidence."""

import json
import shutil
from dataclasses import replace
from types import SimpleNamespace

import pytest
from test_resource_training import bundle, spt_proposals, training

from smartsom.config import load_resolved_run
from smartsom.config.codec import ConfigurationError, canonical_json, digest
from smartsom.config.training import ResolvedTrainingRun, episode_input
from smartsom.experiments import evaluation
from smartsom.experiments.audit import audit_run
from smartsom.experiments.evidence import write_json
from smartsom.learning.resource_policy import ResourceCheckpointPolicy


@pytest.fixture
def saved(tmp_path, monkeypatch):
    monkeypatch.setattr("smartsom.learning.checkpoint.require_backend", lambda p: {})
    resolved, checkpoint = bundle(tmp_path)
    write_json(
        tmp_path / "resolved_training.json",
        {"schema": "smartsom.resolved-training/v1", "resolved": training()},
    )
    original = ResourceCheckpointPolicy.__init__
    monkeypatch.setattr(
        ResourceCheckpointPolicy,
        "__init__",
        lambda self, resolved, **kwargs: original(
            self, resolved, predictor=spt_proposals, **kwargs
        ),
    )
    return tmp_path, checkpoint, resolved


def opts(**kwargs):
    return SimpleNamespace(**kwargs)


def test_default_five_model_only_runs_and_all_replays(saved, tmp_path):
    source, checkpoint, _ = saved
    result = evaluation.evaluate_checkpoint(
        source, opts(), output_root=tmp_path / "evaluations"
    )
    assert result.status == "completed", result.results
    assert (result.completed, result.failed, result.engineering_failures) == (5, 0, 0)
    assert result.checkpoint == checkpoint
    assert len(result.results) == 5
    assert {row["algorithm_id"] for row in result.results} == {"model"}
    assert all(row["replay"]["status"] == "passed" for row in result.results)
    assert all(
        row["replay"]["joint_replay"]["status"] == "passed" for row in result.results
    )
    assert json.loads((result.run_dir / "summary.json").read_text())["completed"] == 5
    assert not (source / "run.json").exists()


def test_one_materialization_per_replication_and_paired_seeds(
    saved, tmp_path, monkeypatch
):
    source, _, _ = saved
    calls = []
    original = ResolvedTrainingRun.episode

    def episode(self, index, **kwargs):
        calls.append((index, kwargs["root_seed"]))
        return original(self, index, **kwargs)

    monkeypatch.setattr(ResolvedTrainingRun, "episode", episode)
    result = evaluation.evaluate_checkpoint(
        source,
        opts(replications=2, baselines=("spt",)),
        output_root=tmp_path / "evaluations",
    )
    assert result.status == "completed", result.results
    assert len(calls) == 2
    for replication in range(2):
        paired = [row for row in result.results if row["replication"] == replication]
        assert len({row["world_sha256"] for row in paired}) == 1
        assert len({row["world_seed"] for row in paired}) == 1
        algorithm_seeds = [
            next(s["value"] for s in row["seeds"] if s["domain"] == "algorithm")
            for row in paired
        ]
        assert len(set(algorithm_seeds)) == 2
        resolved = [
            load_resolved_run(result.run_dir / row["run_dir"] / "resolved_run.yaml")
            for row in paired
        ]
        assert episode_input(resolved[0]) == episode_input(resolved[1])
        assert resolved[0].arrival_provenance.effective_seed == next(
            s.value for s in resolved[0].seeds if s.domain == "demand"
        )


def test_seed_changes_do_not_hide_reused_scenario_or_recorded_overlap(saved, tmp_path):
    source, _, _ = saved
    entries, _ = evaluation._prepare(
        evaluation._select_checkpoint(source),
        evaluation._Options(replications=2, seed=202),
    )
    frozen_world = episode_input(entries[0][0])
    world_sha = digest(frozen_world)
    (source / "episodes.jsonl").write_text(
        canonical_json({"input_sha256": world_sha}) + "\n"
    )
    write_json(
        source / "validation_inputs.json",
        [{"episode": frozen_world, "world_seed": 303}],
    )
    result = evaluation.evaluate_checkpoint(
        source, opts(replications=2, seed=202), output_root=tmp_path / "evaluations"
    )
    coverage = json.loads((result.run_dir / "summary.json").read_text())[
        "input_coverage"
    ]
    assert coverage["evaluation_inputs"] == 2
    assert coverage["evaluation_unique_worlds"] == 2
    assert coverage["training_history"]["overlapping_evaluation_worlds"] == [world_sha]
    assert coverage["training_history"]["all_training_samples_covered"] is False
    assert coverage["validation"]["worlds_disjoint"] is False
    assert coverage["cases"][0]["same_training_factory"]
    assert coverage["cases"][0]["same_training_workload"]


def test_missing_history_is_unavailable_not_disjoint(saved, tmp_path):
    source, _, _ = saved
    result = evaluation.evaluate_checkpoint(
        source, opts(replications=1), output_root=tmp_path / "evaluations"
    )
    coverage = json.loads((result.run_dir / "run.json").read_text())["input_coverage"]
    assert coverage["training_history"] == {"status": "unavailable"}
    assert coverage["validation"] == {"status": "unavailable"}


def test_default_reconstruction_does_not_open_authoring_source(saved, tmp_path):
    source, _, _ = saved
    snapshot = json.loads((source / "resolved_training.json").read_text())
    snapshot["resolved"]["run"]["scenario"] = "/missing/historical/scenario.yaml"
    snapshot["resolved"]["base"]["run"]["scenario"] = (
        "/missing/historical/scenario.yaml"
    )
    write_json(source / "resolved_training.json", snapshot)
    result = evaluation.evaluate_checkpoint(
        source, opts(replications=1), output_root=tmp_path / "evaluations"
    )
    assert result.status == "completed", result.results


def test_explicit_scenario_without_snapshot(saved, tmp_path):
    source, checkpoint, resolved = saved
    (source / "resolved_training.json").unlink()
    result = evaluation.evaluate_checkpoint(
        checkpoint,
        opts(replications=1, scenarios=(resolved.run.scenario,)),
        output_root=tmp_path / "evaluations",
    )
    assert result.status == "completed", result.results
    assert result.results[0]["case_id"] != "training"


def test_checkpoint_selection_and_errors_retain_failed_output(saved, tmp_path):
    source, checkpoint, _ = saved
    assert evaluation._select_checkpoint(checkpoint).selection == "explicit"
    assert evaluation._select_checkpoint(source).directory == checkpoint
    root = tmp_path / "outputs"
    with pytest.raises(ConfigurationError, match="no best") as caught:
        evaluation.evaluate_checkpoint(
            source, opts(checkpoint="best"), output_root=root
        )
    record = json.loads((caught.value.run_dir / "run.json").read_text())
    assert record["status"] == "failed"
    assert record["stage"] == "checkpoint_selection"
    assert record["results"] == []
    controlled = tmp_path / "controlled"
    update = controlled / "checkpoints/update-000004"
    update.mkdir(parents=True)
    shutil.copytree(checkpoint, update / "inference")
    shutil.copy2(source / "resolved_training.json", update / "resolved_training.json")
    write_json(controlled / "checkpoints/last.json", {"checkpoint": "update-000004"})
    outer = tmp_path / "outer"
    outer.mkdir()
    write_json(outer / "run.json", {"paths": {"training": "../controlled"}})
    selected = evaluation._select_checkpoint(outer)
    assert selected.directory == update / "inference"
    assert selected.snapshot == update / "resolved_training.json"
    assert evaluation._select_checkpoint(update).directory == selected.directory


@pytest.mark.parametrize(
    "values",
    [
        {"replications": 0},
        {"seed": True},
        {"deterministic": 1},
        {"checkpoint": "latest"},
        {"baselines": "spt"},
        {"scenarios": ("",)},
    ],
)
def test_bad_options_allocate_no_output(saved, tmp_path, values):
    with pytest.raises(ConfigurationError):
        evaluation.evaluate_checkpoint(
            saved[0], opts(**values), output_root=tmp_path / "outputs"
        )
    assert not (tmp_path / "outputs").exists()


def test_stall_is_noncompletion_and_only_prefix_replay(saved, tmp_path, monkeypatch):
    source, _, _ = saved
    monkeypatch.setattr(
        "test_evaluation.spt_proposals", lambda d: {v.agent_id: 0 for v in d.views}
    )
    result = evaluation.evaluate_checkpoint(
        source, opts(replications=1), output_root=tmp_path / "evaluations"
    )
    assert result.status == "completed_with_failures", result.results
    assert (result.completed, result.failed, result.engineering_failures) == (0, 1, 0)
    row = result.results[0]
    assert row["reason"] == "policy_stalled" and row["makespan"] is None
    assert row["replay"]["status"] == "partial_verified"
    assert row["replay"]["schedule_replay"] == "not_applicable_incomplete_run"


def test_engineering_error_is_distinct(saved, tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("injected unavailable model")

    monkeypatch.setattr(evaluation, "run_one", fail)
    result = evaluation.evaluate_checkpoint(
        saved[0], opts(replications=1), output_root=tmp_path / "evaluations"
    )
    assert result.status == "failed" and result.engineering_failures == 1
    assert result.results[0]["makespan"] is None
    assert result.results[0]["replay"]["status"] == "unavailable_no_run_evidence"


def test_stochastic_flag_and_no_replay_are_explicit(saved, tmp_path, monkeypatch):
    original = evaluation.run_one
    flags = []

    def run(resolved, **kwargs):
        flags.append(kwargs["deterministic"])
        return original(resolved, **kwargs)

    monkeypatch.setattr(evaluation, "run_one", run)
    result = evaluation.evaluate_checkpoint(
        saved[0],
        opts(replications=1, deterministic=False, full_replay=False),
        output_root=tmp_path / "evaluations",
    )
    assert result.status == "completed", result.results
    assert flags == [False]
    assert result.results[0]["replay"] == {"status": "not_requested"}
    assert next(
        s["consumed"] for s in result.results[0]["seeds"] if s["domain"] == "algorithm"
    )


def test_corruption_never_passes_full_audit(saved, tmp_path):
    result = evaluation.evaluate_checkpoint(
        saved[0], opts(replications=1), output_root=tmp_path / "evaluations"
    )
    directory = result.run_dir / result.results[0]["run_dir"]
    (directory / "trace.jsonl").write_text("{}\n")
    with pytest.raises(ValueError, match="digest mismatch"):
        audit_run(directory)


def test_ineligible_cp_retains_failed_preparation_without_execution(saved, tmp_path):
    with pytest.raises(ConfigurationError, match="does not support") as caught:
        evaluation.evaluate_checkpoint(
            saved[0], opts(baselines=("cp",)), output_root=tmp_path / "outputs"
        )
    record = json.loads((caught.value.run_dir / "run.json").read_text())
    assert record["status"] == "failed" and record["stage"] == "input_preparation"
    assert not (caught.value.run_dir / "evidence/runs").exists()


def test_static_schedule_audit_without_logistics(tmp_path):
    from smartsom.config import resolve_run
    from smartsom.config.authoring import create_template
    from smartsom.config.models import RecordingSpec

    project = create_template("minimal_jsp", tmp_path / "static")
    resolved = resolve_run(project / "run.yaml")
    resolved = replace(
        resolved,
        run=resolved.run.model_copy(
            update={"recording": RecordingSpec(observations="hash")}
        ),
    )
    result = evaluation.run_one(resolved)
    assert not (result.run_dir / "execution_schedule.json").exists()
    audited = audit_run(result.run_dir)
    assert audited["status"] == "passed" and audited["joint_replay"] is None


def test_explicit_static_cp_uses_same_world_with_recorded_extra_information(
    tmp_path, monkeypatch
):
    from smartsom.config import resolve_run
    from smartsom.config.authoring import create_template
    from smartsom.config.models import EpisodeBudget

    project = create_template("minimal_jsp", tmp_path / "static")
    base = resolve_run(project / "run.yaml")
    monkeypatch.setattr(evaluation.importlib.util, "find_spec", lambda name: object())
    # Build the CP spec through the same public baseline alias table.
    algorithms = evaluation._algorithms(
        SimpleNamespace(algorithm=lambda: base.algorithm), ("cp",)
    )
    options = evaluation._Options(replications=1)
    world, _ = evaluation.study_roots(options.seed, "static", 0, "")
    base = evaluation._case_world(project / "scenario.yaml", world)
    bound = evaluation._bind(
        base,
        algorithms[1][1],
        "pyjobshop.cp_sat",
        "static",
        0,
        options,
        EpisodeBudget(),
    )
    assert episode_input(bound) == episode_input(base)
    assert bound.scenario.visibility == "full_static"
    assert bound.algorithm.algorithm.required_information == "full_static"
    assert base.scenario.visibility == "decision_context"


def test_keyboard_interrupt_retains_row_and_stops_further_runs(
    saved, tmp_path, monkeypatch
):
    def interrupt(resolved, **kwargs):
        directory = evaluation.Path(resolved.run.output_root) / "interrupted-child"
        directory.mkdir(parents=True)
        raise evaluation.RunFailedError(directory, KeyboardInterrupt())

    monkeypatch.setattr(evaluation, "run_one", interrupt)
    with pytest.raises(KeyboardInterrupt):
        evaluation.evaluate_checkpoint(
            saved[0], opts(), output_root=tmp_path / "outputs"
        )
    records = list((tmp_path / "outputs").glob("*/run.json"))
    record = json.loads(records[0].read_text())
    assert record["status"] == "interrupted" and record["pending"] == 4
    assert record["results"][0]["status"] == "interrupted"


def test_portable_model_and_evaluation_packages_use_copied_checkpoint(saved, tmp_path):
    from smartsom.experiments.packaging import (
        export_experiment,
        export_model,
        import_bundle,
    )

    archive = export_model(saved[1], tmp_path / "model.zip")
    imported = import_bundle(archive, tmp_path / "imported-model")
    selected = evaluation._select_checkpoint(imported)
    assert selected.directory.is_relative_to(imported)
    assert selected.snapshot == imported / "resolved_training.json"
    result = evaluation.evaluate_checkpoint(
        imported, opts(replications=1), output_root=tmp_path / "evaluations"
    )
    assert result.status == "completed", result.results
    archive = export_experiment(result.run_dir, tmp_path / "evaluation.zip")
    imported_eval = import_bundle(archive, tmp_path / "imported-evaluation")
    selected = evaluation._select_checkpoint(imported_eval)
    assert selected.directory.is_relative_to(imported_eval)
    assert selected.directory != result.checkpoint
    audited = audit_run(imported_eval / result.results[0]["run_dir"])
    assert audited["status"] == "passed"
    assert selected.snapshot.is_relative_to(imported_eval)
    assert audit_run(imported_eval)["status"] == "passed"


def test_outer_audit_checks_coverage_duplicate_runs_and_result_identity(
    saved, tmp_path
):
    result = evaluation.evaluate_checkpoint(
        saved[0], opts(replications=2), output_root=tmp_path / "evaluations"
    )
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
    record = json.loads(path.read_text())
    original = json.loads(path.read_text())
    record["results"][1]["run_dir"] = record["results"][0]["run_dir"]
    write_json(path, record)
    assert audit_run(outer)["status"] == "failed"
    record = original
    record["results"][0]["makespan"] += 1
    write_json(path, record)
    assert audit_run(outer)["status"] == "failed"
    record["results"].pop()
    write_json(path, record)
    with pytest.raises(ValueError, match="coverage mismatch"):
        audit_run(outer)


def test_solver_without_incumbent_has_no_execution_to_replay(tmp_path, monkeypatch):
    from smartsom.algorithms.solver import ScheduleSolution, SolverStatus
    from smartsom.config import resolve_run
    from smartsom.config.authoring import create_template
    from smartsom.config.models import EpisodeBudget

    project = create_template("minimal_jsp", tmp_path / "static")
    base = resolve_run(project / "run.yaml")
    monkeypatch.setattr(evaluation.importlib.util, "find_spec", lambda name: object())
    algorithm = evaluation._algorithms(
        SimpleNamespace(algorithm=lambda: base.algorithm), ("cp",)
    )[1][1]
    options = evaluation._Options(replications=1)
    world, _ = evaluation.study_roots(options.seed, "static", 0, "")
    base = evaluation._case_world(project / "scenario.yaml", world)
    bound = evaluation._bind(
        base, algorithm, "cp", "static", 0, options, EpisodeBudget()
    )
    bound = replace(
        bound, run=bound.run.model_copy(update={"output_root": str(tmp_path / "runs")})
    )
    monkeypatch.setattr(
        "smartsom.experiments.runner.build_provider",
        lambda algorithm: SimpleNamespace(
            solve=lambda request: ScheduleSolution(
                (), SolverStatus.TIME_LIMIT, None, None, 0.01
            )
        ),
    )
    with pytest.raises(evaluation.RunFailedError) as caught:
        evaluation.run_one(bound)
    assert evaluation._failure(caught.value.cause) == ("solver_no_incumbent", False)
    audited = audit_run(caught.value.run_dir)
    assert audited["status"] == "partial_verified"
    assert audited["checks"] == ["artifact_integrity_without_execution"]


@pytest.mark.parametrize("kind", ["model", "experiment"])
def test_zip_source_imports_durable_model_without_a_caller_import(
    saved, tmp_path, kind
):
    from smartsom.experiments.packaging import export_experiment, export_model

    if kind == "model":
        archive = export_model(saved[1], tmp_path / "model.zip")
    else:
        model_archive = export_model(saved[1], tmp_path / "model.zip")
        previous = evaluation.evaluate_checkpoint(
            model_archive, opts(replications=1), output_root=tmp_path / "previous"
        )
        archive = export_experiment(previous.run_dir, tmp_path / "experiment.zip")
        shutil.rmtree(previous.run_dir)
    archive_bytes = archive.read_bytes()
    shutil.rmtree(saved[1])
    result = evaluation.evaluate_checkpoint(
        archive, opts(replications=1), output_root=tmp_path / "evaluations"
    )
    assert result.status == "completed", result.results
    assert result.results[0]["replay"]["status"] == "passed"
    imported = result.run_dir / "evidence/imported-model"
    assert result.checkpoint.is_relative_to(imported)
    record = json.loads((result.run_dir / "run.json").read_text())
    assert record["stage"] == "finished"
    assert record["model_input"]["sha256"] == evaluation.file_hash(archive)
    assert record["paths"]["imported_model"] == "evidence/imported-model"
    assert evaluation.Path(record["checkpoint"]["training_snapshot"]).is_relative_to(
        imported
    )
    assert archive.read_bytes() == archive_bytes
    assert all(
        evaluation.Path(row["checkpoint"]["path"]).is_relative_to(imported)
        for row in record["results"]
    )


def test_missing_model_has_durable_failure_metadata_and_exception_directory(tmp_path):
    with pytest.raises(ConfigurationError, match="no last") as caught:
        evaluation.evaluate_checkpoint(
            tmp_path / "missing", opts(), output_root=tmp_path / "outputs"
        )
    record = json.loads((caught.value.run_dir / "run.json").read_text())
    summary = json.loads((caught.value.run_dir / "summary.json").read_text())
    assert record["status"] == summary["status"] == "failed"
    assert record["stage"] == "checkpoint_selection"
    assert record["checkpoint"] is None
    assert record["results"] == []
    assert record["error"].startswith("ConfigurationError:")
    with pytest.raises(ConfigurationError, match="no saved checkpoint reference"):
        evaluation._select_checkpoint(caught.value.run_dir)


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

    (saved[0] / "resolved_training.json").write_text("{invalid JSON")
    archive = export_model(saved[1], tmp_path / "model.zip")
    with pytest.raises((ConfigurationError, ValueError)) as caught:
        evaluation.evaluate_checkpoint(
            archive, opts(), output_root=tmp_path / "outputs"
        )
    root = caught.value.run_dir
    record = json.loads((root / "run.json").read_text())
    assert record["status"] == "failed" and record["stage"] == "input_preparation"
    assert (
        root / "evidence/imported-model/payload/resolved_training.json"
    ).read_text() == "{invalid JSON"
    assert evaluation.Path(record["checkpoint"]["path"]).is_dir()
    assert not (root / "evidence/runs").exists()


def test_import_interruption_retains_verified_files_and_attaches_run_dir(
    saved, tmp_path, monkeypatch
):
    from smartsom.experiments.packaging import export_model

    archive = export_model(saved[1], tmp_path / "model.zip")
    original = evaluation.import_bundle
    interrupted = KeyboardInterrupt("stop after verified extraction")

    def stop(source, destination):
        original(source, destination)
        raise interrupted

    monkeypatch.setattr(evaluation, "import_bundle", stop)
    with pytest.raises(KeyboardInterrupt) as caught:
        evaluation.evaluate_checkpoint(
            archive, opts(), output_root=tmp_path / "outputs"
        )
    assert caught.value is interrupted
    root = interrupted.run_dir
    record = json.loads((root / "run.json").read_text())
    assert record["status"] == "interrupted" and record["stage"] == "model_import"
    assert (
        root / "evidence/imported-model/payload/checkpoint/checkpoint.json"
    ).is_file()
    assert record["results"] == []


def test_failure_metadata_error_does_not_replace_original_exception(
    saved, tmp_path, monkeypatch
):
    original = evaluation.write_json
    failure = RuntimeError("input preparation failed")

    def write(path, value):
        if value.get("status") == "failed":
            raise OSError("metadata directory unavailable")
        return original(path, value)

    def prepare(*args):
        raise failure

    monkeypatch.setattr(evaluation, "write_json", write)
    monkeypatch.setattr(evaluation, "_prepare", prepare)
    with pytest.raises(RuntimeError) as caught:
        evaluation.evaluate_checkpoint(
            saved[0], opts(), output_root=tmp_path / "outputs"
        )
    assert caught.value is failure
    assert failure.run_dir.is_dir()
    assert "failure metadata could not be saved" in failure.__notes__[0]


def test_outer_json_selection_precedes_preserved_materialized_alias(saved, tmp_path):
    source, checkpoint, _ = saved
    outer = tmp_path / "outer-current"
    old = outer / "checkpoints/last"
    new = outer / "evidence/training/current/checkpoints/update-000008"
    shutil.copytree(checkpoint, old)
    shutil.copytree(checkpoint, new / "inference")
    shutil.copy2(source / "resolved_training.json", new / "resolved_training.json")
    write_json(
        outer / "run.json",
        {
            "schema": "smartsom.experiment/v2",
            "paths": {"training": "evidence/training/current"},
        },
    )
    write_json(
        outer / "checkpoints/last.json",
        {"checkpoint": "../evidence/training/current/checkpoints/update-000008"},
    )
    original = (old / "checkpoint.json").read_bytes()
    selected = evaluation._select_checkpoint(outer)
    assert selected.directory == new / "inference"
    assert selected.snapshot == new / "resolved_training.json"
    assert (old / "checkpoint.json").read_bytes() == original


def test_unreadable_checkpoint_manifest_retains_selection_failure(saved, tmp_path):
    (saved[1] / "checkpoint.json").write_text("{invalid JSON")
    with pytest.raises(ConfigurationError, match="cannot read") as caught:
        evaluation.evaluate_checkpoint(
            saved[1], opts(), output_root=tmp_path / "outputs"
        )
    record = json.loads((caught.value.run_dir / "run.json").read_text())
    assert record["status"] == "failed" and record["stage"] == "checkpoint_selection"
    assert record["results"] == []


def test_replay_interruption_retains_completed_physical_run_reference(
    saved, tmp_path, monkeypatch
):
    def stop(*args):
        raise KeyboardInterrupt("replay interrupted")

    monkeypatch.setattr(evaluation, "audit_run", stop)
    with pytest.raises(KeyboardInterrupt) as caught:
        evaluation.evaluate_checkpoint(
            saved[0], opts(replications=2), output_root=tmp_path / "outputs"
        )
    root = caught.value.run_dir
    record = json.loads((root / "run.json").read_text())
    assert record["status"] == "interrupted" and record["pending"] == 1
    row = record["results"][0]
    assert row["status"] == "interrupted" and row["makespan"] is None
    assert row["replay"]["status"] == "interrupted"
    assert (root / row["run_dir"] / "trace.jsonl").is_file()
