"""Public workflow contracts survive replacing the underlying physics."""

import json
from pathlib import Path

import pytest

from smartsom.api import (
    EvaluationOptions,
    evaluate,
    load_config,
    prepare,
    resume,
    run,
    train,
)
from smartsom.config.codec import digest
from smartsom.experiments.packaging import export_model, verify_bundle


def recipe(name, root):
    config = load_config(f"configs/runs/{name}_production.yaml")
    config.output.root = str(root)
    config.training.total_steps = 128
    config.training.steps_per_update = 64
    config.algorithm.batch_size = 32
    config.algorithm.hidden_sizes = (8,)
    config.algorithm.n_epochs = 2
    config.validation.every_updates = 1
    config.validation.replications = 1
    config.logging.verbose = False
    config.runtime.num_envs = 2
    return config


@pytest.mark.learning
@pytest.mark.parametrize("backend", ["sb3", "rllib", "marl"])
def test_public_resume_matches_uninterrupted_weights(tmp_path, backend):
    torch = pytest.importorskip("torch")
    pytest.importorskip("sb3_contrib" if backend == "sb3" else "ray")
    config = recipe(backend, tmp_path)
    full = train(config)
    stopped = train(
        config,
        on_progress=lambda row: (
            {"stop": "pruned"} if row["stage"] == "validation" else None
        ),
    )
    assert stopped.status == "pruned" and stopped.environment_steps == 64
    continued = resume(stopped.run_dir)
    assert continued.run_dir == stopped.run_dir
    assert continued.status == "completed" and continued.environment_steps == 128
    if backend == "sb3":
        from sb3_contrib import MaskablePPO

        a = MaskablePPO.load(full.last_checkpoint / "model.zip").policy.state_dict()
        b = MaskablePPO.load(
            continued.last_checkpoint / "model.zip"
        ).policy.state_dict()
        assert a.keys() == b.keys()
        assert all(torch.equal(a[key], b[key]) for key in a)
    else:
        for member in full.last_checkpoint.glob("*.pt"):
            a = torch.load(member, weights_only=True)
            b = torch.load(continued.last_checkpoint / member.name, weights_only=True)
            assert a.keys() == b.keys()
            assert all(torch.equal(a[key], b[key]) for key in a), member.name
    full_rows = (full.training_dir / "episodes.jsonl").read_text()
    resumed_rows = (continued.training_dir / "episodes.jsonl").read_text()
    assert full_rows == resumed_rows
    from smartsom.experiments.training_audit import audit_training

    audit = audit_training(continued.run_dir)
    assert audit["environment_steps"] == 128 and audit["status"] == "passed"
    with pytest.raises(ValueError, match="budget exhausted"):
        resume(continued.run_dir)


def test_run_result_and_independent_recording(tmp_path):
    config = load_config("configs/runs/production_hand.yaml")
    config.output.root = str(tmp_path)
    recorded = run(config, verbose=False)
    plain = run(config, verbose=False, record=False)
    assert recorded.status == plain.status == "completed"
    assert recorded.simulation_result.makespan == plain.simulation_result.makespan == 8
    assert recorded.simulation_result.final_state == plain.simulation_result.final_state
    assert {p.name for p in plain.run_dir.iterdir()} == {"run.json"}


@pytest.mark.learning
def test_model_bundle_and_evaluation_without_trace(tmp_path):
    pytest.importorskip("sb3_contrib")
    config = recipe("sb3", tmp_path)
    result = train(config)
    archive = export_model(result.run_dir, tmp_path / "model.zip")
    assert verify_bundle(archive)["kind"] == "model"
    evaluated = evaluate(
        archive,
        EvaluationOptions(
            replications=2, baselines=("spt",), record=False, verbose=False
        ),
        output_root=tmp_path,
    )
    assert len(evaluated.results) == 4
    for row in evaluated.results:
        directory = evaluated.run_dir / row["run_dir"]
        assert {p.name for p in directory.iterdir()} == {"run.json"}
        assert row["replay"]["status"] == (
            "passed" if row["status"] == "completed" else "partial_verified"
        )
    groups = {}
    for row in evaluated.results:
        manifest = json.loads(
            (evaluated.run_dir / row["run_dir"] / "run.json").read_text()
        )
        groups.setdefault(row["replication"], []).append(
            digest(manifest["inputs"]["scenario"])
        )
    assert all(len(set(values)) == 1 for values in groups.values())
    from smartsom.experiments.report import load_report_data

    report = load_report_data(evaluated.run_dir)
    coverage = report["evaluations"][0]
    assert not coverage["warnings"]
    assert all(
        row["missing"] == 0 and row["requested"] == 2 for row in coverage["coverage"]
    )
    assert len(coverage["pairs"]) == 2
    assert all(
        row["status"] in ("paired", "not_completed") for row in coverage["pairs"]
    )
    assert all(row["training_seed"] == config.seed for row in coverage["pairs"])


def test_frozen_recipe_does_not_reopen_authoring_files(tmp_path):
    from smartsom.config.experiment import prepare_frozen

    config = load_config("configs/runs/sb3_production.yaml")
    prepared = prepare(config)
    candidate = config.model_copy(deep=True)
    candidate.algorithm.learning_rate = 0.001
    candidate.output.root = str(tmp_path)
    rebound = prepare_frozen(candidate, prepared)
    assert rebound.resolved.scenario == prepared.resolved.scenario
    assert rebound.resolved.algorithm.learning_rate == 0.001
    assert rebound.scientific_sha256 != prepared.scientific_sha256


@pytest.mark.learning
@pytest.mark.parametrize("backend", ["sb3", "rllib", "marl"])
def test_process_sampling_preserves_logical_streams(tmp_path, backend):
    torch = pytest.importorskip("torch")
    pytest.importorskip("sb3_contrib" if backend == "sb3" else "ray")
    config = recipe(backend, tmp_path)
    config.training.total_steps = 64
    config.validation.enabled = False
    sequential = train(config)
    config.runtime.sampling_processes = 2
    parallel = train(config)
    assert sequential.environment_steps == parallel.environment_steps == 64
    assert (sequential.training_dir / "episodes.jsonl").read_text() == (
        parallel.training_dir / "episodes.jsonl"
    ).read_text()
    if backend == "sb3":
        from sb3_contrib import MaskablePPO

        a = MaskablePPO.load(
            sequential.last_checkpoint / "model.zip"
        ).policy.state_dict()
        b = MaskablePPO.load(parallel.last_checkpoint / "model.zip").policy.state_dict()
        assert all(torch.equal(a[key], b[key]) for key in a)
    else:
        for member in sequential.last_checkpoint.glob("*.pt"):
            a = torch.load(member, weights_only=True)
            b = torch.load(parallel.last_checkpoint / member.name, weights_only=True)
            assert all(torch.equal(a[key], b[key]) for key in a), member.name


@pytest.mark.learning
@pytest.mark.parametrize("backend", ["sb3", "rllib", "marl"])
def test_grid_custom_hooks_resume_and_inference(tmp_path, backend):
    pytest.importorskip("sb3_contrib" if backend == "sb3" else "ray")
    from test_training_extensions import research_spec

    from smartsom.experiments.training_audit import audit_training

    config = recipe(backend, tmp_path)
    config.algorithm.extensions = research_spec(backend == "marl")
    full = train(config)
    stopped = train(
        config,
        on_progress=lambda event: (
            {"stop": "pruned"} if event["stage"] == "validation" else None
        ),
    )
    continued = resume(stopped.run_dir)
    assert (full.training_dir / "episodes.jsonl").read_text() == (
        continued.training_dir / "episodes.jsonl"
    ).read_text()
    assert audit_training(continued.run_dir)["environment_steps"] == 128
    result = evaluate(
        full.run_dir,
        EvaluationOptions(replications=1, record=False, verbose=False),
        output_root=tmp_path,
    )
    assert len(result.results) == 1 and result.results[0]["replay"]["status"] == (
        "passed" if result.results[0]["status"] == "completed" else "partial_verified"
    )


def test_grid_batch_pairs_worlds_restores_and_detects_changed_results(
    tmp_path, monkeypatch
):
    from pathlib import Path

    import smartsom.config.production as production
    from smartsom.config import resolve_study
    from smartsom.config.study import semantic_run
    from smartsom.experiments.batch import run_batch

    materializations = []
    materialize = production.materialize

    def counted(*args):
        materializations.append(args[-1])
        return materialize(*args)

    monkeypatch.setattr(production, "materialize", counted)

    source = Path("configs/runs/production_hand.yaml").resolve().parents[2]
    path = tmp_path / "study.yaml"
    path.write_text(
        json.dumps(
            {
                "schema": "smartsom.study/v1",
                "seed": 17,
                "replications": 2,
                "cases": [
                    {
                        "id": "simple",
                        "scenario": str(
                            source / "configs/scenarios/production_hand.yaml"
                        ),
                    }
                ],
                "algorithms": [
                    {
                        "id": "spt",
                        "config": str(source / "configs/algorithms/spt.yaml"),
                    },
                    {
                        "id": "first",
                        "config": str(
                            source / "configs/algorithms/first_feasible.yaml"
                        ),
                    },
                ],
                "output_root": str(tmp_path / "runs"),
            }
        )
    )
    study = resolve_study(path)
    assert len(study.entries) == 4
    assert len(materializations) == 2
    for replication in range(2):
        rows = [entry for entry in study.entries if entry.replication == replication]
        assert len({digest(entry.resolved.resolved.scenario) for entry in rows}) == 1
        assert len({entry.resolved.resolved.algorithm_seed for entry in rows}) == 2
    serial = run_batch(study, workers=1)
    parallel = run_batch(study, workers=2)
    assert serial.completed == parallel.completed == 4
    assert serial.failed == parallel.failed == 0
    assert run_batch(resume=serial.study_dir).completed == 4
    summary = json.loads((serial.study_dir / "summary.json").read_text())
    child = Path(summary["runs"][0]["run_dir"])
    assert {p.name for p in child.iterdir()} == {"run.json", "trace.jsonl"}
    manifest = json.loads((child / "run.json").read_text())
    manifest["result"]["tick"] += 1
    (child / "run.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="result differs"):
        run_batch(resume=serial.study_dir)
    assert (
        semantic_run(study.entries[0].resolved)["scenario"]["seed"]
        == study.entries[0].resolved.resolved.scenario.seed
    )


@pytest.mark.learning
@pytest.mark.parametrize("backend", ["sb3", "rllib", "marl"])
def test_direct_training_controls_stop_resume_and_disable_saving(tmp_path, backend):
    from dataclasses import replace

    from smartsom.experiments.training import train_one
    from smartsom.experiments.training_controls import TrainingControls

    pytest.importorskip("sb3_contrib" if backend == "sb3" else "ray")
    config = recipe(backend, tmp_path)
    config.validation.enabled = False
    prepared = prepare(config)
    controls = TrainingControls(
        checkpoint_every_updates=1,
        validation=None,
        save_best=False,
        stop_after_updates=1,
        num_envs=2,
    )
    stopped = train_one(prepared, controls=controls)
    assert stopped.status == "interrupted" and stopped.environment_steps == 64
    continued = train_one(
        prepared,
        controls=replace(
            controls, stop_after_updates=None, resume_from=stopped.last_checkpoint
        ),
    )
    assert continued.status == "completed" and continued.environment_steps == 128
    assert continued.run_dir == stopped.run_dir
    discarded = train_one(
        prepared, controls=replace(controls, save_last=False, save_best=False)
    )
    assert discarded.status == "interrupted" and discarded.environment_steps == 64
    assert discarded.checkpoint_dir is None and discarded.last_checkpoint is None
    assert not list((discarded.run_dir / "checkpoints").glob("update-*"))
    assert not list((discarded.run_dir / "evidence").rglob("model.zip"))
    assert not list((discarded.run_dir / "evidence").rglob("*.pt"))


@pytest.mark.learning
def test_external_evaluation_uses_frozen_algorithm_and_keeps_failed_cases(
    tmp_path, monkeypatch
):
    from pathlib import Path
    from types import SimpleNamespace

    import smartsom.experiments.production as production
    from smartsom.experiments.evaluation import evaluate_checkpoint

    pytest.importorskip("sb3_contrib")
    config = recipe("sb3", tmp_path)
    algorithm_source = tmp_path / "algorithm.yaml"
    algorithm_source.write_bytes(Path(config.algorithm.source).read_bytes())
    config.algorithm.source = str(algorithm_source)
    config.training.total_steps = 64
    config.validation.enabled = False
    trained = train(config)
    algorithm_source.unlink()
    execute = production.run
    failures = []

    def fail_once(*args, **kwargs):
        if kwargs.get("policy") is not None and not failures:
            failures.append(True)
            raise RuntimeError("injected inference failure")
        return execute(*args, **kwargs)

    monkeypatch.setattr(production, "run", fail_once)
    result = evaluate_checkpoint(
        trained.run_dir,
        SimpleNamespace(
            replications=2,
            scenarios=(config.scenario,),
            baselines=("spt",),
            verbose=False,
        ),
        output_root=tmp_path,
    )
    assert result.status == "failed" and result.engineering_failures == 1
    assert len(result.results) == 4
    assert (
        result.results[0]["engineering_failure"]
        and result.results[0]["run_dir"] is None
    )
    assert result.results[0]["reason"] == "RuntimeError"
    assert all(
        row["replay"]["status"]
        == ("passed" if row["status"] == "completed" else "partial_verified")
        for row in result.results[1:]
    )
    assert (
        sum(
            row["status"] == "completed"
            for row in result.results
            if row["algorithm_id"] == "builtin.spt"
        )
        == 2
    )


@pytest.mark.learning
def test_checkpoint_baselines_are_paired_auditable_and_protected(tmp_path):
    pytest.importorskip("sb3_contrib")
    from smartsom.experiments.audit import audit_run

    config = recipe("sb3", tmp_path)
    config.training.total_steps = 64
    config.validation.enabled = False
    first = train(config)
    config.seed += 1
    second = train(config)
    result = evaluate(
        first.run_dir,
        EvaluationOptions(
            replications=2,
            baselines=(f"checkpoint:{second.run_dir}", "spt"),
            verbose=False,
        ),
        output_root=tmp_path,
    )
    assert len(result.results) == 6 and result.engineering_failures == 0
    model = [r for r in result.results if r["algorithm_id"] == "model"]
    baseline = [
        r for r in result.results if r["algorithm_id"].startswith("checkpoint-")
    ]
    assert len(model) == len(baseline) == 2
    assert (
        model[0]["checkpoint"]["manifest_sha256"]
        != baseline[0]["checkpoint"]["manifest_sha256"]
    )
    assert [r["world_sha256"] for r in model] == [r["world_sha256"] for r in baseline]
    assert audit_run(result.run_dir)["status"] == (
        "passed" if result.failed == 0 else "partial_verified"
    )
    for checkpoint in (first.last_checkpoint, second.last_checkpoint):
        assert (checkpoint.parent / "references" / checkpoint.name).is_dir()
    manifest_path = result.run_dir / "run.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["results"][0]["makespan"] = 999
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="differs from child evidence"):
        audit_run(result.run_dir)
    unrecorded = evaluate(
        first.run_dir,
        EvaluationOptions(replications=1, record=False, verbose=False),
        output_root=tmp_path,
    )
    assert audit_run(unrecorded.run_dir)["status"] == "partial_verified"


@pytest.mark.learning
@pytest.mark.parametrize("backend", ["sb3", "rllib", "marl"])
def test_initialize_from_carries_observation_state_into_fresh_training(
    tmp_path, backend
):
    pytest.importorskip("sb3_contrib" if backend == "sb3" else "ray")
    from test_training_extensions import research_spec

    from smartsom.experiments.training_audit import audit_training

    config = recipe(backend, tmp_path)
    config.training.total_steps = 64
    config.validation.enabled = False
    config.algorithm.extensions = research_spec(backend == "marl")
    first = train(config)
    saved = json.loads((first.last_checkpoint / "extension_state.json").read_text())
    second = train(config, initialize_from=first.run_dir)
    assert second.run_dir != first.run_dir
    assert second.environment_steps == 64
    rows = [
        json.loads(line)
        for line in (second.training_dir / "episodes.jsonl").read_text().splitlines()
    ]
    rows.extend(
        json.loads((second.last_checkpoint / "active_episodes.json").read_text())
    )
    initial = next(
        row["extension_state"]["initial"] for row in rows if row["episode"] == 0
    )
    expected = {(r["kind"], r["role"]): r for r in saved["components"]}
    for row in initial["components"]:
        if row["kind"] == "observation":
            assert row == expected[(row["kind"], row["role"])]
        elif row["kind"] == "reward":
            assert row["state"] == {"calls": 0, "episodes": 0}
    assert audit_training(second.run_dir)["status"] == "passed"


@pytest.mark.learning
def test_exported_algorithm_remains_reusable_by_original_run_entry(tmp_path):
    pytest.importorskip("sb3_contrib")
    from smartsom.experiments.audit import audit_run
    from smartsom.experiments.packaging import import_bundle

    config = recipe("sb3", tmp_path)
    config.training.total_steps = 64
    config.validation.enabled = False
    trained = train(config)
    archive = export_model(trained.run_dir, tmp_path / "model.zip")
    imported = import_bundle(archive, tmp_path / "imported")
    config.algorithm.source = str(imported / "checkpoint_algorithm.json")
    prepared = prepare(config, training=False)
    assert (
        prepared.resolved.algorithm.max_jobs
        == json.loads((trained.last_checkpoint / "checkpoint.json").read_text())[
            "algorithm"
        ]["max_jobs"]
    )
    assert Path(prepared.resolved.algorithm.checkpoint).is_relative_to(imported)
    result = run(config, verbose=False)
    manifest = json.loads((result.run_dir / "run.json").read_text())
    assert manifest["checkpoint"]["path"] == prepared.resolved.algorithm.checkpoint
    assert (
        manifest["learning_contract"]["algorithm"]["max_jobs"]
        == json.loads((trained.last_checkpoint / "checkpoint.json").read_text())[
            "algorithm"
        ]["max_jobs"]
    )
    assert audit_run(result.run_dir)["status"] == (
        "passed" if result.status == "completed" else "partial_verified"
    )
    # Direct run controls override checkpoint defaults and remain auditable even
    # when a decision budget stops before the first physical tick.
    config.training.max_decisions = 1
    config.logging.observations = "full"
    limited = run(config, verbose=False)
    manifest = json.loads((limited.run_dir / "run.json").read_text())
    assert limited.status == "truncated"
    assert manifest["learning_contract"]["limits"]["max_decisions"] == 1
    assert manifest["learning_contract"]["observations"] == "full"
    assert audit_run(limited.run_dir)["status"] == "partial_verified"
    with pytest.raises(ValueError, match="without a checkpoint"):
        prepare(config)


@pytest.mark.learning
def test_stochastic_direct_run_reuses_explicit_algorithm_seed(tmp_path):
    pytest.importorskip("sb3_contrib")
    from smartsom.experiments.packaging import import_bundle
    from smartsom.experiments.runner import run_one

    config = recipe("sb3", tmp_path)
    config.training.total_steps = 64
    config.validation.enabled = False
    trained = train(config)
    package = import_bundle(
        export_model(trained.run_dir, tmp_path / "model.zip"), tmp_path / "imported"
    )
    config.algorithm.source = str(package / "checkpoint_algorithm.json")
    frozen = prepare(config, training=False)
    a = run_one(frozen, verbose=False, deterministic=False)
    b = run_one(frozen, verbose=False, deterministic=False)
    assert a.simulation_result == b.simulation_result
    assert (a.run_dir / "trace.jsonl").read_bytes() == (
        b.run_dir / "trace.jsonl"
    ).read_bytes()
