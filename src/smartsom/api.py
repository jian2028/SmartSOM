"""Small public experiment API. Framework imports occur only when running them."""

import json
import os
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from smartsom.config.codec import ConfigurationError, canonical_json, digest
from smartsom.config.experiment import (
    EvaluationOptions,
    ExperimentConfig,
    PreparedExperiment,
    load_config,
    load_preset,
    prepare,
    preview,
    training_identity,
)
from smartsom.experiments.evidence import source_identity, write_json

__all__ = [
    "ExperimentConfig",
    "EvaluationOptions",
    "load_config",
    "load_preset",
    "show_config",
    "train",
    "train_prepared",
    "run",
    "evaluate",
    "train_evaluate",
    "resume",
    "batch_train",
    "search",
    "SimulationRunResult",
]


@dataclass(frozen=True)
class TrainingResult:
    run_dir: Path
    training_dir: Path
    last_checkpoint: Path | None
    best_checkpoint: Path | None
    environment_steps: int
    learner_updates: int
    ppo_updates: int
    status: str


@dataclass(frozen=True)
class ExperimentResult:
    training: TrainingResult
    evaluation: object | None


@dataclass(frozen=True)
class SimulationRunResult:
    run_dir: Path
    evidence_dir: Path
    simulation_result: object
    status: str = "completed"


def show_config(config: ExperimentConfig) -> dict:
    return preview(config)


def _allocate(config: ExperimentConfig, prepared, kind: str):
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    identity = f"{stamp}-{config.output.name}-{uuid4().hex[:10]}"
    root = Path(config.output.root).resolve() / identity
    root.mkdir(parents=True)
    for name in ("config", "logs", "checkpoints", "evaluation", "evidence", "reports"):
        (root / name).mkdir()
    (root / "config/experiment.json").write_text(prepared.config_json + "\n")
    (root / "config/origins.json").write_text(prepared.origins_json + "\n")
    if getattr(prepared, "validation_json", None) is not None:
        (root / "config/validation_inputs.json").write_text(
            prepared.validation_json + "\n"
        )
    record = {
        "schema": "smartsom.experiment/v2",
        "id": identity,
        "name": config.output.name,
        "kind": kind,
        "status": "running",
        "created_at": datetime.now(UTC).isoformat(),
        "tags": list(config.output.tags),
        "seed": config.seed,
        "scientific_sha256": prepared.scientific_sha256,
        "provider": prepared.resolved.algorithm.algorithm.provider,
        "source": source_identity(),
        "paths": {"config": "config/experiment.json", "logs": "logs"},
    }
    write_json(root / "run.json", record)
    return root, record


def _finish(root, record, status, **fields):
    record.update(status=status, **fields)
    record["finished_at"] = datetime.now(UTC).isoformat()
    write_json(root / "run.json", record)


def _training_controls(config, initialize_from=None, *, validation_json=None):
    from smartsom.experiments.training_controls import (
        TrainingControls,
        ValidationControls,
    )

    validation = (
        ValidationControls(
            **{
                k: getattr(config.validation, k)
                for k in type(config.validation).model_fields
                if k != "enabled"
            }
        )
        if config.validation.enabled
        else None
    )
    return TrainingControls(
        checkpoint_every_updates=config.checkpointing.every_updates,
        keep_last=config.checkpointing.keep_last,
        save_last=config.checkpointing.save_last,
        save_best=config.checkpointing.save_best,
        initialize_from=Path(initialize_from).resolve() if initialize_from else None,
        validation=validation,
        device=config.runtime.device,
        numerical_threads=config.runtime.numerical_threads,
        num_envs=config.runtime.num_envs,
        sampling_processes=config.runtime.sampling_processes,
        validation_inputs_json=validation_json,
    )


def train(
    config: ExperimentConfig,
    *,
    initialize_from: str | Path | None = None,
    on_progress=None,
) -> TrainingResult:
    prepared = prepare(config, require_dependencies=True)
    return train_prepared(
        prepared, initialize_from=initialize_from, on_progress=on_progress
    )


def train_prepared(
    prepared: PreparedExperiment, *, initialize_from=None, on_progress=None
) -> TrainingResult:
    """Execute a verified frozen recipe without rereading authoring paths."""
    from smartsom.config.snapshots import validate_resolved
    from smartsom.config.training import ResolvedTrainingRun
    from smartsom.learning.checkpoint import require_backend

    if not isinstance(prepared, PreparedExperiment) or not isinstance(
        prepared.resolved, ResolvedTrainingRun
    ):
        raise TypeError("train_prepared requires a frozen training recipe")
    # Freeze a detached copy, so edits to the caller's object cannot alter a run.
    config = ExperimentConfig.model_validate_json(prepared.config_json)
    validate_resolved(prepared.resolved.base)
    if (
        digest(
            training_identity(
                prepared.resolved, config.runtime, prepared.validation_json
            )
        )
        != prepared.scientific_sha256
    ):
        raise ConfigurationError("frozen training input identity mismatch")
    if (
        prepared.resolved.run.seed != config.seed
        or prepared.resolved.run.budget.environment_steps != config.training.total_steps
        or prepared.resolved.algorithm.algorithm.parameters.n_steps
        != config.training.steps_per_update
    ):
        raise ConfigurationError(
            "frozen recipe disagrees with the recorded configuration"
        )
    require_backend(prepared.resolved.algorithm.algorithm.provider)
    if bool(config.validation.enabled and config.validation.scenarios) != (
        prepared.validation_json is not None
    ):
        raise ConfigurationError(
            "external validation cases require their frozen input snapshot"
        )
    controls = _training_controls(config, validation_json=prepared.validation_json)
    if config.logging.tensorboard or config.logging.wandb:
        from smartsom.telemetry.training import TrainingDisplay

        TrainingDisplay.preflight(config.logging)
    root, record = _allocate(config, prepared, "training")
    resolved = replace(
        prepared.resolved,
        run=prepared.resolved.run.model_copy(
            update={"output_root": str(root / "evidence/training")}
        ),
    )
    try:
        if initialize_from is not None:
            from smartsom.experiments.packaging import import_bundle, model_locator

            source = Path(initialize_from).resolve()
            if source.is_file():
                source = import_bundle(source, root / "evidence/initialized-model")
            controls = replace(controls, initialize_from=model_locator(source))
            record["initialize_from"] = str(initialize_from)
        return _execute_training(root, record, config, resolved, controls, on_progress)
    except BaseException as exc:
        _record_failure(root, record, exc)
        raise


def _record_failure(root, record, exc):
    """Keep the original exception when writing failure metadata also fails."""
    try:
        if getattr(exc, "run_dir", None):
            record["paths"]["training"] = str(Path(exc.run_dir).relative_to(root))
        _finish(
            root,
            record,
            "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
            failure={"exception": type(exc).__name__, "message": str(exc)},
        )
    except Exception as write_error:
        exc.add_note(f"failure metadata could not be saved: {write_error!r}")


def _execute_training(root, record, config, resolved, controls, on_progress=None):
    from smartsom.experiments.training import train_one
    from smartsom.telemetry.training import TrainingDisplay

    try:
        with TrainingDisplay(root, config, on_progress=on_progress) as display:
            kwargs = {"on_progress": display}
            if controls is not None:
                kwargs["controls"] = controls
            trained = train_one(resolved, **kwargs)
        record["paths"]["training"] = str(trained.run_dir.relative_to(root))
        last = trained.last_checkpoint
        best = getattr(trained, "best_checkpoint", None)
        for name, target in (("last", last), ("best", best)):
            if target is not None:
                target = Path(target).resolve()
                link = root / "checkpoints" / name
                relative = Path("../") / target.relative_to(root)
                write_json(link.with_suffix(".json"), {"checkpoint": str(relative)})
                # Portable archives materialize old shortcut directories. Keep
                # those historical bytes; the current JSON reference supersedes them.
                if link.is_dir() and not link.is_symlink():
                    continue
                temporary = link.with_name(f".{name}-{uuid4().hex}.tmp")
                temporary.symlink_to(relative, target_is_directory=True)
                os.replace(temporary, link)
        if last:
            record["paths"]["checkpoint"] = str(Path(last).relative_to(root))
        status = getattr(trained, "status", "completed")
        _finish(root, record, status)
        return TrainingResult(
            root,
            trained.run_dir,
            last,
            best,
            trained.environment_steps,
            trained.learner_updates,
            trained.environment_steps // config.training.steps_per_update,
            status,
        )
    except BaseException as exc:
        _record_failure(root, record, exc)
        raise


def run(config: ExperimentConfig, *, on_progress=None) -> SimulationRunResult:
    from smartsom.experiments.runner import run_one

    prepared = prepare(config, training=False, require_dependencies=True)
    frozen = ExperimentConfig.model_validate_json(prepared.config_json)
    root, record = _allocate(frozen, prepared, "simulation")
    resolved = replace(
        prepared.resolved,
        run=prepared.resolved.run.model_copy(
            update={"output_root": str(root / "evidence/runs")}
        ),
    )
    try:
        result = run_one(resolved, on_progress=on_progress)
        record["paths"]["evaluation"] = str(result.run_dir.relative_to(root))
        _finish(root, record, "completed", makespan=result.simulation_result.makespan)
        return SimulationRunResult(root, result.run_dir, result.simulation_result)
    except BaseException as exc:
        try:
            if getattr(exc, "run_dir", None):
                record["paths"]["evaluation"] = str(exc.run_dir.relative_to(root))
            _finish(
                root,
                record,
                "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                failure={"exception": type(exc).__name__, "message": str(exc)},
            )
        except Exception as write_error:
            exc.add_note(f"failure metadata could not be saved: {write_error!r}")
        raise


def evaluate(
    source: str | Path,
    config: EvaluationOptions | None = None,
    *,
    output_root: str | Path | None = None,
):
    from smartsom.experiments.evaluation import evaluate_checkpoint

    return evaluate_checkpoint(
        source, config or EvaluationOptions(), output_root=output_root
    )


def train_evaluate(
    config: ExperimentConfig, *, initialize_from=None
) -> ExperimentResult:
    frozen = ExperimentConfig.model_validate_json(canonical_json(config))
    frozen._origins = config.origins()
    frozen._baseline = config._baseline.copy()
    frozen._owner = config._owner
    if frozen.evaluation.checkpoint == "last" and not frozen.checkpointing.save_last:
        raise ConfigurationError(
            "evaluation selects last but checkpointing.save_last is false"
        )
    if frozen.evaluation.checkpoint == "best" and not (
        frozen.validation.enabled and frozen.checkpointing.save_best
    ):
        raise ConfigurationError(
            "evaluation selects best but best saving/validation is disabled"
        )
    trained = train(frozen, initialize_from=initialize_from)
    evaluated = None
    if trained.status in {"completed", "early_stopped"}:
        record = json.loads((trained.run_dir / "run.json").read_text())
        record["kind"] = "train_evaluate"
        record["training_status"] = trained.status
        record["status"] = record["evaluation_status"] = "running"
        write_json(trained.run_dir / "run.json", record)
        try:
            evaluated = evaluate(
                trained.run_dir,
                frozen.evaluation,
                output_root=trained.run_dir / "evaluation",
            )
        except BaseException as exc:
            try:
                directory = getattr(exc, "run_dir", None)
                if directory is not None:
                    record["paths"]["evaluation"] = str(
                        Path(directory).relative_to(trained.run_dir)
                    )
                status = (
                    "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
                )
                record["evaluation_status"] = status
                _finish(
                    trained.run_dir,
                    record,
                    status,
                    failure={"exception": type(exc).__name__, "message": str(exc)},
                )
            except Exception as write_error:
                exc.add_note(
                    f"combined evaluation failure metadata could not be saved: {write_error!r}"
                )
            raise
        record["paths"]["evaluation"] = str(
            evaluated.run_dir.relative_to(trained.run_dir)
        )
        record["evaluation_status"] = evaluated.status
        record["status"] = (
            evaluated.status if evaluated.status != "completed" else trained.status
        )
        write_json(trained.run_dir / "run.json", record)
    return ExperimentResult(trained, evaluated)


def resume(source: str | Path, *, on_progress=None):
    from smartsom.experiments.packaging import model_locator
    from smartsom.experiments.training_lifecycle import load_resumable_training

    source = Path(source).resolve()
    roots = [p for p in (source, *source.parents) if (p / "run.json").is_file()]
    if not roots:
        raise ConfigurationError(
            "resume requires a v2 experiment with update checkpoints; use initialize-from for model-only artifacts"
        )
    root = roots[0]
    record = json.loads((root / "run.json").read_text())
    if record.get("schema") != "smartsom.experiment/v2":
        raise ConfigurationError("unsupported experiment identity")
    checkpoint = model_locator(source)
    if checkpoint.name == "inference":
        checkpoint = checkpoint.parent
    resolved = load_resumable_training(checkpoint)
    resolved = replace(
        resolved,
        run=resolved.run.model_copy(
            update={"output_root": str(root / "evidence/training")}
        ),
    )
    config = ExperimentConfig.model_validate_json(
        (root / "config/experiment.json").read_text()
    )
    validation_path = root / "config/validation_inputs.json"
    controls = replace(
        _training_controls(
            config,
            validation_json=validation_path.read_text()
            if validation_path.is_file()
            else None,
        ),
        resume_from=checkpoint,
    )
    record.setdefault("attempts", []).append(
        {
            "status": record["status"],
            "paths": dict(record["paths"]),
            "resumed_from": str(checkpoint.relative_to(root)),
        }
    )
    record["status"] = "running"
    record.pop("failure", None)
    write_json(root / "run.json", record)
    return _execute_training(root, record, config, resolved, controls, on_progress)


def batch_train(
    configs=None,
    *,
    output_root=None,
    max_concurrent=1,
    resume=None,
    retry_failed=False,
    on_progress=None,
):
    from smartsom.experiments.learning_study import run_learning_batch

    return run_learning_batch(
        configs,
        output_root=output_root,
        max_concurrent=max_concurrent,
        resume=resume,
        retry_failed=retry_failed,
        on_progress=on_progress,
    )


def search(config=None, *, resume=None, retry_failed=False, on_progress=None):
    from smartsom.experiments.learning_study import search as execute_search

    return execute_search(
        config, resume=resume, retry_failed=retry_failed, on_progress=on_progress
    )
