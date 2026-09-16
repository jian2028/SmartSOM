"""Small public experiment API. Framework imports occur only when running them."""

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from smartsom.config.codec import ConfigurationError, canonical_json
from smartsom.config.experiment import (
    EvaluationOptions,
    ExperimentConfig,
    PreparedExperiment,
    load_config,
    load_preset,
    prepare,
    preview,
)
from smartsom.experiments.evidence import source_identity, write_json

_UNSET = object()

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
        "provider": getattr(
            prepared.resolved.algorithm, "algorithm", prepared.resolved.algorithm
        ).provider,
        "paths": {"config": "config/experiment.json", "logs": "logs"},
    }
    try:
        root.mkdir(parents=True)
        write_json(root / "run.json", record)
        for name in (
            "config",
            "logs",
            "checkpoints",
            "evaluation",
            "evidence",
            "reports",
        ):
            (root / name).mkdir()
        (root / "config/experiment.json").write_text(prepared.config_json + "\n")
        (root / "config/origins.json").write_text(prepared.origins_json + "\n")
        if getattr(prepared, "validation_json", None) is not None:
            (root / "config/validation_inputs.json").write_text(
                prepared.validation_json + "\n"
            )
        record["source"] = source_identity()
        write_json(root / "run.json", record)
    except BaseException as exc:
        if root.is_dir():
            exc.run_dir = root
            try:
                _finish(
                    root,
                    record,
                    "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                    failure={"exception": type(exc).__name__, "message": str(exc)},
                )
            except Exception as metadata_error:
                exc.add_note(f"failure metadata could not be saved: {metadata_error}")
            if kind == "training" and isinstance(exc, Exception):
                from smartsom.experiments.training import TrainingFailedError

                raise TrainingFailedError(root, exc) from exc
        raise
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
    from smartsom.config.production import ProductionRecipe
    from smartsom.experiments.production_training import train_prepared as execute

    if not isinstance(prepared, PreparedExperiment) or not isinstance(
        prepared.resolved, ProductionRecipe
    ):
        raise TypeError(
            "train_prepared requires a frozen grid experiment recipe; old model encodings require retraining"
        )
    return execute(prepared, initialize_from=initialize_from, on_progress=on_progress)


def run(
    config: ExperimentConfig,
    *,
    on_progress=None,
    render_mode=None,
    verbose=None,
    record=True,
) -> SimulationRunResult:

    if isinstance(config, (str, Path)):
        config = load_config(config)
    prepared = prepare(config, training=False, require_dependencies=True)
    from smartsom.experiments.runner import run_one

    result = run_one(
        prepared,
        on_progress=on_progress,
        render_mode=render_mode,
        verbose=verbose,
        record=record,
    )
    return SimulationRunResult(
        result.run_dir,
        result.run_dir,
        result.simulation_result,
        result.simulation_result.status,
    )


def evaluate(
    source: str | Path,
    config: EvaluationOptions | None = None,
    *,
    output_root: str | Path | None = None,
    render_mode=_UNSET,
    verbose=None,
    record=None,
    render_case=_UNSET,
    render_replication=None,
):
    from smartsom.experiments.production_evaluation import (
        evaluate as evaluate_checkpoint,
    )

    options = (config or EvaluationOptions()).model_dump(mode="json", by_alias=True)
    for key, value in {
        "verbose": verbose,
        "record": record,
        "render_replication": render_replication,
    }.items():
        if value is not None:
            options[key] = value
    for key, value in {"render_mode": render_mode, "render_case": render_case}.items():
        if value is not _UNSET:
            options[key] = value
    options = EvaluationOptions.model_validate_json(json.dumps(options))

    return evaluate_checkpoint(source, options, output_root=output_root)


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
    from smartsom.experiments.production_training import resume as execute

    return execute(source, on_progress=on_progress)


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
