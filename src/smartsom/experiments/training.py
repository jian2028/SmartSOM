"""Public training entry and controls for the shared grid experiment lifecycle."""

from dataclasses import dataclass
from pathlib import Path

from smartsom.config.codec import digest, primitive


@dataclass(frozen=True, slots=True)
class TrainingResult:
    run_dir: Path
    checkpoint_dir: Path | None
    environment_steps: int
    learner_updates: int
    status: str = "completed"
    last_checkpoint: Path | None = None
    best_checkpoint: Path | None = None


class TrainingFailedError(RuntimeError):
    def __init__(self, run_dir: Path, cause: BaseException):
        self.run_dir, self.cause = run_dir, cause
        super().__init__(f"training failed in {run_dir}: {cause}")


def apply_training_controls(prepared, controls):
    """Bind direct-entry controls identically for a real train and a startup probe."""
    from dataclasses import replace

    from smartsom.config.codec import canonical_json
    from smartsom.config.experiment import ExperimentConfig, PreparedExperiment
    from smartsom.config.production import ProductionRecipe, recipe_identity

    if not isinstance(prepared, PreparedExperiment) or not isinstance(
        prepared.resolved, ProductionRecipe
    ):
        raise TypeError(
            "train_one requires a prepared grid experiment; old action and observation encodings require retraining"
        )
    config = ExperimentConfig.model_validate_json(prepared.config_json)
    validation_json = prepared.validation_json
    if controls is not None:
        from smartsom.experiments.training_controls import TrainingControls

        if not isinstance(controls, TrainingControls):
            raise TypeError("controls requires TrainingControls")
        for name in ("device", "numerical_threads", "num_envs", "sampling_processes"):
            setattr(config.runtime, name, getattr(controls, name))
        for name in ("keep_last", "save_last", "save_best"):
            setattr(config.checkpointing, name, getattr(controls, name))
        config.checkpointing.every_updates = controls.checkpoint_every_updates
        if controls.validation is not None:
            from smartsom.config.experiment import ValidationOptions
            from smartsom.config.validation import (
                freeze_validation_cases,
                validate_frozen_cases,
            )

            config.validation = ValidationOptions.model_validate_json(
                canonical_json(
                    {
                        **primitive(config.validation),
                        **primitive(controls.validation),
                        "enabled": True,
                    }
                )
            )
            validation_json = controls.validation_inputs_json
            if config.validation.scenarios:
                if validation_json is None:
                    validation_json = freeze_validation_cases(
                        prepared.resolved, config.validation
                    )
                validate_frozen_cases(validation_json, config.validation)
            elif validation_json is not None:
                raise ValueError(
                    "validation snapshots require declared external scenarios"
                )
        else:
            config.validation.enabled = False
            config.validation.scenarios = ()
            validation_json = None
    recipe = prepared.resolved
    return replace(
        prepared,
        config_json=canonical_json(config),
        validation_json=validation_json,
        scientific_sha256=digest(recipe_identity(recipe, config, validation_json)),
    )


def train_one(prepared, *, on_progress=None, controls=None) -> TrainingResult:
    """Preserve lifecycle controls while executing only the canonical grid backend."""
    from smartsom.experiments.production_training import train_prepared

    prepared = apply_training_controls(prepared, controls)

    def progress(event):
        decision = on_progress(event) if on_progress else None
        if (
            controls
            and controls.stop_after_updates
            and event["stage"] == "learning_metrics"
            and event["ppo_updates"] >= controls.stop_after_updates
        ):
            return {"stop": "interrupted"}
        return decision

    arguments = {
        "on_progress": progress,
        "initialize_from": controls.initialize_from if controls else None,
    }
    if controls and controls.resume_from:
        checkpoint = Path(controls.resume_from)
        roots = [
            root
            for root in checkpoint.parents
            if (root / "config/grid_recipe.json").is_file()
        ]
        if not roots:
            raise ValueError(
                "resume requires a grid experiment with update checkpoints"
            )
        arguments.update(root=roots[0], resume_from=checkpoint)
    result = train_prepared(prepared, **arguments)
    return TrainingResult(
        result.run_dir,
        result.last_checkpoint or result.best_checkpoint,
        result.environment_steps,
        result.learner_updates,
        result.status,
        result.last_checkpoint,
        result.best_checkpoint,
    )
