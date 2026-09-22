"""Execute a frozen experiment through the sole grid simulation loop."""

import json
from dataclasses import dataclass
from pathlib import Path

from smartsom.config.codec import ConfigurationError, digest
from smartsom.config.experiment import ExperimentConfig, PreparedExperiment
from smartsom.config.production import ProductionRecipe, recipe_identity
from smartsom.experiments.production import ProductionResult
from smartsom.telemetry.runtime import operation


@dataclass(frozen=True, slots=True)
class RunResult:
    run_dir: Path
    simulation_result: ProductionResult


class RunFailedError(RuntimeError):
    def __init__(self, run_dir: Path, cause: BaseException):
        self.run_dir, self.cause = run_dir, cause
        super().__init__(f"run failed in {run_dir}: {cause}")


@operation("run")
def run_one(
    prepared,
    *,
    on_progress=None,
    deterministic=True,
    render_mode=None,
    verbose=None,
    record=True,
    output_root=None,
):
    """Consume detached inputs; no source YAML is reopened during execution."""
    if not isinstance(prepared, PreparedExperiment) or not isinstance(
        prepared.resolved, ProductionRecipe
    ):
        raise TypeError(
            "run_one requires a prepared grid experiment; migrate matrix factories with explicit grid and ports"
        )
    from smartsom.experiments.production import run as execute

    config = ExperimentConfig.model_validate_json(prepared.config_json)
    recipe = prepared.resolved
    if (
        recipe.algorithm.provider.startswith(("sb3.", "rllib."))
        and not recipe.algorithm.checkpoint
    ):
        raise ConfigurationError(
            "evaluation requires a trained checkpoint; use train first"
        )
    if (
        digest(recipe_identity(recipe, config, prepared.validation_json))
        != prepared.scientific_sha256
    ):
        raise ValueError("frozen run scientific identity mismatch")
    try:
        directory = execute(
            recipe.scenario,
            recipe.algorithm,
            output_root=output_root or config.output.root,
            name=config.output.name,
            render_mode=render_mode,
            verbose=config.logging.verbose if verbose is None else verbose,
            record=record,
            debug=config.logging.debug,
            observations=config.logging.observations,
            on_progress=on_progress,
            deterministic=deterministic,
            policy_seed=recipe.algorithm_seed,
            limits=config.training,
            full_replay=config.evaluation.full_replay,
            input_metadata={
                "workload_authoring": json.loads(
                    recipe.workload_source_json or recipe.workload_json
                ),
                "scenario_authoring": json.loads(recipe.settings_json),
            },
            experiment={
                "config": json.loads(prepared.config_json),
                "origins": json.loads(prepared.origins_json),
                "scientific_sha256": prepared.scientific_sha256,
                "algorithm_seed": recipe.algorithm_seed,
                "validation_inputs": json.loads(prepared.validation_json)
                if prepared.validation_json
                else None,
            },
        )
    except Exception as exc:
        if hasattr(exc, "run_dir"):
            raise RunFailedError(exc.run_dir, exc) from exc
        raise
    manifest = json.loads((directory / "run.json").read_text())
    state = manifest["result"]
    return RunResult(
        directory,
        ProductionResult(
            state["tick"] if manifest["status"] == "completed" else None,
            manifest["status"],
            state["return"],
            len(state["completed"]),
            state,
        ),
    )
