"""Construct and close the selected real training backend, with no PPO updates."""

from pathlib import Path
from uuid import uuid4

from smartsom.config.codec import primitive
from smartsom.experiments.evidence import source_identity, write_json
from smartsom.learning.checkpoint import require_backend


def probe_training_backend(resolved, controls, *, output_root=None):
    from smartsom.config.experiment import PreparedExperiment

    if not isinstance(resolved, PreparedExperiment):
        raise TypeError("backend probe requires a prepared grid experiment")
    from smartsom.experiments.training import apply_training_controls

    return probe_grid_backend(
        apply_training_controls(resolved, controls), output_root=output_root
    )


def probe_grid_backend(prepared, *, output_root=None):
    from smartsom.config.codec import digest
    from smartsom.config.experiment import ExperimentConfig
    from smartsom.config.production import recipe_identity
    from smartsom.experiments.production_training import ProductionEvidence, identity
    from smartsom.learning.production_sampling import ProductionSamplingSpec
    from smartsom.learning.training_state import isolated_rng

    config = ExperimentConfig.model_validate_json(prepared.config_json)
    recipe = prepared.resolved
    if (
        digest(recipe_identity(recipe, config, prepared.validation_json))
        != prepared.scientific_sha256
    ):
        raise ValueError("backend probe frozen input identity mismatch")
    dependencies = require_backend(recipe.algorithm.provider)
    root = Path(output_root) if output_root else Path(config.output.root) / ".probes"
    directory = root.resolve() / f"probe-{uuid4().hex}"
    directory.mkdir(parents=True)
    report = {
        "provider": recipe.algorithm.provider,
        "run_dir": str(directory),
        "source": source_identity(),
        "dependencies": dependencies,
        "algorithm": primitive(recipe.algorithm),
        "identity": identity(prepared, config),
        "configured_budget": primitive(config.training),
        "topology_semantics": "configured",
    }
    if recipe.algorithm.provider == "sb3.maskable_ppo":
        from smartsom.learning.production import train_sb3 as construct
    else:
        from smartsom.learning.production_ray import train_ray as construct
    try:
        with isolated_rng():
            result = construct(
                recipe.scenario,
                recipe.algorithm,
                directory / "backend",
                total_steps=config.training.total_steps,
                rollout_steps=config.training.steps_per_update,
                runtime=config.runtime,
                probe_only=True,
                sampling_spec=ProductionSamplingSpec(recipe, config.seed),
                evidence=ProductionEvidence(directory / "backend"),
            )
        report.update(status="passed", **result)
        write_json(directory / "probe.json", report)
        return report
    except BaseException as exc:
        exc.run_dir = directory
        try:
            write_json(
                directory / "probe.json",
                {**report, "status": "failed", "reason": str(exc)},
            )
        except Exception as metadata_error:
            exc.add_note(
                f"probe failure metadata also could not be saved: {metadata_error}"
            )
        raise
