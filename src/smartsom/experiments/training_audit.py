"""Regenerate scientific episodes and verify the compact learning ledger."""

import json
from pathlib import Path
from typing import Literal

from pydantic import Field

from smartsom.config.codec import digest, read_model
from smartsom.config.models import StrictModel
from smartsom.config.production import ProductionRecipe
from smartsom.config.snapshots import validate_resolved
from smartsom.config.training import (
    EPISODE_SEED_VERSION,
    ResolvedTrainingRun,
    framework_seed,
)


class TrainingSnapshot(StrictModel):
    schema_id: Literal["smartsom.resolved-training/v1"] = Field(alias="schema")
    resolved: ResolvedTrainingRun | ProductionRecipe


def load_training_snapshot(path: Path) -> ResolvedTrainingRun:
    resolved = read_model(path, TrainingSnapshot)[0].resolved
    if isinstance(resolved, ProductionRecipe):
        from smartsom.domain.production import validate_production_scenario

        validate_production_scenario(resolved.scenario)
        return resolved
    validate_resolved(resolved.base)
    if (
        resolved.episode_seed_version != EPISODE_SEED_VERSION
        or resolved.framework_seed
        != framework_seed(resolved.run.seed, resolved.algorithm.algorithm.provider)
    ):
        raise ValueError("training snapshot seed authority mismatch")
    if resolved.base.run.seed != resolved.run.seed:
        raise ValueError("training snapshot base seed mismatch")
    return resolved


def audit_training(run_dir: Path) -> dict:
    """Verify grid learning ledgers; never silently execute a historical core."""
    path = Path(run_dir)
    if not any(
        (parent / "config/grid_recipe.json").is_file()
        for parent in (path, *path.parents)
    ):
        raise ValueError(
            "historical matrix training evidence requires its recorded source checkout for replay"
        )
    return audit_grid_training(path)


def audit_grid_training(source):
    """Replay recorded indices through the same public adapter and frozen inputs."""
    from smartsom.config.experiment import ExperimentConfig
    from smartsom.experiments.production_training import (
        ProductionEvidence,
        verify_checkpoint,
    )
    from smartsom.learning.production_sampling import GridStream, ProductionSamplingSpec

    source = Path(source).resolve()
    root = next(
        parent
        for parent in (source, *source.parents)
        if (parent / "config/grid_recipe.json").is_file()
    )
    record = json.loads((root / "run.json").read_text())
    from smartsom.experiments.catalog import contained_path

    if (source / "update.json").is_file():
        checkpoint = source
    elif source == root:
        checkpoint = contained_path(root, record["paths"]["checkpoint"])
    else:
        attempts = [
            item
            for item in (record, *record.get("attempts", []))
            if item.get("paths", {}).get("training")
            and contained_path(root, item["paths"]["training"]) == source
        ]
        if len(attempts) != 1 or not attempts[0]["paths"].get("checkpoint"):
            raise ValueError("requested training attempt has no retained checkpoint")
        checkpoint = contained_path(root, attempts[0]["paths"]["checkpoint"])
    update = verify_checkpoint(checkpoint)
    payload = json.loads((checkpoint / "recipe.json").read_text())
    config = ExperimentConfig.model_validate_json(json.dumps(payload["config"]))
    recipe = ProductionRecipe(**payload["recipe"])
    rows = [
        json.loads(line)
        for line in (checkpoint / "episodes.jsonl").read_text().splitlines()
    ]
    active = json.loads((checkpoint / "active_episodes.json").read_text())
    seen, decisions, quotas = set(), 0, [0] * config.runtime.num_envs
    stream_rows = [[] for _ in quotas]
    for row in rows + active:
        stream, local = row["stream_id"], row["local_episode"]
        if (
            type(stream) is not int
            or not 0 <= stream < len(quotas)
            or type(local) is not int
            or local < 0
        ):
            raise ValueError("invalid grid stream identity")
        if row["episode"] != local * len(quotas) + stream or row["episode"] in seen:
            raise ValueError("duplicate or inconsistent grid episode identity")
        seen.add(row["episode"])
        stream_rows[stream].append(local)
        env = GridStream(
            ProductionSamplingSpec(recipe, config.seed), stream, len(quotas)
        )
        try:
            env.local_episode = env.adapter.episode_index = local - 1
            if env.adapter.hooks:
                env.adapter.hooks.runtime.load_state_dict(
                    row["extension_state"]["initial"]
                )
            env.reset()
            for step in row["steps"]:
                action = dict(step["indices"]) if env.resource else step["action_index"]
                env.step(action)
            actual = ProductionEvidence.row(env.snapshot())
            if digest(actual) != digest(row):
                raise ValueError(
                    f"grid episode {row['episode']} decision/state/reward replay mismatch"
                )
        finally:
            env.env.close()
        decisions += len(row["steps"])
        quotas[stream] += len(row["steps"])
    if any(sorted(indices) != list(range(len(indices))) for indices in stream_rows):
        raise ValueError("grid episode ledger coverage has gaps")
    if (
        len(set(quotas)) != 1
        or decisions != update["steps"]
        or decisions > config.training.total_steps
    ):
        raise ValueError("grid sampling quota or checkpoint budget mismatch")
    return {
        "status": "passed",
        "training_status": record["status"],
        "budget_completed": decisions == config.training.total_steps,
        "planned_environment_steps": config.training.total_steps,
        "environment_steps": decisions,
        "episodes": len(rows) + len(active),
        "completed_episodes": sum(r["end_reason"] == "completed" for r in rows),
        "failed_episodes": sum(r["end_reason"] != "completed" for r in rows),
        "num_envs": len(quotas),
        "learner_updates": update["updates"],
        "ppo_updates": decisions // config.training.steps_per_update,
        "provider": recipe.algorithm.provider,
        "checkpoint": str(checkpoint),
        **(
            {
                "agent_steps": sum(
                    len(step["indices"])
                    for row in rows + active
                    for step in row["steps"]
                ),
                "physical_actions": sum(
                    len(step["actions"])
                    for row in rows + active
                    for step in row["steps"]
                ),
                "conflicts": sum(
                    step.get("conflict_count", 0)
                    for row in rows + active
                    for step in row["steps"]
                ),
                "team_return_definition": "one shared return; never sum over resources",
            }
            if recipe.algorithm.provider == "rllib.resource_ppo"
            else {}
        ),
    }
