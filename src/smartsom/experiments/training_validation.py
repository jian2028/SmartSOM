"""Fixed validation worlds and conservative checkpoint selection, outside PPO."""

from smartsom.config.codec import digest
from smartsom.config.study import study_roots


def validation_inputs(resolved, controls):
    """Reuse the frozen materializer with the exact paired-study world roots."""
    from smartsom.config.experiment import PreparedExperiment
    from smartsom.config.production import ProductionRecipe

    recipe = resolved.resolved if isinstance(resolved, PreparedExperiment) else resolved
    if not isinstance(recipe, ProductionRecipe):
        raise TypeError("validation requires a prepared grid recipe")
    result = []
    for replication in range(controls.replications):
        world, _ = study_roots(controls.seed, controls.case_id, replication, "")
        episode = recipe.episode(world)
        result.append(
            {
                "input_id": digest([world, replication, episode]),
                "case_id": controls.case_id,
                "world_seed": world,
                "replication": replication,
                "episode": episode,
            }
        )
    return tuple(result)


def select_best(candidate, incumbent, controls):
    """Return (replace, reason); never compare survivor means from different cases."""
    successful = candidate["successful_inputs"]
    if not successful:
        return False, "no_completed_episodes"
    strict = controls.best_mode == "all_complete" or (
        controls.best_mode == "custom" and controls.failure_policy == "ineligible"
    )
    if strict and candidate["completed"] != candidate["episodes"]:
        return False, "requires_all_complete"
    if incumbent is None:
        return True, "first_eligible"
    if controls.best_mode == "completion_first":
        if candidate["completed"] != incumbent["completed"]:
            return candidate["completed"] > incumbent["completed"], "completion_count"
    if successful != incumbent["successful_inputs"]:
        return False, "different_successful_inputs"
    metric = controls.metric if controls.best_mode == "custom" else "makespan"
    direction = controls.direction if controls.best_mode == "custom" else "min"
    current, previous = candidate["metrics"][metric], incumbent["metrics"][metric]
    if current is None or previous is None:
        raise ValueError(f"validation metric is unavailable: {metric}")
    improvement = previous - current if direction == "min" else current - previous
    return (
        improvement > controls.min_delta,
        "metric" if improvement > controls.min_delta else "tie_or_no_improvement",
    )
