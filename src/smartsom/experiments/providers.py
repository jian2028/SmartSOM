"""Construct rules and learned policies for the single grid execution contract."""

from smartsom.algorithms.production import (
    GreedyProductionPolicy,
    RandomProductionPolicy,
    ScriptedProductionPolicy,
)


def build_provider(
    algorithm,
    scenario=None,
    *,
    seed=None,
    deterministic=True,
    limits=None,
    observations=None,
):
    provider = getattr(
        algorithm,
        "provider",
        getattr(getattr(algorithm, "algorithm", None), "provider", None),
    )
    rules = {
        "builtin.greedy",
        "builtin.spt",
        "builtin.first_feasible",
        "builtin.random",
        "builtin.scripted",
    }
    learning = {"sb3.maskable_ppo", "rllib.ppo", "rllib.resource_ppo"}
    if provider not in rules | learning:
        raise ValueError(f"unsupported provider for grid execution: {provider!r}")
    if scenario is None:
        raise ValueError("provider construction requires the current grid scenario")
    seed = scenario.seed if seed is None else seed
    if provider in {"builtin.greedy", "builtin.spt", "builtin.first_feasible"}:
        return GreedyProductionPolicy(
            scenario.factory,
            seed,
            rule=provider.removeprefix("builtin."),
            quality_mode=algorithm.quality_mode,
        )
    if provider == "builtin.random":
        return RandomProductionPolicy(scenario.factory, seed)
    if provider == "builtin.scripted":
        return ScriptedProductionPolicy(algorithm.commands)
    if algorithm.checkpoint is None:
        raise ValueError("evaluation requires a trained checkpoint")
    from smartsom.learning.production import LearnedProductionDriver

    return LearnedProductionDriver(
        algorithm.checkpoint,
        scenario,
        deterministic=deterministic,
        seed=seed,
        limits=limits,
        observations=observations,
    )
