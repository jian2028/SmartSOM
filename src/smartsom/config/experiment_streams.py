"""Explicit opt-in random roots; legacy derivation stays byte-for-byte equivalent."""

from smartsom.config.production import named_seed

SCHEMA = "smartsom.experiment-seed-streams/v1"


def roots(experiment, policy_seed):
    world = (
        experiment.data_seed if experiment.world_seed is None else experiment.world_seed
    )
    environment = (
        world if experiment.environment_seed is None else experiment.environment_seed
    )
    return {"world": world, "environment": environment, "policy": policy_seed}


def explicit(experiment):
    return any(
        getattr(experiment, name) is not None
        for name in (
            "world_seed",
            "environment_seed",
            "policy_seed",
            "training_environment_seeds",
        )
    )


def frozen_streams(experiment, policy_seed, *, split, replication):
    if not explicit(experiment):
        return None
    selected = roots(experiment, policy_seed)
    world = named_seed(selected["world"], f"{split}:{replication}")
    environment = named_seed(selected["environment"], f"{split}:{replication}")
    path = experiment.training_environment_seeds or "environment"
    return {
        "schema": SCHEMA,
        "roots": selected,
        "split": split,
        "replication": replication,
        "world_seed": world,
        "environment_root": environment,
        "environment_seed": named_seed(environment, "environment"),
        "training_environment_seeds": path,
        "training_environment_root": policy_seed if path == "policy" else environment,
        "episode_domain": "environment:{index}:{episode}",
        "policy_domain": "policy:{group}",
    }


def training_episode_seed(recipe, policy_seed, index, episode):
    streams = recipe.get("seed_contract")
    if streams is not None:
        if streams.get("schema") != SCHEMA or streams.get(
            "training_environment_seeds"
        ) not in ("environment", "policy"):
            raise ValueError("unsupported frozen training seed contract")
        root = streams.get("training_environment_root")
        expected = (
            policy_seed
            if streams["training_environment_seeds"] == "policy"
            else streams.get("environment_root")
        )
        if type(root) is not int or not 0 <= root < 2**64 or root != expected:
            raise ValueError("frozen training seed root is inconsistent")
        return named_seed(root, f"environment:{index}:{episode}")
    if "data_seed" in recipe:
        return named_seed(recipe["data_seed"], f"environment:{index}:{episode}")
    return named_seed(policy_seed, f"training:{index}:{episode}")
