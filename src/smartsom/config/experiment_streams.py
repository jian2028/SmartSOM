"""Explicit opt-in random roots; legacy derivation stays byte-for-byte equivalent."""

from smartsom.config.production import named_seed

SCHEMA = "smartsom.experiment-seed-streams/v1"
SCHEMA_V2 = "smartsom.experiment-seed-streams/v2"


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
        "schema": SCHEMA_V2 if path == "legacy_policy" else SCHEMA,
        "roots": selected,
        "split": split,
        "replication": replication,
        "world_seed": world,
        "environment_root": environment,
        "environment_seed": named_seed(environment, "environment"),
        "training_environment_seeds": path,
        "training_environment_root": policy_seed
        if path in ("policy", "legacy_policy")
        else environment,
        "episode_domain": ("training" if path == "legacy_policy" else "environment")
        + ":{index}:{episode}",
        "policy_domain": "policy:{group}",
    }


def frozen_world_streams(reference, policy_seed, path):
    """Direct frozen-world seeds have no split hash between author and simulator."""
    return {
        "schema": SCHEMA_V2,
        "roots": {
            "world": reference.data_seed,
            "environment": reference.environment_seed,
            "policy": policy_seed,
        },
        "split": "train",
        "replication": 0,
        "world_seed": reference.data_seed,
        "environment_root": reference.environment_seed,
        "environment_seed": reference.environment_seed,
        "training_environment_seeds": path,
        "training_environment_root": policy_seed
        if path in ("policy", "legacy_policy")
        else reference.environment_seed,
        "episode_domain": ("training" if path == "legacy_policy" else "environment")
        + ":{index}:{episode}",
        "policy_domain": "policy:{group}",
    }


def training_episode_seed(recipe, policy_seed, index, episode):
    streams = recipe.get("seed_contract")
    if streams is not None:
        path = streams.get("training_environment_seeds")
        schema = streams.get("schema")
        if not (
            (schema == SCHEMA and path in ("environment", "policy"))
            or (
                schema == SCHEMA_V2
                and path in ("environment", "policy", "legacy_policy")
            )
        ):
            raise ValueError("unsupported frozen training seed contract")
        root = streams.get("training_environment_root")
        expected = (
            policy_seed
            if path in ("policy", "legacy_policy")
            else streams.get("environment_root")
        )
        if type(root) is not int or not 0 <= root < 2**64 or root != expected:
            raise ValueError("frozen training seed root is inconsistent")
        domain = "training" if path == "legacy_policy" else "environment"
        if streams.get("episode_domain") != domain + ":{index}:{episode}":
            raise ValueError("frozen training seed domain is inconsistent")
        return named_seed(root, f"{domain}:{index}:{episode}")
    if "data_seed" in recipe:
        return named_seed(recipe["data_seed"], f"environment:{index}:{episode}")
    return named_seed(policy_seed, f"training:{index}:{episode}")
