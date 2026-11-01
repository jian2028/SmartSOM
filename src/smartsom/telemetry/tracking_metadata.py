"""Tracking labels derived from frozen scientific inputs, never policy RNG."""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from smartsom.config.codec import read_document


def metadata(root, experiment, *, seed=None, variant=None, tags=(), short_name=None):
    root = Path(root)
    documents = {"experiment": experiment}
    for kind in ("factory", "workload", "algorithm"):
        path = root / "inputs" / (kind + ".yaml")
        if path.is_file():
            documents[kind] = read_document(path)
    algorithm = documents.get("algorithm", {})
    policies = algorithm.get("production", {}).get("policies", {})
    learning = [p for p in policies.values() if p.get("kind") == "learning"]
    method, observation = "custom", "custom"
    if not learning:
        default = algorithm.get("production", {}).get("default")
        target = policies.get(default, {}).get("implementation", {}).get("target", "")
        for known in ("rule", "random"):
            if target == f"smartsom.policies.{known}.create_policy":
                method, observation = known, "base"
    if len(learning) == 1:
        target = learning[0].get("learner", {}).get("target", "")
        for known in ("ppo", "dqn"):
            if target == f"smartsom.learning.{known}.create_learner":
                method = known
        target = learning[0].get("observation", {}).get("target", "")
        for package, label in (("base", "base"), ("social", "sl")):
            if (
                target
                == f"smartsom.representations.{package}.observation.create_observation"
            ):
                observation = label
    factory = documents.get("factory", {}).get("factory", {})
    factory_id = factory.get("factory_id", "custom")
    scale = {
        "base_small": "small",
        "base_medium": "medium",
        "base_target": "medium",
    }.get(factory_id, "custom")
    control = {"base": "distributed", "central": "centralized"}.get(
        algorithm.get("control"), algorithm.get("control", "custom")
    )
    short_name = (
        short_name or Path(experiment.get("run", {}).get("output", root.name)).name
    )
    label = variant or observation
    parts = [short_name, scale, method, label]
    if learning and seed is not None:
        parts.append(f"s{seed}")
    automatic = [
        f"algorithm:{method}",
        f"observation:{observation}",
        f"map:{scale}",
        f"control:{control}",
    ]
    return {
        "name": "－".join(parts),
        "tags": list(
            dict.fromkeys(
                [*automatic, *experiment.get("run", {}).get("tags", []), *tags]
            )
        ),
        "documents": documents,
        "comparison": {
            "method": method,
            "method_identity": hashlib.sha256(
                json.dumps(algorithm, sort_keys=True).encode()
            ).hexdigest(),
            "reference_identities": {
                method: hashlib.sha256(
                    json.dumps(
                        {
                            **algorithm,
                            "production": {"fixed_reference": method},
                        },
                        sort_keys=True,
                    ).encode()
                ).hexdigest()
                for method in ("rule", "random")
            },
            "observation": observation,
            "map": scale,
            "factory_id": factory_id,
            "control": control,
            "variant": label,
        },
    }


def method_identity(comparison, method):
    return comparison.get("reference_identities", {}).get(
        method, comparison.get("method_identity")
    )


def launch_group(name, manifest):
    timestamp = datetime.fromtimestamp(
        manifest.get("started_unix", datetime.now(UTC).timestamp()), UTC
    )
    identity = str(manifest.get("run_id") or uuid4().hex)[:8]
    return f"{name}-{timestamp:%Y%m%dT%H%M%SZ}-{identity}"
