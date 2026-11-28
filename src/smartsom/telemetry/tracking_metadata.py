"""Tracking labels derived from frozen scientific inputs, never policy RNG."""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from smartsom.config.codec import read_document


def metadata(
    root,
    experiment,
    *,
    seed=None,
    variant=None,
    tags=(),
    short_name=None,
    run_meta=None,
    facts=None,
    name_field="short_title",
):
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
        "small_factory": "small",
        "small_target": "small-wide",
        "medium_factory": "medium",
        "base_medium": "medium",
    }.get(factory_id, "custom")
    if factory_id == "base_target":
        # This historical ID was reused by both compact and wide Medium.
        grid = factory.get("grid", {})
        scale = {(27, 12): "medium", (29, 19): "medium-wide"}.get(
            (grid.get("width"), grid.get("height")), "custom"
        )
    control = {"base": "distributed", "central": "centralized"}.get(
        algorithm.get("control"), algorithm.get("control", "custom")
    )
    if method == "custom" and policies:
        method = next(
            (
                name
                for name, policy in policies.items()
                if policy.get("kind") == "learning"
            ),
            algorithm["production"]["default"],
        )
        if learning:
            from smartsom.config.base_diagnostic import BaseAlgorithm
            from smartsom.config.components import (
                absolute_components,
                isolated_random,
                network_ownership,
            )

            options = absolute_components(
                BaseAlgorithm.model_validate(algorithm), root / "inputs"
            )
            with isolated_random():
                ownership = network_ownership(options)
            if any(row["mode"] == "global" for row in ownership.values()):
                control = "centralized"
    from smartsom.config.run_meta import select_display_text

    title_prefix = (
        select_display_text(run_meta, name_field, authored_only=True)
        if run_meta and name_field != "default"
        else None
    )
    short_name = (
        title_prefix
        or short_name
        or Path(experiment.get("run", {}).get("output", root.name)).name
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
    if isinstance(run_meta, dict) and run_meta.get("short_title"):
        automatic.append(f"title:{run_meta['short_title']}")
    host_name = (
        facts.get("host", {}).get("name")
        if isinstance(facts, dict) and isinstance(facts.get("host"), dict)
        else None
    )
    if host_name:
        automatic.append(f"host:{host_name}")
    meta_tags = (
        run_meta.get("tags", [])
        if isinstance(run_meta, dict) and isinstance(run_meta.get("tags"), list)
        else []
    )
    return {
        "name": "－".join(parts),
        "tags": list(
            dict.fromkeys(
                [
                    *automatic,
                    *experiment.get("run", {}).get("tags", []),
                    *meta_tags,
                    *tags,
                ]
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
