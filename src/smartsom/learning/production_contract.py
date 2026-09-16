"""Framework-free semantic compatibility checks for grid learning checkpoints."""

import json

from smartsom.config.codec import digest, primitive
from smartsom.trace.production import state_hash

OBSERVATION_CONTRACT = "smartsom.grid-observation/v1"
ACTION_CONTRACT = "smartsom.grid-actions/v1"


def factory_identity(factory):
    """Ignore display names and set ordering, retaining physical slot/binding order.

    Resource IDs and geometry stay in the identity. Slot and binding sequences are
    intentionally not sorted: they can define deterministic transfer choices.
    """
    data = primitive(factory)
    data.pop("name", None)
    data["operation_types"] = sorted(data["operation_types"])
    data["grid"]["blocked_cells"] = sorted(
        data["grid"]["blocked_cells"], key=lambda cell: (cell["x"], cell["y"])
    )
    for field, key in (
        ("machines", "machine_id"),
        ("agvs", "agv_id"),
        ("buffers", "buffer_id"),
        ("ports", "port_id"),
        ("inspection_stations", "inspection_station_id"),
        ("scrap_bins", "scrap_bin_id"),
        ("chargers", "charger_id"),
    ):
        data[field] = sorted(data[field], key=lambda row: row[key])
        for row in data[field]:
            row.pop("name", None)
            if field == "machines":
                row["operation_types"] = sorted(row["operation_types"])
                row["quality_modes"] = sorted(
                    row["quality_modes"], key=lambda mode: mode["quality_mode_id"]
                )
    return digest(data)


def validate_model_contract(manifest, scenario, algorithm=None):
    """Reject weights whose semantic encoding cannot be reused by this adapter."""
    if (
        manifest.get("schema") != "smartsom.production-checkpoint/v1"
        or manifest.get("observation_contract") != OBSERVATION_CONTRACT
        or manifest.get("action_contract") != ACTION_CONTRACT
    ):
        raise ValueError(
            "checkpoint action/observation encoding is incompatible; retrain with the current core"
        )
    original_factory = manifest.get("scenario", {}).get("factory")
    if original_factory is None:
        compatible = manifest.get("factory_hash") == state_hash(scenario.factory)
    else:
        if manifest.get("factory_hash") != state_hash(original_factory):
            raise ValueError(
                "checkpoint factory hash disagrees with its frozen factory"
            )
        compatible = factory_identity(original_factory) == factory_identity(
            scenario.factory
        )
    if not compatible:
        raise ValueError("checkpoint factory differs from evaluation factory")
    if (
        manifest.get("quality_probability_visibility", "public")
        != scenario.quality_probability_visibility
    ):
        raise ValueError(
            "checkpoint observation visibility differs from evaluation scenario"
        )
    if algorithm is None:
        return
    from smartsom.config.production import AlgorithmConfig

    original = AlgorithmConfig.model_validate_json(json.dumps(manifest["algorithm"]))
    fields = (
        "provider",
        "max_jobs",
        "time_scale",
        "count_scale",
        "hidden_sizes",
        "activation",
    )
    mismatches = [
        name for name in fields if getattr(original, name) != getattr(algorithm, name)
    ]
    for name in ("observation", "role_observations", "network"):
        before = getattr(original.extensions, name, None)
        after = getattr(algorithm.extensions, name, None)
        if primitive(before) != primitive(after):
            mismatches.append("extensions." + name)
    if mismatches:
        raise ValueError(
            "initialization has incompatible model inputs or network: "
            + ", ".join(mismatches)
        )


def validate_checkpoint_manifest(manifest, scenario, algorithm=None):
    """Validate inference metadata before opening weights or allocating a run."""
    import re

    from smartsom.config.production import AlgorithmConfig

    validate_model_contract(manifest, scenario, algorithm)
    recorded = AlgorithmConfig.model_validate_json(json.dumps(manifest["algorithm"]))
    provider = recorded.provider
    if manifest.get("provider") != provider or provider not in {
        "sb3.maskable_ppo",
        "rllib.ppo",
        "rllib.resource_ppo",
    }:
        raise ValueError("checkpoint provider differs from its algorithm")
    for field in ("environment_steps", "learner_updates"):
        if type(manifest.get(field)) is not int or manifest[field] <= 0:
            raise ValueError(f"checkpoint {field} requires a positive trained count")
    if provider == "sb3.maskable_ppo":
        expected = {"policy"}
        members = {"model.zip"}
    else:
        expected = {"default_policy"}
        if provider == "rllib.resource_ppo":
            factory = scenario.factory
            expected = {
                role
                for role, active in (
                    ("agv_policy", factory.agvs),
                    ("machine_policy", factory.machines),
                    ("quality_policy", factory.inspection_stations),
                    (
                        "buffer_policy",
                        factory.inspection_stations
                        or any(b.role != "system_output" for b in factory.buffers),
                    ),
                )
                if active
            }
        modules = manifest.get("modules")
        if (
            not isinstance(modules, list)
            or len(modules) != len(expected)
            or set(modules) != expected
        ):
            raise ValueError(
                "checkpoint role modules differ from the resource structure"
            )
        members = {f"{role}.pt" for role in expected}
    weights = manifest.get("weights")
    if (
        not isinstance(weights, dict)
        or set(weights) != expected
        or any(
            not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None
            for value in weights.values()
        )
    ):
        raise ValueError("checkpoint role weight identities are invalid")
    if not members <= {row.get("path") for row in manifest.get("files", [])}:
        raise ValueError("checkpoint is missing required model weight files")
