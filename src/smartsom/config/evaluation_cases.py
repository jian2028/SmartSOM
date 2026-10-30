"""Portable fixed-case inventories, including exact imported physical worlds."""

import json
from pathlib import Path

from smartsom.config.codec import digest, primitive
from smartsom.config.scenario_snapshot import scenario_from_snapshot


def load_cases(path, *, split, factory, workload):
    data = json.loads(Path(path).read_text())
    if (
        data.get("schema") not in {"smartsom.fixed-cases/v1", "smartsom.case-recipe/v1"}
        or data.get("split") != split
    ):
        raise ValueError("fixed cases require the matching split and schema")
    if data.get("factory_sha256") != digest(primitive(factory)) or data.get(
        "workload_sha256"
    ) != digest(workload.model_dump(mode="json")):
        raise ValueError("fixed cases differ from the configured factory/workload")
    cases = data.get("cases", [])
    if not cases or len({c["id"] for c in cases}) != len(cases):
        raise ValueError("fixed cases require nonempty unique case IDs")
    identities = set()
    keys = set()
    for case in cases:
        if case["condition"] not in {"normal", "low", "medium", "high"}:
            raise ValueError("unknown fixed case condition")
        if type(case["seed"]) is not int:
            raise ValueError("fixed case seed must be an integer")
        key = (case["condition"], case["seed"])
        if key in keys:
            raise ValueError("duplicate condition/seed in fixed cases")
        keys.add(key)
        if data["schema"] == "smartsom.case-recipe/v1":
            from smartsom.config.base_diagnostic import generate_scenario

            case["scenario"] = primitive(
                generate_scenario(
                    factory,
                    workload,
                    case["seed"],
                    case["condition"],
                    data["drain_ticks"],
                )
            )
            if case.get("scenario_sha256") != digest(case["scenario"]):
                raise ValueError(
                    "case recipe no longer reproduces its frozen historical world"
                )
        scenario = scenario_from_snapshot(case["scenario"])
        if primitive(scenario.factory) != primitive(factory):
            raise ValueError("imported scenario factory mismatch")
        identity = digest(primitive(scenario))
        if identity in identities:
            raise ValueError("duplicate physical world in fixed case split")
        identities.add(identity)
        seeds = case.get("action_seeds")
        if (
            not seeds
            or any(type(s) is not int for s in seeds)
            or len(set(seeds)) != len(seeds)
        ):
            raise ValueError("fixed cases require unique explicit action_seeds")
    return cases


def validate_panel_separation(panels, experiment, factory, workload, drain):
    """Compare actual worlds even when only one split is explicitly imported."""
    from smartsom.config.base_diagnostic import generate_scenario, seed_for

    if not panels:
        return
    identities = {}
    for split, namespace in (("validation", 80000), ("test", 90000)):
        if split in panels:
            identities[split] = {digest(c["scenario"]) for c in panels[split]}
        else:
            identities[split] = {
                digest(
                    primitive(
                        generate_scenario(
                            factory,
                            workload,
                            seed_for(namespace, index // 4) % (2**31),
                            ("normal", "low", "medium", "high")[index % 4],
                            drain,
                        )
                    )
                )
                for index in range(getattr(experiment, f"{split}_cases"))
            }
    if identities["validation"] & identities["test"]:
        raise ValueError("validation and test contain the same physical world")
