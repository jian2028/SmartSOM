"""The opt-in semantic identity separates execution from scientific settings."""

import json
from dataclasses import replace

from smartsom.config.codec import canonical_json, digest
from smartsom.config.experiment_v3 import PreparedComposition
from smartsom.config.scientific_contract import CONTRACT, identity
from smartsom.experiments.tuning_session import scientific_identity


def fixture(config):
    return PreparedComposition(
        canonical_json(config),
        "{}",
        "{}",
        "{}",
        "{}",
        "[]",
        "[]",
        '{"author.yaml":"original-source-hash"}',
        "derived",
        '{"workload":{"seed":101},"authoring":{"path":"author.yaml"}}',
    )


def test_legacy_adaptive_identity_is_unchanged():
    prepared = fixture({"runtime": {"numerical_threads": 1, "max_concurrent": 2}})
    data = prepared.__dict__.copy()
    config = json.loads(data.pop("config_json"))
    config["runtime"] = {}
    data.pop("scientific_sha256")
    assert scientific_identity(prepared) == digest(
        {"contract": "smartsom.adaptive-continuation/v1", "config": config, **data}
    )


def test_new_contract_retention_and_resource_policy_do_not_change_science():
    config = {
        "interface_contract": CONTRACT,
        "runtime": {"device": "cpu", "numerical_threads": 1, "max_concurrent": 2},
        "validation": {"updates": [8, 32, 64]},
        "checkpointing": {},
    }
    prepared = fixture(config)
    config["checkpointing"]["retention"] = {"mode": "latest_full_and_best"}
    config["runtime"].update(numerical_threads=2, max_concurrent=4)
    changed = replace(prepared, config_json=canonical_json(config))
    assert identity(prepared) == identity(changed) == scientific_identity(changed)
    assert changed.origins_json == prepared.origins_json
    config["validation"]["updates"] = [8, 64]
    assert identity(prepared) != identity(
        replace(changed, config_json=canonical_json(config))
    )


def test_resolved_world_and_policy_are_scientific_inputs():
    prepared = fixture({"interface_contract": CONTRACT})
    for field in ("scenario_json", "policies_json", "parameters_json"):
        assert identity(prepared) != identity(
            replace(
                prepared,
                **{
                    field: '{"changed":{"implementation":{"kind":"rule","name":"spt"}}}'
                    if field == "policies_json"
                    else '{"changed":true}'
                },
            )
        )


def test_resolved_case_locators_are_provenance_not_scientific_inputs():
    config = {"interface_contract": CONTRACT, "validation": {"scenarios": ["old.yaml"]}}
    case = {
        "seed": 101,
        "scenario": {"resolved_world": True},
        "recipe": {
            "settings": {
                "factory": "old-factory.yaml",
                "workload": "old-workload.yaml",
                "horizon": 4096,
            },
            "authoring": {"path": "old.yaml"},
            "workload": {"jobs": 128},
        },
    }
    prepared = replace(fixture(config), validation_json=canonical_json([case]))
    config["validation"]["scenarios"] = ["relocated.yaml"]
    case["recipe"]["settings"].update(
        factory="relocated-factory.yaml", workload="relocated-workload.yaml"
    )
    case["recipe"]["authoring"] = {"path": "relocated.yaml"}
    relocated = replace(
        prepared,
        config_json=canonical_json(config),
        validation_json=canonical_json([case]),
    )
    assert identity(prepared) == identity(relocated)
    case["recipe"]["settings"]["horizon"] = 2048
    assert identity(prepared) != identity(
        replace(relocated, validation_json=canonical_json([case]))
    )
