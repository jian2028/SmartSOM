"""Opt-in semantic freeze; legacy identities keep their historical algorithms."""

import json
from dataclasses import asdict

from smartsom.config.codec import digest

CONTRACT = "smartsom.configurable-experiment/v1"


def identity(prepared):
    fields = asdict(prepared)
    config = json.loads(fields["config_json"])
    if config.get("interface_contract") != CONTRACT:
        raise ValueError("semantic freeze requires the explicit interface contract")
    config.pop("output", None)
    config.pop("logging", None)
    # These locators have already been resolved into the frozen payload below.
    for name in ("scenario", "composition"):
        config.pop(name, None)
    if config.get("training"):
        config["training"].pop("parameters", None)
    config.get("runtime", {}).pop("numerical_threads", None)
    config.get("runtime", {}).pop("max_concurrent", None)
    config.get("checkpointing", {}).pop("retention", None)
    for name in ("validation", "evaluation"):
        config.get(name, {}).pop("scenarios", None)
    inputs = json.loads(fields["training_inputs_json"])
    # Original author files/hashes remain in preparation and run provenance.
    # Actual resolved policies, worlds and optimizer settings define science.
    inputs.pop("authoring", None)
    for name in ("factory", "workload"):
        inputs.get("settings", {}).pop(name, None)
    policies = json.loads(fields["policies_json"])
    for declaration in policies.values():
        model = declaration.get("resolved_model")
        if model is not None:
            model.pop("source", None)
            declaration["implementation"].get("model", {}).pop("source", None)
    composition = json.loads(fields["composition_json"])
    composition.pop("pickup_matching", None)
    for group in composition.get("groups", {}).values():
        group.pop("policy", None)
    if isinstance(composition.get("controller"), dict):
        composition["controller"].pop("policy", None)
    cases = {}
    for name in ("validation_json", "evaluation_json"):
        cases[name] = json.loads(fields[name])
        for case in cases[name]:
            recipe = case.get("recipe", {})
            recipe.pop("authoring", None)
            for locator in ("factory", "workload"):
                recipe.get("settings", {}).pop(locator, None)
    return digest(
        {
            "contract": CONTRACT,
            "config": config,
            "training_inputs": inputs,
            "policies": policies,
            "composition": composition,
            **{
                name: json.loads(fields[name])
                for name in (
                    "scenario_json",
                    "parameters_json",
                )
            },
            **cases,
        }
    )
