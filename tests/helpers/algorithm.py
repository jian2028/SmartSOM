"""Concise current-component fixtures; no historical schema parser."""

from copy import deepcopy

from smartsom.config.base_diagnostic import (
    BaseAlgorithm,
    DecisionProtocol,
    LearnerParameters,
)


def algorithm_document(recipe):
    recipe = dict(recipe)
    if "schema" in recipe:
        raise ValueError("test recipes name current components, not historical schemas")
    method = recipe.pop("method", "ppo")
    control = recipe.pop("control", "base")
    arm = recipe.pop("social_arm", "base")
    pool = recipe.pop("social_pool", "attention")
    bindings, providers = recipe.pop("bindings", {}), recipe.pop("providers", {})
    observation, network = recipe.pop("observation", None), recipe.pop("network", None)
    protocol = DecisionProtocol().model_dump(mode="json")
    stop = recipe.pop("allow_stop", True)
    protocol.update(
        pre={"accept": {"allow_stop": stop}},
        post={"accept": {"allow_stop": stop}},
        dispatcher={"pickup": {"allow_empty_wait": False}},
    )
    protocol["dispatcher"]["pickup"]["allow_empty_wait"] = recipe.pop(
        "allow_empty_wait", False
    )
    for k in tuple(recipe):
        if k in protocol:
            protocol[k] = recipe.pop(k)
    params = LearnerParameters.model_validate(recipe).model_dump(mode="json")
    names = {method, *bindings.values()}
    policies = {}
    for name in names:
        if name in ("ppo", "dqn"):
            representation = "base" if arm == "base" else "social"
            spec = {
                "kind": "learning",
                "learner": {
                    "target": f"smartsom.learning.{name}.create_learner",
                    "params": params,
                },
                "observation": observation
                or {
                    "target": f"smartsom.representations.{representation}.observation.create_observation",
                    "params": {"arm": arm} if arm != "base" else {},
                },
                "encoder": {
                    "target": f"smartsom.representations.{representation}.encoder.create_encoder",
                    "params": {"pool": pool} if arm != "base" else {},
                },
            }
            if network:
                spec.pop("encoder")
                spec["network"] = network
        else:
            obs = (
                "smartsom.observations.central.create_observation"
                if control == "centralized"
                else (
                    "smartsom.observations.rules.create_random_observation"
                    if name == "random"
                    else "smartsom.observations.rules.create_observation"
                )
            )
            spec = {
                "kind": "policy",
                "implementation": providers.get(
                    name, {"target": f"smartsom.policies.{name}.create_policy"}
                ),
                "observation": observation or {"target": obs},
            }
        policies[name] = spec
    for spec in policies.values():
        for kind in ("observation", "network", "implementation"):
            if kind in spec and "path" in spec[kind]:
                spec[kind].setdefault(
                    "factory",
                    "create_policy" if kind == "implementation" else "create_" + kind,
                )
    result = {
        "schema": "smartsom.algorithm-components/v2",
        "control": control,
        "action_representation": {"default": "direct"},
        "production": {"default": method, "policies": policies, "bindings": bindings},
        "decision_protocol": protocol,
    }
    if control == "centralized":
        for name in ("pre", "post", "dispatcher"):
            result["decision_protocol"].pop(name)
    return BaseAlgorithm.model_validate(result).model_dump(mode="json", by_alias=True)


def algorithm_options(**recipe):
    return BaseAlgorithm.model_validate(algorithm_document(recipe))


def with_protocol(options, **fields):
    data = deepcopy(options.model_dump(mode="json", by_alias=True))
    if "allow_stop" in fields:
        value = fields.pop("allow_stop")
        for group in ("pre", "post"):
            data["decision_protocol"][group]["accept"]["allow_stop"] = value
    if "allow_empty_wait" in fields:
        data["decision_protocol"]["dispatcher"]["pickup"]["allow_empty_wait"] = (
            fields.pop("allow_empty_wait")
        )
    if fields:
        raise ValueError(fields)
    return BaseAlgorithm.model_validate(data)
