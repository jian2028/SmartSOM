"""Explicit one-way migration. Never called by the current runtime parser."""

import copy
import hashlib
import json
import shutil
from itertools import product
from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, model_validator

from smartsom.config.base_diagnostic import (
    BaseAlgorithm,
    DecisionProtocol,
    LearnerParameters,
    Strict,
)
from smartsom.domain.base_observation import ObservationVersion


class PolicyProvider(Strict):
    path: str
    factory: str = "create_policy"
    sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")


class PythonModule(Strict):
    path: str
    sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")


class LegacyAlgorithm(Strict):
    schema_id: Literal["smartsom.base-algorithm/v1"] = Field(alias="schema")
    method: Literal["ppo", "dqn", "rule", "random"] = "ppo"
    control: Literal["base", "centralized"] = "base"
    observation_schema: ObservationVersion = "native-v2"
    observation: PythonModule | None = None
    network: PythonModule | None = None
    social_arm: Literal[
        "base",
        "machine",
        "dispatcher-pickup",
        "dispatcher-delivery",
        "dispatcher",
        "pre",
        "post",
        "input",
        "all",
    ] = "base"
    social_pool: Literal["attention", "mean-max"] = "attention"
    replay_capacity: int = Field(default=10000, gt=0)
    target_update_interval: int = Field(default=100, gt=0)
    epsilon: float = Field(default=0.1, ge=0, le=1)
    allow_stop: bool = True
    allow_empty_wait: bool = False
    bindings: dict[str, str] = Field(default_factory=dict)
    providers: dict[str, PolicyProvider] = Field(default_factory=dict)
    width: int = Field(default=32, ge=8)
    learning_rate: float = Field(default=0.0003, gt=0)
    epochs: int = Field(default=3, gt=0)
    batch_size: int = Field(default=256, gt=0)
    clip: float = Field(default=0.2, gt=0, lt=1)
    entropy: float = Field(default=0.01, ge=0)
    reevaluate_ticks: int = Field(default=10, gt=0)
    time_scale: float = Field(default=200.0, gt=0)
    distance_scale: float = Field(default=50.0, gt=0)
    count_scale: float = Field(default=20.0, gt=0)
    quality_penalty: float = Field(default=1000.0, ge=0)
    learning_reward_scale: float = Field(default=1.0, gt=0, allow_inf_nan=False)
    critic_initial_scale: float = Field(default=1.0, gt=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def valid_bindings(self):
        from smartsom.algorithms.base_policy import BUILTINS, DECISION_POINTS

        points = (
            {"central.assign"} if self.control == "centralized" else DECISION_POINTS
        )
        if set(self.bindings) - points:
            raise ValueError("unknown Base role.stage policy binding")
        if set(self.providers) & BUILTINS:
            raise ValueError("custom providers cannot replace builtin identifiers")
        if set(self.bindings.values()) - BUILTINS - set(self.providers):
            raise ValueError("policy binding names an undeclared provider")
        names = {self.bindings.get(point, self.method) for point in points}
        if self.control == "centralized" and (
            names & {"ppo", "dqn"} or self.social_arm != "base"
        ):
            raise ValueError(
                "centralized control supports rule/random/custom; its DRL learner is external"
            )
        if {"ppo", "dqn"} <= names:
            raise ValueError("one experiment selects PPO or DQN, not both")
        if set(self.providers) - set(self.bindings.values()):
            raise ValueError("declared policy provider is unused")
        return self


def migrate_algorithm(document):
    if document.get("schema") != "smartsom.base-algorithm/v1":
        raise ValueError(
            "migration supports smartsom.base-algorithm/v1 only; historical versions require their archive"
        )
    old = LegacyAlgorithm.model_validate(document)
    if old.observation_schema != "native-v2":
        raise ValueError(
            "native-v1 observations require archived source; migration cannot silently change layout"
        )
    from smartsom.algorithms.base_policy import DECISION_POINTS

    points = {"central.assign"} if old.control == "centralized" else DECISION_POINTS
    names = {old.bindings.get(point, old.method) for point in points}
    default = old.method if old.method in names else sorted(names)[0]
    policies = {}
    for name in sorted(names):
        if name in {"ppo", "dqn"}:
            social = old.social_arm != "base"
            representation = "social" if social else "base"
            observation = {
                "target": f"smartsom.representations.{representation}.observation.create_observation",
                "params": {"arm": old.social_arm} if social else {},
            }
            encoder = {
                "target": f"smartsom.representations.{representation}.encoder.create_encoder",
                "params": {"pool": old.social_pool} if social else {},
            }
            policy = {
                "kind": "learning",
                "learner": {
                    "target": f"smartsom.learning.{name}.create_learner",
                    "params": {
                        k: getattr(old, k) for k in LearnerParameters.model_fields
                    },
                },
                "observation": observation,
                "encoder": encoder,
            }
            if old.observation:
                policy["observation"] = {
                    "path": old.observation.path,
                    "factory": "create_observation",
                }
            if old.network:
                policy.pop("encoder")
                policy["network"] = {
                    "path": old.network.path,
                    "factory": "create_network",
                }
            policies[name] = policy
        elif name in {"rule", "random"}:
            policies[name] = {
                "kind": "policy",
                "implementation": {"target": f"smartsom.policies.{name}.create_policy"},
            }
        else:
            spec = old.providers[name]
            policies[name] = {
                "kind": "policy",
                "implementation": {"path": spec.path, "factory": spec.factory},
            }
        if old.observation and policies[name]["kind"] == "policy":
            policies[name]["observation"] = {
                "path": old.observation.path,
                "factory": "create_observation",
            }
    result = {
        "schema": "smartsom.algorithm-components/v1",
        "control": old.control,
        "production": {
            "default": default,
            "bindings": old.bindings,
            "policies": policies,
        },
        "traffic": {"mover": {"target": "smartsom.traffic.rule_mover.create_mover"}},
    }
    result["decision_protocol"] = {
        k: getattr(old, k)
        for k in DecisionProtocol.model_fields
        if k not in {"pre", "post", "dispatcher"}
    }
    if old.control == "base":
        result["decision_protocol"].update(
            pre={"accept": {"allow_stop": old.allow_stop}},
            post={"accept": {"allow_stop": old.allow_stop}},
            dispatcher={"pickup": {"allow_empty_wait": old.allow_empty_wait}},
        )
    BaseAlgorithm.model_validate(result)
    return result


def migrate_experiment(raw, legacy, source):
    """Preserve coupled old axes (one allow_stop axis still makes two variants)."""
    result = copy.deepcopy(raw)
    for key in ("output", "calibration", "checkpoint"):
        value = result.get("run", {}).get(key)
        if value:
            result["run"][key] = str(
                (source.parent / Path(value).expanduser()).resolve()
            )
    if "matrix" not in result:
        return result
    matrix, groups = {}, {"production": {}, "decision_protocol": {}}
    protocol_fields = set(DecisionProtocol.model_fields) - {"pre", "post", "dispatcher"}
    for key, values in result["matrix"].items():
        if not key.startswith("algorithm."):
            if key in {"experiment.run.calibration", "experiment.run.checkpoint"}:
                values = [
                    str((source.parent / Path(value).expanduser()).resolve())
                    if value
                    else value
                    for value in values
                ]
            matrix[key] = values
            continue
        field = key.removeprefix("algorithm.")
        if not isinstance(values, list) or not values:
            raise ValueError(f"invalid migration matrix axis: {key}")
        if field in {
            "control",
            "observation",
            "network",
            "providers",
            "observation_schema",
        } or field.startswith(("observation.", "network.", "providers.")):
            raise ValueError(
                f"matrix axis {key} needs explicit manual component migration"
            )
        group = (
            "decision_protocol"
            if field in protocol_fields | {"allow_stop", "allow_empty_wait"}
            else "production"
        )
        groups[group][field] = values
    for group, axes in groups.items():
        if not axes:
            continue
        variants = []
        for values in product(*axes.values()):
            document = copy.deepcopy(legacy)
            for key, value in zip(axes, values, strict=True):
                node = document
                parts = key.split(".")
                for part in parts[:-1]:
                    node = node.setdefault(part, {})
                node[parts[-1]] = value
            variants.append(migrate_algorithm(document)[group])
        matrix["algorithm." + group] = variants
    result["matrix"] = matrix
    return result


def migrate(source, destination):
    from smartsom.config.codec import read_document
    from smartsom.config.components import absolute_components, validate_components

    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.exists():
        raise ValueError("migration destination already exists; choose a new directory")
    raw = read_document(source)
    experiment = raw.get("schema") == "smartsom.base-experiment/v1"
    algorithm_path = (
        (source.parent / raw["algorithm"]).resolve() if experiment else source
    )
    legacy_document = read_document(algorithm_path)
    if (
        experiment
        and legacy_document.get("schema") == "smartsom.algorithm-components/v1"
    ):
        from smartsom.config.scenario_migration import migrate_scene_experiment

        return migrate_scene_experiment(source, destination)
    converted = migrate_algorithm(legacy_document)
    legacy = LegacyAlgorithm.model_validate(legacy_document)
    source_checks = {}
    for name, spec in [
        ("observation", legacy.observation),
        ("network", legacy.network),
        *legacy.providers.items(),
    ]:
        if spec is None:
            continue
        original = (algorithm_path.parent / spec.path).resolve()
        actual = hashlib.sha256(original.read_bytes()).hexdigest()
        if spec.sha256 and spec.sha256 != actual:
            raise ValueError(f"legacy source mismatch: {name}: {original}")
        source_checks[name] = {
            "path": str(original),
            "sha256": actual,
            "pinned": spec.sha256 is not None,
        }
    migrated_experiment = (
        migrate_experiment(raw, legacy_document, source) if experiment else None
    )
    declarations = [converted]
    for production in (
        (migrated_experiment or {}).get("matrix", {}).get("algorithm.production", [])
    ):
        declarations.append({**converted, "production": production})
    extensions = {}
    # Include components activated only by a matrix variant (e.g. rule -> PPO).
    # Validate every variant before creating any output directory.
    for declaration in declarations:
        options = BaseAlgorithm.model_validate(declaration)
        validate_components(absolute_components(options, algorithm_path.parent))
        for key, spec in options.component_items():
            if spec.path:
                extensions.setdefault(
                    spec.path, "extensions/" + key.replace(".", "-") + ".py"
                )

    def rebase(value):
        if isinstance(value, dict):
            if isinstance(value.get("path"), str) and value["path"] in extensions:
                value["path"] = extensions[value["path"]]
            for child in value.values():
                rebase(child)
        elif isinstance(value, list):
            for child in value:
                rebase(child)

    destination.mkdir(parents=True)
    try:
        if extensions:
            (destination / "extensions").mkdir()
        for original, relative in extensions.items():
            shutil.copyfile(algorithm_path.parent / original, destination / relative)
        rebase(converted)
        if migrated_experiment:
            rebase(migrated_experiment.get("matrix", {}))
        options = BaseAlgorithm.model_validate(converted)
        (destination / "algorithm.yaml").write_text(
            yaml.safe_dump(converted, sort_keys=False)
        )
        if experiment:
            ex = migrated_experiment
            for name in ("factory", "workload"):
                shutil.copyfile(
                    source.parent / ex[name], destination / (name + ".yaml")
                )
                ex[name] = name + ".yaml"
            ex["algorithm"] = "algorithm.yaml"
            (destination / "experiment.yaml").write_text(
                yaml.safe_dump(ex, sort_keys=False)
            )
        report = {
            "source": str(source),
            "destination": str(destination),
            "old_schema": "smartsom.base-algorithm/v1",
            "new_schema": "smartsom.algorithm-components/v1",
            "source_checks": source_checks,
            "matrix_axes": list((migrated_experiment or {}).get("matrix", {})),
            "run_paths": (migrated_experiment or {}).get("run", {}),
            "effective_default": converted["production"]["default"],
            "mappings": {
                "method/bindings/providers": "production",
                "allow_stop": "decision_protocol.pre/post.accept.allow_stop",
                "allow_empty_wait": "decision_protocol.dispatcher.pickup.allow_empty_wait",
                "social_arm/social_pool": "observation/encoder.params",
                "implicit SharedMover": "traffic.mover",
            },
            "defaults": options.model_dump(mode="json", by_alias=True),
            "checkpoint_compatibility": "new experiment only; resume old checkpoints with archived source",
        }
        (destination / "migration.json").write_text(json.dumps(report, indent=2))
    except BaseException:
        shutil.rmtree(destination)
        raise
    return report
