"""Compile author files into detached, frozen native execution objects."""

import copy
import hashlib
import json
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path

from smartsom.config.authoring_v4 import AlgorithmV2, ExperimentV4
from smartsom.config.codec import ConfigurationError, canonical_json, digest, primitive
from smartsom.config.compositions import Binding, CompositionFile, Group
from smartsom.config.experiment_v3 import (
    CompositionInputs,
    ExecutionConfig,
    TrainingOptionsV3,
    prepare_detached,
)
from smartsom.config.factory_design import load_factory_design_file
from smartsom.config.policies import PolicyFile
from smartsom.config.production import (
    ScenarioFile,
    WorkloadFile,
    materialize,
    named_seed,
)
from smartsom.config.travel_time import freeze_transport


@dataclass(frozen=True)
class AuthorEntry:
    id: str
    task: str
    prepared: object
    sources: dict
    heterogeneity: dict
    workload: dict


@dataclass(frozen=True)
class AuthorPlan:
    experiment: ExperimentV4
    entries: tuple[AuthorEntry, ...]
    authoring: dict

    def summary(self):
        return {
            "status": "checked",
            "input_type": "experiment-v4",
            "task": self.experiment.task,
            "count": len(self.entries),
            "execution": primitive(self.experiment.execution),
            "output": primitive(self.experiment.output),
            "entries": [
                {
                    "id": e.id,
                    "task": e.task,
                    "sources": e.sources,
                    "H": e.heterogeneity,
                    "V": {
                        "level": e.workload.get(
                            "selected_level", e.workload.get("level")
                        ),
                        "measured_v": e.workload.get("V", e.workload.get("measured_v")),
                        "data_seed": e.workload.get("data_seed"),
                        "split": e.workload.get("split"),
                    },
                    "groups": json.loads(e.prepared.policies_json),
                    "training": primitive(e.prepared.config.training),
                    "validation_cases": len(json.loads(e.prepared.validation_json)),
                    "evaluation_cases": len(json.loads(e.prepared.evaluation_json))
                    if e.task != "train"
                    else 0,
                    "frozen_evaluation_cases": len(
                        json.loads(e.prepared.evaluation_json)
                    ),
                    "scientific_sha256": e.prepared.scientific_sha256,
                }
                for e in self.entries
            ],
            "scope": "input checks only; no workers, learners, Ray or performance measurement",
        }


def _validated(model, data):
    try:
        return model.model_validate_json(canonical_json(data))
    except ValueError as exc:
        raise ConfigurationError(str(exc)) from exc


def _read_source(path):
    """Parse and hash exactly the same byte snapshot."""
    import yaml

    from smartsom.config.experiment import _UniqueLoader

    try:
        raw = path.read_bytes()
        document = yaml.load(raw.decode("utf-8"), Loader=_UniqueLoader)
        if not isinstance(document, dict):
            raise ValueError("expected a YAML mapping")
        return document, hashlib.sha256(raw).hexdigest()
    except (OSError, ValueError, yaml.YAMLError) as exc:
        raise ConfigurationError(f"{path}: {exc}") from exc


def _overrides(items):
    import yaml

    from smartsom.config.codec import _UniqueLoader

    result = []
    for item in items:
        if "=" not in item:
            raise ConfigurationError("--set requires NAMESPACE.FIELD=VALUE")
        key, value = item.split("=", 1)
        if (
            key.split(".", 1)[0]
            not in {"factory", "workload", "algorithm", "experiment"}
            or "." not in key
        ):
            raise ConfigurationError(
                "--set requires factory/workload/algorithm/experiment namespace"
            )
        if any(
            key == k or key.startswith(k + ".") or k.startswith(key + ".")
            for k, _ in result
        ):
            raise ConfigurationError("duplicate/overlapping override")
        result.append((key, yaml.load(value, Loader=_UniqueLoader)))
    return result


def _apply(data, changes, namespace):
    value = copy.deepcopy(data)
    for key, item in changes:
        prefix, field = key.split(".", 1)
        if prefix != namespace:
            continue
        node = value
        parts = field.split(".")
        for part in parts[:-1]:
            if part not in node or not isinstance(node[part], dict):
                raise ConfigurationError(f"unknown configuration field: {key}")
            node = node[part]
        if parts[-1] not in node:
            raise ConfigurationError(f"unknown configuration field: {key}")
        node[parts[-1]] = item
    return value


def _algorithm(algorithm, origin, training):
    policies, groups, bindings = {}, {}, {}

    def add(group, role, policy):
        declaration = _validated(
            PolicyFile,
            {
                "schema": "smartsom.frozen-policy/v1",
                "role": role,
                "implementation": primitive(policy),
            },
        )
        if group in policies and primitive(policies[group]) != primitive(declaration):
            raise ConfigurationError(
                "sharing group has conflicting observation/action/network policies"
            )
        policies[group] = declaration
        groups[group] = Group(policy=str(origin))

    if algorithm.mode == "central":
        add("central", "central", algorithm.controller)
        composition = CompositionFile(
            schema="smartsom.frozen-composition/v1",
            pickup_matching=str(origin),
            controller=groups["central"],
        )
    else:
        for role, setting in algorithm.agents.items():
            group = setting.group or role
            add(group, role, setting.default)
            overrides = {}
            for entity, override in setting.overrides.items():
                add(override.group, role, override.policy)
                overrides[entity] = override.group
            bindings[role] = Binding(default=group, overrides=overrides)
        composition = CompositionFile(
            schema="smartsom.frozen-composition/v1",
            pickup_matching=str(origin),
            groups=groups,
            bindings=bindings,
        )
    fresh = tuple(
        g for g, p in policies.items() if p.implementation.kind == "new_model"
    )
    if training and not fresh:
        # Model policies explicitly selected for learning can initialize new training.
        fresh = tuple(
            g for g, p in policies.items() if p.implementation.kind == "model"
        )
    if training and not fresh:
        raise ConfigurationError(
            "training requires at least one model agent; rules are evaluation tasks"
        )
    return composition, policies, fresh


def compile_experiment(
    path,
    *,
    task=None,
    factory=None,
    workload=None,
    algorithm=None,
    sets=(),
    seed=None,
    data_seed=None,
    background=None,
    tuning=None,
    extension_modules=(),
    require_dependencies=False,
    evaluation_overrides=None,
    display_overrides=None,
    purpose=None,
):
    """Read inputs once and compile without allocating output or launching processes."""
    from smartsom.algorithms.rule_registry import load_rule_modules
    from smartsom.config.reliability import (
        factory_heterogeneity,
        factory_reliability_outages,
        validate_factory_conditions,
    )

    modules = load_rule_modules(extension_modules)
    owner = Path(path).expanduser().resolve()
    original, owner_hash = _read_source(owner)
    source_hashes = {owner: owner_hash}
    changes = _overrides(sets)
    explicit = {
        "experiment.task": task,
        "experiment.seed": seed,
        "experiment.data_seed": data_seed,
        "experiment.execution.background": background,
        "experiment.execution.tuning": tuning,
    }
    for key, value in explicit.items():
        if value is not None and any(
            key == changed
            or key.startswith(changed + ".")
            or changed.startswith(key + ".")
            for changed, _ in changes
        ):
            raise ConfigurationError(f"conflicting flag and --set override: {key}")
    data = primitive(_validated(ExperimentV4, original))
    if task is not None:
        data["task"] = task
    for key, val in (("seed", seed), ("data_seed", data_seed)):
        if val is not None:
            data[key] = val
    if background is not None:
        data["execution"]["background"] = background
    if tuning is not None:
        data["execution"]["tuning"] = tuning
    if evaluation_overrides:
        data["evaluation"].update(evaluation_overrides)
    if display_overrides:
        data["logging"].update(display_overrides)
    # Selectors precede field overrides and narrow matrix axes.
    for key, selected in (("factory", factory), ("workload", workload)):
        if selected is not None:
            selected = str(Path(selected).expanduser().resolve())
            if data["matrix"] is not None:
                data["matrix"]["factories" if key == "factory" else "workloads"] = [
                    selected
                ]
            else:
                data[key] = selected
    if algorithm is not None:
        selected = str(Path(algorithm).expanduser().resolve())
        if data["matrix"] is not None and data["matrix"].get("algorithms") is not None:
            data["matrix"]["algorithms"] = [selected]
        else:
            data["algorithm"] = selected
    data = _apply(data, changes, "experiment")
    data["output"]["root"] = str((owner.parent / data["output"]["root"]).resolve())
    experiment = _validated(ExperimentV4, data)
    training = experiment.task != "evaluate"
    if not training and any(
        key.startswith("experiment.training.")
        or key.startswith("algorithm.learner.parameters.")
        or key in {"experiment.training", "algorithm.learner.parameters"}
        for key, _ in changes
    ):
        raise ConfigurationError(
            "training budget/optimizer overrides do not apply to evaluate"
        )
    if not training and (
        experiment.runtime.num_envs != 1 or experiment.runtime.sampling_processes
    ):
        raise ConfigurationError(
            "evaluation uses native case concurrency; learner sampling settings do not apply"
        )
    if experiment.execution.executor == "tune" and experiment.execution.tuning == "off":
        raise ConfigurationError("executor=tune requires tuning=recommend or auto")
    if experiment.execution.tuning != "off":
        experiment = experiment.model_copy(
            update={
                "execution": experiment.execution.model_copy(
                    update={"executor": "tune"}
                )
            }
        )
    if experiment.execution.tuning != "off" or experiment.execution.executor == "tune":
        if not training or experiment.task != "train-evaluate":
            raise ConfigurationError(
                "tuning currently requires a learning train-evaluate task"
            )
    if experiment.task == "train-evaluate":
        from smartsom.experiments.commands import _checkpoint_conditions

        # Budget/validation/checkpoint requirements do not depend on file format.
        _checkpoint_conditions(experiment)
    fs = experiment.matrix.factories if experiment.matrix else (experiment.factory,)
    ws = experiment.matrix.workloads if experiment.matrix else (experiment.workload,)
    algorithms = (
        experiment.matrix.algorithms
        if experiment.matrix and experiment.matrix.algorithms is not None
        else (experiment.algorithm,)
    )
    seeds = (
        experiment.matrix.seeds
        if experiment.matrix and experiment.matrix.seeds is not None
        else (experiment.seed,)
    )
    factory_docs = {}
    for value in fs:
        p = (owner.parent / value).resolve()
        document, source_hashes[p] = load_factory_design_file(p)
        factory_data = primitive(document)
        factory_data.setdefault("reliability", None)
        document = _validated(type(document), _apply(factory_data, changes, "factory"))
        factory_docs[p] = document
    validate_factory_conditions(tuple(factory_docs.values()))
    algorithm_docs = {}
    for value in algorithms:
        algorithm_path = (owner.parent / value).resolve()
        if algorithm_path in algorithm_docs:
            continue
        raw, source_hashes[algorithm_path] = _read_source(algorithm_path)
        method = _validated(AlgorithmV2, raw)
        method = _validated(
            AlgorithmV2, _apply(primitive(method), changes, "algorithm")
        )
        if training and method.mode == "rules":
            raise ConfigurationError("rules mode supports evaluate only")
        if method.mode == "rules" and experiment.runtime.device != "cpu":
            raise ConfigurationError("framework-free rules require runtime.device=cpu")
        composition, policies, train_groups = _algorithm(
            method, algorithm_path, training
        )
        reward = primitive(method.learner.reward) if method.learner else None
        if reward is not None:
            reward.setdefault("roles", {})
            for role, setting in method.agents.items():
                if setting.reward is not None:
                    if role + "_policy" in reward["roles"]:
                        raise ConfigurationError(
                            "duplicate role reward in learner and agent"
                        )
                    reward["roles"][role + "_policy"] = primitive(setting.reward)
        algorithm_docs[algorithm_path] = (
            method,
            composition,
            policies,
            train_groups,
            reward,
        )
    entries, identities = [], set()
    provenance = {
        "schema": "smartsom.authoring-sources/v1",
        "experiment": {"path": str(owner), "document": original},
        "overrides": list(changes),
        "extension_modules": modules,
    }
    if purpose is not None:
        provenance["purpose"] = purpose
    workload_docs = {}
    for index, (f, w, algorithm_ref, policy_seed) in enumerate(
        product(fs, ws, algorithms, seeds)
    ):
        fp, wp = (owner.parent / f).resolve(), (owner.parent / w).resolve()
        algorithm_path = (owner.parent / algorithm_ref).resolve()
        method, composition, policies, train_groups, reward = algorithm_docs[
            algorithm_path
        ]
        document = factory_docs[fp]
        if wp not in workload_docs:
            workload_docs[wp], source_hashes[wp] = _read_source(wp)
        raw_workload = workload_docs[wp]
        if raw_workload.get("schema") == "smartsom.workload/v3":
            from smartsom.config.workload_v3 import WorkloadV3
            from smartsom.workloads.workload_v3 import materialize_workload

            wc = _validated(WorkloadV3, raw_workload)
            wc = _validated(WorkloadV3, _apply(primitive(wc), changes, "workload"))

            def dataset(split, replication):
                if wc.demands is not None or wc.profile is not None:
                    simple = WorkloadFile(
                        schema="smartsom.frozen-workload/v1",
                        demands=wc.demands,
                        profile=wc.profile,
                    )
                    ds = named_seed(experiment.data_seed, f"{split}:{replication}")
                    if wc.profile is not None:
                        env = experiment.runtime.environment
                        settings = ScenarioFile(
                            schema="smartsom.frozen-scenario/v1",
                            factory=str(fp),
                            workload=str(wp),
                            mode=wc.mode,
                            tick_limit=wc.tick_limit or env.tick_limit or 1000,
                        )
                        generated = materialize(document.factory, simple, settings, ds)
                        simple = WorkloadFile(
                            schema="smartsom.frozen-workload/v1",
                            demands=generated.demands,
                        )
                    return (
                        simple,
                        {
                            "kind": "fixed" if wc.demands is not None else "profile",
                            "data_seed": ds,
                            "split": split,
                            "measured_v": None,
                            "level": None,
                        },
                        wc.tick_limit
                        or experiment.runtime.environment.tick_limit
                        or 1000,
                        wc.mode,
                    )
                frozen = materialize_workload(
                    wc,
                    named_seed(experiment.data_seed, f"{split}:{replication}"),
                    split,
                )
                return (
                    _validated(
                        WorkloadFile,
                        {
                            **primitive(frozen.workload),
                            "schema": "smartsom.frozen-workload/v1",
                        },
                    ),
                    frozen.provenance,
                    frozen.tick_limit,
                    frozen.mode,
                )
        else:
            raise ConfigurationError("workload must use smartsom.workload/v3")

        def world(split, replication):
            wf, info, ticks, mode = dataset(split, replication)
            env = primitive(experiment.runtime.environment)
            env["tick_limit"] = env["tick_limit"] or ticks
            env["mode"] = mode
            ds = named_seed(experiment.data_seed, f"{split}:{replication}")
            env["outages"] = primitive(
                factory_reliability_outages(
                    document, named_seed(ds, "faults"), env["tick_limit"]
                )
            )
            settings = _validated(
                ScenarioFile,
                {
                    "schema": "smartsom.frozen-scenario/v1",
                    "factory": str(fp),
                    "workload": str(wp),
                    **env,
                },
            )
            settings = freeze_transport(settings, owner)
            scenario = materialize(
                document.factory, wf, settings, named_seed(ds, "environment")
            )
            return scenario, settings, wf, info, ds

        scenario, settings, wf, info, ds = world(
            "train" if training else "evaluation", 0
        )
        cases = {}
        for split, options in (
            ("validation", experiment.validation),
            ("evaluation", experiment.evaluation),
        ):
            rows = []
            if (
                split == "validation" and training and options.enabled
            ) or split == "evaluation":
                for rep in range(options.replications):
                    sc, st, work, metadata, case_seed = world(split, rep)
                    rows.append(
                        {
                            "case": "0",
                            "replication": rep,
                            "seed": case_seed,
                            "scenario": primitive(sc),
                            "recipe": {
                                "settings": primitive(st),
                                "workload": primitive(work),
                                "data_seed": case_seed,
                                "workload_provenance": metadata,
                            },
                        }
                    )
            cases[split] = rows
        train_options = None
        if training:
            train_options = TrainingOptionsV3.model_validate_json(
                canonical_json(
                    {
                        **primitive(experiment.training),
                        "mode": method.mode,
                        "backend": method.learner.backend,
                        "algorithm": method.learner.algorithm,
                        "groups": train_groups,
                        "gamma": method.learner.gamma,
                        "reward": reward,
                    }
                )
            )
        runtime = primitive(experiment.runtime)
        runtime.pop("environment")
        validation_options = {
            **primitive(experiment.validation),
            "seed": named_seed(experiment.data_seed, "validation"),
        }
        evaluation_options = {
            **primitive(experiment.evaluation),
            "seed": named_seed(experiment.data_seed, "evaluation"),
        }
        config = ExecutionConfig.model_validate_json(
            canonical_json(
                {
                    "schema": "smartsom.execution-config/v2",
                    "seed": policy_seed,
                    "training": primitive(train_options),
                    "runtime": runtime,
                    "validation": validation_options,
                    "evaluation": evaluation_options,
                    "checkpointing": primitive(experiment.checkpointing),
                    "logging": primitive(experiment.logging),
                    "output": primitive(experiment.output),
                }
            )
        )
        config._owner = owner
        sources = {
            "factory": str(fp),
            "workload": str(wp),
            "algorithm": str(algorithm_path),
            "experiment": str(owner),
        }
        source_data = {
            "factory": primitive(document),
            "workload": primitive(wc),
            "algorithm": primitive(method),
            "experiment": primitive(experiment),
        }
        metadata = {
            "data_seed": ds,
            "workload_provenance": info,
            "authoring": {
                **provenance,
                "sources": sources,
                "documents": source_data,
                "input_sha256": {
                    str(p): source_hashes[p] for p in (owner, fp, wp, algorithm_path)
                },
            },
        }
        detached = CompositionInputs(
            scenario,
            settings,
            wf,
            composition,
            {"schema": "smartsom.pickup-matching/v2", "name": method.pickup_matching},
            policies,
            {g: algorithm_path for g in policies},
            method.learner.parameters if method.learner else {},
            cases["validation"],
            cases["evaluation"],
            {sources[k]: digest(v) for k, v in source_data.items()},
            metadata,
        )
        prepared = prepare_detached(
            config,
            training=training,
            require_dependencies=require_dependencies,
            inputs=detached,
        )
        from smartsom.algorithms.rule_registry import freeze_rule

        declarations = json.loads(prepared.policies_json)
        for declaration in declarations.values():
            impl = declaration["implementation"]
            if impl["kind"] == "rule":
                from smartsom.algorithms.production_rules import RULES

                if (
                    impl["name"] in RULES.get(declaration["role"], ())
                    and impl["parameters"]
                ):
                    raise ConfigurationError(
                        "builtin rules currently expose no parameters"
                    )
                declaration["resolved_rule"] = freeze_rule(
                    declaration["role"],
                    impl["name"],
                    version=impl.get("version"),
                    parameters=impl["parameters"],
                    code_sha256=impl.get("code_sha256"),
                )
        from dataclasses import replace

        prepared = replace(prepared, policies_json=canonical_json(declarations))
        prepared = replace(prepared, scientific_sha256=scientific_identity(prepared))
        identity = digest(
            {
                "factory": {
                    "factory": primitive(document.factory),
                    "reliability": primitive(document.reliability),
                },
                "workload": primitive(wc),
                "algorithm": primitive(method),
                "policy_seed": policy_seed,
                "data_seed": experiment.data_seed,
            }
        )
        if identity in identities:
            raise ConfigurationError("duplicate execution combination")
        identities.add(identity)
        entries.append(
            AuthorEntry(
                f"entry-{index + 1:04d}",
                experiment.task,
                prepared,
                sources,
                asdict(factory_heterogeneity(document)),
                info,
            )
        )
    return AuthorPlan(experiment, tuple(entries), provenance)


def scientific_identity(prepared):
    """v4 scientific identity excludes recorded execution, display and path origins."""
    config = json.loads(prepared.config_json)
    for key in ("logging", "output", "scenario", "composition"):
        config.pop(key, None)
    composition = json.loads(prepared.composition_json)
    composition.pop("pickup_matching", None)
    for group in composition.get("groups", {}).values():
        group.pop("policy", None)
    if composition.get("controller"):
        composition["controller"].pop("policy", None)
    recipe = json.loads(prepared.training_inputs_json)
    recipe.pop("authoring", None)

    def without_paths(value):
        result = copy.deepcopy(value)
        result["settings"].pop("factory", None)
        result["settings"].pop("workload", None)
        return result

    recipe = without_paths(recipe)
    cases = {}
    for label, encoded in (
        ("validation", prepared.validation_json),
        ("evaluation", prepared.evaluation_json),
    ):
        rows = json.loads(encoded)
        for row in rows:
            row["recipe"] = without_paths(row["recipe"])
        cases[label] = rows
    policies = json.loads(prepared.policies_json)
    for declaration in policies.values():
        impl = declaration["implementation"]
        if impl["kind"] == "model":
            impl["model"].pop("source", None)
            if declaration.get("resolved_model"):
                declaration["resolved_model"].pop("source", None)
                declaration["resolved_model"].pop("original_source", None)
    return digest(
        {
            "config": config,
            "scenario": json.loads(prepared.scenario_json),
            "composition": composition,
            "policies": policies,
            "parameters": json.loads(prepared.parameters_json),
            "training_inputs": recipe,
            **cases,
        }
    )
