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
from smartsom.config.experiment_streams import explicit as explicit_streams
from smartsom.config.experiment_streams import (
    frozen_streams,
    frozen_world_streams,
    roots,
)
from smartsom.config.experiment_v3 import (
    CompositionInputs,
    ExecutionConfig,
    TrainingOptionsV3,
    prepare_detached,
)
from smartsom.config.factory_design import load_factory_design_file
from smartsom.config.frozen_inputs import resolve_world
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
        from smartsom.experiments.evaluation_resume import selected_cases

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
                    "evaluation_cases": len(selected_cases(e.prepared))
                    if e.task != "train"
                    else 0,
                    **(
                        {
                            "initial_control_cases": len(
                                selected_cases(e.prepared, "initial_control")
                            )
                        }
                        if e.prepared.config.evaluation.initial_control
                        else {}
                    ),
                    "frozen_evaluation_cases": len(
                        json.loads(e.prepared.evaluation_json)
                    ),
                    "scientific_sha256": e.prepared.scientific_sha256,
                    **(
                        {
                            "seed_contract": json.loads(
                                e.prepared.training_inputs_json
                            )["seed_contract"]
                        }
                        if "seed_contract"
                        in json.loads(e.prepared.training_inputs_json)
                        else {}
                    ),
                    **(
                        {"task_reward": primitive(e.prepared.config.task_reward)}
                        if e.prepared.config.task_reward is not None
                        else {}
                    ),
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


def _supplied_overrides(document, changes):
    """Track explicit inputs after validated overrides, without injecting defaults."""
    result = copy.deepcopy(document)
    for key, value in changes:
        if not key.startswith("experiment."):
            continue
        parts = key.split(".")[1:]
        target = result
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = copy.deepcopy(value)
    return result


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


def _algorithm(algorithm, origin, training, selected_groups=None):
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
    if training and selected_groups is not None:
        if not selected_groups or len(set(selected_groups)) != len(selected_groups):
            raise ConfigurationError("training groups must be nonempty and unique")
        if any(
            group not in policies or policies[group].implementation.kind == "rule"
            for group in selected_groups
        ):
            raise ConfigurationError(
                "training groups must select existing model groups"
            )
        fresh = tuple(selected_groups)
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
    batch_calibration=None,
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
    supplied = copy.deepcopy(original)
    if task is not None:
        data["task"] = task
    for key, val in (("seed", seed), ("data_seed", data_seed)):
        if val is not None:
            data[key] = val
            supplied[key] = val
    if background is not None:
        data["execution"]["background"] = background
    if tuning is not None:
        data["execution"]["tuning"] = tuning
    if evaluation_overrides:
        data["evaluation"].update(evaluation_overrides)
        supplied.setdefault("evaluation", {}).update(evaluation_overrides)
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
    # Shared directory calibration applies only to its learning stage. Resolve
    # it before mode-dependent defaults/validation, without rereading the YAML.
    if batch_calibration and data["task"] == "train-evaluate":
        for key, value in batch_calibration.items():
            if key not in {
                "calibration_level",
                "calibration_seconds",
                "calibration_candidate",
            }:
                raise ConfigurationError(f"unsupported batch calibration field: {key}")
            changes.append((f"experiment.execution.{key}", value))
    # A section override replaces its mapping; restore typed defaults before
    # inspecting it, while retaining only the fields explicitly supplied.
    data = primitive(_validated(ExperimentV4, _apply(data, changes, "experiment")))
    execution_fields = set(original.get("execution", {}))
    for key, value in changes:
        if key == "experiment.execution":
            execution_fields = set(value)
        elif key.startswith("experiment.execution."):
            execution_fields.add(key.split(".")[2])

    # New daily train-evaluate inputs opt into online execution. Explicit native
    # or off inputs and old frozen plans retain their selected path.
    if (
        data["task"] == "train-evaluate"
        and tuning is None
        and not {"tuning", "executor"}.intersection(execution_fields)
    ):
        data["execution"]["tuning"] = "auto"
    if (
        data["execution"]["tuning"] == "off"
        and "calibration_level" not in execution_fields
    ):
        data["execution"]["calibration_level"] = "off"
    if (
        data["execution"]["tuning"] == "recommend"
        and data["execution"]["calibration_level"] == "online"
    ):
        if "calibration_level" in execution_fields:
            raise ConfigurationError(
                "recommend requires quick or full calibration, not online"
            )
        data["execution"]["calibration_level"] = "quick"
    if (
        data["execution"]["calibration_level"] in {"online", "off"}
        and data["execution"].get("calibration_seconds") is not None
    ):
        raise ConfigurationError(
            "online/off execution has no isolated calibration timeout"
        )
    data["output"]["root"] = str((owner.parent / data["output"]["root"]).resolve())
    experiment = _validated(ExperimentV4, data)
    if experiment.frozen_inputs is not None:
        authored = _supplied_overrides(supplied, changes)
        if "data_seed" in authored or "environment" in authored.get("runtime", {}):
            raise ConfigurationError(
                "frozen_inputs owns data seeds and complete environment physics"
            )
        if any(
            "replications" in authored.get(phase, {})
            for phase in ("validation", "evaluation")
        ):
            raise ConfigurationError("frozen cases cannot mix with phase replications")
        if any(key.startswith("workload.") for key, _ in changes):
            raise ConfigurationError("workload overrides cannot modify frozen worlds")
    if (
        experiment.execution.tuning != "off"
        and experiment.execution.calibration_level == "online"
        and experiment.execution.scheduling != "adaptive"
    ):
        raise ConfigurationError(
            "online feedback requires adaptive scheduling; use calibration_level=off for a fixed layout"
        )
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
        else (
            experiment.seed
            if experiment.policy_seed is None
            else experiment.policy_seed,
        )
    )
    if (
        experiment.execution.tuning == "auto"
        and experiment.execution.calibration_level == "online"
        and "max_concurrent" not in execution_fields
    ):
        experiment = experiment.model_copy(
            update={
                "execution": experiment.execution.model_copy(
                    update={
                        "max_concurrent": len(fs)
                        * len(ws)
                        * len(algorithms)
                        * len(seeds)
                    }
                )
            }
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
            method,
            algorithm_path,
            training,
            experiment.training.groups if training else None,
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
            if experiment.task_reward is not None:
                reward["task"] = primitive(experiment.task_reward)
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
    frozen_sources = {}

    def read_frozen_source(path):
        if path not in frozen_sources:
            frozen_sources[path] = _read_source(path)
        return frozen_sources[path]

    for index, (f, w, algorithm_ref, policy_seed) in enumerate(
        product(fs, ws, algorithms, seeds)
    ):
        fp = (owner.parent / f).resolve()
        frozen = experiment.frozen_inputs
        first_input = (
            (frozen.training if training else frozen.evaluation[0]) if frozen else None
        )
        wp = (owner.parent / (first_input.world if frozen else w)).resolve()
        algorithm_path = (owner.parent / algorithm_ref).resolve()
        method, composition, policies, train_groups, reward = algorithm_docs[
            algorithm_path
        ]
        seed_roots = roots(experiment, policy_seed)
        document = factory_docs[fp]
        if not frozen and wp not in workload_docs:
            workload_docs[wp], source_hashes[wp] = _read_source(wp)
        raw_workload = workload_docs[wp] if not frozen else {}
        if frozen:
            wc = frozen
        elif raw_workload.get("schema") == "smartsom.workload/v3":
            from smartsom.config.workload_v3 import WorkloadV3
            from smartsom.workloads.workload_v3 import (
                apply_workload_timing,
                materialize_workload,
            )

            wc = _validated(WorkloadV3, raw_workload)
            wc = _validated(WorkloadV3, _apply(primitive(wc), changes, "workload"))

            def dataset(split, replication):
                if wc.demands is not None or wc.profile is not None:
                    simple = WorkloadFile(
                        schema="smartsom.frozen-workload/v1",
                        demands=wc.demands,
                        profile=wc.profile,
                    )
                    ds = named_seed(seed_roots["world"], f"{split}:{replication}")
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
                    demands, timing = apply_workload_timing(
                        wc, simple.demands, document.factory, ds, split
                    )
                    if timing is not None:
                        simple = WorkloadFile(
                            schema="smartsom.frozen-workload/v1", demands=demands
                        )
                    return (
                        simple,
                        {
                            "kind": "fixed" if wc.demands is not None else "profile",
                            "data_seed": ds,
                            "split": split,
                            "measured_v": None,
                            "level": None,
                            **({"timing": timing} if timing is not None else {}),
                        },
                        wc.tick_limit
                        or experiment.runtime.environment.tick_limit
                        or 1000,
                        wc.mode,
                    )
                frozen = materialize_workload(
                    wc,
                    named_seed(seed_roots["world"], f"{split}:{replication}"),
                    split,
                    factory=document.factory,
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

        frozen_recipes = {}

        def world(split, replication):
            if frozen:
                reference = (
                    frozen.training
                    if split == "train"
                    else getattr(frozen, split)[replication]
                )
                sc, st, work, metadata, case_seed, recipe = resolve_world(
                    reference,
                    owner=owner,
                    factory=document.factory,
                    factory_path=fp,
                    split=split,
                    read_source=read_frozen_source,
                    source_hashes=source_hashes,
                    timing=frozen.timing,
                )
                if split == "train":
                    recipe["seed_contract"] = frozen_world_streams(
                        reference, policy_seed, experiment.training_environment_seeds
                    )
                frozen_recipes[split, replication] = recipe
                return sc, st, work, metadata, case_seed
            wf, info, ticks, mode = dataset(split, replication)
            env = primitive(experiment.runtime.environment)
            env["tick_limit"] = env["tick_limit"] or ticks
            env["mode"] = mode
            ds = named_seed(seed_roots["world"], f"{split}:{replication}")
            environment_root = named_seed(
                seed_roots["environment"], f"{split}:{replication}"
            )
            env["outages"] = primitive(
                factory_reliability_outages(
                    document, named_seed(environment_root, "faults"), env["tick_limit"]
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
                document.factory,
                wf,
                settings,
                named_seed(environment_root, "environment"),
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
                for rep in range(
                    len(getattr(frozen, split))
                    if frozen
                    else options.frozen_replications
                    if split == "evaluation"
                    else options.replications
                ):
                    sc, st, work, metadata, case_seed = world(split, rep)
                    rows.append(
                        {
                            "case": getattr(frozen, split)[rep].id if frozen else "0",
                            "replication": 0 if frozen else rep,
                            "seed": case_seed,
                            **(
                                {
                                    "seed_semantics": "data_seed",
                                    "data_seed": case_seed,
                                    "environment_seed": sc.seed,
                                    "policy_root": getattr(frozen, split)[
                                        rep
                                    ].policy_seed,
                                }
                                if frozen
                                else {}
                            ),
                            "scenario": primitive(sc),
                            "recipe": {
                                "settings": primitive(st),
                                "workload": primitive(work),
                                "data_seed": case_seed,
                                "workload_provenance": metadata,
                                **(
                                    {
                                        "seed_contract": frozen_streams(
                                            experiment,
                                            policy_seed,
                                            split=split,
                                            replication=rep,
                                        )
                                    }
                                    if explicit_streams(experiment) and not frozen
                                    else {}
                                ),
                                **frozen_recipes.get((split, rep), {}),
                            },
                        }
                    )
            cases[split] = rows
        train_options = None
        if training:
            authored_training = primitive(experiment.training)
            authored_training.pop("groups", None)
            if experiment.evaluation.initial_control:
                authored_training["record_initial"] = True
            train_options = TrainingOptionsV3.model_validate_json(
                canonical_json(
                    {
                        **authored_training,
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
            "seed": named_seed(seed_roots["world"], "validation"),
        }
        evaluation_options = {
            **primitive(experiment.evaluation),
            "seed": named_seed(seed_roots["world"], "evaluation"),
        }
        config = ExecutionConfig.model_validate_json(
            canonical_json(
                {
                    "schema": "smartsom.execution-config/v2",
                    **(
                        {
                            "interface_contract": experiment.interface_contract
                            or "smartsom.configurable-experiment/v1"
                        }
                        if experiment.interface_contract is not None
                        or (training and experiment.training.groups is not None)
                        else {}
                    ),
                    "seed": policy_seed,
                    "training": primitive(train_options),
                    **(
                        {"task_reward": primitive(experiment.task_reward)}
                        if experiment.task_reward is not None
                        else {}
                    ),
                    "runtime": runtime,
                    "validation": validation_options,
                    "evaluation": evaluation_options,
                    "checkpointing": primitive(experiment.checkpointing),
                    "logging": primitive(experiment.logging),
                    "diagnostics": primitive(experiment.diagnostics),
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
            **(
                {
                    "seed_contract": frozen_streams(
                        experiment,
                        policy_seed,
                        split="train" if training else "evaluation",
                        replication=0,
                    )
                }
                if explicit_streams(experiment) and not frozen
                else {}
            ),
            **frozen_recipes.get(("train" if training else "evaluation", 0), {}),
            "authoring": {
                **provenance,
                "sources": sources,
                "documents": source_data,
                "input_sha256": {
                    str(p): source_hashes[p]
                    for p in (
                        (owner, fp, wp, algorithm_path) if not frozen else source_hashes
                    )
                },
            },
        }
        if frozen:
            metadata["frozen_input_contract"] = {
                **metadata["frozen_input_contract"],
                "cases": {
                    phase: [
                        copy.deepcopy(case["recipe"]["frozen_input_contract"])
                        for case in cases[phase]
                    ]
                    for phase in ("validation", "evaluation")
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
    from smartsom.config.scientific_contract import CONTRACT, identity

    if config.get("interface_contract") == CONTRACT:
        return identity(prepared)
    for key in ("logging", "output", "scenario", "composition", "diagnostics"):
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
