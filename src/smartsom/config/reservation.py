"""Compile reservation four-file inputs for the public check/run interface."""

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from smartsom.algorithms.base_policy import uses_learning
from smartsom.config.base_diagnostic import BaseExperiment, parse_inputs
from smartsom.config.codec import (
    ConfigurationError,
    _UniqueLoader,
    read_document,
    safe_yaml_dump,
    safe_yaml_load,
)
from smartsom.domain.base_decisions import BASE_CONTRACT


def is_reservation(path):
    return bool(
        path
        and read_document(path).get("schema")
        in {"smartsom.base-experiment/v1", "smartsom.resolved-experiment/v1"}
    )


@dataclass(frozen=True)
class ReservationPlan:
    experiment: BaseExperiment
    documents: tuple[tuple[str, str], ...]
    source_files: tuple[tuple[str, str, str], ...]
    machines: int
    agvs: int
    orders: int
    overrides: tuple[str, ...] = ()
    components: tuple[tuple[str, bytes], ...] = ()
    component_report: dict = field(default_factory=dict)
    operations: dict = field(default_factory=dict)
    origins: dict = field(default_factory=dict)
    scientific_sha256: str = ""

    def materialize(self, root):
        for name, content in self.components:
            target = Path(root) / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        for name, content in self.documents:
            (Path(root) / f"{name}.yaml").write_text(content)
        return Path(root) / "experiment.yaml"

    def summary(self):
        from smartsom.algorithms.base_policy import DECISION_POINTS
        from smartsom.config.base_diagnostic import BaseAlgorithm

        algorithm = BaseAlgorithm.model_validate(
            safe_yaml_load(dict(self.documents)["algorithm"])
        )
        points = (
            ["central.assign"]
            if algorithm.control == "centralized"
            else sorted(DECISION_POINTS)
        )
        return {
            "status": "checked",
            "components": self.component_report,
            "algorithm": safe_yaml_load(dict(self.documents)["algorithm"]),
            "evaluation": self.experiment.evaluation.model_dump(
                mode="json", by_alias=True
            ),
            "operations": self.operations,
            "origins": self.origins,
            "scientific_sha256": self.scientific_sha256,
            "overrides": list(self.overrides),
            "action_modes": {
                point: algorithm.action_representation.mode(point) for point in points
            },
            "contract": BASE_CONTRACT,
            "train_seeds": list(self.experiment.train_seeds),
            "validation": self.experiment.validation.model_dump(mode="json"),
            "task": self.experiment.run.task,
            "output": self.experiment.run.output,
            "machines": self.machines,
            "agvs": self.agvs,
            "orders": self.orders,
            "drain_ticks": self.experiment.drain_ticks,
            "episode_end": self.experiment.episode_end,
            "condition_scenarios": self.experiment.condition_scenarios,
            "source_files": {
                name: {"path": path, "sha256": digest}
                for name, path, digest in self.source_files
            },
            "execution": "background"
            if self.experiment.launch.mode == "background"
            else "foreground",
            "evidence": "development-diagnostic",
        }


def compile_reservation(
    path,
    *,
    sets=(),
    task=None,
    seed=None,
    background=None,
    tuning=None,
    profile=None,
    operations=None,
    matrix_sets=None,
    learning_rate=None,
    _case_cache=None,
):
    path = Path(path).expanduser().resolve()
    snapshots = {}

    def read(name, source):
        raw = source.read_bytes()
        snapshots[name] = hashlib.sha256(raw).hexdigest()
        data = yaml.load(raw, Loader=_UniqueLoader)
        if not isinstance(data, dict):
            raise ConfigurationError(f"expected a YAML mapping: {source}")
        return data

    documents = {"experiment": read("experiment", path)}
    if documents["experiment"].get("schema") == "smartsom.resolved-experiment/v1":
        # Recompile a current frozen snapshot through its authored Scenario.
        # Expanded runtime files are never accepted as historical authoring.
        document = documents["experiment"]
        document.pop("factory")
        document.pop("workload")
        document.update(
            schema="smartsom.base-experiment/v1", scenario="authoring/scenario.yaml"
        )
    if documents["experiment"].get("schema") != "smartsom.base-experiment/v1":
        raise ConfigurationError(
            "historical experiment configuration requires its archived source; "
            "see docs/archive/index.json"
        )
    if "matrix" in documents["experiment"]:
        if matrix_sets is None:
            raise ConfigurationError(
                "matrix inputs require the shared check/run compiler"
            )
        documents["experiment"].pop("matrix")
    sets = tuple(matrix_sets or ()) + tuple(sets)
    paths = {"experiment": path}
    if (
        "scenario" not in documents["experiment"]
        or {"factory", "workload"} & documents["experiment"].keys()
    ):
        raise ConfigurationError(
            "authored experiments require Scenario; old inputs require archived source"
        )
    for name in ("scenario", "algorithm"):
        paths[name] = (path.parent / documents["experiment"][name]).resolve()
        documents[name] = read(name, paths[name])
    for condition, reference in (
        documents["experiment"].get("condition_scenarios", {}).items()
    ):
        name = f"condition_{condition}"
        paths[name] = (path.parent / reference).resolve()
        documents[name] = read(name, paths[name])
    if learning_rate is not None:
        policies = documents["algorithm"].get("production", {}).get("policies", {})
        names = [
            name for name, spec in policies.items() if spec.get("kind") == "learning"
        ]
        if len(names) != 1:
            raise ConfigurationError("--learning-rate requires one configured learner")
        sets += (
            f"algorithm.production.policies.{names[0]}.learner.params.learning_rate={learning_rate}",
        )
    for item in sets:
        key, separator, raw = item.partition("=")
        names = key.split(".")
        if not separator or len(names) < 2 or names[0] not in documents:
            raise ConfigurationError(
                "--set requires scenario/algorithm/experiment.field=value"
            )
        if names[0] == "experiment" and names[1] in (
            "factory",
            "workload",
            "algorithm",
            "scenario",
            "condition_scenarios",
            "schema",
        ):
            raise ConfigurationError(
                "edit file references in the Experiment before compilation"
            )
        node = documents[names[0]]
        for name in names[1:-1]:
            if node.get(name) is None:
                node[name] = {}
            node = node[name]
            if not isinstance(node, dict):
                raise ConfigurationError(f"override parent is not a mapping: {key}")
        node[names[-1]] = safe_yaml_load(raw)
    from smartsom.config.scenario_authoring import expand_documents

    scene_snapshot = safe_yaml_dump(documents["scenario"], sort_keys=False).encode()
    expand_documents(documents)
    ex = documents["experiment"]
    run = ex.setdefault("run", {})
    if task is not None:
        run["task"] = task
    if seed is not None:
        run["seed"] = seed
        ex["train_seeds"] = [seed]
    from smartsom.config.launch import resolve_operations, scientific_identity

    requested = dict(operations or {})
    if background is not None:
        requested["launch"] = {"mode": "background" if background else "foreground"}
    execution, origins = resolve_operations(ex, profile=profile, cli=requested)
    for key in ("launch", "tracking", "concurrency", "resources", "display"):
        ex[key] = execution[key]
    run["max_concurrent"] = execution["concurrency"]["max_concurrent"]
    run["resource_execution"] = (
        "adaptive" if execution["concurrency"]["mode"] == "adaptive" else "fixed"
    )
    if execution["output_root"]:
        explicit_output = any(
            item.startswith("experiment.run.output=") for item in sets
        )
        name = Path(run["output"]).name if explicit_output else path.stem
        run["output"] = str(Path(execution["output_root"]).resolve() / name)
    if tuning is not None:
        run["tune"] = tuning
    for name in ("output", "calibration", "checkpoint"):
        if run.get(name):
            run[name] = str((path.parent / Path(run[name]).expanduser()).resolve())
    from smartsom.config.experiment_layout import location

    layout = location(path)
    if layout and "output" not in run:
        repository, study, variant = layout
        run["output"] = str(repository / "runs" / study / "trials" / variant)
    run.setdefault("output", str((path.parent / "runs/reservation-smoke").resolve()))
    for name in ("factory", "workload", "algorithm"):
        ex[name] = name + ".yaml"
    for condition in ex.get("condition_scenarios", {}):
        ex["condition_scenarios"][condition] = f"condition_{condition}.yaml"
    experiment, factory, workload, algorithm = parse_inputs(documents)
    if experiment.validation_split == "validation":
        # Persist inferred naming before shorthand panels become frozen references.
        documents["experiment"].pop("periodic_evaluation", None)
        documents["experiment"].pop("periodic_evaluation_cases", None)
        documents["experiment"]["validation"] = experiment.validation.model_dump(
            mode="json"
        )
        documents["experiment"]["validation_cases"] = experiment.validation_cases
    from smartsom.telemetry.experiment_tracking import resolved

    documents["experiment"]["tracking"] = resolved(experiment.tracking)
    execution["tracking"] = documents["experiment"]["tracking"]
    from smartsom.config.components import (
        absolute_components,
        freeze_components,
        isolated_random,
        network_ownership,
        validate_components,
    )

    try:
        absolute = absolute_components(algorithm, paths["algorithm"].parent)
        validate_components(absolute)
        documents["algorithm"], components, component_report = freeze_components(
            algorithm, paths["algorithm"].parent
        )
        if uses_learning(absolute):
            with isolated_random():
                component_report["network_ownership"] = network_ownership(absolute)
    except (ValueError, ImportError, OSError, TypeError) as exc:
        raise ConfigurationError(str(exc)) from exc
    imported = {}
    panels = {}
    import math

    from smartsom.config.codec import digest, primitive
    from smartsom.config.evaluation_cases import (
        condition_input_identity,
        inline_panels,
        inline_test_cases,
        load_cases,
        validate_panel_separation,
    )

    case_drain = experiment.drain_ticks
    if experiment.run.calibration and Path(experiment.run.calibration).is_file():
        case_drain = json.loads(Path(experiment.run.calibration).read_text())[
            "drain_ticks"
        ]
    if experiment.evaluation.mode == "completion":
        case_drain = math.ceil(
            case_drain * experiment.evaluation.completion_drain_multiplier
        )

    panel_sources = dict(experiment.case_sets)
    authored_panels = inline_panels(experiment)
    panel_sources.update({split: None for split in authored_panels})
    for split, reference in panel_sources.items():
        source = (path.parent / reference).resolve() if reference is not None else None
        cache = _case_cache if _case_cache is not None else {}
        cache_key = (
            hashlib.sha256(source.read_bytes()).hexdigest()
            if source
            else digest(
                {
                    split: authored_panels[split].model_dump(mode="json"),
                    "drain_ticks": case_drain,
                }
            ),
            digest(primitive(factory)),
            digest(workload.model_dump(mode="json")),
            split,
            json.dumps(
                condition_input_identity(experiment._condition_inputs), sort_keys=True
            ),
        )
        if cache_key not in cache:
            cache[cache_key] = (
                load_cases(
                    source,
                    split=split,
                    factory=factory,
                    workload=workload,
                    condition_inputs=experiment._condition_inputs,
                )
                if source
                else inline_test_cases(
                    authored_panels[split],
                    split=split,
                    factory=factory,
                    workload=workload,
                    condition_inputs=experiment._condition_inputs,
                    drain_ticks=case_drain,
                )
            )
        cases = cache[cache_key]
        if split in experiment.case_set_sha256 and experiment.case_set_sha256[
            split
        ] != digest(cases):
            raise ConfigurationError(
                "fixed case inventory digest differs from the frozen experiment"
            )
        panels[split] = cases
        imported[split] = {digest(c["scenario"]) for c in cases}
        documents["experiment"].setdefault("case_set_sha256", {})[split] = digest(cases)
        target = f"cases/{split}.json"
        frozen_cases = {
            "schema": "smartsom.fixed-cases/v1",
            "split": split,
            "factory_sha256": digest(primitive(factory)),
            "workload_sha256": digest(workload.model_dump(mode="json")),
            "cases": cases,
        }
        if experiment.condition_scenarios:
            frozen_cases["condition_inputs_sha256"] = condition_input_identity(
                experiment._condition_inputs
            )
        components += ((target, json.dumps(frozen_cases, sort_keys=True).encode()),)
        documents["experiment"].setdefault("case_sets", {})[split] = target
        if split == "test":
            documents["experiment"].pop("test", None)
        if split == "validation":
            documents["experiment"]["validation"].pop("seeds", None)
            documents["experiment"]["validation"].pop("conditions", None)
        documents["experiment"][f"{split}_cases"] = len(cases)
    if imported.get(experiment.validation_split, set()) & imported.get("test", set()):
        raise ConfigurationError(
            "periodic_evaluation and test contain the same physical world"
        )
    validate_panel_separation(panels, experiment, factory, workload, case_drain)
    experiment, factory, workload, algorithm = parse_inputs(documents)
    evaluation = documents["experiment"].get("evaluation", {})
    extra_sources = []
    for section, reference_field in (
        ("recovery", "calibration"),
        ("learning_target", "reference"),
    ):
        spec = evaluation.get(section) or {}
        if not spec.get(reference_field):
            continue
        source = (path.parent / spec[reference_field]).resolve()
        raw = source.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        report = json.loads(raw)
        if section == "recovery":
            spec["calibration_sha256"] = digest
        if section == "learning_target":
            if (
                report.get("schema") != "smartsom.evaluation-summary/v1"
                or report.get("split") != experiment.validation_split
            ):
                raise ConfigurationError(
                    "CAR reference must be a provenance-bearing periodic_evaluation research-summary.json"
                )
            from smartsom.telemetry.evaluation_summary import reference_definition

            expected = reference_definition(experiment, factory, workload, algorithm)
            panel = report.get("reference_panel", {})
            if (
                panel.get("definition") != expected["definition"]
                or panel.get("sha256") != expected["sha256"]
            ):
                raise ConfigurationError(
                    "CAR Rule reference differs from the frozen periodic_evaluation panel definition"
                )
            if len(panel.get("worlds", [])) != experiment.validation_cases:
                raise ConfigurationError(
                    "CAR Rule reference has incomplete frozen world inventory"
                )
            if (
                algorithm.traffic.mover.target
                != "smartsom.traffic.rule_mover.create_mover"
            ):
                raise ConfigurationError(
                    "CAR common Rule reference currently requires the deterministic built-in Mover"
                )
            rule = report.get("methods", report).get("rule", {}).get("all", {})
            if not rule.get("complete_inventory") or not report.get("source_sha256"):
                raise ConfigurationError(
                    "CAR Rule reference lacks complete inventory or source identity"
                )
            cost = rule.get("metrics", rule).get("joint_cost")
            if cost is None or cost < 0:
                raise ConfigurationError(
                    "CAR reference requires a complete Rule cost panel"
                )
            if spec.get("rule_cost") is not None and spec["rule_cost"] != cost:
                raise ConfigurationError("CAR rule_cost differs from frozen reference")
            spec["rule_cost"] = cost
            spec["reference_sha256"] = digest
        frozen = "evaluation-references/" + digest + ".json"
        spec[reference_field] = frozen
        extra_sources.append((frozen, raw))
    components = tuple(components) + tuple(extra_sources)
    for condition, (
        condition_factory,
        condition_workload,
    ) in experiment._condition_inputs.items():
        expanded = {
            "factory": primitive(condition_factory),
            "workload": condition_workload.model_dump(mode="json", by_alias=True),
        }
        components += (
            (
                f"expanded/condition_{condition}.json",
                json.dumps(expanded, sort_keys=True).encode(),
            ),
        )
    if scene_snapshot is not None:
        components += (("authoring/scenario.yaml", scene_snapshot),)
    experiment, factory, workload, algorithm = parse_inputs(documents)
    if (
        experiment.evaluation.recovery.action != "disabled"
        and experiment.run.task not in ("calibrate", "evaluate")
    ):
        raise ConfigurationError(
            "controlled recovery is an explicit calibration/evaluation task"
        )
    machine_ids = {m.machine_id for m in factory.machines}
    for candidate in experiment.evaluation.recovery.candidates:
        if any(fault.machine_id not in machine_ids for fault in candidate.faults):
            raise ConfigurationError("recovery fault references an unknown machine")
    # Persist fully expanded typed inputs; compilation never runs in rollout loops.
    documents["experiment"] = experiment.model_dump(mode="json", by_alias=True)
    documents["workload"] = workload.model_dump(mode="json", by_alias=True)
    documents["algorithm"] = algorithm.model_dump(mode="json", by_alias=True)
    documents["experiment"]["train_seeds"] = list(experiment.train_seeds)
    canonical = tuple(
        (name, safe_yaml_dump(value, sort_keys=True))
        for name, value in documents.items()
    )
    r = experiment.run
    if (
        algorithm.control == "centralized"
        and algorithm.learning_policy is None
        and {"ppo", "dqn"} & set(r.methods)
    ):
        raise ConfigurationError(
            "centralized methods must be rule, random or configured"
        )
    if r.tune != "off" and (
        not experiment.train_seeds
        or len(set(experiment.train_seeds)) != len(experiment.train_seeds)
    ):
        raise ConfigurationError("Tune requires nonempty unique train_seeds")
    if r.tune != "off" and r.task != "train":
        raise ConfigurationError("Tune requires a learning train task")
    if not factory.agvs:
        raise ConfigurationError("reservation execution requires at least one AGV")
    from smartsom.config.base_diagnostic import generate_scenario
    from smartsom.engine.production import ProductionSimulator

    simulator = ProductionSimulator(
        generate_scenario(factory, workload, 0, "normal", experiment.drain_ticks),
        contract="base",
    )
    routes = simulator.protocol.routes
    reachable = routes.distances((tuple(next(iter(simulator.agvs.values()))["cell"]),))
    if any(
        cell not in reachable for cells in routes.ports.values() for cell in cells
    ) or any(tuple(agv["cell"]) not in reachable for agv in simulator.agvs.values()):
        raise ConfigurationError(
            "all AGV starts and interaction points must be statically connected"
        )
    if (
        r.task in ("train", "evaluate")
        and not r.calibration
        and not experiment.drain_frozen
        and experiment.evaluation.recovery.action == "disabled"
    ):
        raise ConfigurationError(
            "train/evaluate require run.calibration; use smoke for bounded checks"
        )
    if r.task == "train" and not uses_learning(algorithm):
        raise ConfigurationError("train requires a configured learning policy")
    if (
        r.task == "evaluate"
        and (
            bool({"ppo", "dqn"} & set(r.methods))
            or ("configured" in r.methods and uses_learning(algorithm))
        )
        and not r.checkpoint
    ):
        raise ConfigurationError("learning evaluation requires run.checkpoint")
    if r.task == "evaluate" and any(
        getattr(experiment, f"{split}_cases") % 4
        for split in (experiment.validation_split, "test")
        if split not in experiment.case_sets
    ):
        raise ConfigurationError(
            "evaluation cases must balance the four fault conditions"
        )
    sources = tuple((name, str(p), snapshots[name]) for name, p in paths.items())
    return ReservationPlan(
        experiment,
        canonical,
        sources,
        len(factory.machines),
        len(factory.agvs),
        workload.orders,
        tuple(sets)
        + tuple(
            f"{key}={value}"
            for key, value in (
                ("task", task),
                ("seed", seed),
                ("background", background),
                ("tuning", tuning),
            )
            if value is not None
        ),
        components,
        component_report,
        execution,
        origins,
        scientific_identity(documents),
    )


def calibration_identity(paths):
    """Execution paths/budgets do not alter calibrated factory/workload semantics."""
    if "scenario" in paths:
        from smartsom.config.codec import primitive
        from smartsom.config.scenario_authoring import load_scene

        factory, workload = load_scene(paths["scenario"])[0].expand()
        values = dict(
            factory=primitive(factory),
            workload=workload.model_dump(mode="json", by_alias=True),
            algorithm=read_document(paths["algorithm"]),
        )
    else:
        values = {
            name: read_document(paths[name])
            for name in ("factory", "workload", "algorithm")
        }
    for name, path in paths.items():
        if name.startswith("condition_"):
            from smartsom.config.base_diagnostic import _expand_condition_scene
            from smartsom.config.codec import primitive

            selected_factory, selected_workload = _expand_condition_scene(
                json.dumps(read_document(path), sort_keys=True)
            )
            values[name] = {
                "factory": primitive(selected_factory),
                "workload": selected_workload.model_dump(mode="json", by_alias=True),
            }
    values["algorithm"].pop("method", None)
    values["initial_drain_ticks"] = read_document(paths["experiment"]).get(
        "drain_ticks", 2000
    )
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()
