"""Compile reservation four-file inputs for the public check/run interface."""

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from smartsom.algorithms.base_policy import uses_learning
from smartsom.config.base_diagnostic import BaseExperiment, parse_inputs
from smartsom.config.codec import ConfigurationError, _UniqueLoader, read_document
from smartsom.domain.base_decisions import BASE_CONTRACT


def is_reservation(path):
    return bool(
        path and read_document(path).get("schema") == "smartsom.base-experiment/v1"
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
        return {
            "status": "checked",
            "components": self.component_report,
            "algorithm": yaml.safe_load(dict(self.documents)["algorithm"]),
            "evaluation": self.experiment.evaluation.model_dump(
                mode="json", by_alias=True
            ),
            "operations": self.operations,
            "origins": self.origins,
            "scientific_sha256": self.scientific_sha256,
            "overrides": list(self.overrides),
            "contract": BASE_CONTRACT,
            "train_seeds": list(self.experiment.train_seeds),
            "validation": self.experiment.validation.model_dump(mode="json"),
            "task": self.experiment.run.task,
            "output": self.experiment.run.output,
            "machines": self.machines,
            "agvs": self.agvs,
            "orders": self.orders,
            "drain_ticks": self.experiment.drain_ticks,
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
    if documents["experiment"].get("schema") != "smartsom.base-experiment/v1":
        raise ConfigurationError(
            "historical experiment configuration requires its archived source; "
            "see docs/migration/current-runtime.md and docs/archive/index.json"
        )
    if "matrix" in documents["experiment"]:
        if matrix_sets is None:
            raise ConfigurationError(
                "matrix inputs require the shared check/run compiler"
            )
        documents["experiment"].pop("matrix")
    sets = tuple(matrix_sets or ()) + tuple(sets)
    paths = {"experiment": path}
    scene_input = "scenario" in documents["experiment"]
    if scene_input and {"factory", "workload"} & documents["experiment"].keys():
        raise ConfigurationError(
            "use scenario or the legacy factory/workload pair, not both"
        )
    for name in (
        ("scenario", "algorithm")
        if scene_input
        else ("factory", "workload", "algorithm")
    ):
        paths[name] = (path.parent / documents["experiment"][name]).resolve()
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
        if scene_input and key.startswith("workload."):
            from smartsom.config.scenario_migration import WORKLOAD_PATHS

            _, field, *rest = key.split(".")
            if field not in WORKLOAD_PATHS:
                raise ConfigurationError(
                    f"use an explicit scenario field instead of {key}"
                )
            key = ".".join(["scenario", WORKLOAD_PATHS[field], *rest])
        names = key.split(".")
        if not separator or len(names) < 2 or names[0] not in documents:
            raise ConfigurationError(
                "--set requires scenario/algorithm/experiment.field=value (expanded inputs also accept factory/workload)"
            )
        if names[0] == "experiment" and names[1] in (
            "factory",
            "workload",
            "algorithm",
            "scenario",
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
        node[names[-1]] = yaml.safe_load(raw)
    scene_snapshot = None
    if scene_input:
        from smartsom.config.scenario_authoring import expand_documents

        scene_snapshot = yaml.safe_dump(documents["scenario"], sort_keys=False).encode()
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
    experiment, factory, workload, algorithm = parse_inputs(documents)
    from smartsom.telemetry.experiment_tracking import resolved

    documents["experiment"]["tracking"] = resolved(experiment.tracking)
    execution["tracking"] = documents["experiment"]["tracking"]
    from smartsom.config.components import (
        absolute_components,
        freeze_components,
        validate_components,
    )

    try:
        absolute = absolute_components(algorithm, paths["algorithm"].parent)
        validate_components(absolute)
        documents["algorithm"], components, component_report = freeze_components(
            algorithm, paths["algorithm"].parent
        )
    except (ValueError, ImportError, OSError, TypeError) as exc:
        raise ConfigurationError(str(exc)) from exc
    imported = {}
    panels = {}
    from smartsom.config.codec import digest
    from smartsom.config.evaluation_cases import load_cases

    for split, reference in experiment.case_sets.items():
        source = (path.parent / reference).resolve()
        from smartsom.config.codec import primitive

        cache = _case_cache if _case_cache is not None else {}
        cache_key = (
            hashlib.sha256(source.read_bytes()).hexdigest(),
            digest(primitive(factory)),
            digest(workload.model_dump(mode="json")),
            split,
        )
        if cache_key not in cache:
            cache[cache_key] = load_cases(
                source, split=split, factory=factory, workload=workload
            )
        cases = cache[cache_key]
        panels[split] = cases
        imported[split] = {digest(c["scenario"]) for c in cases}
        documents["experiment"].setdefault("case_set_sha256", {})[split] = digest(cases)
        target = f"cases/{split}.json"
        from smartsom.config.codec import primitive

        frozen_cases = {
            "schema": "smartsom.fixed-cases/v1",
            "split": split,
            "factory_sha256": digest(primitive(factory)),
            "workload_sha256": digest(workload.model_dump(mode="json")),
            "cases": cases,
        }
        components += ((target, json.dumps(frozen_cases, sort_keys=True).encode()),)
        documents["experiment"]["case_sets"][split] = target
        documents["experiment"][f"{split}_cases"] = len(cases)
    if imported.get("validation", set()) & imported.get("test", set()):
        raise ConfigurationError("validation and test contain the same physical world")
    import math

    from smartsom.config.evaluation_cases import validate_panel_separation

    case_drain = experiment.drain_ticks
    if experiment.run.calibration and Path(experiment.run.calibration).is_file():
        case_drain = json.loads(Path(experiment.run.calibration).read_text())[
            "drain_ticks"
        ]
    if experiment.evaluation.mode == "completion":
        case_drain = math.ceil(
            case_drain * experiment.evaluation.completion_drain_multiplier
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
                or report.get("split") != "validation"
            ):
                raise ConfigurationError(
                    "CAR reference must be a provenance-bearing validation research-summary.json"
                )
            from smartsom.telemetry.evaluation_summary import reference_definition

            expected = reference_definition(experiment, factory, workload, algorithm)
            panel = report.get("reference_panel", {})
            if (
                panel.get("definition") != expected["definition"]
                or panel.get("sha256") != expected["sha256"]
            ):
                raise ConfigurationError(
                    "CAR Rule reference differs from the frozen validation panel definition"
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
        (name, yaml.safe_dump(value, sort_keys=True))
        for name, value in documents.items()
    )
    r = experiment.run
    if algorithm.control == "centralized" and {"ppo", "dqn"} & set(r.methods):
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
        for split in ("validation", "test")
        if split not in experiment.case_sets
    ):
        raise ConfigurationError(
            "evaluation cases must balance the four fault conditions"
        )
    if uses_learning(algorithm):
        component_report["network_ownership"] = {
            role: {
                "mode": algorithm.learning_policy.parameter_sharing.mode(role),
                "entities": list(owners),
                "networks": len(owners)
                if algorithm.learning_policy.parameter_sharing.mode(role)
                == "independent"
                else 1,
                "parameters": algorithm.for_role(role).learner_parameters.model_dump(
                    mode="json"
                ),
            }
            for role, owners in algorithm._owner_ids.items()
        }
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
    values["algorithm"].pop("method", None)
    values["initial_drain_ticks"] = read_document(paths["experiment"]).get(
        "drain_ticks", 2000
    )
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()
