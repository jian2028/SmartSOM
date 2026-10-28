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
    providers: tuple[tuple[str, bytes], ...] = ()
    overrides: tuple[str, ...] = ()
    modules: tuple[tuple[str, bytes], ...] = ()
    operations: dict = field(default_factory=dict)
    origins: dict = field(default_factory=dict)
    scientific_sha256: str = ""

    def materialize(self, root):
        for name, content in self.modules:
            target = Path(root) / "modules" / (name + ".py")
            target.parent.mkdir(exist_ok=True)
            target.write_bytes(content)
        for name, content in self.providers:
            target = Path(root) / "providers" / (name + ".py")
            target.parent.mkdir(exist_ok=True)
            target.write_bytes(content)
        for name, content in self.documents:
            (Path(root) / f"{name}.yaml").write_text(content)
        return Path(root) / "experiment.yaml"

    def summary(self):
        return {
            "status": "checked",
            "operations": self.operations,
            "origins": self.origins,
            "scientific_sha256": self.scientific_sha256,
            "overrides": list(self.overrides),
            "contract": BASE_CONTRACT,
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
    for name in ("factory", "workload", "algorithm"):
        paths[name] = (path.parent / documents["experiment"][name]).resolve()
        documents[name] = read(name, paths[name])
    for item in sets:
        key, separator, raw = item.partition("=")
        names = key.split(".")
        if not separator or len(names) < 2 or names[0] not in documents:
            raise ConfigurationError(
                "--set requires factory/workload/algorithm/experiment.field=value"
            )
        if names[0] == "experiment" and names[1] in (
            "factory",
            "workload",
            "algorithm",
            "schema",
        ):
            raise ConfigurationError(
                "edit file references in the Experiment before compilation"
            )
        node = documents[names[0]]
        for name in names[1:-1]:
            node = node.setdefault(name, {})
            if not isinstance(node, dict):
                raise ConfigurationError(f"override parent is not a mapping: {key}")
        node[names[-1]] = yaml.safe_load(raw)
    ex = documents["experiment"]
    run = ex.setdefault("run", {})
    if task is not None:
        run["task"] = task
    if seed is not None:
        run["seed"] = seed
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
    run.setdefault("output", str((path.parent / "runs/reservation-smoke").resolve()))
    for name in ("factory", "workload", "algorithm"):
        ex[name] = name + ".yaml"
    experiment, factory, workload, algorithm = parse_inputs(documents)
    from smartsom.telemetry.experiment_tracking import resolved

    documents["experiment"]["tracking"] = resolved(experiment.tracking)
    execution["tracking"] = documents["experiment"]["tracking"]
    providers = []
    for name, spec in algorithm.providers.items():
        if not name.isidentifier():
            raise ConfigurationError("provider identifiers must be Python identifiers")
        source = (paths["algorithm"].parent / spec.path).resolve()
        raw = source.read_bytes()
        checksum = hashlib.sha256(raw).hexdigest()
        if spec.sha256 is not None and spec.sha256 != checksum:
            raise ConfigurationError("policy provider source mismatch")
        providers.append((name, raw))
        documents["algorithm"]["providers"][name] = {
            "path": "providers/" + name + ".py",
            "factory": spec.factory,
            "sha256": checksum,
        }
    modules = []
    schemas = {}
    from smartsom.config.python_module import load_module

    for name in ("observation", "network"):
        spec = getattr(algorithm, name)
        if spec is None:
            continue
        source = (paths["algorithm"].parent / spec.path).resolve()
        raw = source.read_bytes()
        checksum = hashlib.sha256(raw).hexdigest()
        if spec.sha256 is not None and spec.sha256 != checksum:
            raise ConfigurationError(f"{name} module source mismatch")
        module = load_module(
            spec.model_copy(
                update={
                    "path": str(source),
                    "sha256": checksum,
                }
            )
        )
        schema = getattr(module, "INPUT_SCHEMA", None)
        required = ("encode",) if name == "observation" else ("create_network", "pack")
        if (
            not isinstance(schema, str)
            or not schema
            or any(not callable(getattr(module, member, None)) for member in required)
        ):
            raise ConfigurationError(f"invalid {name} module interface")
        schemas[name] = schema
        modules.append((name, raw))
        documents["algorithm"][name] = {
            "path": "modules/" + name + ".py",
            "sha256": checksum,
        }
    if algorithm.network is not None and schemas.get("network") != schemas.get(
        "observation"
    ):
        raise ConfigurationError("observation/network INPUT_SCHEMA mismatch")
    if (
        algorithm.observation is not None
        and uses_learning(algorithm)
        and algorithm.network is None
    ):
        raise ConfigurationError(
            "custom learning observation requires a matching network file"
        )
    experiment, factory, workload, algorithm = parse_inputs(documents)
    # Persist fully expanded typed inputs; compilation never runs in rollout loops.
    documents["experiment"] = experiment.model_dump(mode="json", by_alias=True)
    documents["workload"] = workload.model_dump(mode="json", by_alias=True)
    documents["algorithm"] = algorithm.model_dump(mode="json", by_alias=True)
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
    if r.task in ("train", "evaluate") and not r.calibration:
        raise ConfigurationError(
            "train/evaluate require run.calibration; use smoke for bounded checks"
        )
    if r.task == "train" and not uses_learning(algorithm):
        raise ConfigurationError(
            "train requires PPO or DQN; evaluate accepts rule/random"
        )
    if (
        r.task == "evaluate"
        and (
            bool({"ppo", "dqn"} & set(r.methods))
            or ("configured" in r.methods and uses_learning(algorithm))
        )
        and not r.checkpoint
    ):
        raise ConfigurationError("learning evaluation requires run.checkpoint")
    if r.task == "evaluate" and (
        experiment.validation_cases % 4 or experiment.test_cases % 4
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
        tuple(providers),
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
        tuple(modules),
        execution,
        origins,
        scientific_identity(documents),
    )


def calibration_identity(paths):
    """Execution paths/budgets do not alter calibrated factory/workload semantics."""
    values = {
        name: read_document(paths[name])
        for name in ("factory", "workload", "algorithm")
    }
    values["algorithm"].pop("method", None)
    values["initial_drain_ticks"] = read_document(paths["experiment"]).get(
        "drain_ticks", 2000
    )
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()
