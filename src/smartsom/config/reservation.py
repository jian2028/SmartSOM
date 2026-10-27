"""Compile reservation four-file inputs for the public check/run interface."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import yaml

from smartsom.algorithms.base_policy import uses_learning
from smartsom.config.base_diagnostic import BaseExperiment, parse_inputs
from smartsom.config.codec import ConfigurationError, _UniqueLoader
from smartsom.config.experiment_v3 import read_document
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

    def materialize(self, root):
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
            if self.experiment.run.background
            else "foreground",
            "evidence": "development-diagnostic",
        }


def compile_reservation(
    path, *, sets=(), task=None, seed=None, background=None, tuning=None
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
    if background is not None:
        run["background"] = background
    if tuning is not None:
        run["tune"] = tuning
    for name in ("output", "calibration", "checkpoint"):
        if run.get(name):
            run[name] = str((path.parent / Path(run[name]).expanduser()).resolve())
    run.setdefault("output", str((path.parent / "runs/reservation-smoke").resolve()))
    for name in ("factory", "workload", "algorithm"):
        ex[name] = name + ".yaml"
    experiment, factory, workload, algorithm = parse_inputs(documents)
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
    if r.tune != "off" and (
        not experiment.train_seeds
        or len(set(experiment.train_seeds)) != len(experiment.train_seeds)
    ):
        raise ConfigurationError("Tune requires nonempty unique train_seeds")
    if r.tune != "off" and r.task != "train":
        raise ConfigurationError("reservation Tune requires a PPO train task")
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
        raise ConfigurationError("train requires PPO; evaluate accepts rule/random")
    if (
        r.task == "evaluate"
        and (
            "ppo" in r.methods
            or ("configured" in r.methods and uses_learning(algorithm))
        )
        and not r.checkpoint
    ):
        raise ConfigurationError("PPO evaluation requires run.checkpoint")
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
