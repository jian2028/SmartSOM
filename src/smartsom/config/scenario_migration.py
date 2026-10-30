"""Explicit, create-only migration from current four-file authoring to scenes."""

import hashlib
import json
import shutil
from pathlib import Path

import yaml

from smartsom.config.base_diagnostic import BaseAlgorithm, BaseWorkload
from smartsom.config.codec import read_document
from smartsom.config.components import freeze_components
from smartsom.config.factory_design import load_factory_design_file
from smartsom.config.scenario_authoring import ScenarioFile, save_scene

WORKLOAD_PATHS = {
    "orders": "jobs.orders.total",
    "initial_orders": "jobs.orders.initial",
    "mean_interarrival": "jobs.arrivals.mean_interval_ticks",
    "families": "jobs.routes.families",
    "optional_repeat_probability": "jobs.routes.repeat_first_probability",
    "base_ticks": "jobs.processing.base_ticks",
    "workload_rho": "jobs.processing.relative_variation",
    "urgent_probability": "jobs.due_dates.urgent_probability",
    "normal_due": "jobs.due_dates.normal_multiplier",
    "urgent_due": "jobs.due_dates.urgent_multiplier",
    "fault_intervals": "factory.machines.defaults.breakdown.intervals",
    "repair": "factory.machines.defaults.breakdown.repair",
}


def migrate_scene_experiment(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.exists():
        raise ValueError("migration destination already exists; choose a new directory")
    ex = read_document(source)
    if "scenario" in ex:
        raise ValueError("experiment already references a scenario")
    paths = {
        name: (source.parent / ex[name]).resolve()
        for name in ("factory", "workload", "algorithm")
    }
    envelope, _ = load_factory_design_file(paths["factory"])
    workload = BaseWorkload.model_validate(read_document(paths["workload"]))
    scene = ScenarioFile.from_factory(envelope, workload)
    algorithm = BaseAlgorithm.model_validate(read_document(paths["algorithm"]))
    frozen, components, _ = freeze_components(algorithm, paths["algorithm"].parent)
    matrix = {}
    for path, values in ex.get("matrix", {}).items():
        if path.startswith("workload."):
            _, field, *rest = path.split(".")
            if field not in WORKLOAD_PATHS:
                raise ValueError(f"cannot migrate matrix path: {path}")
            path = ".".join(["scenario", WORKLOAD_PATHS[field], *rest])
        elif path.startswith("factory."):
            raise ValueError(f"factory matrix needs explicit scene authoring: {path}")
        matrix[path] = values
    if matrix:
        ex["matrix"] = matrix
    for name in ("factory", "workload"):
        ex.pop(name)
    ex.update(scenario="scenario.yaml", algorithm="algorithm.yaml")
    # Resolve references without moving or changing evidence files.
    for split, reference in ex.get("case_sets", {}).items():
        ex["case_sets"][split] = str((source.parent / reference).resolve())
    for key in ("output", "checkpoint", "calibration"):
        if ex.get("run", {}).get(key):
            ex["run"][key] = str((source.parent / ex["run"][key]).resolve())
    for key, field in (("recovery", "calibration"), ("learning_target", "reference")):
        spec = ex.get("evaluation", {}).get(key, {})
        if spec.get(field):
            spec[field] = str((source.parent / spec[field]).resolve())
    report = {
        "source": str(source),
        "destination": str(destination),
        "mappings": {"factory + workload": "scenario.yaml", **WORKLOAD_PATHS},
        "defaults": "Common values extracted; per-entity differences preserved",
        "geometry": "Machines and AGVs are 1x1; non-1x1 machines rejected",
        "checkpoint_compatibility": "Not established; historical runs use frozen source",
        "source_files": {
            name: {
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for name, path in {"experiment": source, **paths}.items()
        },
    }
    destination.mkdir(parents=True)
    try:
        save_scene(destination / "scenario.yaml", scene)
        (destination / "algorithm.yaml").write_text(
            yaml.safe_dump(frozen, sort_keys=False)
        )
        for name, payload in components:
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
        (destination / "experiment.yaml").write_text(
            yaml.safe_dump(ex, sort_keys=False)
        )
        (destination / "migration-report.json").write_text(json.dumps(report, indent=2))
    except BaseException:
        shutil.rmtree(destination)
        raise
    return report
