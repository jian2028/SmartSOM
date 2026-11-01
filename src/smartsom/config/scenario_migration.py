"""Explicit, create-only migration from current four-file authoring to scenes."""

import hashlib
import json
import shutil
from copy import deepcopy
from itertools import product
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


def freeze_matrix_components(algorithm, matrix, base):
    """Copy every matrix component, including bundles and leaf-path overrides."""
    axes = {k: v for k, v in matrix.items() if k.startswith("algorithm.")}
    document = algorithm.model_dump(mode="json", by_alias=True)
    declarations = [document]
    for values in product(*axes.values()) if axes else ():
        variant = deepcopy(document)
        for key, value in zip(axes, values, strict=True):
            node = variant
            parts = key.split(".")[1:]
            for part in parts[:-1]:
                node = node.setdefault(part, {})
            node[parts[-1]] = deepcopy(value)
        declarations.append(variant)
    files, references, declarations_by_source = {}, {}, {}
    for declaration in declarations:
        options = BaseAlgorithm.model_validate(declaration)
        frozen, bundle, _ = freeze_components(options, base)
        bundle = dict(bundle)
        files.update(
            (k, v) for k, v in bundle.items() if k.startswith("components/builtin/")
        )
        for key, spec in options.component_items():
            node = frozen
            for part in key.split("."):
                node = node[part]
            component_key = "algorithm." + key
            varied = any(
                axis == component_key
                or axis.startswith(component_key + ".")
                or component_key.startswith(axis + ".")
                for axis in axes
            )
            if declaration is document and not varied:
                original_node = document
                for part in key.split("."):
                    original_node = original_node[part]
                original_node["sha256"] = node["sha256"]
            if not spec.path:
                continue
            # Distinct variants of the same policy must not overwrite each
            # other's sources. The digest covers the entire declared bundle.
            old_folder = "components/" + key.replace(".", "-")
            folder = "components/extensions/" + node["sha256"]
            files.update(
                (folder + name[len(old_folder) :], payload)
                for name, payload in bundle.items()
                if name.startswith(old_folder + "/")
            )
            relocated = {}
            for field in ("path", "source_root"):
                original = getattr(spec, field)
                if original:
                    replacement = folder + node[field][len(old_folder) :]
                    ref = ("algorithm." + key + "." + field, original)
                    references.setdefault(ref, set()).add(replacement)
                    relocated[field] = replacement
            declarations_by_source[(component_key, spec.path, spec.source_root)] = (
                relocated
            )

    for axis in axes:
        if axis.endswith(".source_root") and any(
            field == axis.removesuffix("source_root") + "path" and len(choices) > 1
            for (field, _), choices in references.items()
        ):
            # Changing the bundle relocates its entry too. Independent scalar
            # axes cannot express that correlated rewrite without changing the
            # matrix; require the already supported complete declaration.
            raise ValueError(
                f"matrix field {axis} also relocates the entry path; "
                "use complete component declarations"
            )

    def rebase(value, prefix):
        if isinstance(value, dict):
            if isinstance(value.get("path"), str):
                relocated = declarations_by_source.get(
                    (prefix, value["path"], value.get("source_root"))
                )
                if relocated is not None:
                    return {**value, **relocated}
            return {k: rebase(v, prefix + "." + k) for k, v in value.items()}
        if isinstance(value, list):
            return [rebase(v, prefix) for v in value]
        if isinstance(value, str):
            choices = references.get((prefix, value), {value})
            if len(choices) != 1:
                raise ValueError(
                    f"matrix field {prefix} uses incompatible source scopes; "
                    "use complete component declarations"
                )
            return next(iter(choices))
        return value

    return (
        rebase(document, "algorithm"),
        {key: rebase(values, key) for key, values in matrix.items()},
        tuple(sorted(files.items())),
    )


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
    frozen, source_matrix, components = freeze_matrix_components(
        algorithm, ex.get("matrix", {}), paths["algorithm"].parent
    )
    matrix = {}
    for path, values in source_matrix.items():
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
