"""Portable scenario projects and read-only previews using existing input contracts."""

import hashlib
import json
from importlib.resources import files
from pathlib import Path

import yaml

from smartsom.config.codec import ConfigurationError, digest, primitive
from smartsom.config.factory_design import load_factory_design_file
from smartsom.config.production import (
    ScenarioFile,
    WorkloadFile,
    materialize,
    named_seed,
    read_file,
)
from smartsom.domain.production import (
    Demand,
    ProductionStep,
    validate_production_scenario,
)
from smartsom.workloads import import_fjs


def _templates() -> dict:
    resource = files("smartsom.config").joinpath("authoring_templates.json")
    return json.loads(resource.read_text(encoding="utf-8"))["templates"]


def list_templates() -> tuple[dict, ...]:
    """Describe the built-in project templates without importing learner backends."""
    return tuple(
        {"name": name, "description": row["description"], "learning": row["learning"]}
        for name, row in _templates().items()
    )


def _project_documents(documents: dict) -> dict:
    return {
        **documents,
        "algorithm.yaml": {
            "schema": "smartsom.algorithm/v1",
            "algorithm": {"provider": "builtin.spt"},
        },
        "run.yaml": {
            "schema": "smartsom.run/v1",
            "scenario": "scenario.yaml",
            "algorithm": "algorithm.yaml",
            "seed": 101,
            "output_root": "runs",
        },
    }


def _write_project(target: str | Path, documents: dict) -> Path:
    # Do not follow a target symlink or reuse an existing, even empty, directory.
    directory = Path(target).expanduser().absolute()
    directory.mkdir(parents=True)
    for name, document in documents.items():
        content = (
            json.dumps(document, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
            if name.endswith(".json")
            else yaml.safe_dump(document, sort_keys=False, allow_unicode=True)
        )
        (directory / name).write_text(content, encoding="utf-8")
    return directory.resolve()


def create_template(name: str, target: str | Path) -> Path:
    """Create a self-contained project; return its directory and refuse overwrites.

    Every project contains scenario.yaml and an SPT run.yaml. The marl_micro
    project also contains train.yaml and learning-algorithm.yaml, preserving the
    existing micro training recipe. Creation never starts a run or training job.
    """
    templates = _templates()
    if name not in templates:
        raise ConfigurationError(
            f"unknown scenario template {name!r}; choose from {', '.join(templates)}"
        )
    return _write_project(target, _project_documents(templates[name]["documents"]))


def import_fjs_project(
    source: str | Path,
    target: str | Path,
    *,
    instance_id: str | None = None,
    factory: str | Path | None = None,
    machine_map: dict[str, str] | None = None,
) -> Path:
    """Import FJS processing data using an explicitly authored grid factory.

    M1, M2, ... map to identical factory IDs unless machine_map is supplied.
    A catalog type must have exactly the FJS operation's eligible machines.
    Layouts and omitted capabilities are never guessed from a distance matrix.
    """
    path = Path(source).expanduser().resolve()
    try:
        imported = import_fjs(
            path, instance_id=path.stem if instance_id is None else instance_id
        )
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != imported.provenance.source_sha256:
            raise ValueError("FJS source changed while preparing its export")
        if factory is None:
            raise ValueError(
                "FJS requires an explicit grid factory; supply factory= or --factory (layout and ports cannot be inferred)"
            )
        design_file, _ = load_factory_design_file(factory)
        design = design_file.factory
        mapping = {m.machine_id: m.machine_id for m in imported.factory.machines}
        if machine_map:
            if set(machine_map) - mapping.keys():
                raise ValueError("machine_map references an unknown FJS machine")
            mapping.update(machine_map)
        if len(set(mapping.values())) != len(mapping) or not set(mapping.values()) <= {
            m.machine_id for m in design.machines
        }:
            raise ValueError(
                "machine_map must map each FJS machine to a distinct known factory machine"
            )
        capabilities = {
            kind: {m.machine_id for m in design.machines if kind in m.operation_types}
            for kind in design.operation_types
        }
        demands = []
        for order in imported.workload.orders:
            for job in order.jobs:
                steps = []
                for operation in job.operations:
                    times = {}
                    for mode in operation.modes:
                        machine = mapping[mode.machine_id]
                        if machine in times:
                            raise ValueError(
                                "FJS alternatives repeat a machine; the grid workload requires one nominal time per machine"
                            )
                        times[machine] = mode.nominal_ticks
                    types = [
                        kind
                        for kind, machines in capabilities.items()
                        if machines == set(times)
                    ]
                    if not types:
                        raise ValueError(
                            f"factory catalog has no type with exactly the eligible machines for {operation.operation_id}: {sorted(times)}"
                        )
                    steps.append(
                        ProductionStep(
                            operation.operation_id,
                            types[0],
                            min(times.values()),
                            tuple(times.items()),
                        )
                    )
                demands.append(Demand(job.job_id, tuple(steps)))
        workload = WorkloadFile(
            schema="smartsom.workload/v2",
            demands=tuple(demands),
            provenance=imported.provenance,
        )
        settings = ScenarioFile(
            schema="smartsom.scenario/v2",
            factory="factory.yaml",
            workload="workload.yaml",
            tick_limit=10000,
        )
        validate_production_scenario(materialize(design, workload, settings, 101))
    except (OSError, ValueError) as exc:
        raise ConfigurationError(f"{path}: {exc}") from exc
    documents = {
        "factory.yaml": primitive(design_file),
        "workload.yaml": primitive(workload),
        "scenario.yaml": primitive(settings),
    }
    directory = _write_project(target, _project_documents(documents))
    (directory / "source.fjs").write_bytes(raw)
    return directory


def preview_scenario(path: str | Path, seed: int = 101) -> dict:
    """Validate and materialize public inputs without constructing a simulator."""
    scenario_path = Path(path).expanduser().resolve()
    if scenario_path.is_dir():
        scenario_path /= "scenario.yaml"
    try:
        settings = read_file(scenario_path, ScenarioFile)
        factory_path = (scenario_path.parent / settings.factory).resolve()
        workload_path = (scenario_path.parent / settings.workload).resolve()
        design_file, factory_hash = load_factory_design_file(factory_path)
        workload = read_file(workload_path, WorkloadFile)
        case = materialize(design_file.factory, workload, settings, seed)
        validate_production_scenario(case)
    except (OSError, ValueError) as exc:
        raise ConfigurationError(f"{scenario_path}: {exc}") from exc
    factory = case.factory
    return {
        "schema": "smartsom.scenario-preview/v2",
        "status": "valid",
        "scenario": str(scenario_path),
        "seed": seed,
        "simulation_executed": False,
        "workload_source": "profile" if workload.profile else "instance",
        "mode": case.mode,
        "counts": {
            "machines": len(factory.machines),
            "agvs": len(factory.agvs),
            "jobs": len(case.demands),
            "operations": sum(len(d.steps) for d in case.demands),
            "base_processing_modes": sum(
                sum(step.operation_type in m.operation_types for m in factory.machines)
                for d in case.demands
                for step in d.steps
            ),
        },
        "modules": {
            "arrivals": settings.arrivals is not None
            or any(d.release_at for d in case.demands),
            "processing_time": bool(case.processing_samples)
            or case.processing_low != 1
            or case.processing_high != 1,
            "machine_events": bool(case.outages),
            "transport": bool(factory.agvs),
            "buffers": bool(factory.buffers),
            "holding_buffer": any(b.role == "storage" for b in factory.buffers),
            "quality": bool(factory.inspection_stations)
            or any(
                mode.error_rate for m in factory.machines for mode in m.quality_modes
            ),
        },
        "input_sha256": {
            "factory": digest(factory),
            "workload": digest(case.demands),
            "arrivals": digest(
                [(d.demand_id, d.release_at, d.reveal_at) for d in case.demands]
            ),
            "processing_time": digest(
                (case.processing_low, case.processing_high, case.processing_samples)
            ),
            "machine_events": digest(case.outages),
            "quality_draws": digest(case.quality_samples),
        },
        "effective_seeds": {
            name: named_seed(seed, name) for name in ("workload", "arrival")
        },
        "sources": [
            {"role": role, "path": str(source), "sha256": sha}
            for role, source, sha in (
                (
                    "scenario",
                    scenario_path,
                    hashlib.sha256(scenario_path.read_bytes()).hexdigest(),
                ),
                ("factory", factory_path, factory_hash),
                (
                    "workload",
                    workload_path,
                    hashlib.sha256(workload_path.read_bytes()).hexdigest(),
                ),
            )
        ],
    }
