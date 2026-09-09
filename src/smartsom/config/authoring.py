"""Portable scenario projects and read-only previews using existing input contracts."""

import hashlib
import json
from importlib.resources import files
from pathlib import Path

import yaml

from smartsom.config.codec import ConfigurationError, digest, primitive
from smartsom.config.models import (
    AlgorithmFile,
    DispatchRuleAlgorithm,
    FactoryFile,
    InstanceFile,
    RunSpec,
)
from smartsom.config.resolver import _resolve_run_spec
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
    source: str | Path, target: str | Path, *, instance_id: str | None = None
) -> Path:
    """Import a traditional FJS file into a movable, runnable SPT project.

    The source filename stem supplies the default semantic instance ID. Invalid input
    fails before creating the destination; the parser's provenance and original
    bytes are retained in workload.json and source.fjs respectively.
    """
    path = Path(source).expanduser().resolve()
    try:
        imported = import_fjs(
            path, instance_id=path.stem if instance_id is None else instance_id
        )
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != imported.provenance.source_sha256:
            raise ValueError("FJS source changed while preparing its export")
    except (OSError, ValueError) as exc:
        raise ConfigurationError(f"{path}: {exc}") from exc
    documents = {
        "factory.yaml": primitive(
            FactoryFile(schema="smartsom.factory/v1", factory=imported.factory)
        ),
        "workload.json": primitive(
            InstanceFile(
                schema="smartsom.workload-instance/v1",
                workload=imported.workload,
                content_sha256=digest(imported.workload),
                provenance=imported.provenance,
            )
        ),
        "scenario.yaml": {
            "schema": "smartsom.scenario/v1",
            "factory": "factory.yaml",
            "workload": {"kind": "instance", "path": "workload.json"},
        },
    }
    directory = _write_project(target, _project_documents(documents))
    (directory / "source.fjs").write_bytes(raw)
    return directory


def preview_scenario(path: str | Path, seed: int = 101) -> dict:
    """Validate and summarize a scenario, materializing inputs without simulation.

    Accept a scenario file or a project directory containing scenario.yaml.
    References are resolved by the existing resolver from their declaring file.
    This authoring preview includes privileged input counts, not agent observations.
    """
    scenario_path = Path(path).expanduser().resolve()
    if scenario_path.is_dir():
        scenario_path /= "scenario.yaml"
    resolved = _resolve_run_spec(
        RunSpec(
            schema="smartsom.run/v1",
            scenario=str(scenario_path),
            algorithm="__scenario_preview__",
            seed=seed,
            output_root="runs",
        ),
        scenario_path,
        [],
        algorithm_override=AlgorithmFile(
            schema="smartsom.algorithm/v1",
            algorithm=DispatchRuleAlgorithm(provider="builtin.spt"),
        ),
    )
    jobs = tuple(job for order in resolved.workload.orders for job in order.jobs)
    operations = resolved.workload.operations
    transport = resolved.factory.transport
    return {
        "schema": "smartsom.scenario-preview/v1",
        "status": "valid",
        "scenario": str(scenario_path),
        "seed": seed,
        "simulation_executed": False,
        "workload_source": resolved.scenario.workload.kind,
        "visibility": resolved.scenario.visibility,
        "decision_trigger": resolved.scenario.decision_trigger,
        "counts": {
            "machines": len(resolved.factory.machines),
            "agvs": len(transport.agvs)
            if transport and resolved.transport_enabled
            else 0,
            "orders": len(resolved.workload.orders),
            "jobs": len(jobs),
            "operations": len(operations),
            "base_processing_modes": sum(len(op.modes) for op in operations),
        },
        "modules": {
            "arrivals": resolved.arrivals is not None,
            "processing_time": resolved.processing_times is not None,
            "machine_events": resolved.machine_events is not None,
            "transport": resolved.transport_enabled,
            "buffers": resolved.buffers_enabled,
            "holding_buffer": resolved.holding_buffer_enabled,
            "quality": resolved.quality is not None,
        },
        "input_sha256": {
            "factory": resolved.factory_sha256,
            "workload": resolved.workload_sha256,
            "arrivals": resolved.arrivals_sha256,
            "processing_time": resolved.processing_times_sha256,
            "machine_events": resolved.machine_events_sha256,
            "quality_draws": resolved.quality_draws_sha256,
        },
        "effective_seeds": primitive(resolved.seeds),
        "sources": [
            {"role": source.role, "path": str(source.path), "sha256": source.sha256}
            for source in resolved.sources
        ],
    }
