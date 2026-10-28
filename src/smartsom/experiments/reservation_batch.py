"""Base preparation adapter for the existing directory batch/NativeScheduler.

This module freezes inputs only. Execution, admission, failure propagation and
runtime controls remain owned by author_batch and its existing workers.
"""

import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from smartsom.config.codec import ConfigurationError, digest
from smartsom.config.reservation import compile_reservation
from smartsom.experiments.evidence import source_identity, write_json


def compile_files(directory, paths, calibration_overrides, **overrides):
    from smartsom.experiments.author_batch import DirectoryPlan

    if any(value not in (None, "off", 0) for value in calibration_overrides.values()):
        raise ConfigurationError(
            "Base directory execution uses per-Experiment calibration evidence"
        )
    files = []
    output_root = None
    identities = set()
    concurrency = {}
    for path, entry_id, compiled in compile_members(paths, overrides):
        settings = compiled.experiment.run
        if settings.tune != "off":
            raise ConfigurationError(
                "Base batch children use native execution; use standalone run for Tune"
            )
        parent = Path(settings.output).parent
        if output_root is not None and parent != output_root:
            raise ConfigurationError(
                "Base batch run.output paths must share one parent directory"
            )
        output_root = parent
        documents = dict(compiled.documents)
        import yaml

        scientific = {key: yaml.safe_load(value) for key, value in documents.items()}
        from smartsom.config.launch import scientific_identity

        identity = scientific_identity(scientific)
        if identity in identities:
            raise ConfigurationError("duplicate scientific entry in Base batch")
        identities.add(identity)
        batch = compiled.experiment.batch
        if (
            batch.stage in concurrency
            and concurrency[batch.stage] != batch.parallel_files
        ):
            raise ConfigurationError(
                "Base batch stage must use one parallel_files ceiling"
            )
        concurrency[batch.stage] = batch.parallel_files
        files.append(
            {
                "id": entry_id,
                "path": path,
                "compiled": compiled,
                "stage": batch.stage,
                "parallel_files": batch.parallel_files,
                "scientific_sha256": identity,
            }
        )
    shared = files[0]["compiled"].operations
    for row in files[1:]:
        if any(
            row["compiled"].operations[k] != shared[k]
            for k in ("launch", "concurrency", "resources", "display")
        ):
            raise ConfigurationError(
                "directory launch/concurrency/resources/display disagree; select one explicit profile or CLI override"
            )
    return DirectoryPlan(
        directory,
        tuple(files),
        output_root,
        0,
        "off",
        contract="reservation",
        operations=shared,
    )


def allocate_directory(directory_plan):
    from smartsom.experiments.author_batch import RUN_SCHEMA, SCHEMA, STATE_SCHEMA
    from smartsom.experiments.evidence import implementation_identity
    from smartsom.experiments.reservation import allocate

    root = directory_plan.output_root / (
        datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        + "-batch-"
        + directory_plan.directory.name
        + "-"
        + uuid4().hex[:10]
    )
    root.mkdir(parents=True, exist_ok=False)
    rows = []
    for row in directory_plan.files:
        child = root / "experiments" / row["id"]
        allocate(row["compiled"], root=child)
        child_plan = json.loads((child / "plan.json").read_text())
        rows.append(
            {
                "id": row["id"],
                "path": str(row["path"]),
                "child": str(child.relative_to(root)),
                "child_plan_sha256": digest(child_plan),
                "entry_ids": ["run"],
                "scientific_sha256": [row["scientific_sha256"]],
                "task": row["compiled"].experiment.run.task,
                "stage": row["stage"],
                "parallel_files": row["parallel_files"],
                "gate": None,
                "contract": "reservation",
            }
        )
    plan = {
        "schema": SCHEMA,
        "runtime_contract": "smartsom.current-runtime/v1",
        "run_id": uuid4().hex,
        "operations": directory_plan.operations,
        "contract": "reservation",
        "directory": str(directory_plan.directory),
        "files": rows,
        "calibration_seconds": 0,
        "calibration_level": "off",
        "calibration_candidate": "latest",
        "source": source_identity(),
        "implementation_sha256": implementation_identity(),
    }
    state = {
        "schema": STATE_SCHEMA,
        "plan_sha256": digest(plan),
        "status": "prepared",
        "stage": "prepared",
        "files": {row["id"]: {"status": "queued"} for row in rows},
        "tune_directory": None,
        "calibration_status": "not_applicable",
        "smoke_status": "not_applicable",
    }
    write_json(root / "plan.json", plan)
    write_json(
        root / "execution.json", {"settings": directory_plan.operations, "history": []}
    )
    write_json(root / "batch.json", state)
    write_json(
        root / "run.json",
        {
            "schema": RUN_SCHEMA,
            "kind": "batch-directory",
            "name": root.name,
            "status": "prepared",
            "stage": "prepared",
            "plan_sha256": digest(plan),
            "source": plan["source"],
        },
    )
    return root


def compile_members(paths, overrides):
    """Expand scientific axes once, before the shared admission queue is created."""
    from itertools import product

    import yaml

    from smartsom.config.codec import read_document
    from smartsom.config.launch import OPERATION_RUN_KEYS

    for path in paths:
        matrix = read_document(path).get("matrix")
        if matrix is None:
            yield path, path.stem, compile_reservation(path, **overrides)
            continue
        if not isinstance(matrix, dict) or not matrix:
            raise ConfigurationError(
                "matrix must be a nonempty mapping of scientific field to values"
            )
        for key, values in matrix.items():
            parts = key.split(".")
            scientific = (
                parts[0] in {"factory", "workload", "algorithm", "experiment"}
                and len(parts) > 1
            )
            if parts[0] == "experiment":
                scientific = scientific and parts[1] not in {
                    "launch",
                    "concurrency",
                    "resources",
                    "tracking",
                    "display",
                    "output_root",
                    "batch",
                    "matrix",
                    "factory",
                    "workload",
                    "algorithm",
                    "schema",
                }
                if parts[1] == "run" and len(parts) > 2:
                    scientific = scientific and parts[2] not in OPERATION_RUN_KEYS
            if not scientific or not isinstance(values, list) or not values:
                raise ConfigurationError(f"invalid scientific matrix axis: {key}")
        for index, values in enumerate(product(*matrix.values()), 1):
            sets = [
                key
                + "="
                + yaml.safe_dump(value, default_flow_style=True)
                .strip()
                .removesuffix("...")
                .strip()
                for key, value in zip(matrix, values, strict=True)
            ]
            compiled = compile_reservation(path, matrix_sets=sets, **overrides)
            yield path, f"{path.stem}--{index:04d}", compiled
