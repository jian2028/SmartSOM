"""Base preparation adapter for the existing directory batch/NativeScheduler.

This module freezes inputs only. Execution, admission, failure propagation and
runtime controls remain owned by author_batch and its existing workers.
"""

import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from smartsom.config.codec import ConfigurationError, digest
from smartsom.config.reservation import compile_reservation
from smartsom.experiments.evidence import source_identity, write_json


def compile_files(
    directory,
    paths,
    calibration_overrides,
    *,
    execution_selection=None,
    resume_missing=False,
    **overrides,
):
    import hashlib
    import os

    import yaml

    from smartsom.config.base_diagnostic import BaseExperiment
    from smartsom.config.experiment_layout import default_batch_root, location
    from smartsom.config.experiment_layout import entry_id as relative_id
    from smartsom.config.launch import scientific_identity
    from smartsom.experiments.author_batch import DirectoryPlan

    if any(value not in (None, "off", 0) for value in calibration_overrides.values()):
        raise ConfigurationError(
            "Base directory execution uses per-Experiment calibration evidence"
        )
    incremental = execution_selection is not None
    canonical = all(location(p) is not None for p in paths)
    # Named files are research entries too; retain legacy canonical output paths.
    research = (
        canonical or incremental or any(p.name != "experiment.yaml" for p in paths)
    )
    locations = {p: location(p) for p in paths} if canonical else {}
    common = (
        Path(os.path.commonpath([str(v[1]) for v in locations.values()]))
        if canonical
        else Path(".")
    )
    files, inventory, selected = [], {}, []
    member_sources = {}
    output_root, identities, concurrency = None, set(), {}
    for path, member_id, compiled in compile_members(paths, overrides):
        if not research and member_id in member_sources:
            raise ConfigurationError(
                f"expanded batch ID {member_id!r} conflicts between {member_sources[member_id]} and {path}"
            )
        member_sources[member_id] = path
        entry = relative_id(path, directory) if directory.is_dir() else member_id
        if canonical:
            repo, study, _ = locations[path]
            member_root = (
                Path(compiled.operations.get("output_root") or repo / "runs") / common
            )
            relative_study = study.relative_to(common)
            task_prefix = f"{study.as_posix().replace('/', '--')}--{member_id}"
        else:
            relative_study = Path(".")
            task_prefix = entry if member_id == path.stem else entry + "--" + member_id
            member_root = Path(compiled.experiment.run.output).parent
            if research:
                member_root = default_batch_root(directory)
        if incremental and compiled.operations.get("output_root"):
            member_root = Path(compiled.operations["output_root"])
        elif any(v.startswith("experiment.run.output=") for v in compiled.overrides):
            member_root = (
                Path(compiled.experiment.run.output) if research else member_root
            )
        if output_root is not None and member_root != output_root:
            raise ConfigurationError("batch entries must use one output root")
        output_root = member_root
        full_science = compiled.scientific_sha256
        inventory[task_prefix] = {
            "entry": entry,
            "scientific_sha256": full_science,
            "source_files": [list(item) for item in compiled.source_files],
            "components": [
                (name, hashlib.sha256(content).hexdigest())
                for name, content in compiled.components
            ],
            "batch": compiled.experiment.batch.model_dump(mode="json"),
        }
        settings = compiled.experiment.run
        if settings.tune != "off":
            raise ConfigurationError(
                "Base batch children use native execution; use standalone run for Tune"
            )
        seeds = (
            compiled.experiment.train_seeds
            if research and settings.task == "train"
            else (settings.seed,)
        )
        for index, seed in enumerate(seeds):
            documents = dict(compiled.documents)
            child = None
            member = compiled
            identity = member_id
            if research:
                # Keep subdirectory identity, including nested named entries.
                trial = member_id if canonical else task_prefix
                child = relative_study / "trials" / trial / f"seed-{seed}"
                document = yaml.safe_load(documents["experiment"])
                document["run"]["seed"] = seed
                document["run"]["output"] = str(output_root / child)
                documents["experiment"] = yaml.safe_dump(document, sort_keys=False)
                member = replace(
                    compiled,
                    scientific_sha256=scientific_identity(
                        {k: yaml.safe_load(v) for k, v in documents.items()}
                    ),
                    experiment=BaseExperiment.model_validate(document),
                    documents=tuple(documents.items()),
                )
                identity = task_prefix + f"--seed-{seed}"
            scoped = (str(relative_study), member.scientific_sha256)
            if scoped in identities:
                raise ConfigurationError("duplicate scientific entry in Base batch")
            identities.add(scoped)
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
                dict(
                    id=identity,
                    path=path,
                    child=str(child) if child else None,
                    study=str(relative_study) if research else None,
                    compiled=member,
                    stage=batch.stage,
                    parallel_files=batch.parallel_files,
                    scientific_sha256=member.scientific_sha256,
                )
            )
    if len({row["id"] for row in files}) != len(files):
        raise ConfigurationError("duplicate expanded batch IDs")
    shared = files[0]["compiled"].operations
    for row in files[1:]:
        if any(
            row["compiled"].operations[k] != shared[k]
            for k in ("launch", "concurrency", "resources", "display")
        ):
            raise ConfigurationError(
                "directory launch/concurrency/resources/display disagree; select one explicit profile or CLI override"
            )
    if incremental:
        from smartsom.experiments.batch_selection import select_tasks

        selected = select_tasks(
            files,
            execution_selection["requests"],
            execution_selection["first_train_seeds"],
            lambda path: relative_id(path, directory),
        )
    return DirectoryPlan(
        directory,
        tuple(files),
        output_root,
        0,
        "off",
        contract="reservation",
        operations=shared,
        research_layout=research,
        incremental=incremental,
        resume_missing=resume_missing,
        selection={**execution_selection, "tasks": selected} if incremental else None,
        inventory=inventory if incremental else None,
    )


def allocate_directory(directory_plan):
    from smartsom.experiments.author_batch import RUN_SCHEMA, SCHEMA, STATE_SCHEMA
    from smartsom.experiments.evidence import implementation_identity
    from smartsom.experiments.reservation import allocate

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    launch_id = uuid4().hex[:10]
    tracking_group = directory_plan.directory.name + "-" + stamp + "-" + launch_id
    root = directory_plan.output_root / (
        stamp + "-batch-" + directory_plan.directory.name + "-" + launch_id
    )
    if directory_plan.research_layout:
        root = directory_plan.output_root
    if root.exists() and directory_plan.incremental and directory_plan.resume_missing:
        from smartsom.experiments.incremental_batch import activate

        return activate(directory_plan)
    if root.exists():
        raise ConfigurationError(
            f"results already exist: {root}; use resume or a new experiment directory"
        )
    root.mkdir(parents=True, exist_ok=False)
    rows = []
    for row in directory_plan.files:
        child = (
            root / row["child"]
            if directory_plan.research_layout
            else root / "experiments" / row["id"]
        )
        allocate(row["compiled"], root=child, tracking_group=tracking_group)
        child_plan = json.loads((child / "plan.json").read_text())
        rows.append(frozen_row(row, child_plan, str(child.relative_to(root))))
    plan = {
        "schema": SCHEMA,
        "incremental": directory_plan.incremental,
        "inventory": directory_plan.inventory,
        "research_layout": directory_plan.research_layout,
        "runtime_contract": "smartsom.current-runtime/v1",
        "run_id": uuid4().hex,
        "tracking_group": tracking_group,
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
        "files": {
            row["id"]: {
                "status": "queued"
                if not directory_plan.incremental
                or row["id"] in directory_plan.selection["tasks"]
                else "deferred"
            }
            for row in rows
        },
        "selection": directory_plan.selection,
        "submission_tasks": directory_plan.selection["tasks"]
        if directory_plan.incremental
        else [row["id"] for row in rows],
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
    if directory_plan.research_layout:
        write_json(root / "manifest.json", {**plan, "kind": "research-batch"})
    return root


def compile_members(paths, overrides):
    """Expand scientific axes once, before the shared admission queue is created."""
    from itertools import product

    import yaml

    from smartsom.config.codec import read_document
    from smartsom.config.launch import OPERATION_RUN_KEYS

    overrides = {**overrides, "_case_cache": {}}
    for path in paths:
        name = path.parent.name if path.name == "experiment.yaml" else path.stem
        matrix = read_document(path).get("matrix")
        if matrix is None:
            yield path, name, compile_reservation(path, **overrides)
            continue
        if not isinstance(matrix, dict) or not matrix:
            raise ConfigurationError(
                "matrix must be a nonempty mapping of scientific field to values"
            )
        for key, values in matrix.items():
            from smartsom.config.experiment_layout import location

            if key == "experiment.run.seed" and location(path) is not None:
                raise ConfigurationError(
                    "research packages use train_seeds, not a run.seed matrix axis"
                )
            parts = key.split(".")
            scientific = (
                parts[0]
                in {"scenario", "factory", "workload", "algorithm", "experiment"}
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
                    "scenario",
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
            yield path, f"{name}--{index:04d}", compiled


def frozen_row(row, child_plan, child):
    """Common immutable task descriptor for initial and appended inventory."""
    return {
        "id": row["id"],
        "path": str(row["path"]),
        "child": child,
        "study": row.get("study"),
        "child_plan_sha256": digest(child_plan),
        "entry_ids": ["run"],
        "scientific_sha256": [row["scientific_sha256"]],
        "task": row["compiled"].experiment.run.task,
        "stage": row["stage"],
        "parallel_files": row["parallel_files"],
        "gate": None,
        "contract": "reservation",
    }
