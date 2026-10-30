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


def compile_files(directory, paths, calibration_overrides, **overrides):
    from smartsom.experiments.author_batch import DirectoryPlan

    if any(value not in (None, "off", 0) for value in calibration_overrides.values()):
        raise ConfigurationError(
            "Base directory execution uses per-Experiment calibration evidence"
        )
    files = []
    output_root = None
    identities = set()
    entry_sources = {}
    concurrency = {}
    from smartsom.config.experiment_layout import location

    research = all(location(p) is not None for p in paths)
    locations = {p: location(p) for p in paths} if research else {}
    if research:
        import os

        common = Path(
            os.path.commonpath([str(value[1]) for value in locations.values()])
        )
    members = []
    for path, entry_id, compiled in compile_members(paths, overrides):
        if research:
            repo, study, variant = locations[path]
            member_root = (
                Path(compiled.operations.get("output_root") or repo / "runs") / common
            )
            if any(v.startswith("experiment.run.output=") for v in compiled.overrides):
                member_root = Path(compiled.experiment.run.output)
            if output_root is not None and member_root != output_root:
                raise ConfigurationError("research entries must use one output root")
            output_root = member_root
            relative_study = study.relative_to(common)
            for seed in (
                compiled.experiment.train_seeds
                if compiled.experiment.run.task == "train"
                else (compiled.experiment.run.seed,)
            ):
                import yaml

                from smartsom.config.base_diagnostic import BaseExperiment

                documents = dict(compiled.documents)
                document = yaml.safe_load(documents["experiment"])
                child = relative_study / "trials" / entry_id / f"seed-{seed}"
                document["run"]["seed"] = seed
                document["run"]["output"] = str(output_root / child)
                documents["experiment"] = yaml.safe_dump(document, sort_keys=False)
                from smartsom.config.launch import scientific_identity

                member = replace(
                    compiled,
                    scientific_sha256=scientific_identity(
                        {k: yaml.safe_load(v) for k, v in documents.items()}
                    ),
                    experiment=BaseExperiment.model_validate(document),
                    documents=tuple(documents.items()),
                )
                members.append(
                    (
                        path,
                        f"{study.as_posix().replace('/', '--')}--{entry_id}--seed-{seed}",
                        member,
                        str(child),
                        str(relative_study),
                    )
                )
        else:
            members.append((path, entry_id, compiled, None, None))
    for path, entry_id, compiled, child, study in members:
        if entry_id in entry_sources:
            raise ConfigurationError(
                f"expanded batch ID {entry_id!r} conflicts between "
                f"{entry_sources[entry_id]} and {path}"
            )
        entry_sources[entry_id] = path
        settings = compiled.experiment.run
        if settings.tune != "off":
            raise ConfigurationError(
                "Base batch children use native execution; use standalone run for Tune"
            )
        parent = Path(settings.output).parent
        if not research and output_root is not None and parent != output_root:
            raise ConfigurationError(
                "Base batch run.output paths must share one parent directory"
            )
        if not research:
            output_root = parent
        documents = dict(compiled.documents)
        import yaml

        scientific = {key: yaml.safe_load(value) for key, value in documents.items()}
        from smartsom.config.launch import scientific_identity

        identity = scientific_identity(scientific)
        scoped_identity = (study, identity) if research else identity
        if scoped_identity in identities:
            raise ConfigurationError("duplicate scientific entry in Base batch")
        identities.add(scoped_identity)
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
                "child": child,
                "study": study,
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
        research_layout=research,
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
        rows.append(
            {
                "id": row["id"],
                "path": str(row["path"]),
                "child": str(child.relative_to(root)),
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
        )
    plan = {
        "schema": SCHEMA,
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
            yield path, f"{name}--{index:04d}", compiled
