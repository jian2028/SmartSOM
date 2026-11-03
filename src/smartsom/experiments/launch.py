"""Shared launch, read-only comparison and operation-only continuation helpers."""

import time
from pathlib import Path

from smartsom.config.codec import read_document
from smartsom.config.launch import flatten, resolve_experiment, scientific_documents
from smartsom.config.reservation import compile_reservation
from smartsom.experiments.evidence import write_json


def compile_launch(
    value,
    *,
    ids=None,
    tags=None,
    selectors=None,
    first_train_seeds=None,
    resume_missing=False,
    **overrides,
):
    path = resolve_experiment(value)
    incremental = (
        first_train_seeds is not None or resume_missing or selectors is not None
    )
    if incremental and not path.is_dir():
        raise ValueError("execution seed selection requires an Experiment directory")
    entry = path / "experiment.yaml" if path.is_dir() else path
    if entry.is_file() and (
        overrides.get("tuning") not in (None, "off")
        or read_document(entry).get("run", {}).get("tune", "off") != "off"
    ):
        if incremental:
            raise ValueError("incremental batches require native directory execution")
        return compile_reservation(entry, **overrides)
    from smartsom.config.experiment_layout import location

    if path.is_file() and location(path) is not None:
        from smartsom.experiments.reservation_batch import compile_files

        if ids or tags:
            raise ValueError("--id/--tag require an Experiment directory")
        return compile_files(path.parent, [path], {}, **overrides)
    if path.is_dir():
        from smartsom.experiments.author_batch import compile_directory

        return compile_directory(
            path,
            ids=ids,
            tags=tags,
            selectors=selectors,
            first_train_seeds=first_train_seeds,
            resume_missing=resume_missing,
            **overrides,
        )
    if ids or tags:
        raise ValueError("--id/--tag require an Experiment directory")
    if "matrix" in read_document(path):
        from smartsom.experiments.reservation_batch import compile_files

        return compile_files(path, [path], {}, **overrides)
    return compile_reservation(path, **overrides)


def run_launch(plan, *, keep_open=False, interactive=False):
    from smartsom.experiments.author_batch import DirectoryPlan

    if isinstance(plan, DirectoryPlan) and plan.incremental:
        from smartsom.experiments.incremental_batch import submission

        # Retain the claim until foreground completion or the background launcher's
        # verified owner handshake. No second submission can alter this selection.
        with submission(plan.output_root):
            return _run_launch(plan, keep_open=keep_open, interactive=interactive)
    return _run_launch(plan, keep_open=keep_open, interactive=interactive)


def _run_launch(plan, *, keep_open=False, interactive=False):
    from importlib.util import find_spec

    from smartsom.experiments.author_batch import DirectoryPlan
    from smartsom.telemetry.runtime import display_options

    members = (
        [row["compiled"] for row in plan.files]
        if isinstance(plan, DirectoryPlan)
        else [plan]
    )
    for member in members:
        if member.experiment.run.task not in {"train", "evaluate"}:
            continue
        tracking = member.operations["tracking"]
        for module, enabled in (
            ("wandb", tracking["mode"] != "disabled"),
            ("tensorboard", tracking["tensorboard"]),
        ):
            if enabled and find_spec(module) is None:
                raise ValueError(
                    f"{module} is enabled but missing; install the {module} extra or explicitly disable it"
                )
    display = plan.operations.get("display", {})
    with display_options(
        progress=display.get("progress"),
        format=display.get("log_format"),
        every_seconds=display.get("summary_interval"),
    ):
        if interactive and plan.operations["launch"]["mode"] == "foreground":
            from smartsom.experiments.foreground import run as foreground
            from smartsom.telemetry.runtime import OVERRIDES

            if isinstance(plan, DirectoryPlan):
                from smartsom.experiments.author_batch import (
                    _runtime_initialize,
                    allocate,
                    load,
                )

                root = allocate(plan)
                _, saved, state = load(root)
                _runtime_initialize(
                    root,
                    saved,
                    state,
                    limit=plan.operations["concurrency"]["max_concurrent"],
                    keep_open=keep_open,
                )
            else:
                from smartsom.experiments.reservation import allocate

                if keep_open:
                    raise ValueError("--keep-open requires a directory scheduler")
                root = allocate(plan)
            return foreground(root, display_options=OVERRIDES.get())
        if isinstance(plan, DirectoryPlan):
            from smartsom.experiments.author_batch import run

            return run(
                plan,
                background=plan.operations["launch"]["mode"] == "background",
                runtime_limit=plan.operations["concurrency"]["max_concurrent"],
                keep_open=keep_open,
            )
        from smartsom.experiments.reservation import run

        if keep_open:
            raise ValueError("--keep-open requires a directory scheduler")
        return run(plan)


def update_execution(root, *, profile=None, operations=None):
    """Validate every effective child config before recording operation overrides."""
    from smartsom.config.base_diagnostic import load_inputs
    from smartsom.config.launch import resolve_operations
    from smartsom.experiments import runtime_control
    from smartsom.experiments.catalog import read_json, require_current
    from smartsom.experiments.control import ACTIVE, alive, processes, read

    pending = {}

    def prepare(directory):
        directory = Path(directory).resolve()
        if directory in pending:
            return
        require_current(directory)
        owner = read(directory)
        if (
            owner
            and owner.get("status") in ACTIVE
            and alive(owner["owner"], processes())
        ):
            raise ValueError(
                "change live concurrency through batch controls; tracking/profile changes require a stopped driver"
            )
        previous = read_json(directory / "execution.json")
        settings, origins = resolve_operations(
            previous.get("settings", {}), profile=profile, cli=operations
        )
        if settings.get("output_root") != previous.get("settings", {}).get(
            "output_root"
        ):
            raise ValueError(
                "resume cannot relocate an existing run; output_root is for new runs"
            )
        plan = read_json(directory / "plan.json")
        if plan.get("schema") == "smartsom.author-batch-plan/v1":
            from smartsom.experiments.author_batch import load

            _, plan, _ = load(directory)
        else:
            from smartsom.experiments.reservation import load

            load(directory)
            load_inputs(
                directory / "inputs/experiment.yaml", execution_settings=settings
            )
        pending[directory] = (previous, settings, origins)
        children = (
            {directory / row["child"] for row in plan.get("files", [])}
            if isinstance(plan.get("files"), list)
            else set()
        )
        children.update(
            Path(row["root"])
            for row in runtime_control.snapshot(directory).get("additions", {}).values()
        )
        for child in sorted(children):
            prepare(child)

    # Preparation is read-only, including operation history and runtime commands.
    # An invalid late child cannot leave an earlier child partially updated.
    prepare(root)
    for directory, (previous, settings, origins) in pending.items():
        if settings != previous.get("settings"):
            history = previous.get("history", []) + [
                {"time": time.time(), "settings": settings, "origins": origins}
            ]
            write_json(
                directory / "execution.json",
                {"settings": settings, "origins": origins, "history": history},
            )
        if runtime_control.snapshot(directory):
            runtime_control.submit(
                directory, "limit", value=settings["concurrency"]["max_concurrent"]
            )
            runtime_control.submit(
                directory, "mode", value=settings["concurrency"]["mode"]
            )
            budget = {
                k: v
                for k, v in {
                    "cpus": settings["resources"]["cpus"],
                    "memory": int(settings["resources"]["memory_gib"] * 1024**3)
                    if settings["resources"]["memory_gib"]
                    else None,
                }.items()
                if v is not None
            }
            if budget:
                runtime_control.submit(directory, "budget", value=budget)


def compare(left, right, *, roots=None):
    from smartsom.experiments.author_batch import DirectoryPlan
    from smartsom.experiments.catalog import read_json, resolve_run

    def content(value):
        path = Path(value).expanduser()
        if path.is_file():
            plan = compile_reservation(path)
        elif path.is_dir() and (path / "plan.json").exists():
            plan = None
        else:
            try:
                plan = compile_launch(value)
            except ValueError:
                path, plan = resolve_run(value, roots), None
        if isinstance(plan, DirectoryPlan):
            return {
                row["id"]: {
                    "scientific": scientific_documents(
                        {
                            k: __import__("yaml").safe_load(v)
                            for k, v in row["compiled"].documents
                        }
                    ),
                    "operations": row["compiled"].operations,
                }
                for row in plan.files
            }
        if plan is not None:
            import yaml

            docs = {k: yaml.safe_load(v) for k, v in plan.documents}
            return {
                "scientific": scientific_documents(docs),
                "operations": plan.operations,
                "origins": plan.origins,
            }
        inputs = path / "inputs"
        if not (inputs / "experiment.yaml").exists():
            batch = read_json(path / "plan.json")
            if batch.get("schema") == "smartsom.author-batch-plan/v1":
                from smartsom.experiments.batch_inventory import read_view

                batch, _ = read_view(path)
                return {
                    row["id"]: content(path / row["child"]) for row in batch["files"]
                }
            raise ValueError(
                "historical snapshot format requires archived config inspection"
            )
        docs = {
            name: read_document(inputs / (name + ".yaml"))
            for name in ("experiment", "factory", "workload", "algorithm")
        }
        execution = read_json(path / "execution.json")
        return {
            "scientific": scientific_documents(docs),
            "operations": execution.get("settings", {}),
            "origins": execution.get("origins", {}),
        }

    a, b = flatten(content(left)), flatten(content(right))
    return [
        {
            "field": key,
            "left": a.get(key),
            "right": b.get(key),
            "left_present": key in a,
            "right_present": key in b,
        }
        for key in sorted(a.keys() | b.keys())
        if a.get(key) != b.get(key) or (key in a) != (key in b)
    ]


def check_resource_budget(root, *, prepared=None):
    """Apply explicit single-run admission budgets without changing scientific work.

    This is a logical admission check, not OS process isolation. Directory/Tune
    schedulers additionally reserve aggregate resources and observe active workers.
    """
    import psutil

    from smartsom.config.base_diagnostic import load_inputs

    if prepared is None:
        ex, *_ = load_inputs(Path(root) / "inputs/experiment.yaml")
    else:
        from smartsom.experiments.reservation_runtime import prepared_identity

        inputs, expected = prepared
        if prepared_identity(inputs[-1]) != expected:
            raise ValueError("prepared input identity changed before admission")
        ex = inputs[0]
    budget = ex.resources
    if budget.cpus is None and budget.memory_gib is None:
        return
    required_memory = max(256 * 1024**2, psutil.Process().memory_info().rss)
    accepted = (budget.cpus is None or budget.cpus >= 1) and (
        budget.memory_gib is None
        or budget.memory_gib * 1024**3 >= required_memory * 1.25
    )
    write_json(
        Path(root) / "resource-admission.json",
        {
            "accepted": accepted,
            "requested_cpus": 1,
            "observed_or_floor_memory": required_memory,
            "memory_peak_factor": 1.25,
            "budget": budget.model_dump(),
            "scope": "single native experiment",
        },
    )
    if not accepted:
        raise ValueError(
            "resource budget cannot admit this experiment; increase CPU/memory budget without changing scientific inputs"
        )
