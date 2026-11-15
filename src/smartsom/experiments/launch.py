"""Shared launch, read-only comparison and operation-only continuation helpers."""

import time
from pathlib import Path

from smartsom.config.codec import read_document
from smartsom.config.launch import flatten, resolve_experiment, scientific_documents
from smartsom.config.reservation import compile_reservation
from smartsom.experiments.evidence import write_json


def compile_launch(value, **overrides):
    from smartsom.config.rllib_compatibility import validate_plan

    return validate_plan(_compile_launch(value, **overrides))


def _compile_launch(
    value,
    *,
    ids=None,
    tags=None,
    selectors=None,
    first_train_seeds=None,
    resume_missing=False,
    search=None,
    **overrides,
):
    path = resolve_experiment(value)
    if search is not None:
        from smartsom.experiments.tune_hpo import compile_search

        entry = path / "experiment.yaml" if path.is_dir() else path
        if ids or tags or selectors or first_train_seeds is not None or resume_missing:
            raise ValueError("search uses one explicit experiment and training seed")
        return compile_search(entry, search, **overrides)
    incremental = (
        first_train_seeds is not None or resume_missing or selectors is not None
    )
    if incremental and not path.is_dir():
        raise ValueError("execution seed selection requires an Experiment directory")
    from smartsom.config.experiment_layout import location

    if path.is_file() and location(path) is not None:
        from smartsom.experiments.reservation_batch import compile_files

        if ids or tags:
            raise ValueError("--id/--tag require an Experiment directory")
        return compile_files(path.parent, [path], **overrides)
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

        return compile_files(path, [path], **overrides)
    compiled = compile_reservation(path, **overrides)
    if (
        compiled.experiment.run.task == "train"
        and len(compiled.experiment.train_seeds) > 1
    ):
        from smartsom.experiments.reservation_batch import compile_files

        return compile_files(path, [path], **overrides)
    if (
        compiled.experiment.run.task == "train"
        and compiled.experiment.run.seed != compiled.experiment.train_seeds[0]
    ):
        return compile_reservation(
            path, **{**overrides, "seed": compiled.experiment.train_seeds[0]}
        )
    return compiled


def run_launch(plan, *, keep_open=False, interactive=False):
    from smartsom.config.rllib_compatibility import validate_plan
    from smartsom.experiments.author_batch import DirectoryPlan

    validate_plan(plan)
    if isinstance(plan, DirectoryPlan) and plan.incremental:
        from smartsom.experiments.incremental_batch import submission

        # Retain the claim until foreground completion or the background launcher's
        # verified owner handshake. No second submission can alter this selection.
        with submission(plan.output_root):
            return _run_launch(plan, keep_open=keep_open, interactive=interactive)
    return _run_launch(plan, keep_open=keep_open, interactive=interactive)


def _run_launch(plan, *, keep_open=False, interactive=False):
    if keep_open:
        raise ValueError("--keep-open was removed; Tune runs a frozen trial manifest")
    from importlib.util import find_spec

    from smartsom.experiments.author_batch import DirectoryPlan
    from smartsom.experiments.tune_hpo import SearchPlan
    from smartsom.telemetry.runtime import display_options

    if isinstance(plan, SearchPlan):
        from smartsom.experiments.tune_hpo import run

        return run(plan)
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
                from smartsom.experiments.author_batch import allocate

                root = allocate(plan)
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
                "execution settings cannot change while the Tune driver is active"
            )
        previous = read_json(directory / "execution.json")
        settings, origins = resolve_operations(
            {}, base=previous.get("settings", {}), profile=profile, cli=operations
        )
        if any((directory / "ray").glob("**/tuner.pkl")):
            for key in ("resources", "trial_resources", "concurrency"):
                if settings[key] != previous["settings"].get(key):
                    raise ValueError(
                        f"Tune restore retains frozen {key}; choose these settings before a new run"
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
