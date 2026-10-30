"""Public API for the current four-file experiment runtime."""

import json

from smartsom.config.codec import ConfigurationError


def check_experiment(path, **overrides):
    """Validate and explain resolved inputs without connecting tracking."""
    from smartsom.experiments.launch import compile_launch

    return compile_launch(path, **overrides).summary()


def run_experiment(path, **overrides):
    from smartsom.experiments.launch import compile_launch, run_launch

    return run_launch(compile_launch(path, **overrides))


def train(path, **overrides):
    return run_experiment(path, task="train", **overrides)


def evaluate(path, **overrides):
    return run_experiment(path, task="evaluate", **overrides)


def resume(
    source,
    *,
    background=None,
    retry_failed=False,
    roots=None,
    profile=None,
    operations=None,
    interactive=False,
):
    """Resume matching current frozen code, inputs and continuation state."""
    from smartsom.experiments import control, runtime_control
    from smartsom.experiments.catalog import require_current, resolve_run
    from smartsom.experiments.launch import update_execution

    root = resolve_run(source, roots)
    require_current(root)
    owner = control.read(root)
    if (
        owner
        and owner.get("status") in control.ACTIVE
        and control.alive(owner["owner"], control.processes())
    ):
        if profile or operations or background is not None:
            raise ConfigurationError(
                "live resume only releases pause; change live resources through batch controls"
            )
        if runtime_control.snapshot(root):
            return runtime_control.submit(root, "resume")
        control.clear_pause(root)
        return {"directory": str(root), "status": "resume_requested"}
    if profile or operations or background is not None:
        changes = dict(operations or {})
        if background is not None:
            changes["launch"] = {"mode": "background" if background else "foreground"}
        update_execution(root, profile=profile, operations=changes)
    saved = (
        json.loads((root / "execution.json").read_text())
        if (root / "execution.json").exists()
        else {}
    )
    if background is None:
        background = (
            saved.get("settings", {}).get("launch", {}).get("mode", "foreground")
            == "background"
        )
    control.clear_pause(root)
    plan_path = root / "plan.json"
    schema = (
        json.loads(plan_path.read_text()).get("schema") if plan_path.exists() else None
    )
    if interactive and not background:
        from smartsom.experiments.foreground import run as foreground
        from smartsom.telemetry.runtime import OVERRIDES

        if schema == "smartsom.reservation-plan/v1":
            from smartsom.experiments.reservation import load
        elif schema == "smartsom.author-batch-plan/v1":
            from smartsom.experiments.author_batch import load
        else:
            raise ConfigurationError("historical recovery requires archived source")
        load(root)
        return foreground(
            root,
            resume=True,
            retry_failed=retry_failed,
            display_options=OVERRIDES.get(),
        )
    if schema == "smartsom.reservation-plan/v1":
        from smartsom.experiments.reservation import resume as execute

        return execute(root, background=background)
    if schema == "smartsom.author-batch-plan/v1":
        from smartsom.experiments.author_batch import execute_saved, load

        load(root)
        if background:
            from smartsom.experiments.background import launch

            return launch(root, resume=True, retry_failed=retry_failed)
        return execute_saved(root, retry_failed=retry_failed)
    raise ConfigurationError(
        "historical experiment recovery requires its archived source and environment; "
        "see docs/migration/current-runtime.md and docs/archive/index.json"
    )


def batch_run(
    directory,
    *,
    background=None,
    runtime_limit=None,
    keep_open=False,
    ids=None,
    **overrides,
):
    from smartsom.experiments.launch import compile_launch, run_launch

    operations = dict(overrides.pop("operations", {}) or {})
    if runtime_limit is not None:
        operations.setdefault("concurrency", {})["max_concurrent"] = runtime_limit
    return run_launch(
        compile_launch(
            directory,
            ids=ids,
            background=background,
            operations=operations,
            **overrides,
        ),
        keep_open=keep_open,
    )
