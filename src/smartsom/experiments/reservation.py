"""Public reservation execution adapter; physics/policies share one implementation."""

import hashlib
import json
import shutil
import time
from pathlib import Path
from types import SimpleNamespace

from smartsom.algorithms.base_policy import build_policy
from smartsom.config.base_diagnostic import load_inputs
from smartsom.experiments import reservation_runtime as runtime
from smartsom.experiments.control import StopRequested
from smartsom.telemetry.runtime import operation


def smoke(args):
    ex, factory, workload, options, paths = load_inputs(args.experiment)
    runtime.freeze(
        args.output, paths, "smoke", {"tick_budget_per_method": ex.run.smoke_ticks}
    )
    rows = []
    deadline = time.monotonic() + ex.run.seconds
    for method in ex.run.methods:
        current = (
            options
            if method == "configured"
            else options.model_copy(
                update={"method": method, "bindings": {}, "providers": {}}
            )
        )
        policy = build_policy(current, ex.run.seed)
        row, rewards, critic = runtime.episode(
            factory,
            workload,
            current,
            policy,
            ex.run.seed,
            "normal",
            ex.drain_ticks,
            max_ticks=ex.run.smoke_ticks,
            wall_deadline=deadline,
            diagnostics=ex.run.diagnostics,
            metrics_level=ex.run.metrics,
            metric_interval=ex.run.metric_interval,
        )
        row["method"] = method
        if getattr(policy, "training", False):
            policy.update(
                rewards, terminal=row["status"] == "completed", final_critic=critic
            )
            policy.save(Path(args.output) / "last.pt")
            row["updates"] = policy.updates
            row["losses"] = policy.last_losses
        rows.append(row)
        runtime.write_json(
            Path(args.output) / "progress.json",
            {"status": "smoke", "methods_done": len(rows)},
        )
    runtime.write_json(
        Path(args.output) / "summary.json",
        {"kind": "bounded-smoke-not-convergence", "results": rows},
    )
    runtime.write_json(
        Path(args.output) / "progress.json",
        {"status": "completed", "methods_done": len(rows)},
    )


SCHEMA = "smartsom.reservation-plan/v1"


def load(root):
    root = Path(root).resolve()
    plan = json.loads((root / "plan.json").read_text())
    if plan["schema"] != SCHEMA:
        raise ValueError("not a reservation execution plan")
    source = runtime.source_identity()
    from smartsom.experiments.evidence import runtime_source_matches

    if not runtime_source_matches(plan["source"]["environment"], source["environment"]):
        raise ValueError("frozen runtime identity mismatch")
    if any(
        plan["source"][key] != source[key]
        for key in ("commit", "implementation_sha256")
    ):
        raise ValueError("frozen source identity mismatch")
    for name, digest in plan["files"].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
            raise ValueError("frozen input identity mismatch")
    return plan


def allocate(plan, *, root=None):
    root = Path(root or plan.experiment.run.output)
    if root.exists():
        raise ValueError(f"output already exists: {root}")
    root.mkdir(parents=True)
    inputs = root / "inputs"
    inputs.mkdir()
    plan.materialize(inputs)
    # Snapshot referenced evidence as well as authoring documents.
    for name in ("calibration", "checkpoint"):
        path = getattr(plan.experiment.run, name)
        if path:
            shutil.copyfile(
                path, inputs / (name + (".pt" if name == "checkpoint" else ".json"))
            )
    runtime.write_json(
        root / "plan.json",
        {
            "schema": SCHEMA,
            "source": runtime.source_identity(),
            "files": {
                str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in inputs.rglob("*")
                if p.is_file()
            },
        },
    )
    runtime.write_json(root / "batch.json", {"status": "allocated"})
    runtime.write_json(root / "authoring.json", plan.summary())
    return root


@operation("reservation")
def run(plan):
    root = allocate(plan)
    if plan.experiment.run.background:
        from smartsom.experiments.background import launch

        return launch(root)
    return execute_saved(root)


def arguments(root):
    ex, *_ = load_inputs(root / "inputs/experiment.yaml")
    settings = ex.run
    return SimpleNamespace(
        experiment=root / "inputs/experiment.yaml",
        output=root,
        native=True,
        seed=settings.seed,
        seconds=settings.seconds,
        ticks=ex.train_ticks,
        calibration=root / "inputs/calibration.json" if settings.calibration else None,
        cases=settings.calibration_cases if settings.task == "calibrate" else None,
        max_drain=settings.max_drain,
        checkpoint=str(root / "inputs/checkpoint.pt") if settings.checkpoint else None,
        method=list(settings.methods),
        selection=settings.selection,
        split=settings.split,
        action_seeds=settings.action_seeds,
    ), settings


def execute_saved(root):
    from smartsom.telemetry.runtime import bind

    root = Path(root).resolve()
    load(root)
    bind(root)
    args, settings = arguments(root)
    progress = root / "progress.json"
    if progress.exists():
        previous = json.loads(progress.read_text())
        if previous["status"] in (
            "completed",
            "wall_budget_reached",
            "recommended",
            "frozen",
        ):
            return {"directory": str(root), **previous}
    if settings.tune != "off":
        from smartsom.experiments.reservation_tune import execute as tune

        runtime.write_json(progress, {"status": "running", "stage": "tuning"})
        try:
            return tune(root)
        except BaseException as exc:
            status = (
                "stopped"
                if isinstance(exc, (StopRequested, KeyboardInterrupt, SystemExit))
                else "failed"
            )
            runtime.write_json(progress, {"status": status, "error": str(exc)})
            runtime.write_json(root / "batch.json", {"status": status})
            raise
    attempts_file = root / "attempts.json"
    attempts = json.loads(attempts_file.read_text()) if attempts_file.exists() else []
    while True:
        try:
            {
                "smoke": smoke,
                "calibrate": runtime.calibrate,
                "train": runtime.train,
                "evaluate": runtime.evaluate,
            }[settings.task](args)
            result = {"directory": str(root), **json.loads(progress.read_text())}
            runtime.write_json(root / "batch.json", result)
            return result
        except BaseException as exc:
            stopped = isinstance(exc, (StopRequested, KeyboardInterrupt, SystemExit))
            attempts.append(
                {
                    "time": time.time(),
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "stopped": stopped,
                }
            )
            runtime.write_json(attempts_file, attempts)
            runtime.write_json(progress, {"status": "stopped" if stopped else "failed"})
            runtime.write_json(
                root / "batch.json", {"status": "stopped" if stopped else "failed"}
            )
            failures = sum(not item["stopped"] for item in attempts)
            if (
                stopped
                or settings.task not in {"train", "evaluate", "calibrate"}
                or failures > settings.max_failures
                or isinstance(exc, (ValueError, AssertionError))
            ):
                raise
            # A new session restores the latest atomic complete-update generation.
            load(root)


@operation("reservation-resume")
def resume(root, *, background=False):
    from smartsom.experiments.background import _check_owner, launch

    root = Path(root).resolve()
    load(root)
    progress = root / "progress.json"
    if progress.exists() and json.loads(progress.read_text())["status"] in (
        "completed",
        "wall_budget_reached",
        "recommended",
        "frozen",
    ):
        return execute_saved(root)
    _check_owner(root)
    return launch(root, resume=True) if background else execute_saved(root)


def execute(args):
    """Reject unsupported legacy switches rather than silently ignoring intent."""
    from smartsom.config.codec import ConfigurationError
    from smartsom.config.reservation import compile_reservation

    supported = {
        "command",
        "run_config",
        "config",
        "set",
        "task",
        "seed",
        "steps",
        "learning_rate",
        "checkpoint",
        "output_root",
        "name",
        "background",
        "tune",
        "verbose",
        "progress",
        "log_format",
        "summary_interval",
        "progress_title",
        "debug",
    }
    # Parser defaults include a few empty containers and fixed presentation fields.
    from smartsom.experiments.cli import _parser

    if args.config and args.run_config:
        raise ConfigurationError("provide only one experiment configuration")
    defaults = vars(_parser().parse_args([args.command]))
    for key, value in vars(args).items():
        if key not in supported and value != defaults.get(key):
            raise ConfigurationError(
                f"reservation execution does not support --{key.replace('_', '-')}"
            )
    changes = list(args.set)
    if args.steps is not None and args.task != "train":
        raise ConfigurationError(
            "--steps requires --task train; smoke uses experiment.run.smoke_ticks"
        )
    for key, target in (
        ("steps", "experiment.train_ticks"),
        ("learning_rate", "algorithm.learning_rate"),
        ("checkpoint", "experiment.run.checkpoint"),
    ):
        if getattr(args, key, None) is not None:
            changes.append(target + "=" + json.dumps(getattr(args, key)))
    if args.output_root is not None or args.name is not None:
        changes.append(
            "experiment.run.output="
            + json.dumps(
                str(
                    (
                        Path(args.output_root or "runs") / (args.name or "reservation")
                    ).resolve()
                )
            )
        )
    plan = compile_reservation(
        args.config or args.run_config,
        sets=changes,
        task=args.task,
        seed=args.seed,
        background=args.background,
        tuning=args.tune,
    )
    return plan.summary() if args.command == "check" else run(plan)
