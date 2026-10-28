"""Public reservation execution adapter; physics/policies share one implementation."""

import hashlib
import json
import shutil
import time
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from smartsom.algorithms.base_policy import build_policy
from smartsom.config.base_diagnostic import load_inputs
from smartsom.experiments import reservation_runtime as runtime
from smartsom.experiments.control import PauseRequested, StopRequested
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
            "runtime_contract": "smartsom.current-runtime/v1",
            "run_id": uuid4().hex,
            "scientific_sha256": plan.scientific_sha256,
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
    runtime.write_json(
        root / "execution.json",
        {"settings": plan.operations, "origins": plan.origins, "history": []},
    )
    saved = json.loads((root / "plan.json").read_text())
    print(
        f"Run: {saved['run_id']}\nOutput: {root}\nMonitor: smartsom attach {root}",
        flush=True,
    )
    return root


@operation("reservation")
def run(plan):
    root = allocate(plan)
    if plan.experiment.launch.mode == "background":
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
    from smartsom.experiments.launch import check_resource_budget

    check_resource_budget(root)
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
                "paused"
                if isinstance(exc, PauseRequested)
                else "stopped"
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
            status = (
                "paused"
                if isinstance(exc, PauseRequested)
                else "stopped"
                if stopped
                else "failed"
            )
            runtime.write_json(progress, {"status": status})
            runtime.write_json(root / "batch.json", {"status": status})
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
