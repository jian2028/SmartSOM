"""Frozen current experiment directories, scheduled by the shared NativeScheduler."""

import json
import re
import time
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from pathlib import Path

from smartsom.config.codec import ConfigurationError, digest
from smartsom.experiments.control import requested
from smartsom.experiments.evidence import (
    implementation_identity,
    runtime_source_matches,
    source_identity,
    write_json,
)
from smartsom.telemetry.runtime import CURRENT, bind, operation

SCHEMA = "smartsom.author-batch-plan/v1"
STATE_SCHEMA = "smartsom.author-batch-state/v1"
RUN_SCHEMA = "smartsom.author-batch-run/v1"
CALIBRATION_SECONDS = {"online": 0, "off": 0, "quick": 5 * 60, "full": 30 * 60}
FINAL = {"completed", "failed", "stopped"}


@dataclass(frozen=True)
class DirectoryPlan:
    directory: Path
    files: tuple
    output_root: Path
    calibration_seconds: float
    calibration_level: str = "off"
    calibration_candidate: str = "latest"
    selection: dict | None = None
    contract: str = "reservation"
    operations: dict | None = None

    def summary(self):
        return {
            "status": "checked",
            "input_type": "reservation-directory",
            "directory": str(self.directory),
            "output_root": str(self.output_root),
            "count_files": len(self.files),
            "files": [
                {
                    "id": row["id"],
                    "stage": row["stage"],
                    **row["compiled"].summary(),
                }
                for row in self.files
            ],
        }


def compile_directory(
    directory,
    *,
    require_dependencies=False,
    calibration_seconds=None,
    calibration_level=None,
    calibration_candidate=None,
    ids=None,
    tags=None,
    **overrides,
):
    """Check every direct YAML member, failing on an unsupported or duplicate input."""
    directory = Path(directory).expanduser().resolve()
    if not directory.is_dir():
        raise ConfigurationError(
            f"batch-run requires an Experiment directory: {directory}"
        )
    if calibration_level is not None and calibration_level not in CALIBRATION_SECONDS:
        raise ConfigurationError("calibration level must be online, off, quick or full")
    if calibration_seconds is not None and (
        not isinstance(calibration_seconds, (int, float))
        or isinstance(calibration_seconds, bool)
        or not 0 <= calibration_seconds < float("inf")
    ):
        raise ConfigurationError(
            "batch calibration budget must be finite and nonnegative"
        )
    calibration_overrides = {
        key: value
        for key, value in (
            ("calibration_level", calibration_level),
            ("calibration_candidate", calibration_candidate),
            ("calibration_seconds", calibration_seconds),
        )
        if value is not None
    }
    if calibration_seconds == 0:
        # V4 represents no isolated timeout as None; the shared budget below
        # still validates that an explicit zero is only used with online/off.
        calibration_overrides["calibration_seconds"] = None
    paths = sorted(
        (
            p
            for p in directory.iterdir()
            if p.is_file() and p.suffix.lower() in {".yaml", ".yml"}
        ),
        key=lambda p: p.name,
    )
    if not paths:
        raise ConfigurationError(
            "batch-run directory contains no Experiment YAML files"
        )
    inventory = {}
    for path in paths:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", path.stem):
            raise ConfigurationError(f"unsafe batch Experiment name: {path.name}")
        if path.stem in inventory:
            raise ConfigurationError(f"duplicate batch Experiment name: {path.stem}")
        inventory[path.stem] = path
    if ids is not None:
        if not ids or len(set(ids)) != len(ids):
            raise ConfigurationError("batch IDs must be nonempty and unique")
        unknown = set(ids) - set(inventory)
        if unknown:
            raise ConfigurationError(
                "unknown batch Experiment IDs: " + ", ".join(sorted(unknown))
            )
        paths = [inventory[identity] for identity in ids]
    if tags is not None:
        if not tags or any(not tag for tag in tags) or len(set(tags)) != len(tags):
            raise ConfigurationError("batch tags must be nonempty and unique")
        from smartsom.config.codec import read_document

        matched = []
        for path in paths:
            document = read_document(path)
            output = document.get("run", {})
            if not isinstance(output, dict):
                raise ConfigurationError(f"{path.name}: output must be a mapping")
            authored = output.get("tags", [])
            if not isinstance(authored, list) or any(
                type(tag) is not str for tag in authored
            ):
                raise ConfigurationError(
                    f"{path.name}: run.tags must be a list of strings"
                )
            if set(tags) <= set(authored):
                matched.append(path)
            elif ids is not None:
                raise ConfigurationError(
                    f"{path.stem}: requested ID does not match all tags"
                )
        paths = matched
        if not paths:
            raise ConfigurationError("batch selection matched no Experiments")
    from smartsom.config.reservation import is_reservation

    reservation_inputs = [is_reservation(path) for path in paths]
    if any(reservation_inputs):
        if not all(reservation_inputs):
            raise ConfigurationError(
                "mixing Base and V4 contracts in one batch is unsupported"
            )
        from smartsom.experiments.reservation_batch import compile_files

        return compile_files(directory, paths, calibration_overrides, **overrides)
    raise ConfigurationError(
        "historical batch inputs require archived source; see docs/archive/index.json"
    )


def allocate(directory_plan):
    from smartsom.experiments.reservation_batch import allocate_directory

    return allocate_directory(directory_plan)


def load(root):
    from smartsom.experiments.reservation import load as load_reservation

    root = Path(root).expanduser().resolve()
    plan = json.loads((root / "plan.json").read_text())
    state = json.loads((root / "batch.json").read_text())
    if plan.get("contract") != "reservation":
        raise ValueError("historical batch recovery requires archived source")
    if (
        plan.get("schema") != SCHEMA
        or state.get("schema") != STATE_SCHEMA
        or digest(plan) != state.get("plan_sha256")
    ):
        raise ValueError("frozen directory batch changed")
    if (
        not runtime_source_matches(plan["source"], source_identity())
        or implementation_identity() != plan["implementation_sha256"]
    ):
        raise ValueError("batch source/dependency identity changed; create a new run")
    for row in plan["files"]:
        child = (root / row["child"]).resolve()
        if not child.is_relative_to(root / "experiments"):
            raise ValueError("batch child escapes its run")
        child_plan = load_reservation(child)
        if digest(child_plan) != row["child_plan_sha256"]:
            raise ValueError("frozen child plan changed")
    return root, plan, state


def _save(root, state):
    write_json(root / "batch.json", state)
    record = json.loads((root / "run.json").read_text())
    record.update(status=state["status"], stage=state["stage"], files=state["files"])
    write_json(root / "run.json", record)


def _quiet_worker(root, parent_root):
    """Keep child diagnostics in child logs while the parent owns the Rich view."""
    from smartsom.experiments.control import CURRENT as CONTROL
    from smartsom.experiments.control import PauseRequested, Scope, StopRequested

    root = Path(root)
    scope = Scope()
    scope.inherit_driver(parent_root)
    token = CONTROL.set(scope)
    status = "failed"
    try:
        (root / "logs").mkdir(parents=True, exist_ok=True)
        with (
            (root / "logs/driver-stdout.log").open("a") as stdout,
            (root / "logs/driver-stderr.log").open("a") as stderr,
        ):
            with redirect_stdout(stdout), redirect_stderr(stderr):
                from smartsom.experiments.reservation import execute_saved

                result = execute_saved(root)
                if result["status"] in {"frozen", "wall_budget_reached"}:
                    # Scheduling completion is separate from scientific completion.
                    write_json(
                        root / "batch.json",
                        {
                            **result,
                            "status": "completed",
                            "native_status": result["status"],
                        },
                    )
        status = result.get("status", "completed")
        return result
    except (StopRequested, PauseRequested) as error:
        status = "paused" if isinstance(error, PauseRequested) else "stopped"
        return {"directory": str(root), "status": status}
    except KeyboardInterrupt:
        status = "stopped"
        raise
    finally:
        scope.finish(status)
        CONTROL.reset(token)


def _publish(root, plan, state):
    from smartsom.telemetry.current_progress import presentation

    session = CURRENT.get()
    if session is None:
        return
    session.configure_batch(
        [
            dict(id=row["id"], stage=row["stage"], barrier_group=str(root), requires=[])
            for row in plan["files"]
        ]
    )
    session.phase(state["stage"])
    with session.batch_updates():
        for row in plan["files"]:
            status = state["files"][row["id"]]["status"]
            progress_path = root / row["child"] / "progress.json"
            progress = (
                json.loads(progress_path.read_text()) if progress_path.exists() else {}
            )
            details, total, unit = presentation(progress, task=row["task"])
            session.update(
                row["id"],
                {
                    **details,
                    "display_name": row["id"],
                    "status": status,
                    "stage": details["stage"] if progress else status,
                    "scheduler_status": status,
                    "context": f"stage {row['stage']} · {row['task']}",
                },
                total=total,
                unit=unit,
                final=status in FINAL,
            )


def _gate(root, row):
    path = root / row["child"] / "cases.jsonl"
    cases = (
        [json.loads(line) for line in path.read_text().splitlines()]
        if path.exists()
        else []
    )
    if any(case.get("engineering_failure") for case in cases):
        raise RuntimeError(f"{row['id']}: engineering failure in evaluation")
    return {"cases": len(cases), "passed": True}


def _request_owned_stop(root):
    from smartsom.experiments.control import read
    from smartsom.experiments.control import write_json as control_write

    owner = read(root)
    if owner and owner.get("status") in {"running", "stop_requested", "stopping"}:
        control_write(root / "control/stop.json", {"id": owner["id"]})


def _runtime_initialize(root, plan, state, *, limit=None, keep_open=False):
    from smartsom.experiments import runtime_control as runtime

    settings_path = Path(root) / "execution.json"
    operations = (
        json.loads(settings_path.read_text()).get("settings", {})
        if settings_path.exists()
        else plan.get("operations", {})
    )
    default = max((row.get("parallel_files", 1) for row in plan["files"]), default=1)
    runtime.initialize(
        root,
        limit or operations.get("concurrency", {}).get("max_concurrent", default),
        mode=operations.get("concurrency", {}).get("mode", "manual"),
        keep_open=keep_open,
    )
    with runtime.transaction(root) as saved:
        resources = operations.get("resources", {})
        if resources.get("cpus") is not None:
            saved["budget"].setdefault("cpus", resources["cpus"])
        if resources.get("memory_gib") is not None:
            saved["budget"].setdefault("memory", int(resources["memory_gib"] * 1024**3))
        for row in plan["files"]:
            saved["tasks"].setdefault(
                row["id"], {"status": state["files"][row["id"]]["status"]}
            )


def _run_children(root, plan, state, rows, *, parallel_files):
    from smartsom.experiments import runtime_control as runtime
    from smartsom.experiments.runtime_workers import NativeScheduler

    _runtime_initialize(root, plan, state, limit=parallel_files)
    jobs = {
        row["id"]: dict(
            root=str(root / row["child"]),
            status=state["files"][row["id"]]["status"],
            own_slots=len(row.get("entry_ids", [])) > 1,
        )
        for row in rows
    }
    for job in jobs.values():
        if job["status"] not in {"completed", "failed"}:
            job["status"] = "queued"
    scheduler = NativeScheduler(root, jobs)
    try:
        while True:
            if requested(root):
                with runtime.transaction(root) as saved:
                    saved["stopping"] = True
            observed = scheduler.tick()
            for row in rows:
                state["files"][row["id"]].update(jobs[row["id"]])
            state["status"] = observed["status"]
            _save(root, state)
            _publish(root, plan, state)
            if not scheduler.unfinished:
                break
            time.sleep(0.25)
    except BaseException:
        scheduler.close()
        raise
    return not observed["stopping"] and all(
        jobs[row["id"]]["status"]
        in {"completed", "stopped", "removed", "partial_removed", "partial_stopped"}
        for row in rows
    )


@operation("batch-directory")
def execute_saved(root, *, retry_failed=False):
    from smartsom.experiments.locking import exclusive_lock

    root, plan, state = load(root)
    bind(root)
    from smartsom.experiments.queue_admission import initialize

    session = CURRENT.get()
    if session:
        session.total_tasks = len(plan["files"])
        _publish(root, plan, state)
    with exclusive_lock(root / "driver.lock"):
        from smartsom.experiments import runtime_control as runtime

        initialize(root, supported=plan.get("calibration_level") in {"off", "online"})
        _runtime_initialize(root, plan, state)
        if state["status"] in {
            "stopped",
            "failed",
            "partial_stopped",
            "partial_removed",
        }:
            runtime.restart(root)
        try:
            if state["status"] == "completed":
                for row in plan["files"]:
                    if row["task"] == "evaluate":
                        state["files"][row["id"]]["gate"] = _gate(root, row)
                _publish(root, plan, state)
                return {
                    "directory": str(root),
                    "status": "completed",
                    "files": state["files"],
                }
            if state["status"] == "failed" and not retry_failed:
                return {
                    "directory": str(root),
                    "status": "failed",
                    "files": state["files"],
                    "error": state.get("error"),
                    "retry_required": True,
                }
            state.pop("error", None)
            if retry_failed:
                for item in state["files"].values():
                    if item["status"] == "failed":
                        item["status"] = "queued"
            stages = sorted({row["stage"] for row in plan["files"]})
            for stage in stages:
                rows = [row for row in plan["files"] if row["stage"] == stage]
                for row in rows:
                    blocked = [
                        identity
                        for identity in row.get("requires", [])
                        if state["files"][identity]["status"] != "completed"
                    ]
                    if blocked:
                        raise RuntimeError(
                            f"{row['id']}: dependencies did not complete: {blocked}"
                        )
                if all(
                    state["files"][row["id"]]["status"] == "completed" for row in rows
                ):
                    for row in rows:
                        if row["task"] == "evaluate":
                            state["files"][row["id"]]["gate"] = _gate(root, row)
                    continue
                if requested(root):
                    state.update(status="stopped", stage="stopped")
                    break
                state.update(status="running", stage=f"stage_{stage}")
                _save(root, state)
                _publish(root, plan, state)
                unfinished = [
                    row
                    for row in rows
                    if state["files"][row["id"]]["status"] != "completed"
                ]
                if not _run_children(
                    root,
                    plan,
                    state,
                    unfinished,
                    parallel_files=rows[0]["parallel_files"],
                ):
                    if requested(root):
                        state.update(status="stopped", stage="stopped")
                        break
                    raise RuntimeError(f"batch stage {stage} did not complete")
                for row in rows:
                    if (
                        row["task"] == "evaluate"
                        and state["files"][row["id"]]["status"] == "completed"
                    ):
                        state["files"][row["id"]]["gate"] = _gate(root, row)
                        _save(root, state)
            else:
                from smartsom.experiments.runtime_workers import NativeScheduler

                tail = NativeScheduler(root)
                while True:
                    tail.tick()
                    if runtime.finish_if_drained(
                        root,
                        unfinished=tail.unfinished,
                        failed=any(j["status"] == "failed" for j in tail.jobs.values()),
                    ):
                        break
                    _publish(root, plan, state)
                    time.sleep(0.25)
                state.update(status=runtime.snapshot(root)["status"], stage="completed")
        except KeyboardInterrupt:
            state.update(status="stopped", stage="stopped")
            for file_state in state["files"].values():
                if file_state["status"] == "running":
                    file_state["status"] = "stopped"
        except BaseException as exc:
            state.update(
                status="failed", stage="failed", error=f"{type(exc).__name__}: {exc}"
            )
            for file_state in state["files"].values():
                if file_state["status"] == "running":
                    file_state["status"] = "failed"
        runtime.publish(root, active=0, status=state["status"])
        _save(root, state)
        _publish(root, plan, state)
        return {
            "directory": str(root),
            "status": state["status"],
            "files": state["files"],
            "error": state.get("error"),
        }


def run(
    directory_plan,
    *,
    background=False,
    display_options=None,
    runtime_limit=None,
    keep_open=False,
):
    root = allocate(directory_plan)
    _, plan, state = load(root)
    _runtime_initialize(root, plan, state, limit=runtime_limit, keep_open=keep_open)
    if background:
        from smartsom.experiments.background import launch

        return launch(root, display_options=display_options)
    return execute_saved(root, display_options=display_options)
