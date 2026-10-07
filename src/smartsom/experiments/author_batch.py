"""A frozen directory of V4 Experiments with one parent lifecycle.

Directory scheduling is execution metadata. Each child is still compiled by
the ordinary V4 compiler and retains its own scientific identity and inputs.
"""

import json
import multiprocessing
import re
import time
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from smartsom.config.codec import ConfigurationError, digest, primitive
from smartsom.config.experiment_v4 import compile_experiment
from smartsom.experiments.composable import implementation_identity
from smartsom.experiments.control import requested
from smartsom.experiments.evidence import (
    runtime_source_matches,
    source_identity,
    write_json,
)
from smartsom.telemetry.runtime import CURRENT, bind, operation

SCHEMA = "smartsom.author-batch-plan/v1"
STATE_SCHEMA = "smartsom.author-batch-state/v1"
RUN_SCHEMA = "smartsom.author-batch-run/v1"
CALIBRATION_SECONDS = {"off": 0, "quick": 5 * 60, "full": 30 * 60}
FINAL = {"completed", "failed", "stopped"}


@dataclass(frozen=True)
class DirectoryPlan:
    directory: Path
    files: tuple
    output_root: Path
    calibration_seconds: float
    calibration_level: str = "quick"
    calibration_candidate: str = "latest"

    def summary(self):
        return {
            "status": "checked",
            "input_type": "experiment-v4-directory",
            "directory": str(self.directory),
            "output_root": str(self.output_root),
            "calibration_seconds": self.calibration_seconds,
            "calibration_level": self.calibration_level,
            "calibration_candidate": self.calibration_candidate,
            "count_files": len(self.files),
            "count_entries": sum(len(row["compiled"].entries) for row in self.files),
            "files": [
                {
                    "id": row["id"],
                    "path": str(row["path"]),
                    "stage": row["stage"],
                    "parallel_files": row["parallel_files"],
                    "gate": row["gate"],
                    "task": row["compiled"].experiment.task,
                    "count": len(row["compiled"].entries),
                    "entries": [
                        {"id": e.id, "scientific_sha256": e.prepared.scientific_sha256}
                        for e in row["compiled"].entries
                    ],
                }
                for row in self.files
            ],
            "scope": "input checks only; no run directory, learner, Ray or calibration",
        }


def compile_directory(
    directory,
    *,
    require_dependencies=False,
    calibration_seconds=None,
    calibration_level=None,
    calibration_candidate=None,
):
    """Check every direct YAML member, failing on an unsupported or duplicate input."""
    directory = Path(directory).expanduser().resolve()
    if not directory.is_dir():
        raise ConfigurationError(
            f"batch-run requires an Experiment directory: {directory}"
        )
    if calibration_level is not None and calibration_level not in CALIBRATION_SECONDS:
        raise ConfigurationError("calibration level must be off, quick or full")
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
    files, scientific, names = [], set(), set()
    output_root = None
    for index, path in enumerate(paths):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", path.stem):
            raise ConfigurationError(f"unsafe batch Experiment name: {path.name}")
        if path.stem in names:
            raise ConfigurationError(f"duplicate batch Experiment name: {path.stem}")
        names.add(path.stem)
        try:
            compiled = compile_experiment(
                path, require_dependencies=require_dependencies
            )
        except Exception as exc:
            raise ConfigurationError(f"{path.name}: {exc}") from exc
        if output_root is None:
            output_root = Path(compiled.experiment.output.root)
        elif output_root != Path(compiled.experiment.output.root):
            raise ConfigurationError("batch Experiments must share one output.root")
        for entry in compiled.entries:
            if entry.prepared.scientific_sha256 in scientific:
                raise ConfigurationError(
                    f"duplicate scientific entry in batch directory: {path.name}/{entry.id}"
                )
            scientific.add(entry.prepared.scientific_sha256)
        metadata = compiled.experiment.batch
        if metadata and metadata.gate and compiled.experiment.task != "evaluate":
            raise ConfigurationError(f"{path.name}: batch gate requires evaluate")
        files.append(
            {
                "id": path.stem,
                "path": path,
                "compiled": compiled,
                "stage": metadata.stage if metadata else index,
                "parallel_files": metadata.parallel_files if metadata else 1,
                "gate": primitive(metadata.gate)
                if metadata and metadata.gate
                else None,
                "annotated": metadata is not None,
            }
        )
    if any(row["annotated"] for row in files) and not all(
        row["annotated"] for row in files
    ):
        raise ConfigurationError(
            "batch.stage must be present in every Experiment or none"
        )
    for stage in {row["stage"] for row in files}:
        limits = {row["parallel_files"] for row in files if row["stage"] == stage}
        if len(limits) != 1:
            raise ConfigurationError(
                f"batch stage {stage} has conflicting parallel_files"
            )
    learning = [
        row for row in files if row["compiled"].experiment.task == "train-evaluate"
    ]
    if learning:
        stages = {row["stage"] for row in learning}
        if len(stages) != 1:
            raise ConfigurationError(
                "shared calibration requires train-evaluate files in one stage"
            )
        settings = {
            (
                row["compiled"].experiment.execution.mode,
                row["compiled"].experiment.execution.scheduling,
                row["compiled"].experiment.execution.preflight,
                row["compiled"].experiment.execution.preflight_coverage,
            )
            for row in learning
        }
        if len(settings) != 1:
            raise ConfigurationError(
                "learning files must agree on shared tuning settings"
            )
        if any(
            row["compiled"].experiment.task != "train-evaluate"
            for row in files
            if row["stage"] in stages
        ):
            raise ConfigurationError(
                "shared learning stage cannot contain rule/native files"
            )
    settings = learning[0]["compiled"].experiment.execution if learning else None
    level = calibration_level or (settings.calibration_level if settings else "quick")
    candidate = calibration_candidate or (
        settings.calibration_candidate if settings else "latest"
    )
    if (
        learning
        and calibration_level is None
        and any(
            row["compiled"].experiment.execution.calibration_level != level
            for row in learning
        )
    ):
        raise ConfigurationError("learning files must agree on calibration level")
    if (
        learning
        and calibration_candidate is None
        and any(
            row["compiled"].experiment.execution.calibration_candidate != candidate
            for row in learning
        )
    ):
        raise ConfigurationError("learning files must agree on calibration candidate")
    if calibration_seconds is None and learning:
        values = {
            row["compiled"].experiment.execution.calibration_seconds for row in learning
        }
        if len(values) != 1:
            raise ConfigurationError("learning files must agree on calibration timeout")
        calibration_seconds = values.pop()
    if calibration_seconds is None:
        calibration_seconds = CALIBRATION_SECONDS[level]
    if (
        not isinstance(calibration_seconds, (int, float))
        or isinstance(calibration_seconds, bool)
        or not 0 <= calibration_seconds < float("inf")
        or (level == "off" and calibration_seconds != 0)
        or (level != "off" and calibration_seconds == 0)
    ):
        raise ConfigurationError("batch calibration budget must be zero only when off")
    files.sort(key=lambda row: (row["stage"], row["id"]))
    return DirectoryPlan(
        directory,
        tuple(files),
        output_root,
        float(calibration_seconds),
        level,
        candidate,
    )


def _tune_inputs(directory_plan, *, output_root, parent_digest):
    from smartsom.experiments.tuning_batch import BatchInputs

    learning = [
        r
        for r in directory_plan.files
        if r["compiled"].experiment.task == "train-evaluate"
    ]
    if not learning:
        return None
    setting = learning[0]["compiled"].experiment.execution
    entries = tuple(
        {
            "experiment_id": f"{row['id']}__{entry.id}",
            "file_id": row["id"],
            "prepared": asdict(entry.prepared),
            "control_spec": {},
            "baseline_concurrency": row["compiled"].experiment.execution.max_concurrent,
        }
        for row in learning
        for entry in row["compiled"].entries
    )
    return BatchInputs(
        entries,
        mode=setting.mode,
        execution=setting.scheduling,
        active_limit=directory_plan.calibration_seconds,
        calibration_level=directory_plan.calibration_level,
        calibration_candidate=directory_plan.calibration_candidate,
        output_root=str(output_root),
        provenance={"kind": "author-batch", "parent_plan_sha256": parent_digest},
        preflight=setting.preflight,
        preflight_coverage=setting.preflight_coverage,
    )


def allocate(directory_plan):
    """Freeze the whole batch and children before starting any control or learner."""
    from smartsom.experiments.author_driver import allocate as allocate_author
    from smartsom.experiments.tuning_batch import allocate_batch, preflight

    root = directory_plan.output_root / (
        datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        + "-batch-"
        + directory_plan.directory.name
        + "-"
        + uuid4().hex[:10]
    )
    # Check framework dependencies and all learning entries before allocating.
    tentative = _tune_inputs(
        directory_plan, output_root=root / "performance", parent_digest="preflight"
    )
    if tentative:
        preflight(tentative)
    root.mkdir(parents=True, exist_ok=False)
    saved_files = []
    for row in directory_plan.files:
        child = root / "experiments" / row["id"]
        allocate_author(row["compiled"], root=child)
        child_plan = json.loads((child / "plan.json").read_text())
        saved_files.append(
            {
                "id": row["id"],
                "path": str(row["path"]),
                "source_sha256": json.loads(
                    row["compiled"].entries[0].prepared.training_inputs_json
                )["authoring"]["input_sha256"][str(row["path"])],
                "child": str(child.relative_to(root)),
                "child_plan_sha256": digest(child_plan),
                "entry_ids": [e.id for e in row["compiled"].entries],
                "scientific_sha256": [
                    e.prepared.scientific_sha256 for e in row["compiled"].entries
                ],
                "task": row["compiled"].experiment.task,
                "stage": row["stage"],
                "parallel_files": row["parallel_files"],
                "gate": row["gate"],
            }
        )
    plan = {
        "schema": SCHEMA,
        "directory": str(directory_plan.directory),
        "files": saved_files,
        "calibration_seconds": directory_plan.calibration_seconds,
        "calibration_level": directory_plan.calibration_level,
        "calibration_candidate": directory_plan.calibration_candidate,
        "source": source_identity(),
        "implementation_sha256": implementation_identity(),
    }
    write_json(root / "plan.json", plan)
    tune_directory = None
    if tentative:
        inputs = _tune_inputs(
            directory_plan, output_root=root / "performance", parent_digest=digest(plan)
        )
        tune_directory, _, _ = allocate_batch(inputs)
        tune_directory = str(tune_directory.relative_to(root))
    state = {
        "schema": STATE_SCHEMA,
        "plan_sha256": digest(plan),
        "status": "prepared",
        "stage": "prepared",
        "files": {row["id"]: {"status": "queued"} for row in saved_files},
        "tune_directory": tune_directory,
        "calibration_status": (
            "skipped"
            if directory_plan.calibration_level == "off"
            else "queued"
            if tune_directory
            else "not_applicable"
        ),
        "smoke_status": "queued",
    }
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


def load(root):
    from smartsom.experiments.author_driver import load as load_author
    from smartsom.experiments.tuning_batch import load_run

    root = Path(root).expanduser().resolve()
    plan = json.loads((root / "plan.json").read_text())
    state = json.loads((root / "batch.json").read_text())
    if (
        plan.get("schema") != SCHEMA
        or state.get("schema") != STATE_SCHEMA
        or digest(plan) != state.get("plan_sha256")
    ):
        raise ValueError("frozen V4 directory batch changed")
    if (
        not runtime_source_matches(plan["source"], source_identity())
        or implementation_identity() != plan["implementation_sha256"]
    ):
        raise ValueError("batch source/dependency identity changed; create a new run")
    for row in plan["files"]:
        child = (root / row["child"]).resolve()
        if not child.is_relative_to(root / "experiments"):
            raise ValueError("batch child escapes its run")
        _, child_plan, _ = load_author(child)
        if digest(child_plan) != row["child_plan_sha256"]:
            raise ValueError("frozen child plan changed")
        if [e["scientific_sha256"] for e in child_plan["entries"]] != row[
            "scientific_sha256"
        ]:
            raise ValueError("batch scientific entry identity changed")
    if state.get("tune_directory"):
        child = (root / state["tune_directory"]).resolve()
        if not child.is_relative_to(root / "performance"):
            raise ValueError("batch Tune directory escapes its run")
        _, tune_plan, _ = load_run(child)
        if tune_plan.get("provenance", {}).get("parent_plan_sha256") != digest(plan):
            raise ValueError("Tune plan belongs to a different directory batch")
        expected = {
            f"{row['id']}__{entry}"
            for row in plan["files"]
            if row["task"] == "train-evaluate"
            for entry in row["entry_ids"]
        }
        if {e["experiment_id"] for e in tune_plan["entries"]} != expected:
            raise ValueError("shared Tune entry identities changed")
    return root, plan, state


def _save(root, state):
    write_json(root / "batch.json", state)
    record = json.loads((root / "run.json").read_text())
    record.update(status=state["status"], stage=state["stage"], files=state["files"])
    write_json(root / "run.json", record)


def _smoke_all(root, plan, state):
    """Exercise each frozen entry before the shared performance clock starts."""
    from smartsom.experiments.composable import prepared_from_run
    from smartsom.experiments.preflight import smoke

    if state.get("smoke_status") == "completed":
        return
    saved = root / "smoke.json"
    evidence = (
        json.loads(saved.read_text())
        if saved.exists()
        else {
            "schema": "smartsom.author-batch-smoke/v1",
            "entries": {},
            "status": "running",
            "started_at": time.time(),
            "tick_limit": 128,
        }
    )
    state.update(stage="smoke", smoke_status="running")
    _save(root, state)
    total = sum(len(row["entry_ids"]) for row in plan["files"])

    def show():
        if CURRENT.get():
            CURRENT.get().configure_preflight(
                {
                    "status": "passed"
                    if evidence["status"] == "completed"
                    else evidence["status"],
                    "level": "full",
                    "coverage": "each",
                    "checks_done": total,
                    "checks_total": total,
                    "smoke_done": len(evidence["entries"]),
                    "smoke_total": total,
                    "smoke_skipped": 0,
                    "current": evidence.get("current"),
                    "entries": evidence["entries"],
                }
            )

    show()
    for row in plan["files"]:
        child = root / row["child"]
        child_plan = json.loads((child / "plan.json").read_text())
        for entry in child_plan["entries"]:
            identity = f"{row['id']}__{entry['id']}"
            if evidence["entries"].get(identity, {}).get("status") == "passed":
                continue
            if requested(root):
                raise KeyboardInterrupt("batch stopped during smoke")
            prepared = prepared_from_run(child / entry["snapshot"])
            outcome = smoke(prepared, cancelled=lambda: requested(root))
            evidence["entries"][identity] = outcome
            evidence["current"] = identity
            write_json(saved, evidence)
            show()
            if outcome["status"] != "passed":
                evidence["status"] = "failed"
                write_json(saved, evidence)
                show()
                raise RuntimeError(f"{identity}: smoke {outcome}")
    evidence.update(status="completed", finished_at=time.time())
    write_json(saved, evidence)
    show()
    state["smoke_status"] = "completed"
    _save(root, state)


@operation("tune")
def _tune_worker(root, *, recommend_only, retry_failed=False):
    from smartsom.experiments.tuning_batch import execute_batch, load_run

    root, plan, state = load_run(root)
    bind(root)
    if CURRENT.get():
        CURRENT.get().kind = "tune"
    return execute_batch(
        root,
        plan,
        state,
        recommend_only=recommend_only,
        retry_failed=retry_failed,
        display=CURRENT.get(),
    )


def _quiet_worker(
    root, parent_root, *, tuning=False, recommend_only=False, retry_failed=False
):
    """Keep child diagnostics in child logs while the parent owns the Rich view."""
    from smartsom.experiments.control import CURRENT as CONTROL
    from smartsom.experiments.control import Scope

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
                if tuning:
                    result = _tune_worker(
                        root,
                        recommend_only=recommend_only,
                        retry_failed=retry_failed,
                        display_options={"progress": "off", "verbose": False},
                    )
                else:
                    from smartsom.experiments.author_driver import execute_saved

                    result = execute_saved(
                        root, display_options={"progress": "off", "verbose": False}
                    )
        status = result.get("status", "completed")
        return result
    except KeyboardInterrupt:
        status = "stopped"
        raise
    finally:
        scope.finish(status)
        CONTROL.reset(token)


def _child_snapshot(root):
    path = root / "logs/progress.json"
    try:
        return json.loads(path.read_text()) if path.is_file() else {}
    except (OSError, ValueError):
        return {}


def _last_error_line(message):
    lines = [line.strip() for line in str(message or "").splitlines() if line.strip()]
    return lines[-1][:240] if lines else ""


def _tune_failure(state):
    for identity, row in state.get("entries", {}).items():
        if row.get("status") == "failed":
            failure = row.get("failure") or {}
            message = _last_error_line(failure.get("message"))
            if message:
                return f"{identity}: {message}"
    return _last_error_line((state.get("failure") or {}).get("message"))


def _publish(root, plan, state):
    session = CURRENT.get()
    if session is None:
        return
    session.phase(state["stage"])
    tune = root / state["tune_directory"] if state.get("tune_directory") else None
    tune_snapshot = _child_snapshot(tune) if tune else {}
    by_id = {row["id"]: row for row in tune_snapshot.get("tasks", ())}
    tune_state_path = tune / "batch.json" if tune else None
    try:
        tune_entries = (
            json.loads(tune_state_path.read_text()).get("entries", {})
            if tune_state_path
            else {}
        )
    except (OSError, ValueError):
        tune_entries = {}
    with session.batch_updates():
        for row in plan["files"]:
            key = row["id"]
            status = state["files"][key]["status"]
            if row["task"] == "train-evaluate":
                completed = sum(
                    tune_entries.get(f"{key}__{entry}", {}).get("status") == "completed"
                    for entry in row["entry_ids"]
                )
            else:
                child = root / row["child"] / "batch.json"
                try:
                    child_entries = json.loads(child.read_text()).get("entries", {})
                except (OSError, ValueError):
                    child_entries = {}
                completed = sum(
                    child_entries.get(entry, {}).get("status") == "completed"
                    for entry in row["entry_ids"]
                )
            values = {}
            if row["task"] == "evaluate":
                finished = requested_cases = 0
                active_case = None
                for entry in row["entry_ids"]:
                    entry_root = root / row["child"] / "entries" / entry
                    snapshot = _child_snapshot(entry_root)
                    for task in snapshot.get("tasks", ()):
                        metrics = task.get("values", {})
                        if "evaluation_requested" not in metrics:
                            continue
                        finished += metrics.get("evaluation_finished") or 0
                        requested_cases += metrics.get("evaluation_requested") or 0
                        if metrics.get("evaluation_case_active"):
                            active_case = metrics
                values = {
                    "workflow": {"mode": "evaluation"},
                    "evaluation_finished": finished,
                    "evaluation_requested": requested_cases,
                    "evaluation_case_active": active_case is not None,
                    "evaluation_tick": (active_case or {}).get("evaluation_tick", 0),
                    "evaluation_tick_limit": (active_case or {}).get(
                        "evaluation_tick_limit", 0
                    ),
                }
            session.update(
                key,
                {
                    "display_name": key,
                    "status": status,
                    "stage": status,
                    "context": f"stage {row['stage']} · {row['task']}",
                    "completed": completed,
                    **values,
                },
                total=len(row["entry_ids"]),
                unit="entries",
                final=status in FINAL,
            )
            if row["task"] == "train-evaluate":
                for entry in row["entry_ids"]:
                    identity = f"{key}__{entry}"
                    child = by_id.get(identity, {})
                    tune_entry = tune_entries.get(identity, {})
                    entry_status = tune_entry.get("status", child.get("status", status))
                    failure = tune_entry.get("failure") or {}
                    reason = _last_error_line(failure.get("message"))
                    event = {
                        "status": entry_status,
                        "stage": child.get("stage", entry_status),
                        "display_name": child.get("name", identity),
                        **child.get("values", {}),
                    }
                    if reason:
                        event["reason"] = reason
                    session.update(
                        identity,
                        event,
                        total=child.get("total"),
                        unit="physical ticks",
                        final=entry_status in FINAL,
                    )
    if tune:
        learning_active = state["stage"] == "calibration" or any(
            state["stage"] == f"stage_{row['stage']}"
            and row["task"] == "train-evaluate"
            for row in plan["files"]
        )
        session.configure_tuning(
            {
                **tune_snapshot.get("tuning", {}),
                "stage": state["stage"],
                "calibration_status": state.get("calibration_status", "queued"),
                "calibration": {
                    **(tune_snapshot.get("tuning", {}).get("calibration") or {}),
                    "level": plan.get("calibration_level", "quick"),
                },
                "batch_training_active": learning_active,
            }
        )


def _evaluation_rows(child, entry_ids):
    rows = []
    for entry_id in entry_ids:
        ledger = json.loads((child / "entries" / entry_id / "stages.json").read_text())
        stage = ledger["stages"]["evaluation"]
        if stage["status"] != "completed":
            raise RuntimeError(f"{entry_id}: evaluation was not completed")
        run_dir = Path(stage["run_dir"]).resolve()
        if not run_dir.is_relative_to(child / "entries" / entry_id):
            raise RuntimeError("evaluation artifact escapes its frozen child")
        rows.extend(json.loads((run_dir / "run.json").read_text()).get("results", []))
    return rows


def _gate(root, row):
    child = root / row["child"]
    cases = _evaluation_rows(child, row["entry_ids"])
    if any(case.get("engineering_failure") for case in cases):
        raise RuntimeError(f"{row['id']}: engineering failure in rule evaluation")
    gate = row["gate"]
    if gate is None:
        return {"cases": len(cases), "passed": True}
    if len(cases) < gate["min_cases"]:
        raise RuntimeError(
            f"{row['id']}: {len(cases)} cases, expected at least {gate['min_cases']}"
        )
    if any(case.get("delivered", 0) < gate["min_deliveries_each"] for case in cases):
        raise RuntimeError(f"{row['id']}: qualified delivery gate failed")
    if gate["require_first_pickup"] and any(
        case.get("dispatcher_diagnostics", {}).get("first_pickup_tick") is None
        for case in cases
    ):
        raise RuntimeError(f"{row['id']}: first-pickup gate failed")
    return {"cases": len(cases), "passed": True}


def _request_owned_stop(root):
    from smartsom.experiments.control import read
    from smartsom.experiments.control import write_json as control_write

    owner = read(root)
    if owner and owner.get("status") in {"running", "stop_requested", "stopping"}:
        control_write(root / "control/stop.json", {"id": owner["id"]})


def _run_children(root, plan, state, rows, *, parallel_files):
    context = multiprocessing.get_context("spawn")
    pending = list(rows)
    active = {}
    stopping = False
    failure = None
    try:
        while pending or active:
            if requested(root):
                stopping = True
            while pending and len(active) < parallel_files and not stopping:
                row = pending.pop(0)
                process = context.Process(
                    target=_quiet_worker, args=(str(root / row["child"]), str(root))
                )
                process.start()
                active[row["id"]] = (process, row)
                state["files"][row["id"]]["status"] = "running"
                _save(root, state)
            for key, (process, row) in list(active.items()):
                if process.is_alive():
                    continue
                process.join()
                child_state = json.loads(
                    (root / row["child"] / "batch.json").read_text()
                )
                status = child_state["status"]
                if process.exitcode and status == "completed":
                    status = "failed"
                state["files"][key]["status"] = status
                state["files"][key]["exit_code"] = process.exitcode
                del active[key]
                if status != "completed":
                    stopping = True
                    if not requested(root):
                        failure = (
                            f"{key}: child ended {status} (exit {process.exitcode})"
                        )
                        _request_owned_stop(root)
            _save(root, state)
            _publish(root, plan, state)
            if not active and (stopping or not pending):
                break
            time.sleep(0.25)
    except BaseException:
        _request_owned_stop(root)
        for process, _ in active.values():
            process.join()
        for key, (process, row) in active.items():
            child = root / row["child"] / "batch.json"
            state["files"][key].update(
                status=json.loads(child.read_text())["status"]
                if child.is_file()
                else "failed",
                exit_code=process.exitcode,
            )
        _save(root, state)
        raise
    if failure:
        raise RuntimeError(failure)
    return not stopping and all(
        state["files"][row["id"]]["status"] == "completed" for row in rows
    )


def _run_tune(root, plan, state, *, recommend_only, retry_failed=False):
    context = multiprocessing.get_context("spawn")
    tune_root = root / state["tune_directory"]
    process = context.Process(
        target=_quiet_worker,
        args=(str(tune_root), str(root)),
        kwargs={
            "tuning": True,
            "recommend_only": recommend_only,
            "retry_failed": retry_failed,
        },
    )
    process.start()
    try:
        while process.is_alive():
            _publish(root, plan, state)
            time.sleep(0.5)
        process.join()
    except BaseException:
        _request_owned_stop(root)
        process.join()
        raise
    tune_state = json.loads((tune_root / "batch.json").read_text())
    expected = "recommended" if recommend_only else "completed"
    if requested(root):
        from smartsom.experiments.control import StopRequested

        raise StopRequested("parent batch stopped during shared Tune execution")
    if process.exitcode or tune_state["status"] != expected:
        failure = _tune_failure(tune_state)
        raise RuntimeError(
            f"shared Tune {'calibration' if recommend_only else 'training'} "
            f"ended {tune_state['status']} (exit {process.exitcode})"
            + (f": {failure}" if failure else "")
        )
    return tune_state


@operation("batch-directory")
def execute_saved(root, *, retry_failed=False):
    from smartsom.experiments.batch import exclusive_lock

    root, plan, state = load(root)
    bind(root)
    session = CURRENT.get()
    if session:
        session.total_tasks = len(plan["files"]) + sum(
            len(row["entry_ids"])
            for row in plan["files"]
            if row["task"] == "train-evaluate"
        )
    with exclusive_lock(root / "driver.lock"):
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
            _smoke_all(root, plan, state)
            if state.get("tune_directory") and state["calibration_status"] not in {
                "completed",
                "skipped",
            }:
                state.update(status="running", stage="calibration")
                _save(root, state)
                _publish(root, plan, state)
                _run_tune(root, plan, state, recommend_only=True)
                state["calibration_status"] = "completed"
                _save(root, state)
            stages = sorted({row["stage"] for row in plan["files"]})
            for stage in stages:
                rows = [row for row in plan["files"] if row["stage"] == stage]
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
                if rows[0]["task"] == "train-evaluate":
                    for row in rows:
                        state["files"][row["id"]]["status"] = "running"
                    _save(root, state)
                    tune_state = _run_tune(
                        root,
                        plan,
                        state,
                        recommend_only=False,
                        retry_failed=retry_failed,
                    )
                    for row in rows:
                        statuses = [
                            tune_state["entries"][f"{row['id']}__{entry}"]["status"]
                            for entry in row["entry_ids"]
                        ]
                        state["files"][row["id"]]["status"] = (
                            "completed"
                            if all(s == "completed" for s in statuses)
                            else "failed"
                        )
                    _save(root, state)
                else:
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
                        if row["task"] == "evaluate":
                            state["files"][row["id"]]["gate"] = _gate(root, row)
                            _save(root, state)
            else:
                state.update(status="completed", stage="completed")
        except KeyboardInterrupt:
            state.update(status="stopped", stage="stopped")
        except BaseException as exc:
            state.update(
                status="failed", stage="failed", error=f"{type(exc).__name__}: {exc}"
            )
        _save(root, state)
        _publish(root, plan, state)
        return {
            "directory": str(root),
            "status": state["status"],
            "files": state["files"],
            "error": state.get("error"),
        }


def run(directory_plan, *, background=False, display_options=None):
    root = allocate(directory_plan)
    if background:
        from smartsom.experiments.background import launch

        return launch(root, display_options=display_options)
    return execute_saved(root, display_options=display_options)
