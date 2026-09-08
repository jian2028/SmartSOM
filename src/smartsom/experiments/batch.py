"""Single-host orchestration. Workers exclusively execute the ordinary run_one."""

import csv
import fcntl
import json
import multiprocessing
import os
import signal
import threading
import time
from collections import Counter, deque
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from queue import Empty
from uuid import uuid4

from smartsom.config.codec import ConfigurationError, digest, primitive
from smartsom.config.snapshots import load_resolved_run
from smartsom.config.study import PlanEntry, ResolvedStudy, semantic_run
from smartsom.experiments.evidence import (
    _file_digest,
    artifact_digests,
    source_identity,
    write_json,
)
from smartsom.experiments.runner import RunFailedError, run_one


@dataclass(frozen=True, slots=True)
class BatchResult:
    study_dir: Path
    completed: int
    failed: int
    pending: int
    interrupted: bool


@contextmanager
def exclusive_lock(path: Path):
    """Persistent lock inode; unlinking it would permit concurrent owners."""
    with path.open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"study or child is still active: {path}") from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def execution_identity() -> dict:
    source = source_identity()
    package = Path(__file__).resolve().parents[1]
    files = {
        str(p.relative_to(package)): _file_digest(p)
        for p in sorted(package.rglob("*.py"))
    }
    root = package.parents[1]
    for name in ("pyproject.toml", "uv.lock"):
        if (root / name).exists():
            files[name] = _file_digest(root / name)
    return {
        "commit": source["git"]["commit"],
        "python": source["python"],
        "packages": source["packages"],
        "code_sha256": digest(files),
    }


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _save_plan(study: ResolvedStudy, directory: Path) -> None:
    snapshots = directory / "snapshots"
    snapshots.mkdir()
    entries = []
    for entry in study.entries:
        target = snapshots / f"{entry.entry_id}.json"
        write_json(
            target, {"schema": "smartsom.resolved-run/v1", **primitive(entry.resolved)}
        )
        entries.append(
            {key: value for key, value in primitive(entry).items() if key != "resolved"}
            | {
                "snapshot": str(target.relative_to(directory)),
                "snapshot_sha256": _file_digest(target),
            }
        )
    write_json(
        directory / "plan.json",
        {
            "schema": "smartsom.study-plan/v1",
            "plan_sha256": study.plan_sha256,
            "spec": study.spec,
            "sources": study.sources,
            "entries": entries,
        },
    )
    write_json(
        directory / "manifest.json",
        {
            "schema": "smartsom.study-manifest/v1",
            "status": "prepared",
            "source": source_identity(),
            "execution_identity": execution_identity(),
            "plan_file_sha256": _file_digest(directory / "plan.json"),
        },
    )


def _load_plan(directory: Path) -> list[PlanEntry]:
    manifest = _read(directory / "manifest.json")
    if _file_digest(directory / "plan.json") != manifest["plan_file_sha256"]:
        raise ConfigurationError("saved study plan digest mismatch")
    if execution_identity() != manifest["execution_identity"]:
        raise ConfigurationError(
            "source or dependency identity changed; create a new study"
        )
    plan = _read(directory / "plan.json")
    entries = []
    for row in plan["entries"]:
        target = directory / row["snapshot"]
        if (
            target.resolve().parent != (directory / "snapshots").resolve()
            or _file_digest(target) != row["snapshot_sha256"]
        ):
            raise ConfigurationError("saved child snapshot digest or location mismatch")
        resolved = load_resolved_run(target)
        identity = digest(
            [
                row["case_id"],
                row["replication"],
                row["variant_id"],
                row["algorithm_id"],
                semantic_run(resolved),
            ]
        )
        if identity != row["entry_id"]:
            raise ConfigurationError("saved child scientific identity mismatch")
        entries.append(
            PlanEntry(
                identity,
                row["case_id"],
                row["algorithm_id"],
                row["replication"],
                row["variant_id"],
                tuple(row["disabled"]),
                resolved,
            )
        )
    if (
        len({e.entry_id for e in entries}) != len(entries)
        or digest([e.entry_id for e in entries]) != plan["plan_sha256"]
    ):
        raise ConfigurationError("saved study entries mismatch")
    return entries


def _attempt_status(path: Path, entry: PlanEntry) -> dict:
    if (path / "interrupted.json").exists():
        return {"status": "incomplete", "attempt_dir": str(path), "run_dir": None}
    if (path / "failed.json").exists():
        return {"status": "failed", "attempt_dir": str(path), "run_dir": None}
    assignment = path / "run.json"
    if assignment.exists():
        assigned = path / _read(assignment)["run_dir"]
        if assigned.resolve().parent != (path / "runs").resolve():
            raise ConfigurationError("invalid attempt run assignment")
        runs = [assigned]
    else:
        runs = sorted((path / "runs").glob("*"))
    if not runs:
        return {"status": "incomplete", "attempt_dir": str(path), "run_dir": None}
    if len(runs) != 1:
        raise ConfigurationError(f"attempt has multiple child runs: {path}")
    run_dir = runs[0]
    manifest_path = run_dir / "manifest.json"
    manifest = _read(manifest_path) if manifest_path.exists() else {}
    state = manifest.get("status", "running")
    if state == "completed":
        required = {
            "summary.json",
            "trace.jsonl",
            "resolved_run.yaml",
            "realized_instance.json",
            "metrics.jsonl",
            "progress.log",
        }
        artifacts = artifact_digests(run_dir)
        if not required.issubset(artifacts) or artifacts != manifest.get("artifacts"):
            raise ConfigurationError(
                f"completed run evidence is missing or changed: {run_dir}"
            )
        restored = load_resolved_run(run_dir / "resolved_run.yaml")
        if semantic_run(restored) != semantic_run(entry.resolved):
            raise ConfigurationError(f"completed run inputs differ: {run_dir}")
        summary = _read(run_dir / "summary.json")
        if (
            summary.get("status") != "completed"
            or type(summary.get("makespan")) is not int
        ):
            raise ConfigurationError(f"invalid completed summary: {run_dir}")
    else:
        summary = {}
    return {
        "status": state if state in ("completed", "failed") else "incomplete",
        "attempt_dir": str(path),
        "run_dir": str(run_dir),
        **{
            name: summary.get(name)
            for name in (
                "makespan",
                "passing_rate",
                "passed_jobs",
                "defective_jobs",
                "delivered_jobs",
            )
        },
    }


def _latest(directory: Path, entry: PlanEntry) -> dict:
    attempts = sorted((directory / "children" / entry.entry_id).glob("attempt-*"))
    if not attempts:
        return {"status": "pending", "run_dir": None}
    return _attempt_status(attempts[-1], entry)


def _worker(
    directory: str, entry: PlanEntry, attempt: str, messages, identity: dict
) -> None:
    # Ctrl+C belongs to the coordinator; its second request signals workers explicitly.
    os.setpgrp()
    signal.signal(signal.SIGINT, signal.default_int_handler)
    directory, attempt = Path(directory), Path(attempt)
    try:
        with exclusive_lock(directory / "locks" / f"{entry.entry_id}.lock"):
            if execution_identity() != identity:
                raise RuntimeError("source changed before worker execution")
            resolved = replace(
                entry.resolved,
                run=entry.resolved.run.model_copy(
                    update={"output_root": str(attempt / "runs")}
                ),
            )

            def progress(value):
                assignment = attempt / "run.json"
                if not assignment.exists():
                    write_json(
                        assignment, {"run_dir": str(value.run_dir.relative_to(attempt))}
                    )
                messages.put((entry.entry_id, primitive(value)))

            run_one(resolved, on_progress=progress)
    except BaseException as exc:
        if isinstance(exc, RunFailedError):
            write_json(
                attempt / "run.json", {"run_dir": str(exc.run_dir.relative_to(attempt))}
            )
        write_json(
            attempt / "worker_failure.json",
            {
                "type": type(exc).__name__,
                "message": str(exc),
                "interrupted": isinstance(exc, KeyboardInterrupt)
                or isinstance(exc, RunFailedError)
                and isinstance(exc.cause, KeyboardInterrupt),
            },
        )
    finally:
        messages.put((entry.entry_id, {"stage": "worker_exit"}))


def _report(
    directory: Path, entries: list[PlanEntry], statuses: dict, *, interrupted: bool
) -> BatchResult:
    rows = []
    for entry in entries:
        rows.append(
            {
                "entry_id": entry.entry_id,
                "case_id": entry.case_id,
                "algorithm_id": entry.algorithm_id,
                "replication": entry.replication,
                "variant_id": entry.variant_id,
                **statuses[entry.entry_id],
            }
        )
    counts = Counter(row["status"] for row in rows)
    completed, failed = counts["completed"], counts["failed"]
    result = BatchResult(
        directory, completed, failed, len(entries) - completed - failed, interrupted
    )
    write_json(
        directory / "summary.json",
        {"schema": "smartsom.study-summary/v1", **primitive(result), "runs": rows},
    )
    fields = [
        "entry_id",
        "case_id",
        "algorithm_id",
        "variant_id",
        "replication",
        "status",
        "makespan",
        "passing_rate",
        "run_dir",
        "attempt_dir",
    ]
    temporary = directory / "summary.csv.tmp"
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(directory / "summary.csv")
    (directory / "summary.md").write_text(
        f"# Study execution\n\nCompleted: {completed}; failed: {failed}; pending/incomplete: {result.pending}.\n\nThis is execution status, not integration acceptance. Failed metrics are missing.\n\n| Case | Algorithm | Variant | Replication | Status | Makespan | Passing rate |\n|---|---|---|---:|---|---:|---:|\n"
        + "".join(
            f"| {r['case_id']} | {r['algorithm_id']} | {r['variant_id']} | {r['replication']} | {r['status']} | {r.get('makespan', '')} | {r.get('passing_rate', '')} |\n"
            for r in rows
        ),
        encoding="utf-8",
    )
    manifest = _read(directory / "manifest.json")
    manifest["status"] = (
        "interrupted"
        if interrupted
        else "completed"
        if completed == len(entries)
        else "finished_with_failures"
    )
    write_json(directory / "manifest.json", manifest)
    return result


def run_batch(
    study: ResolvedStudy | None = None,
    *,
    resume: str | Path | None = None,
    workers: int = 2,
    retry_failed: bool = False,
    on_progress=None,
) -> BatchResult:
    if type(workers) is not int or workers < 1:
        raise ValueError("workers must be a positive integer")
    if (study is None) == (resume is None):
        raise ValueError("provide exactly one resolved study or resume directory")
    if retry_failed and resume is None:
        raise ValueError("retry_failed requires resume")
    if threading.current_thread() is not threading.main_thread():
        raise ValueError("run_batch must coordinate from the main thread")
    if resume is None:
        if not isinstance(study, ResolvedStudy):
            raise TypeError("run_batch accepts ResolvedStudy")
        directory = Path(study.spec.output_root) / (
            datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ") + "-" + uuid4().hex
        )
        directory.mkdir(parents=True)
        _save_plan(study, directory)
    else:
        directory = Path(resume).resolve()
    with exclusive_lock(directory / "coordinator.lock"):
        entries = _load_plan(directory)
        (directory / "locks").mkdir(exist_ok=True)
        for entry in entries:
            with exclusive_lock(directory / "locks" / f"{entry.entry_id}.lock"):
                pass
        statuses = {e.entry_id: _latest(directory, e) for e in entries}
        pending = deque(
            e
            for e in entries
            if statuses[e.entry_id]["status"] in ("pending", "incomplete")
            or retry_failed
            and statuses[e.entry_id]["status"] == "failed"
        )
        by_id = {e.entry_id: e for e in entries}
        context = multiprocessing.get_context("spawn")
        messages = context.Queue()
        active = {}
        interrupts = 0
        stop_started = None
        started = time.monotonic()
        identity = execution_identity()

        def interrupt(signum, frame):
            nonlocal interrupts
            interrupts += 1

        previous = signal.signal(signal.SIGINT, interrupt)
        try:
            with (directory / "progress.log").open(
                "a", encoding="utf-8", buffering=1
            ) as log:
                log.write(f"starting workers={workers} total={len(entries)}\n")
                last_heartbeat = 0.0
                while active or pending and not interrupts:
                    while pending and len(active) < workers and not interrupts:
                        entry = pending.popleft()
                        child = directory / "children" / entry.entry_id
                        child.mkdir(parents=True, exist_ok=True)
                        previous_attempts = list(child.glob("attempt-*"))
                        number = 1 + max(
                            (int(p.name.split("-")[1]) for p in previous_attempts),
                            default=0,
                        )
                        attempt = child / f"attempt-{number:08d}-{uuid4().hex}"
                        attempt.mkdir()
                        process = context.Process(
                            target=_worker,
                            args=(
                                str(directory),
                                entry,
                                str(attempt),
                                messages,
                                identity,
                            ),
                        )
                        process.start()
                        active[entry.entry_id] = (process, attempt)
                        statuses[entry.entry_id] = {
                            "status": "running",
                            "run_dir": None,
                            "attempt_dir": str(attempt),
                        }
                        log.write(
                            f"started case={entry.case_id} algorithm={entry.algorithm_id} replication={entry.replication} variant={entry.variant_id}\n"
                        )
                    if interrupts >= 2:
                        if stop_started is None:
                            stop_started = time.monotonic()
                            for process, _ in active.values():
                                if process.is_alive():
                                    os.kill(process.pid, signal.SIGINT)
                        elif time.monotonic() - stop_started > 5:
                            for process, _ in active.values():
                                if process.is_alive():
                                    process.terminate()
                    try:
                        entry_id, progress = messages.get(timeout=0.1)
                        log.write(
                            json.dumps(
                                {"entry_id": entry_id, **progress}, sort_keys=True
                            )
                            + "\n"
                        )
                        if on_progress is not None:
                            on_progress({"entry_id": entry_id, **progress})
                    except Empty:
                        pass
                    for entry_id, (process, attempt) in list(active.items()):
                        if process.is_alive():
                            continue
                        process.join()
                        statuses[entry_id] = _attempt_status(attempt, by_id[entry_id])
                        # A forced interruption is restartable, even if run_one saved a failure.
                        worker_failure = attempt / "worker_failure.json"
                        if (
                            interrupts >= 2
                            or worker_failure.exists()
                            and _read(worker_failure).get("interrupted")
                        ):
                            if statuses[entry_id]["status"] != "completed":
                                write_json(
                                    attempt / "interrupted.json",
                                    {"restart_from_beginning": True},
                                )
                                statuses[entry_id]["status"] = "incomplete"
                        elif (
                            worker_failure.exists()
                            and statuses[entry_id]["status"] == "incomplete"
                        ):
                            statuses[entry_id]["status"] = "failed"
                            write_json(
                                attempt / "failed.json", {"worker_failure": True}
                            )
                        log.write(
                            f"finished entry={entry_id} status={statuses[entry_id]['status']} exitcode={process.exitcode}\n"
                        )
                        process.close()
                        del active[entry_id]
                    now = time.monotonic()
                    if now - last_heartbeat >= 1 or not active:
                        counts = dict(Counter(s["status"] for s in statuses.values()))
                        event = {
                            "stage": "draining" if interrupts else "batch",
                            "elapsed_seconds": now - started,
                            "counts": counts,
                        }
                        log.write(json.dumps(event, sort_keys=True) + "\n")
                        if on_progress is not None:
                            on_progress(event)
                        last_heartbeat = now
                return _report(
                    directory, entries, statuses, interrupted=bool(interrupts)
                )
        finally:
            signal.signal(signal.SIGINT, previous)
            for process, _ in active.values():
                if process.is_alive():
                    process.terminate()
                process.join()
                process.close()
            messages.close()
            messages.join_thread()
