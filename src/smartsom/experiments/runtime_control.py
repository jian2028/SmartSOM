"""Public single-node execution control, independent of frozen scientific inputs.

Adapters submit commands; the scheduler applies intents and publishes observations.
All writers use one local filesystem lock. Revision is a submission sequence, not
a promise that a worker has reached a checkpoint. No framework SDK is imported.
"""

import copy
import hashlib
import json
import math
import time
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

from smartsom.experiments.evidence import write_json
from smartsom.experiments.locking import acquire, release

SCHEMA = "smartsom.runtime-control/v1"
TERMINAL = {
    "completed",
    "failed",
    "stopped",
    "removed",
    "partial_stopped",
    "partial_removed",
}


def positive(value):
    if type(value) is not int or not 1 <= value <= 1024:
        raise ValueError("concurrency must be an integer between 1 and 1024")
    return value


@contextmanager
def transaction(root):
    directory = Path(root).expanduser().resolve() / "control"
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "runtime.lock").open("a+") as stream:
        while True:
            try:
                acquire(stream)
                break
            except BlockingIOError:
                time.sleep(0.01)
        try:
            path = directory / "runtime.json"
            state = json.loads(path.read_text()) if path.exists() else {}
            yield state
            if state:
                write_json(path, state)
        finally:
            release(stream)


def snapshot(root):
    """Read an atomic observation without creating files or acknowledging commands."""
    path = Path(root).expanduser().resolve() / "control/runtime.json"
    return json.loads(path.read_text()) if path.exists() else {}


def initialize(root, limit, *, mode="manual", keep_open=False):
    positive(limit)
    if mode not in {"manual", "adaptive"}:
        raise ValueError("mode must be manual or adaptive")
    with transaction(root) as state:
        if not state:
            state.update(
                schema=SCHEMA,
                revision=0,
                applied_revision=0,
                limit=limit,
                mode=mode,
                keep_open=bool(keep_open),
                accepting=True,
                paused=False,
                paused_tasks=[],
                stopped_tasks=[],
                removed_tasks=[],
                stopping=False,
                status="prepared",
                active=0,
                tasks={},
                commands={},
                additions={},
                blocking_reason=None,
                budget={},
            )
        return copy.deepcopy(state)


def _removal_target(root, state, task_id):
    tasks = {**state["tasks"], **state["additions"]}
    if task_id not in tasks:
        raise ValueError("unknown task ID")
    task = tasks[task_id]
    if matches(task_id, state.get("removed_tasks", [])):
        raise ValueError("task is already removed or being removed")
    if task.get("status") in TERMINAL:
        raise ValueError("task is no longer active; refresh the selection")
    batch_id = Path(root).expanduser().resolve().name
    short_id = hashlib.sha256(task_id.encode()).hexdigest()[:10]
    # Status and revision bind confirmation to a stable object/intent snapshot;
    # ongoing progress counter updates alone do not invalidate the selection.
    binding = [
        str(Path(root).resolve()),
        task_id,
        state["revision"],
        task.get("status"),
    ]
    token = hashlib.sha256(json.dumps(binding).encode()).hexdigest()
    return dict(
        batch_id=batch_id,
        task_id=task_id,
        short_id=short_id,
        revision=state["revision"],
        confirmation=token,
        task=copy.deepcopy(task),
    )


def removal_target(root, task_id):
    """Read a target-bound challenge; this does not submit or pause anything."""
    state = snapshot(root)
    if not state:
        raise ValueError("runtime control unavailable")
    return _removal_target(root, state, task_id)


def submit(
    root,
    kind,
    *,
    value=None,
    task_id=None,
    command_id=None,
    expected_revision=None,
    confirmation=None,
):
    """Atomically receive an idempotent command; no controller launches workers."""
    if kind not in {
        "limit",
        "delta",
        "mode",
        "pause",
        "resume",
        "stop",
        "close",
        "remove",
        "budget",
    }:
        raise ValueError("unknown runtime command")
    if kind == "limit":
        positive(value)
    if kind == "delta" and (type(value) is not int or value == 0):
        raise ValueError("delta must be a nonzero integer")
    if kind == "mode" and value not in {"manual", "adaptive"}:
        raise ValueError("mode must be manual or adaptive")
    if kind == "budget":
        if not isinstance(value, dict) or not value or set(value) - {"cpus", "memory"}:
            raise ValueError("budget requires cpus and/or memory")
        for key, amount in value.items():
            if (
                isinstance(amount, bool)
                or not isinstance(amount, (int, float))
                or not math.isfinite(amount)
                or amount <= 0
            ):
                raise ValueError("budget values must be positive finite numbers")
            if key == "memory" and type(amount) is not int:
                raise ValueError("memory budget must be integer bytes")
    if task_id is not None and (not isinstance(task_id, str) or not task_id):
        raise ValueError("task ID must be nonempty")
    identity = command_id or uuid4().hex
    payload = dict(kind=kind, value=value, task_id=task_id)
    if kind == "remove":
        payload["confirmation"] = confirmation
    with transaction(root) as state:
        if not state:
            raise ValueError("runtime control is unavailable for this batch")
        prior = state["commands"].get(identity)
        if prior:
            if prior["payload"] != payload:
                raise ValueError("command ID reused with different payload")
            return copy.deepcopy(prior)
        if expected_revision is not None and expected_revision != state["revision"]:
            raise ValueError("runtime revision changed; refresh before retrying")
        if kind == "remove":
            target = _removal_target(root, state, task_id)
            if (
                confirmation != target["confirmation"]
                or expected_revision != target["revision"]
            ):
                raise ValueError(
                    "removal requires fresh target-bound confirmation and revision"
                )
        state["revision"] += 1
        receipt = dict(
            id=identity,
            command_id=identity,
            revision=state["revision"],
            status="received",
            payload=payload,
            received_at=time.time(),
        )
        if state["status"] in TERMINAL:
            receipt.update(
                status="rejected",
                reason="batch has ended; explicitly resume its driver",
            )
        state["commands"][identity] = receipt
        return copy.deepcopy(receipt)


def matches(task, identities):
    # A separator in a user-authored filename must not turn it into a child
    # of another task. Group ownership is carried by drivers, never guessed.
    return task in identities


def _apply(state):
    for receipt in sorted(state["commands"].values(), key=lambda row: row["revision"]):
        if receipt["status"] != "received":
            continue
        command = receipt["payload"]
        kind, task = command["kind"], command["task_id"]
        try:
            if kind in {"limit", "delta"}:
                state["limit"] = positive(
                    command["value"]
                    if kind == "limit"
                    else state["limit"] + command["value"]
                )
            elif kind == "mode":
                state["mode"] = command["value"]
            elif kind == "budget":
                state.setdefault("budget", {}).update(command["value"])
            elif kind == "close":
                state.update(accepting=False, keep_open=False)
            elif task is not None:
                known = set(state["tasks"]) | set(state["additions"])
                if not any(matches(key, [task]) for key in known):
                    raise ValueError("unknown task ID")
                paused, stopped = (
                    set(state["paused_tasks"]),
                    set(state["stopped_tasks"]),
                )
                if matches(task, state.get("removed_tasks", [])):
                    raise ValueError(
                        "removed tasks cannot be resumed; submit a new independent task"
                    )
                if state["tasks"].get(task, {}).get("status") in {
                    "removed",
                    "removing",
                }:
                    raise ValueError("removed tasks cannot be resumed")
                if kind == "pause":
                    paused.add(task)
                elif kind == "stop":
                    stopped.add(task)
                    paused.discard(task)
                elif kind == "resume":
                    paused.discard(task)
                    stopped.discard(task)
                elif kind == "remove":
                    state.setdefault("removed_tasks", []).append(task)
                    paused.discard(task)
                    stopped.discard(task)
                state.update(paused_tasks=sorted(paused), stopped_tasks=sorted(stopped))
            elif kind == "pause":
                state["paused"] = True
            elif kind == "resume":
                state["paused"] = False
            elif kind == "stop":
                state.update(stopping=True, accepting=False)
            receipt.update(
                status="applied",
                applied_at=time.time(),
                result="intent applied; worker transition may still be pending",
            )
        except ValueError as exc:
            receipt.update(status="rejected", reason=str(exc))
        state["applied_revision"] = receipt["revision"]
    return state


def apply(root):
    with transaction(root) as state:
        if state:
            _apply(state)
        return copy.deepcopy(state)


def publish(root, *, active, status=None, blocking_reason=None, tasks=None):
    with transaction(root) as state:
        if not state:
            return {}
        state.update(
            active=active,
            blocking_reason=blocking_reason,
            updated_at=time.time(),
            draining=active > state["limit"],
        )
        if tasks:
            state["tasks"].update(copy.deepcopy(tasks))
        state["status"] = status or (
            "stopping"
            if state["stopping"]
            else "pausing"
            if state["paused"] and active
            else "paused"
            if state["paused"]
            else "draining"
            if active > state["limit"]
            else "running"
        )
        return copy.deepcopy(state)


def restart(root):
    """Explicit driver resume preserves limits/mode and manual task holds."""
    with transaction(root) as state:
        if state:
            state.update(stopping=False, accepting=True, status="prepared", active=0)


def finish_if_drained(root, *, unfinished=False, failed=False):
    """Serialize end-of-queue with submissions so an accepted append cannot vanish."""
    with transaction(root) as state:
        _apply(state)
        pending = any(
            job.get("status", "queued") not in TERMINAL
            for job in state["additions"].values()
        )
        if (
            unfinished
            or pending
            or state["active"]
            or (state["paused"] and not state["stopping"])
        ):
            return False
        if state["keep_open"] and not state["stopping"]:
            return False
        state.update(
            status="stopped"
            if state["stopping"]
            else "failed"
            if failed
            else "partial_removed"
            if state.get("removed_tasks")
            else "partial_stopped"
            if state["stopped_tasks"]
            else "completed",
            accepting=False,
        )
        return True


def append(
    root, directory, *, ids=None, tags=None, command_id=None, expected_revision=None
):
    """Freeze additional independent tasks; never rewrite the parent's manifest."""
    from dataclasses import replace

    from smartsom.config.experiment_v4 import compile_experiment
    from smartsom.experiments.author_batch import compile_directory
    from smartsom.experiments.author_driver import allocate

    root = Path(root).expanduser().resolve()
    identity = command_id or uuid4().hex
    payload = dict(
        kind="add",
        directory=str(Path(directory).expanduser().resolve()),
        ids=ids,
        tags=tags,
        task_id=None,
    )
    prior = snapshot(root).get("commands", {}).get(identity)
    if prior:
        if prior["payload"] != payload:
            raise ValueError("command ID reused with different payload")
        return prior
    compiled = compile_directory(
        directory, require_dependencies=True, ids=ids, tags=tags
    )
    if any(
        row.get("requires")
        or row.get("gate")
        or (row["compiled"].experiment.batch is not None and row["stage"] != 0)
        for row in compiled.files
    ):
        raise ValueError("appended tasks must be independent stage-0 experiments")
    with transaction(root) as state:
        prior = state.get("commands", {}).get(identity)
        if prior:
            if prior["payload"] != payload:
                raise ValueError("command ID reused with different payload")
            return copy.deepcopy(prior)
        if not state or not state["accepting"] or state["status"] in TERMINAL:
            raise ValueError(
                "batch is closed; explicitly resume or create another batch"
            )
        if expected_revision is not None and expected_revision != state["revision"]:
            raise ValueError("runtime revision changed; refresh before retrying")
        known = set(state["tasks"]) | set(state["additions"])
        known.update(
            job["file_id"] for job in state["additions"].values() if job.get("file_id")
        )
        for receipt in state["commands"].values():
            if receipt["payload"]["kind"] == "add":
                known.update(receipt.get("tasks", []))
        frozen_path = root / "plan.json"
        if frozen_path.exists():
            frozen = json.loads(frozen_path.read_text())
            known.update(
                row["experiment_id"]
                for row in frozen.get("entries", [])
                if row.get("experiment_id")
            )
            for row in frozen.get("files", []):
                known.add(row["id"])
                known.update(row["id"] + "__" + entry for entry in row["entry_ids"])
        names = [row["id"] for row in compiled.files]
        expanded = {
            row["id"] + "__" + entry.id
            for row in compiled.files
            for entry in row["compiled"].entries
        }
        if len(names) != len(set(names)) or known.intersection(set(names) | expanded):
            raise ValueError("duplicate task ID")
        jobs = {}
        prepared = []
        for row in compiled.files:
            native = compile_experiment(
                row["path"],
                require_dependencies=True,
                sets=[
                    "experiment.execution.executor=native",
                    'experiment.execution.tuning="off"',
                    'experiment.execution.calibration_level="off"',
                    "experiment.execution.scheduling=fixed",
                    "experiment.execution.max_concurrent=1",
                ],
            )
            if [e.prepared.scientific_sha256 for e in native.entries] != [
                e.prepared.scientific_sha256 for e in row["compiled"].entries
            ]:
                raise ValueError("execution override changed scientific identity")
            prepared.append((row, native))
        for row, native in prepared:
            for entry in native.entries:
                key = (
                    row["id"]
                    if len(native.entries) == 1
                    else row["id"] + "__" + entry.id
                )
                child = allocate(
                    replace(native, entries=(entry,)), root=root / "added" / key
                )
                jobs[key] = dict(
                    id=key, file_id=row["id"], root=str(child), status="queued"
                )
        state["additions"].update(jobs)
        state["revision"] += 1
        receipt = dict(
            id=identity,
            command_id=identity,
            payload=payload,
            received_at=time.time(),
            status="received",
            revision=state["revision"],
            tasks=names,
            directory=str(root),
        )
        state["commands"][identity] = receipt
        return copy.deepcopy(receipt)
