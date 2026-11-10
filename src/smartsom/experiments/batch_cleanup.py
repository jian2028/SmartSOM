"""Preview and permanently delete artifacts of one removed native task."""

import hashlib
import json
import os
import shutil
import stat
from contextlib import ExitStack, contextmanager
from pathlib import Path
from uuid import uuid4

import yaml

from smartsom.config.codec import digest
from smartsom.experiments import control, runtime_control
from smartsom.experiments.batch_inventory import read_view
from smartsom.experiments.locking import exclusive_lock

ARTIFACTS = (
    "checkpoints",
    "continuation",
    "evaluation",
    "validation",
    "test",
    "periodic_evaluation",
    "recordings",
    "logs",
    "tracking",
    "episodes.jsonl",
    "cases.jsonl",
    "cases.jsonl.tmp",
    "case-commits",
    "episodes.jsonl.tmp",
    "last.pt.tmp",
    "interrupted-case.json",
    "research-summary.json",
    "calibration.json",
    "last.pt",
    "last.pt.json",
    "best.pt",
    "best.pt.json",
    "summary.json",
)


def _no_links(path, boundary):
    """Bind all nested metadata; refuse linked, mounted or special artifacts."""
    base_dev = boundary.stat().st_dev
    entries = [path]
    metadata, total = [], 0
    while entries:
        current = entries.pop()
        info = current.lstat()
        if stat.S_ISLNK(info.st_mode) or info.st_dev != base_dev:
            raise ValueError(
                f"linked or mounted artifact is not safely isolated: {current}"
            )
        if stat.S_ISREG(info.st_mode) and info.st_nlink != 1:
            raise ValueError(f"hard-linked artifact is not safely isolated: {current}")
        if not (stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode)):
            raise ValueError(f"special artifact is not safely isolated: {current}")
        metadata.append(
            (
                str(current.relative_to(path)),
                info.st_ino,
                info.st_dev,
                info.st_mode,
                info.st_size,
                info.st_mtime_ns,
            )
        )
        if stat.S_ISREG(info.st_mode):
            total += info.st_size
        if stat.S_ISDIR(info.st_mode):
            entries.extend(current.iterdir())
    return {
        "tree_digest": digest(sorted(metadata)),
        "entries": len(metadata),
        "bytes": total,
    }


def _target(queue, state, task_id):
    queue = Path(queue).resolve()
    tasks = {**state.get("tasks", {}), **state.get("additions", {})}
    job = tasks.get(task_id)
    if not job or job.get("status") != "removed":
        raise ValueError("exact task must first be confirmed removed by the scheduler")
    if not runtime_control.matches(task_id, state.get("removed_tasks", [])):
        raise ValueError("task removal tombstone is missing")
    owner = Path(job.get("owner_root", queue)).resolve()
    if owner != queue:
        raise ValueError(
            "cross-root task cleanup requires ownership synchronization not yet supported"
        )
    owner_id = job.get("owner_task", task_id)
    plan, _ = read_view(owner)
    matches = [row for row in plan["files"] if row["id"] == owner_id]
    if len(matches) != 1:
        raise ValueError("task does not identify exactly one frozen native child")
    row = matches[0]
    relative = Path(row["child"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("invalid child path")
    child = owner / relative
    for part in [
        owner,
        *[
            owner.joinpath(*relative.parts[:i])
            for i in range(1, len(relative.parts) + 1)
        ],
    ]:
        if part.is_symlink():
            raise ValueError("linked child path is not safely isolated")
    if (
        child.resolve() != Path(job.get("root", "")).resolve()
        or not child.is_relative_to(owner)
        or child == owner
    ):
        raise ValueError("task root differs from frozen plan or escapes owner")
    if (
        digest(json.loads((child / "plan.json").read_text()))
        != row["child_plan_sha256"]
    ):
        raise ValueError("frozen child plan changed")
    for other in plan["files"]:
        if other["id"] == owner_id:
            continue
        other_path = (owner / other["child"]).resolve()
        if (
            other_path == child
            or other_path.is_relative_to(child)
            or child.is_relative_to(other_path)
        ):
            raise ValueError("child output overlaps another task")
    return child, row, job


def _settled(child, job):
    owner = control.read(child)
    if owner is None:
        if job.get("pid") is not None or any(
            (child / name).exists() or (child / name).is_symlink() for name in ARTIFACTS
        ):
            raise ValueError("started task has no verified process ownership")
        return
    table = control.processes()
    if (
        owner.get("root") != str(child)
        or not owner.get("finished_at")
        or owner.get("ownership_finalized") is not True
        or owner.get("ownership_error")
        or owner.get("deletion_blockers")
        or owner.get("session_context") != control.session_context()
    ):
        raise ValueError("task process ownership or exit is not fully verified")
    if any(
        control.alive(item, table)
        for item in [owner["owner"], *owner.get("members", [])]
    ) or control.session_members(owner, table):
        raise ValueError("task or owned subprocess still running")


def _inventory(child):
    result = []
    for name in ARTIFACTS:
        path = child / name
        if not path.exists() and not path.is_symlink():
            continue
        result.append({"name": name, "path": str(path), **_no_links(path, child)})
    return result


def known_references(queue, state, child):
    """Inspect enumerable batch task configurations; not a global reference index."""
    plan, _ = read_view(queue)
    roots = {str((queue / row["child"]).resolve()) for row in plan["files"]}
    plan_ids = {row["id"] for row in plan["files"]}
    roots.update(
        job["root"]
        for job in (
            *state.get("tasks", {}).values(),
            *state.get("additions", {}).values(),
        )
        if job.get("root")
    )
    hits, unknown = [], []
    for identity, job in {
        **state.get("tasks", {}),
        **state.get("additions", {}),
    }.items():
        if identity not in plan_ids and not job.get("root"):
            unknown.append(
                {"task_id": identity, "reason": "registered_task_root_unavailable"}
            )
    for text in sorted(roots):
        peer = Path(text).resolve()
        if peer == child:
            continue
        config = peer / "inputs/experiment.yaml"
        if not config.is_file() or config.is_symlink():
            unknown.append(
                {"task_root": str(peer), "reason": "frozen_configuration_unavailable"}
            )
            continue
        try:

            def mapping(value):
                if value is None:
                    return {}
                if not isinstance(value, dict):
                    raise TypeError("configuration section must be a mapping")
                return value

            raw = yaml.safe_load(config.read_text())
            if raw is None:
                raise TypeError("frozen configuration must not be empty")
            doc = mapping(raw)
            run = mapping(doc.get("run"))
            evaluation = mapping(doc.get("evaluation"))
            recovery = mapping(evaluation.get("recovery"))
            target = mapping(evaluation.get("learning_target"))
            fields = {
                "run.checkpoint": run.get("checkpoint"),
                "run.calibration": run.get("calibration"),
                "evaluation.recovery.calibration": recovery.get("calibration"),
                "evaluation.learning_target.reference": target.get("reference"),
            }
            fields.update(
                {
                    f"case_sets.{key}": value
                    for key, value in mapping(doc.get("case_sets")).items()
                }
            )
            for key, value in fields.items():
                if value is None or value == "":
                    continue
                if not isinstance(value, str):
                    unknown.append(
                        {
                            "task_root": str(peer),
                            "field": key,
                            "reason": "non_string_reference",
                        }
                    )
                    continue
                path = Path(value).expanduser()
                if not path.is_absolute():
                    unknown.append(
                        {
                            "task_root": str(peer),
                            "field": key,
                            "reason": "relative_reference_resolution_unverified",
                        }
                    )
                    continue
                resolved = path.resolve()
                if resolved == child or resolved.is_relative_to(child):
                    hits.append(
                        {"task_root": str(peer), "field": key, "path": str(resolved)}
                    )
        except (OSError, TypeError, ValueError, yaml.YAMLError):
            unknown.append(
                {"task_root": str(peer), "reason": "frozen_configuration_unreadable"}
            )
    return {
        "known": hits,
        "unknown": unknown,
        "scope": "frozen configurations of current batch and registered queue tasks; independent roots and external programs are not enumerated",
    }


def preview(queue, task_id):
    root = Path(queue).expanduser().resolve()
    state = runtime_control.snapshot(root)
    child, row, job = _target(root, state, task_id)
    _settled(child, job)
    items = _inventory(child)
    references = known_references(root, state, child)
    payload = {
        "queue": str(root),
        "task_id": task_id,
        "child": str(child),
        "child_plan_sha256": row["child_plan_sha256"],
        "items": items,
        "references": references,
    }
    token = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return {
        **payload,
        "confirmation": token,
        "operation": "permanent_delete",
        "logical_bytes": sum(item["bytes"] for item in items),
        "warning": "Permanent deletion. External references may stop working; independent batches and programs are not indexed.",
        "preserved": [
            "plan.json",
            "inputs",
            "source",
            "control",
            "batch.json",
            "trial.lock",
        ],
    }


@contextmanager
def _cleanup_lease(root, child, task_id):
    """Protect this task and pause new batch admissions without locking scheduling."""
    token = uuid4().hex
    with ExitStack() as locks:
        with runtime_control.transaction(root, wait=False) as state:
            current = state.get("cleanup_active")
            if current and current.get("task_id") != task_id:
                raise ValueError(
                    "another task cleanup is pending; finish or retry it first"
                )
            # Nonblocking child locks also prove a same-task old lease is abandoned.
            locks.enter_context(
                control.registration_lock(child, parents=False, wait=False)
            )
            locks.enter_context(exclusive_lock(child / "trial.lock"))
            verified, _, _ = _target(root, state, task_id)
            if verified != child:
                raise ValueError("task target changed")
            state["cleanup_active"] = {"task_id": task_id, "token": token}
        try:
            yield
        finally:
            with runtime_control.transaction(root) as state:
                active = state.get("cleanup_active", {})
                if active.get("token") == token:
                    del state["cleanup_active"]


def delete(queue, task_id, confirmation):
    """Delete only the previewed artifacts after official removal and exit.

    A receipt records each completed item. An interruption between deletion and
    recording its completion requires manual inspection; it is never guessed.
    """
    root = Path(queue).expanduser().resolve()
    if len(confirmation) != 64 or any(
        c not in "0123456789abcdef" for c in confirmation
    ):
        raise ValueError("confirmation must be the exact preview token")
    if not shutil.rmtree.avoids_symlink_attacks:
        raise ValueError("safe directory deletion is not supported on this platform")
    child, row, job = _target(root, runtime_control.snapshot(root), task_id)
    # Short queue transactions publish/clear the lease; actual deletion does
    # not hold the scheduler lock. Both batch-add implementations check it.
    with _cleanup_lease(root, child, task_id):
        queue_state = runtime_control.snapshot(root)
        child_again, row, job = _target(root, queue_state, task_id)
        if child_again != child:
            raise ValueError("task target changed")
        _settled(child, job)
        receipt_dir = child / "control" / "task-cleanup"
        if receipt_dir.is_symlink():
            raise ValueError("cleanup receipt directory cannot be a symlink")
        receipt = receipt_dir / (confirmation + ".json")
        if receipt.is_symlink():
            raise ValueError("cleanup receipt cannot be a symlink")
        if receipt.exists():
            saved = json.loads(receipt.read_text())
            binding = {
                k: saved[k]
                for k in (
                    "queue",
                    "task_id",
                    "child",
                    "child_plan_sha256",
                    "items",
                    "references",
                )
            }
            if (
                hashlib.sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest()
                != confirmation
                or saved["queue"] != str(root)
                or saved["task_id"] != task_id
                or saved["child"] != str(child)
                or saved["child_plan_sha256"] != row["child_plan_sha256"]
                or len({i["name"] for i in saved["items"]}) != len(saved["items"])
                or any(
                    i["name"] not in ARTIFACTS or i["path"] != str(child / i["name"])
                    for i in saved["items"]
                )
            ):
                raise ValueError("cleanup receipt identity mismatch")
        else:
            saved = preview(root, task_id)
            if saved["confirmation"] != confirmation:
                raise ValueError("preview changed; inspect and confirm again")
            if saved["references"]["known"] or saved["references"]["unknown"]:
                raise ValueError(
                    "known or unresolved task references require preservation"
                )
            receipt_dir.mkdir(exist_ok=True)
            saved["status"] = "in_progress"
            saved["deleted"] = []
            _write_receipt(receipt, saved)
        refs = known_references(root, queue_state, child)
        if refs["known"] or refs["unknown"]:
            raise ValueError("known or unresolved task references require preservation")
        fd = os.open(child, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        name = None
        try:
            for item in saved["items"]:
                name = item["name"]
                path = child / name
                if name in saved["deleted"]:
                    if path.exists() or path.is_symlink():
                        raise ValueError("deleted artifact reappeared; inspect receipt")
                    continue
                if not path.exists() and not path.is_symlink():
                    raise ValueError(
                        "artifact missing after interruption; inspect receipt"
                    )
                if _no_links(path, child) != {
                    k: item[k] for k in ("tree_digest", "entries", "bytes")
                }:
                    raise ValueError("artifact changed after preview; inspect receipt")
                if path.is_dir():
                    shutil.rmtree(name, dir_fd=fd)
                else:
                    os.unlink(name, dir_fd=fd)
                saved["deleted"].append(name)
                _write_receipt(receipt, saved)
        except BaseException:
            saved["status"] = "partial"
            saved["interrupted_item"] = name
            try:
                _write_receipt(receipt, saved)
            except OSError:
                pass
            raise
        finally:
            os.close(fd)
        saved["status"] = "complete"
        _write_receipt(receipt, saved)
        return saved


def _write_receipt(path, payload):
    import tempfile

    fd, temporary = tempfile.mkstemp(prefix=".cleanup-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(payload, stream, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
