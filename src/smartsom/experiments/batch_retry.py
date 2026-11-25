"""Preview and submit explicit, coordinator-validated task retry requests."""

import hashlib
import json
import random
import subprocess
from pathlib import Path

from smartsom.config.codec import digest
from smartsom.experiments import control, queue_admission, runtime_control


def task_view(state, task_id):
    """Reject transiently inconsistent duplicate queue observations."""
    task = state.get("tasks", {}).get(task_id)
    addition = state.get("additions", {}).get(task_id)
    if (
        task is not None
        and not isinstance(task, dict)
        or addition is not None
        and not isinstance(addition, dict)
    ):
        raise ValueError("task state is malformed")
    if task is not None and addition is not None:
        for field in ("status", "root", "pid", "exit_code", "retry_attempt"):
            if field in task and field in addition and task[field] != addition[field]:
                raise ValueError("task state is changing; retry preview later")
    return addition if addition is not None else task


def _retry_settled(child, job):
    """Require finalized ownership and no recorded live processes before retry.

    A non-isolated foreground driver cannot authorize *deletion*, but that
    limitation alone does not rule out retry when its tracked processes exited.
    """
    owner = control.read(child)
    if not isinstance(owner, dict) or owner.get("root") != str(child):
        raise ValueError("failed task ownership is unavailable")
    if not owner.get("finished_at") or owner.get("ownership_finalized") is not True:
        raise ValueError("failed task ownership was not finalized")
    if (
        owner.get("ownership_error")
        or owner.get("session_context") != control.session_context()
    ):
        raise ValueError("failed task exit cannot be verified on this host")
    blockers = set(owner.get("deletion_blockers", [])) - {
        "run has no isolated driver session"
    }
    if blockers:
        raise ValueError(
            "failed task has unresolved process ownership: "
            + "; ".join(sorted(blockers))
        )
    identity = owner.get("owner", {})
    if (
        not isinstance(identity, dict)
        or not isinstance(identity.get("pid"), int)
        or not identity.get("created")
    ):
        raise ValueError("failed task process identity is malformed")
    if job.get("pid") is not None and identity.get("pid") != job["pid"]:
        raise ValueError("failed task owner differs from the scheduler worker")
    try:
        table = control.processes()
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError("cannot verify failed task process ownership") from exc
    if any(
        control.alive(entry, table) for entry in [identity, *owner.get("members", [])]
    ) or control.session_members(owner, table):
        raise ValueError("failed task or owned subprocess is still running")


def inspect_target(queue, task_id, state, *, allow_paused=False, allow_ended=False):
    """Check a failed or explicitly selected paused task and its continuation.

    Caller must revalidate under the coordinator's own scheduling protocol.
    External owners must still be claimed by this queue; this helper does not
    attempt to claim a detached result root.
    """
    from smartsom.experiments.author_batch import load as load_batch
    from smartsom.experiments.incremental_batch import require_recovery
    from smartsom.experiments.reservation import load as load_reservation

    root = Path(queue).resolve()
    if state.get("status") in runtime_control.TERMINAL and not allow_ended:
        raise ValueError(
            "batch has ended; recover the batch through its official driver"
        )
    if state.get("cleanup_active"):
        raise ValueError("batch cleanup in progress; retry later")
    job = task_view(state, task_id)
    allowed = {"failed", "paused"} if allow_paused else {"failed"}
    if not job or job.get("status") not in allowed:
        raise ValueError(
            "retry requires an exact failed task"
            if not allow_paused
            else "retry requires an exact failed or paused task"
        )
    holds = (
        ("removed_tasks", "stopped_tasks")
        if allow_paused
        else ("removed_tasks", "stopped_tasks", "paused_tasks")
    )
    if state.get("stopping") or any(
        runtime_control.matches(task_id, state.get(name, [])) for name in holds
    ):
        raise ValueError("task has an explicit stop, pause or removal hold")
    if job.get("own_slots"):
        raise ValueError("nested multi-worker task retry is not supported")
    owner = Path(job.get("owner_root", root)).resolve()
    claim = queue_admission.snapshot(owner) if owner != root else None
    if owner != root and (claim.get("queue") != str(root) or not claim.get("owner_id")):
        raise ValueError("external owner is detached; retry requires a verified claim")
    _, plan, _ = load_batch(owner)
    owner_id = job.get("owner_task", task_id)
    matches = [row for row in plan["files"] if row["id"] == owner_id]
    if len(matches) != 1:
        raise ValueError("task does not identify one frozen owner plan entry")
    row = matches[0]
    child = (owner / row["child"]).resolve()
    if not child.is_relative_to(owner) or child == owner:
        raise ValueError("task output escapes its owner root")
    if child != Path(job.get("root", "")).resolve():
        raise ValueError("task root differs from its frozen owner plan")
    if row.get("task") != "train":
        raise ValueError("only native training task retry is supported")
    child_plan = load_reservation(child)
    if digest(child_plan) != row["child_plan_sha256"]:
        raise ValueError("frozen child plan differs")
    from smartsom.config.codec import read_document

    experiment = read_document(child / "inputs/experiment.yaml")
    seed = experiment.get("run", {}).get("seed")
    if type(seed) is not int:
        raise ValueError("frozen training seed is unavailable")
    _retry_settled(child, job)
    pointer_path = child / "continuation/current.json"
    pointer_bytes = pointer_path.read_bytes()
    require_recovery(owner, row)
    if pointer_path.read_bytes() != pointer_bytes:
        raise ValueError("continuation pointer changed during inspection")
    pointer = json.loads(pointer_bytes)
    state_bytes = (
        child / "continuation" / pointer["generation"] / "state.json"
    ).read_bytes()
    if hashlib.sha256(state_bytes).hexdigest() != pointer["files"]["state.json"]:
        raise ValueError("continuation state changed during inspection")
    saved = json.loads(state_bytes)
    from smartsom.experiments.reservation_session import tuples

    try:
        random.Random().setstate(tuples(saved["condition_rng"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("condition RNG state cannot be restored") from exc
    return {
        "task_id": task_id,
        "train_seed": seed,
        "owner_root": str(owner),
        "owner_claim_id": claim.get("owner_id") if claim else None,
        "owner_task": owner_id,
        "child": str(child),
        "child_plan_sha256": row["child_plan_sha256"],
        "generation": pointer["generation"],
        "pointer_sha256": hashlib.sha256(pointer_bytes).hexdigest(),
        "model_compatibility": "not_checked_until_worker_load",
        "episodes": len(saved["rows"]),
        "ticks": saved["ticks"],
    }


def inspect_ended_targets(queue, task_ids):
    """Validate the exact stopped-driver subset before changing execution state."""
    if (
        not isinstance(task_ids, (list, tuple))
        or not task_ids
        or any(not isinstance(key, str) or not key for key in task_ids)
        or len(set(task_ids)) != len(task_ids)
    ):
        raise ValueError("retry requires distinct nonempty exact task IDs")
    root = Path(queue).resolve()
    state = runtime_control.snapshot(root)
    from smartsom.experiments.author_batch import load

    _, plan, batch = load(root)
    if pending_retry_targets(state) - set(task_ids) or any(
        job.get("retry_pending") and key not in task_ids
        for key, job in {**state.get("tasks", {}), **state.get("additions", {})}.items()
    ):
        raise ValueError(
            "pending retry outside selected recovery; settle existing retry intents first"
        )
    rows = {row["id"]: row for row in plan["files"]}
    selected = set((batch.get("selection") or {}).get("tasks", rows))
    result = []
    for key in task_ids:
        if key not in rows or key not in selected:
            raise ValueError(
                f"{key}: retry target is outside the frozen batch selection"
            )
        if batch["files"][key]["status"] != "failed":
            raise ValueError(f"{key}: retry requires a failed task")
        row = rows[key]
        if any(
            batch["files"][peer]["status"] != "completed"
            for peer in row.get("requires", [])
        ):
            raise ValueError(f"{key}: retry dependencies have not completed")
        result.append(inspect_target(root, key, state, allow_ended=True))
    return result


def check_driver_recovery(observed, *, enabled):
    """Do not erase uncertain worker/slot evidence during driver recovery."""
    keys = set(observed.get("tasks", {})) | set(observed.get("additions", {}))
    for key in sorted(keys):
        raw = (
            observed.get("tasks", {}).get(key),
            observed.get("additions", {}).get(key),
        )
        if not any(
            isinstance(item, dict)
            and (
                item.get("retry_pending")
                or item.get("retry_once_active")
                or item.get("retry_recovery_required")
            )
            for item in raw
        ):
            continue
        job = task_view(observed, key)
        if not job:
            continue
        if job.get("retry_recovery_required") or (
            job.get("retry_pending") and job.get("status") in {"starting", "running"}
        ):
            raise ValueError(
                f"{key}: previous retry launch requires worker ownership verification"
            )
        if not enabled and (job.get("retry_pending") or job.get("retry_once_active")):
            raise ValueError(
                f"{key}: retry recovery is not enabled; task state retained"
            )


def check_root_recovery(root):
    """Check batch retry evidence before any driver claims or publishes ready."""
    root = Path(root).resolve()
    path = root / "plan.json"
    if not path.is_file():
        return
    plan = json.loads(path.read_text())
    if plan.get("schema") == "smartsom.author-batch-plan/v1":
        check_driver_recovery(
            runtime_control.snapshot(root), enabled=runtime_control._RETRY_ENABLED
        )


def sync_retry_addition(state, key, job):
    """Keep the two official views consistent when restoring an addition."""
    target = state.get("tasks", {}).get(key)
    if target is None:
        return
    if target.get("root") != job.get("root"):
        raise ValueError(f"{key}: addition ownership differs from task view")
    for field in (
        "status",
        "pid",
        "exit_code",
        "retry_pending",
        "retry_once_active",
        "retry_attempt",
        "retry_history",
    ):
        if field in job:
            target[field] = job[field]


def pending_retry_targets(observed):
    """Accepted but unapplied retry requests which must survive driver exit."""
    return {
        receipt.get("payload", {}).get("task_id")
        for receipt in observed.get("commands", {}).values()
        if receipt.get("status") == "received"
        and receipt.get("payload", {}).get("kind") == "retry"
        and receipt.get("payload", {}).get("task_id")
    }


def prepare_stage_jobs(jobs, observed):
    """Keep a paused retry target paused until its command is consumed."""
    pending = pending_retry_targets(observed)
    for key, job in jobs.items():
        if job["status"] not in {"completed", "failed"} and not (
            key in pending and job["status"] == "paused"
        ):
            job["status"] = "queued"


def restore_selected_state(root, plan, state, observed):
    """Reconstruct accepted not-yet-started retries without expanding selection."""
    root = Path(root).resolve()
    restored, updates = [], []
    selected = set((state.get("selection") or {}).get("tasks", []))
    known = {row["id"] for row in plan["files"]}
    for key, job in observed.get("additions", {}).items():
        if key not in known and isinstance(job, dict) and job.get("retry_pending"):
            raise ValueError(
                f"{key}: external-owner retry needs verified queue handoff"
            )
    for row in plan["files"]:
        raw = (
            observed.get("tasks", {}).get(row["id"]),
            observed.get("additions", {}).get(row["id"]),
        )
        # Ordinary additions may legitimately be between their two published
        # views. Only retry journal entries require strict cross-view agreement.
        if not any(
            isinstance(item, dict) and item.get("retry_pending") for item in raw
        ):
            continue
        job = task_view(observed, row["id"])
        if not job or not job.get("retry_pending"):
            continue
        history = job.get("retry_history", [])
        receipt = (
            observed.get("commands", {}).get(history[-1].get("command_id"), {})
            if history
            else {}
        )
        if receipt.get("status") != "applied" or receipt.get(
            "retry_attempt"
        ) != job.get("retry_attempt"):
            raise ValueError(f"{row['id']}: retry journal identity mismatch")
        addition = observed.get("additions", {}).get(row["id"])
        added = (
            isinstance(addition, dict)
            and addition.get("owner_root") == str(root)
            and addition.get("owner_task") == row["id"]
            and Path(addition.get("root", "")).resolve()
            == (root / row["child"]).resolve()
        )
        if plan.get("incremental") and row["id"] not in selected and not added:
            raise ValueError(
                f"{row['id']}: retry task is outside the frozen selection and additions"
            )
        if job.get("status") not in {"queued", "paused"}:
            raise ValueError(
                f"{row['id']}: retry may have started before driver exit; verify worker ownership"
            )
        restored.append(row["id"])
        # Appended members remain in additions and must not expand authored selection.
        if not added and state["files"][row["id"]]["status"] == "failed":
            updates.append((row["id"], job["status"]))
    for identity, status in updates:
        state["files"][identity]["status"] = status
    if (restored or pending_retry_targets(observed)) and state["status"] == "failed":
        state.update(status="prepared", stage="prepared")
    return restored


def preview_failed(queue, task_ids=None, *, include_paused=False):
    """Read-only task snapshot; checks do not authorize execution."""
    root = Path(queue).resolve()
    state = runtime_control.snapshot(root)
    tasks = state.get("tasks", {})
    additions = state.get("additions", {})
    statuses = {"failed", "paused"} if include_paused else {"failed"}
    if task_ids is None:
        selected = sorted(
            key
            for key in set(tasks) | set(additions)
            if any(
                isinstance(job, dict) and job.get("status") in statuses
                for job in (tasks.get(key), additions.get(key))
            )
        )
    else:
        selected = list(task_ids)
        if len(set(selected)) != len(selected):
            raise ValueError("duplicate task ID in retry preview")
    targets = []
    for key in selected:
        item = {"task_id": key}
        try:
            job = task_view(state, key) or {}
            item.update(
                {
                    "status": job.get("status", "unknown"),
                    "exit_code": job.get("exit_code"),
                    "pid": job.get("pid"),
                    "retry_attempt": job.get("retry_attempt"),
                    "error": job.get("error"),
                    "blocking_reason": job.get("blocking_reason"),
                }
            )
            child = Path(job["root"]) if job.get("root") else None
            if child is not None:
                try:
                    attempts = json.loads((child / "attempts.json").read_text())
                    if isinstance(attempts, list) and attempts:
                        item["last_recorded_attempt"] = attempts[-1]
                except (OSError, ValueError, TypeError):
                    pass
            if include_paused and job.get("status") == "paused":
                expected = {
                    name: job.get(name)
                    for name in ("status", "pid", "exit_code", "retry_attempt")
                }
                runtime_control._check_retry_resume(
                    state,
                    key,
                    {"retry_resume_expected": expected},
                    require_enabled=False,
                )
                item.update(
                    check="eligible_for_coordinator_revalidation", action="resume"
                )
            else:
                item.update(
                    check="eligible_for_coordinator_revalidation",
                    action="retry",
                    continuation=inspect_target(
                        root, key, state, allow_paused=include_paused
                    ),
                )
        except (
            OSError,
            ValueError,
            TypeError,
            KeyError,
            RuntimeError,
            subprocess.SubprocessError,
        ) as exc:
            item.update(check="blocked", reason=str(exc))
        targets.append(item)
    binding = {
        "queue": str(root),
        "targets": targets,
        "revision": state.get("revision"),
    }
    return {
        **binding,
        "preview_sha256": digest(binding),
        "warning": "Preview only; final identity, ownership and checkpoint validation remains with the coordinator.",
    }


def submit_preview(queue, preview):
    """Submit exactly the snapshot targets, retaining independent outcomes."""
    root = Path(queue).resolve()
    binding = {
        "queue": preview.get("queue"),
        "targets": preview.get("targets"),
        "revision": preview.get("revision"),
    }
    if preview.get("preview_sha256") != digest(binding) or binding["queue"] != str(
        root
    ):
        raise ValueError("retry preview identity does not match this queue")
    outcomes = []
    for item in binding["targets"]:
        key = item["task_id"]
        if item.get("check") != "eligible_for_coordinator_revalidation":
            outcomes.append(
                {
                    "task_id": key,
                    "status": "rejected",
                    "reason": item.get("reason", "preflight unavailable"),
                }
            )
            continue
        expected = {
            name: item.get(name)
            for name in ("status", "pid", "exit_code", "retry_attempt")
        }
        action = item.get("action", "retry")
        if action == "resume":
            value = {"retry_resume_expected": expected}
        elif action == "retry":
            value = {
                "failure_attempt": expected,
                "pointer_sha256": item["continuation"]["pointer_sha256"],
            }
        else:
            raise ValueError("unknown retry preview action")
        try:
            command_id = hashlib.sha256(
                (preview["preview_sha256"] + "\0" + key).encode()
            ).hexdigest()
            receipt = runtime_control.submit(
                root, action, task_id=key, value=value, command_id=command_id
            )
            outcomes.append(
                {"task_id": key, "status": receipt["status"], "receipt": receipt}
            )
        except (OSError, ValueError, TypeError, KeyError, RuntimeError) as exc:
            outcomes.append({"task_id": key, "status": "rejected", "reason": str(exc)})
    return {"preview_sha256": preview["preview_sha256"], "outcomes": outcomes}
