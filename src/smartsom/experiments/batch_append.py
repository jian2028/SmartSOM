"""Append selected native tasks while retaining their scientific batch ownership."""

import copy
import json
import time
from collections import defaultdict
from pathlib import Path
from uuid import uuid4

from smartsom.config.codec import digest
from smartsom.experiments import queue_admission
from smartsom.experiments import runtime_control as runtime
from smartsom.experiments.locking import exclusive_lock


def compile_sources(
    value,
    *,
    parent,
    ids=None,
    tags=None,
    selectors=None,
    first_train_seeds=None,
    output_root=None,
):
    from smartsom.config.experiment_layout import batch_directory, entry_id, entrypoints
    from smartsom.config.launch import resolve_experiment
    from smartsom.config.reservation import is_reservation
    from smartsom.experiments.author_batch import compile_directory

    value = resolve_experiment(value)
    paths = [value] if value.is_file() else entrypoints(value)
    if selectors is not None and ids is None:
        raise ValueError("--select requires --id and a directory")
    if value.is_file() and ids:
        raise ValueError("--id requires a directory when appending")
    if not paths or any(not is_reservation(path) for path in paths):
        raise ValueError("add requires current Experiment YAML entries")
    if ids is not None:
        lookup = {entry_id(path, value): path for path in paths}
        if not ids or set(ids) - lookup.keys():
            raise ValueError("unknown or duplicate batch Experiment IDs")
        paths = [lookup[identity] for identity in dict.fromkeys(ids)]
    groups = defaultdict(list)
    for path in paths:
        groups[batch_directory(path)].append(path)
    if ids is not None and len(groups) != 1:
        raise ValueError(
            "ordered --id/--select submission requires one owning batch directory"
        )
    if output_root is not None and len(groups) != 1:
        raise ValueError("--output-root requires one owning batch directory")
    plans = []
    for directory, members in groups.items():
        operations = {}
        known = parent.get("known_owners", {}).get(str(directory), [])
        if output_root is not None:
            operations["output_root"] = str(Path(output_root).expanduser().resolve())
        elif len(known) > 1:
            raise ValueError(
                "multiple known output roots for this batch; choose --output-root explicitly"
            )
        elif known:
            operations["output_root"] = known[0]
        # A frozen explicit destination takes precedence for its own batch.
        if str(directory) == parent["directory"]:
            if output_root is not None and Path(operations["output_root"]) != Path(
                parent["root"]
            ):
                raise ValueError(
                    "cannot redirect this queue's own batch to another output root"
                )
            operations["output_root"] = parent["root"]
        compiled = compile_directory(
            directory,
            ids=(
                [entry_id(lookup[key], directory) for key in ids]
                if ids is not None and len(groups) == 1
                else [entry_id(p, directory) for p in members]
            ),
            tags=tags,
            selectors=selectors,
            first_train_seeds=first_train_seeds,
            resume_missing=True,
            operations=operations,
        )
        selected = set(compiled.selection["tasks"])
        if any(row["stage"] != 0 for row in compiled.files if row["id"] in selected):
            raise ValueError("appended tasks must be independent stage-0 experiments")
        plans.append(compiled)
    return plans


def append(
    root,
    value,
    *,
    ids=None,
    tags=None,
    selectors=None,
    command_id=None,
    expected_revision=None,
    first_train_seeds=None,
    output_root=None,
):
    from smartsom.experiments import author_batch
    from smartsom.experiments.batch_inventory import prepare
    from smartsom.experiments.incremental_batch import (
        pending_evaluation,
        require_recovery,
        submission,
    )

    root = Path(root).resolve()
    identity = command_id or uuid4().hex
    payload = dict(
        kind="add",
        directory=str(value),
        ids=ids,
        tags=tags,
        selectors=selectors,
        first_train_seeds=first_train_seeds,
        output_root=str(Path(output_root).expanduser().resolve())
        if output_root is not None
        else None,
        task_id=None,
    )
    prior = runtime.snapshot(root).get("commands", {}).get(identity)
    if prior:
        if prior["payload"] != payload:
            raise ValueError("command ID reused with different payload")
        return prior
    _, parent, _ = author_batch.load(root)
    known_owners = defaultdict(set)
    for job in runtime.snapshot(root).get("additions", {}).values():
        if job.get("owner_root"):
            owner_root = Path(job["owner_root"])
            owner_plan = json.loads((owner_root / "plan.json").read_text())
            known_owners[owner_plan["directory"]].add(str(owner_root))
    plans = compile_sources(
        value,
        parent={
            **parent,
            "root": str(root),
            "known_owners": {k: sorted(v) for k, v in known_owners.items()},
        },
        ids=ids,
        tags=tags,
        selectors=selectors,
        first_train_seeds=first_train_seeds,
        output_root=output_root,
    )
    jobs, claims = {}, []
    # Queue close and append share this transaction. Each source has its own
    # submission/driver claim; only frozen task identities enter this queue.
    with runtime.transaction(root) as state:
        runtime.require_admission_available(state)
        prior = state.get("commands", {}).get(identity)
        if prior:
            if prior["payload"] != payload:
                raise ValueError("command ID reused with different payload")
            return copy.deepcopy(prior)
        if not state or not state["accepting"] or state["status"] in runtime.TERMINAL:
            raise ValueError("batch is closed; use run --resume-missing to reopen it")
        if expected_revision is not None and expected_revision != state["revision"]:
            raise ValueError("runtime revision changed; refresh before retrying")
        try:
            for compiled in plans:
                owner_root = compiled.output_root.resolve()
                if output_root is None:
                    current_owners = set()
                    for job in state.get("additions", {}).values():
                        if job.get("owner_root"):
                            recorded = Path(job["owner_root"])
                            recorded_plan = json.loads(
                                (recorded / "plan.json").read_text()
                            )
                            if recorded_plan["directory"] == str(compiled.directory):
                                current_owners.add(recorded.resolve())
                    if current_owners and current_owners != {owner_root}:
                        raise ValueError(
                            "batch owner changed during submission; retry with its existing --output-root"
                        )
                if owner_root == root:
                    _, plan, owner_state = author_batch.load(root)
                    plan, owner_state = prepare(
                        root, compiled, plan, owner_state, state
                    )
                    _collect(
                        root,
                        owner_root,
                        compiled,
                        plan,
                        owner_state,
                        state,
                        jobs,
                        pending_evaluation,
                        require_recovery,
                    )
                    continue
                with submission(owner_root, adopting_queue=root):
                    if not owner_root.exists():
                        author_batch.allocate(compiled)
                    with exclusive_lock(owner_root / "driver.lock"):
                        _, plan, owner_state = author_batch.load(owner_root)
                        author_batch._runtime_initialize(
                            owner_root, plan, owner_state, wait=False
                        )
                        with runtime.transaction(
                            owner_root, wait=False
                        ) as owner_runtime:
                            plan, owner_state = prepare(
                                owner_root, compiled, plan, owner_state, owner_runtime
                            )
                        _collect(
                            root,
                            owner_root,
                            compiled,
                            plan,
                            owner_state,
                            state,
                            jobs,
                            pending_evaluation,
                            require_recovery,
                        )
                        if not any(
                            job["owner_root"] == str(owner_root)
                            for job in jobs.values()
                        ):
                            continue
                        shared = queue_admission.snapshot(owner_root)
                        if shared.get("queue") != str(root):
                            from smartsom.experiments.control import read

                            owner = read(owner_root)
                            generation = queue_admission.completed_source(
                                owner_root, owner_id=owner["id"] if owner else None
                            )
                            queue_admission.claim(
                                owner_root,
                                root,
                                limit=state["limit"],
                                owner_id=generation,
                            )
                            claims.append(owner_root)
            state["additions"].update(jobs)
            launch_order = state.setdefault("launch_order", [])
            launch_order.extend(key for key in jobs if key not in launch_order)
            state["revision"] += 1
            receipt = dict(
                id=identity,
                command_id=identity,
                payload=payload,
                received_at=time.time(),
                status="received",
                revision=state["revision"],
                tasks=list(jobs),
                directory=str(root),
            )
            state["commands"][identity] = receipt
            return copy.deepcopy(receipt)
        except BaseException:
            # No jobs became visible before the transaction commits. Only claims
            # obtained by this submission can be withdrawn here.
            for owner_root in claims:
                queue_admission.detach(owner_root, root)
            raise


def _collect(
    root,
    owner_root,
    compiled,
    plan,
    owner_state,
    queue_state,
    jobs,
    pending_evaluation,
    require_recovery,
):
    rows = {row["id"]: row for row in plan["files"]}
    for task in compiled.selection["tasks"]:
        row = rows[task]
        key = row["id"] if root == owner_root else plan["run_id"] + "::" + row["id"]
        prior = queue_state.get("additions", {}).get(key)
        if prior is not None:
            continue
        status = owner_state["files"][row["id"]]["status"]
        if root == owner_root:
            status = queue_state.get("tasks", {}).get(key, {}).get("status", status)
        if status in {"failed", "removed", "partial_removed"}:
            continue
        if status == "completed" and not pending_evaluation(owner_root, row):
            continue
        if root == owner_root and status not in {"deferred", "completed"}:
            continue  # already owned by the directory driver's selected stage
        if status not in {"deferred", "queued"}:
            require_recovery(owner_root, row)
        child = owner_root / row["child"]
        jobs[key] = dict(
            id=key,
            file_id=key,
            root=str(child),
            status="queued",
            owner_root=str(owner_root),
            owner_task=row["id"],
            child_plan_sha256=row["child_plan_sha256"],
        )


def synchronize(root, jobs, *, protected=()):
    """Publish receipts and decide detach against the latest accepted queue work.

    Append and detach use the same queue transaction, in queue → owner order.
    A scheduler's older jobs snapshot must not erase an accepted new task or
    release its owning batch while that new task is still queued.
    """
    if not any(job.get("owner_root") for job in jobs.values()):
        return
    with runtime.transaction(root) as latest:
        for key, job in jobs.items():
            if job.get("owner_root") and key in latest.get("additions", {}):
                latest["additions"][key].update(job)
        additions = latest.get("additions", {})
        protected_owners = {
            additions[key].get("owner_root")
            for key in protected
            if isinstance(additions.get(key), dict) and additions[key].get("owner_root")
        }
        eligible = {
            key: job
            for key, job in additions.items()
            if key not in protected and job.get("owner_root") not in protected_owners
        }
        _synchronize_locked(root, eligible)


def _synchronize_locked(root, jobs):
    from smartsom.experiments import author_batch
    from smartsom.experiments.incremental_batch import completion

    groups = defaultdict(dict)
    for key, job in jobs.items():
        if job.get("owner_root"):
            groups[Path(job["owner_root"])][key] = job
    for owner_root, members in groups.items():
        if owner_root == Path(root):
            continue  # its active driver's _save merges these runtime receipts
        if queue_admission.snapshot(owner_root).get("queue") != str(root):
            continue  # completed work has already returned to its original owner
        with exclusive_lock(owner_root / "driver.lock"):
            plan = json.loads((owner_root / "plan.json").read_text())
            state = json.loads((owner_root / "batch.json").read_text())
            if digest(plan) != state.get("plan_sha256"):
                raise ValueError("frozen owning batch plan changed")
            from smartsom.experiments.batch_inventory import extend_view

            plan, state = extend_view(
                owner_root, plan, state, runtime.snapshot(owner_root)
            )
            changed = any(
                state["files"][job["owner_task"]]["status"] != job["status"]
                for job in members.values()
            )
            if changed:
                for job in members.values():
                    state["files"][job["owner_task"]].update(status=job["status"])
                state.update(status=completion(state), stage="external_queue")
                author_batch._save(owner_root, state)
            if all(
                job["status"] in runtime.TERMINAL
                and not job.get("retry_pending")
                and not job.get("retry_recovery_required")
                for job in members.values()
            ):
                shared = queue_admission.snapshot(owner_root)
                if shared.get("queue") == str(root):
                    queue_admission.detach(owner_root, root)
                if plan.get("research_layout"):
                    from smartsom.experiments.research_report import write_reports

                    write_reports(owner_root, plan, state)
