"""Execution selection for an immutable directory inventory, using native recovery."""

from contextlib import contextmanager
from pathlib import Path

from smartsom.config.codec import ConfigurationError, digest
from smartsom.experiments.locking import exclusive_lock


@contextmanager
def submission(root, *, adopting_queue=None):
    """Serialize selection through completion or verified background owner handoff."""
    from smartsom.experiments.background import _check_owner

    root = Path(root).resolve()
    root.parent.mkdir(parents=True, exist_ok=True)
    with exclusive_lock(root.parent / ("." + root.name + ".submission.lock")):
        if root.exists():
            from smartsom.experiments.queue_admission import snapshot

            shared = snapshot(root) if adopting_queue is not None else {}
            if adopting_queue is None or shared.get("queue") != str(adopting_queue):
                _check_owner(root)
        yield


def pending_evaluation(root, row):
    if row["task"] != "train":
        return False
    from smartsom.config.codec import read_document

    child = root / row["child"]
    experiment = read_document(child / "inputs/experiment.yaml")
    return (
        experiment.get("final_evaluation", False)
        and not (child / "evaluation/completed.json").is_file()
    )


def require_recovery(root, row):
    """A started training task must never silently construct a fresh learner."""
    child = root / row["child"]
    if row["task"] == "train" and not (child / "continuation/current.json").is_file():
        raise ConfigurationError(
            f"{row['id']}: started training has no complete continuation; "
            "preserve this result and choose a new output root"
        )


def activate(directory_plan):
    from smartsom.experiments import runtime_control as runtime
    from smartsom.experiments.author_batch import _save, load

    root = directory_plan.output_root
    with exclusive_lock(root / "driver.lock"):
        root, plan, state = load(root)
        if not plan.get("incremental"):
            raise ConfigurationError(
                "existing batch has no full incremental inventory; choose a new output root"
            )
        if plan["directory"] != str(directory_plan.directory) or digest(
            plan["inventory"]
        ) != digest(directory_plan.inventory):
            raise ConfigurationError(
                "batch directory or scientific inventory changed; choose a new output root"
            )
        expected = [
            (row["id"], row["scientific_sha256"], row["child"])
            for row in directory_plan.files
        ]
        actual = [
            (row["id"], row["scientific_sha256"][0], row["child"])
            for row in plan["files"]
        ]
        if expected != actual:
            raise ConfigurationError(
                "batch task identities changed; choose a new output root"
            )
        selected = set(directory_plan.selection["tasks"])
        activated = []
        # Validate all selected recovery candidates before changing the state.
        for row in plan["files"]:
            if row["id"] not in selected:
                continue
            status = state["files"][row["id"]]["status"]
            if status in {"failed", "removed", "partial_removed"}:
                continue  # explicit resume --retry-failed or removal controls own these
            if status == "completed" and not pending_evaluation(root, row):
                continue
            if status not in {"deferred", "queued"}:
                require_recovery(root, row)
            activated.append(row["id"])
        state["selection"] = directory_plan.selection
        state["submission_tasks"] = activated
        for identity in activated:
            state["files"][identity]["status"] = "queued"
        if activated:
            state.update(status="prepared", stage="prepared")
            state.pop("error", None)
            # Closed queues reopen; task-specific pause/stop/removal holds survive.
            runtime.restart(root)
            with runtime.transaction(root) as saved:
                for identity in activated:
                    if identity in saved.get("additions", {}):
                        saved["additions"][identity]["status"] = "queued"
        _save(root, state)
    return root


def selected_rows(plan, state):
    if not plan.get("incremental"):
        return plan["files"]
    selected = set(state["selection"]["tasks"])
    return [
        row
        for row in plan["files"]
        if row["id"] in selected
        and state["files"][row["id"]]["status"]
        not in {"deferred", "failed", "removed", "partial_removed"}
    ]


def completion(state):
    statuses = [item["status"] for item in state["files"].values()]
    return (
        "failed"
        if "failed" in statuses
        else "completed"
        if all(s == "completed" for s in statuses)
        else "partial"
    )


def counts(state):
    values = [item["status"] for item in state["files"].values()]
    return {
        "completed_tasks": values.count("completed"),
        "deferred_tasks": values.count("deferred"),
        "failed_tasks": values.count("failed"),
        "submitted_tasks": len(state.get("submission_tasks", [])),
    }
