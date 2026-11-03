"""Execution selection for an immutable directory inventory, using native recovery."""

import hashlib
import json
from contextlib import contextmanager
from pathlib import Path

from smartsom.config.codec import ConfigurationError
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
    if row["task"] != "train":
        return
    pointer = child / "continuation/current.json"
    try:
        marker = json.loads(pointer.read_text())
        base = (child / "continuation").resolve()
        generation = (base / marker["generation"]).resolve()
        if not generation.is_relative_to(base) or generation == base:
            raise ValueError("continuation generation escapes its root")
        if set(marker["files"]) != {"state.json", "model.pt"}:
            raise ValueError("continuation requires state.json and model.pt")
        for name, expected in marker["files"].items():
            with (generation / name).open("rb") as stream:
                actual = hashlib.file_digest(stream, "sha256").hexdigest()
            if actual != expected:
                raise ValueError("continuation checkpoint digest mismatch")
        saved = json.loads((generation / "state.json").read_text())
        if (
            not isinstance(saved["rows"], list)
            or type(saved["ticks"]) is not int
            or saved["ticks"] < 0
        ):
            raise ValueError("invalid continuation state")
        if "condition_rng" not in saved:
            raise ValueError("continuation lacks RNG state")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ConfigurationError(
            f"{row['id']}: started training has no complete continuation (compatible recovery required): {exc}; "
            "preserve this result and repair recovery or choose a new output root"
        ) from exc


def activate(directory_plan):
    from smartsom.experiments import runtime_control as runtime
    from smartsom.experiments.author_batch import _runtime_initialize, _save, load

    root = directory_plan.output_root
    with exclusive_lock(root / "driver.lock"):
        root, plan, state = load(root)
        from smartsom.experiments.batch_inventory import prepare, validate

        validate(directory_plan, plan)
        selected = set(directory_plan.selection["tasks"])
        activated = []
        controls = runtime.snapshot(root)
        held = (
            controls.get("paused_tasks", [])
            + controls.get("stopped_tasks", [])
            + controls.get("removed_tasks", [])
        )
        if controls.get("paused") or any(
            runtime.matches(row["id"], held)
            for row in plan["files"]
            if row["id"] in selected
            and (
                state["files"][row["id"]]["status"] != "completed"
                or pending_evaluation(root, row)
            )
        ):
            raise ConfigurationError(
                "selected tasks have manual holds; explicitly resume/unhold them first"
            )
        # Validate all selected recovery candidates before changing the state.
        for row in plan["files"]:
            if row["id"] not in selected:
                continue
            status = state["files"][row["id"]]["status"]
            if status in {
                "paused",
                "removed",
                "partial_removed",
            }:
                raise ConfigurationError(
                    f"{row['id']}: manual/stopped state requires explicit resume controls"
                )
            if status == "completed" and not pending_evaluation(root, row):
                continue
            if status not in {"deferred", "queued"}:
                require_recovery(root, row)
            activated.append(row["id"])
        # Membership commits under the same queue lock as active batch-add.
        # Its immutable extension is visible before execution selection changes.
        _runtime_initialize(root, plan, state)
        with runtime.transaction(root) as saved:
            plan, state = prepare(root, directory_plan, plan, state, saved)
        for row in plan["files"]:
            if (
                row["id"] in selected
                and state["files"][row["id"]]["status"] == "deferred"
                and row["id"] not in activated
            ):
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
    rows = [
        row
        for row in plan["files"]
        if row["id"] in selected
        and state["files"][row["id"]]["status"]
        not in {"deferred", "failed", "removed", "partial_removed"}
    ]
    order = {
        identity: index for index, identity in enumerate(state["selection"]["tasks"])
    }
    return sorted(rows, key=lambda row: order[row["id"]])


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
