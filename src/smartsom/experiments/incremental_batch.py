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
    from smartsom.config.codec import digest
    from smartsom.experiments.author_batch import _save, load

    root = directory_plan.output_root
    with exclusive_lock(root / "driver.lock"):
        root, plan, state = load(root)
        if plan["directory"] != str(directory_plan.directory) or digest(
            plan["inventory"]
        ) != digest(directory_plan.inventory):
            raise ConfigurationError(
                "resume-missing requires the exact frozen scientific inventory; use a new run for new tasks"
            )
        if state.get("active_cohorts"):
            raise ConfigurationError(
                "unfinished Tune cohort requires resume or retry --all before selecting missing tasks"
            )
        selected = set(directory_plan.selection["tasks"])
        if not selected <= state["files"].keys():
            raise ConfigurationError(
                "resume-missing cannot append tasks to a frozen Tune run"
            )
        pending = []
        for row in plan["files"]:
            if row["id"] not in selected:
                continue
            status = state["files"][row["id"]]["status"]
            if status == "completed" and not pending_evaluation(root, row):
                continue
            if status not in {"queued", "deferred"}:
                raise ConfigurationError(
                    "started or failed trials require resume or retry --all, not --resume-missing"
                )
            pending.append(row["id"])
        state.update(selection=directory_plan.selection, submission_tasks=pending)
        for identity in pending:
            state["files"][identity]["status"] = "queued"
        if pending:
            state.update(status="prepared", stage="prepared")
        _save(root, state)
    return root


def selected_rows(plan, state, pending_retries=()):
    if not plan.get("incremental"):
        return plan["files"]
    selected = set(state["selection"]["tasks"])
    rows = [
        row
        for row in plan["files"]
        if row["id"] in selected
        and (
            state["files"][row["id"]]["status"]
            not in {"deferred", "failed", "removed", "partial_removed"}
            or (
                row["id"] in pending_retries
                and state["files"][row["id"]]["status"] == "failed"
            )
        )
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
