"""Frozen scientific stage barriers around the shared official Tune execution."""

import json
from pathlib import Path

from smartsom.config.codec import digest
from smartsom.experiments import control
from smartsom.experiments.evidence import write_json
from smartsom.experiments.tune_execution import fit_cohort


def public_metrics(metrics):
    metrics = {
        key: value
        for key, value in metrics.items()
        if key not in {"smartsom_checkpoint_score", "config"}
    }
    if "algorithm_iteration" not in metrics:
        return metrics
    return {
        key: value
        for key, value in metrics.items()
        if key
        in {
            "algorithm_iteration",
            "training_iteration",
            "episodes",
            "physical_ticks",
            "final_evaluation",
            "done",
            "time_total_s",
            "search_evaluation",
            "status",
        }
        or key.startswith("search_objective_")
    }


def _result_report(root, result, *, control_root):
    status = (
        "failed"
        if result.error
        else "completed"
        if result.metrics.get("done")
        else "paused"
        if control.pause_requested(control_root)
        else "stopped"
    )
    metrics = public_metrics(result.metrics)
    report = {
        **metrics,
        "directory": str(root),
        "status": status,
        "native_status": result.metrics.get("status"),
    }
    if result.error:
        report["error"] = str(result.error)
    write_json(root / "batch.json", report)
    return report


def _finish_directory(root, plan, state):
    from smartsom.experiments.author_batch import _publish, _save

    _save(root, state)
    _publish(root, plan, state)
    if plan.get("research_layout"):
        from smartsom.experiments.research_report import write_reports

        write_reports(root, plan, state)
    return {"directory": str(root), **state}


def execute_directory(root, *, retry_failed=False):
    from smartsom.experiments.author_batch import _gate, _publish, _save, load
    from smartsom.experiments.locking import exclusive_lock
    from smartsom.experiments.reservation import arguments
    from smartsom.telemetry.runtime import bind

    root = Path(root).resolve()
    bind(root)
    with exclusive_lock(root / "driver.lock"):
        root, plan, state = load(root)
        selected = set(state.get("submission_tasks", state["files"]))
        operations = json.loads((root / "execution.json").read_text())["settings"]
        cohorts = state.setdefault("tune_cohorts", {})
        active_cohorts = state.setdefault("active_cohorts", {})
        for stage in sorted({row["stage"] for row in plan["files"]}):
            rows = [
                row
                for row in plan["files"]
                if row["stage"] == stage and row["id"] in selected
            ]
            if str(stage) in active_cohorts:
                frozen_ids = set(active_cohorts[str(stage)])
                rows = [row for row in plan["files"] if row["id"] in frozen_ids]
            else:
                rows = [
                    row
                    for row in rows
                    if state["files"][row["id"]]["status"] != "completed"
                ]
            if not rows or all(
                state["files"][row["id"]]["status"] == "completed" for row in rows
            ):
                continue
            if not retry_failed and any(
                state["files"][row["id"]]["status"] == "failed" for row in rows
            ):
                return {"directory": str(root), **state, "retry_required": True}
            identity = digest([row["id"] for row in rows])[:16]
            storage = root / "ray" / f"stage-{stage}-{identity}"
            cohorts[str(storage)] = [row["id"] for row in rows]
            active_cohorts[str(stage)] = cohorts[str(storage)]
            failures = {arguments(root / row["child"])[1].max_failures for row in rows}
            if len(failures) != 1:
                raise ValueError(
                    "one Tune cohort requires one run.max_failures setting"
                )
            state.update(status="running", stage=f"stage-{stage}")
            _save(root, state)
            _publish(root, plan, state)
            results = fit_cohort(
                [root / row["child"] for row in rows],
                storage=storage,
                resources=operations["resources"],
                max_concurrent=operations["concurrency"]["max_concurrent"],
                max_failures=failures.pop(),
                retry_failed=retry_failed,
                task_ids={
                    str((root / row["child"]).resolve()): row["id"] for row in rows
                },
            )
            by_root = {str((root / row["child"]).resolve()): row for row in rows}
            for result in results:
                row = by_root[result.config["trial"]["root"]]
                report = _result_report(root / row["child"], result, control_root=root)
                status = report["status"]
                file_state = state["files"][row["id"]]
                file_state.update(status=status, native_status=report["native_status"])
                if result.error:
                    file_state["error"] = report["error"]
                else:
                    file_state.pop("error", None)
                if status == "completed" and row["task"] == "evaluate":
                    state["files"][row["id"]]["gate"] = _gate(root, row)
            statuses = {state["files"][row["id"]]["status"] for row in rows}
            if statuses != {"completed"}:
                state["status"] = (
                    "paused"
                    if "paused" in statuses
                    else "failed"
                    if "failed" in statuses
                    else "stopped"
                )
                return _finish_directory(root, plan, state)
            active_cohorts.pop(str(stage), None)
            _save(root, state)
            _publish(root, plan, state)
        state.update(status="completed", stage="selection_completed")
        return _finish_directory(root, plan, state)


def execute_single(root, *, retry_failed=False):
    from smartsom.experiments.reservation import arguments, load
    from smartsom.telemetry.runtime import bind

    root = Path(root).resolve()
    load(root)
    bind(root)
    _, settings = arguments(root)
    operations = json.loads((root / "execution.json").read_text())["settings"]
    storage = root / "ray" / "experiment"
    results = fit_cohort(
        [root],
        storage=storage,
        resources=operations["resources"],
        max_concurrent=operations["concurrency"]["max_concurrent"],
        max_failures=settings.max_failures,
        retry_failed=retry_failed,
    )
    return _result_report(root, results[0], control_root=root)
