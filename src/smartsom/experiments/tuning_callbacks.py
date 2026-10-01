"""Pickle-safe Tune callbacks; only the driver writes the batch ledger/display."""

import json
import time
from pathlib import Path

from ray.tune import Callback

from smartsom.config.experiment_v3 import PreparedComposition
from smartsom.experiments.evidence import write_json
from smartsom.experiments.tuning_session import verify_identity
from smartsom.telemetry.runtime import CURRENT


def _selected_workflow(row):
    """Replace the frozen candidate's layout with the admitted measured layout."""
    selected = row.get("selected_prepared")
    if selected is None:
        return None
    from smartsom.telemetry.workflow import describe_prepared

    return describe_prepared(PreparedComposition(**selected), "train-evaluate")


class EvidenceCallback(Callback):
    def __init__(self, root, entries, broker):
        self.root = str(root)
        self.entries = entries
        self.broker = broker
        self._last_display = 0.0

    def on_step_begin(self, iteration, trials, **info):
        from smartsom.experiments.control import boundary, requested

        if requested(self.root) and not any(t.status == "RUNNING" for t in trials):
            boundary(self.root)
        self.broker.refresh()
        unresolved = self.broker.unresolved_failures()
        if unresolved:
            raise RuntimeError(
                "failed actors have incomplete sampling-child ownership; "
                "reservation retained, batch stopped: " + ", ".join(unresolved)
            )
        display = CURRENT.get()
        if display is None or time.monotonic() - self._last_display < 1:
            return
        self._last_display = time.monotonic()
        root = Path(self.root)
        calibration = json.loads((root / "calibration.json").read_text())
        plan = json.loads((root / "plan.json").read_text())
        ledger = json.loads((root / "batch.json").read_text())
        summary = self.broker.summary()
        resources = {
            entry["experiment_id"]: entry for entry in summary.pop("entries", [])
        }
        summary["entries"] = [
            {
                "experiment_id": entry["experiment_id"],
                **resources.get(entry["experiment_id"], {}),
                "status": ledger["entries"][entry["experiment_id"]]["status"]
                if ledger["entries"][entry["experiment_id"]]["status"]
                in {"completed", "failed"}
                else resources.get(entry["experiment_id"], {}).get("status", "queued"),
            }
            for entry in plan["entries"]
        ]
        display.configure_tuning(
            {
                "stage": "running"
                if any(e["status"] == "running" for e in summary["entries"])
                else "waiting_resources",
                **summary,
                "calibration": {
                    **(display.tuning or {}).get("calibration", {}),
                    "active_seconds": calibration["active_seconds"],
                    "wall_seconds": calibration["wall_seconds"],
                    "waiting_seconds": calibration["waiting_seconds"],
                    "limit_seconds": calibration.get("active_limit", 600),
                    "remaining_seconds": max(
                        0,
                        calibration.get("active_limit", 600)
                        - calibration["wall_seconds"],
                    ),
                    "measured": len(calibration["measurements"]),
                    "calibrated": calibration["calibrated"],
                    "reason": calibration.get("reason"),
                    "recommendation": calibration["recommendations"],
                },
            }
        )
        for entry in self.entries:
            if ledger["entries"][entry["experiment_id"]]["status"] in {
                "completed",
                "failed",
            }:
                continue
            try:
                progress = json.loads(
                    (Path(entry["run_dir"]) / "logs/progress.json").read_text()
                )
            except (OSError, ValueError):
                continue
            tasks = progress.get("tasks", [])
            if tasks:
                # A v3 worker's main training task already merges live validation
                # and evaluation ticks, retaining physical training denominators.
                task = next((t for t in tasks if t["id"] == "training"), tasks[0])
                event = {
                    **task.get("values", {}),
                    "stage": task.get("stage", "running"),
                    "status": task.get("status", "running"),
                    "physical_ticks": task.get("completed", 0),
                }
                evaluation = next(
                    (t for t in tasks if t.get("id") == "evaluation"), None
                )
                if evaluation is not None:
                    event.update(evaluation.get("values", {}))
                    if evaluation.get("status") == "running":
                        event["stage"] = "evaluation"
                workflow = _selected_workflow(ledger["entries"][entry["experiment_id"]])
                if workflow is not None:
                    event["workflow"] = workflow
                display.update(
                    entry["experiment_id"],
                    event,
                    total=PreparedComposition(
                        **entry["prepared"]
                    ).config.training.total_ticks,
                    unit="physical ticks",
                )

    def on_trial_start(self, iteration, trials, trial, **info):
        self.broker.note_actor(trial.config["experiment_id"], trial)

    def on_trial_result(self, iteration, trials, trial, result, **info):
        identity = trial.config["experiment_id"]
        self.broker.note_actor(identity, trial)
        record = dict(trial.config["record"])
        record["tuning"] = {"experiment_id": identity}
        marker = verify_identity(
            PreparedComposition(**trial.config["prepared"]),
            record,
            result["checkpoint"],
        )
        root = Path(self.root)
        state = json.loads((root / "batch.json").read_text())
        row = state["entries"][identity]
        row.update(
            status="completed"
            if marker["phase"] == "experiment_complete"
            else "running",
            checkpoint=result["checkpoint"],
            commit_id=marker["commit_id"],
            physical_ticks=marker["physical_ticks"],
            updates=marker["updates"],
            allocation_epoch=result["allocation_epoch"],
        )
        write_json(root / "batch.json", state)
        display = CURRENT.get()
        if display:
            event = {
                "stage": "completed"
                if marker["phase"] == "experiment_complete"
                else "saving",
                "physical_ticks": marker["physical_ticks"],
                "status": row["status"],
            }
            workflow = _selected_workflow(row)
            if workflow is not None:
                event["workflow"] = workflow
            if marker["phase"] == "experiment_complete":
                progress_path = Path(trial.config["run_dir"]) / "logs/progress.json"
                if progress_path.is_file():
                    progress = json.loads(progress_path.read_text())
                    training = next(
                        (
                            item
                            for item in progress["tasks"]
                            if item["id"] == "training"
                        ),
                        None,
                    )
                    if training:
                        for name in (
                            "validation_finished",
                            "validation_requested",
                            "validation_batches_finished",
                        ):
                            if name in training.get("values", {}):
                                event[name] = training["values"][name]
                summary_path = Path(trial.config["run_dir"]) / "evaluation/summary.json"
                if summary_path.is_file():
                    evaluation = json.loads(summary_path.read_text())
                    event["evaluation_requested"] = evaluation["requested"]
                    event["evaluation_finished"] = sum(
                        evaluation[key]
                        for key in ("completed", "truncated", "exceptions")
                    )
            display.update(
                identity,
                event,
                total=PreparedComposition(
                    **trial.config["prepared"]
                ).config.training.total_ticks,
                unit="physical ticks",
                final=row["status"] == "completed",
            )

    def on_trial_error(self, iteration, trials, trial, **info):
        root = Path(self.root)
        state = json.loads((root / "batch.json").read_text())
        row = state["entries"][trial.config["experiment_id"]]
        # Only commit.json establishes resumable progress. Mutable run.json and
        # an in-flight Ray result never overwrite the last verified boundary.
        recovery = Path(trial.config["run_dir"]) / "checkpoints/adaptive-recovery.json"
        if recovery.exists():
            pointer = json.loads(recovery.read_text())
            checkpoint = recovery.parent / pointer["checkpoint"]
            record = {
                **trial.config["record"],
                "tuning": {"experiment_id": trial.config["experiment_id"]},
            }
            marker = verify_identity(
                PreparedComposition(**trial.config["prepared"]), record, checkpoint
            )
            row.update(
                checkpoint=str(checkpoint),
                commit_id=marker["commit_id"],
                physical_ticks=marker["physical_ticks"],
                updates=marker["updates"],
            )
        self.broker.actor_failed(trial.config["experiment_id"], trial)
        from smartsom.experiments.control import requested

        row["status"] = "interrupted" if requested(self.root) else "failed"
        write_json(root / "batch.json", state)
