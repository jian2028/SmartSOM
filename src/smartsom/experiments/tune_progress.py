"""Read-only Tune lifecycle observations published to existing Rich snapshots."""

import json
import time
from contextlib import nullcontext
from pathlib import Path

from ray.tune import Callback

from smartsom.experiments.evidence import write_json


class ScientificProgress(Callback):
    def __init__(self, task_ids=None):
        self.task_ids = task_ids or {}
        self.last_publish = 0.0
        self.terminal = {}
        self.published_terminal = {}

    def setup(self, **info):
        self.last_publish = 0.0
        self.terminal = {}
        self.published_terminal = {}

    def on_step_end(self, iteration, trials, **info):
        now = time.monotonic()
        if now - self.last_publish < 1:
            return
        self.last_publish = now
        self._publish(trials)

    def on_trial_complete(self, iteration, trials, trial, **info):
        self.terminal[trial.trial_id] = "TERMINATED"
        self._publish(trials)

    def on_trial_error(self, iteration, trials, trial, **info):
        self.terminal[trial.trial_id] = "ERROR"
        self._publish(trials)

    def on_trial_start(self, iteration, trials, trial, **info):
        self.terminal.pop(trial.trial_id, None)
        self.published_terminal.pop(trial.trial_id, None)

    def on_experiment_end(self, trials, **info):
        self._publish(trials)

    def _publish(self, trials):
        from smartsom.telemetry.current_progress import presentation
        from smartsom.telemetry.runtime import CURRENT

        session = CURRENT.get()
        with session.batch_updates() if session is not None else nullcontext():
            self._publish_rows(trials, session, presentation)

    def _publish_rows(self, trials, session, presentation):
        for trial in trials:
            tune_status = self.terminal.get(trial.trial_id, trial.status)
            if self.published_terminal.get(trial.trial_id) == tune_status:
                continue
            config = trial.config.get("trial", trial.config)
            template = Path(config["root"])
            root = (
                template / "trials" / trial.trial_id if "search" in config else template
            )
            status = {
                "PENDING": "queued",
                "RUNNING": "running",
                "PAUSED": "waiting",
                "TERMINATED": "completed",
                "ERROR": "failed",
            }[tune_status]
            observation = {
                "trial_id": trial.trial_id,
                "status": status,
                "tune_status": tune_status,
                "updated_unix": time.time(),
            }
            # Bootstrap search directories must remain empty until allocation.
            if (root / "plan.json").exists():
                write_json(root / "tune-status.json", observation)
            if status in {"completed", "failed"}:
                self.published_terminal[trial.trial_id] = tune_status
            if session is None:
                continue
            progress_path = root / "progress.json"
            progress = (
                json.loads(progress_path.read_text()) if progress_path.exists() else {}
            )
            details, total, unit = presentation(progress, task=config["task"])
            session.update(
                self.task_ids.get(str(root), str(root)),
                {
                    **details,
                    **observation,
                    "display_name": root.name,
                    "scheduler_status": status,
                },
                total=total,
                unit=unit,
                final=status in {"completed", "failed"},
            )
