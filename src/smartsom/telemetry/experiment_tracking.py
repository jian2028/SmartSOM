"""One local metric journal and optional W&B sink for native experiments."""

import json
import math
import os
import sys
import threading
import time
import warnings
from numbers import Real
from pathlib import Path
from uuid import uuid4

from smartsom.experiments.evidence import write_json
from smartsom.telemetry.training import isolated_random_state


def resolved(options):
    return {
        "mode": options.mode or os.environ.get("WANDB_MODE", "online"),
        "project": options.project or os.environ.get("WANDB_PROJECT", "SmartSOM"),
        "entity": options.entity or os.environ.get("WANDB_ENTITY"),
        "group": options.group,
        "tensorboard": options.tensorboard,
    }


def offline_choice(error, *, interactive=None):
    warnings.warn(f"W&B startup failed: {type(error).__name__}: {error}", stacklevel=2)
    interactive = sys.stdin.isatty() if interactive is None else interactive
    if interactive:
        answer = (
            input("Continue with local offline W&B recording? [y/N] ").strip().lower()
        )
        if answer in {"y", "yes"}:
            return "offline"
    raise ValueError(
        "W&B online startup failed. Explicitly set experiment.tracking.mode=offline "
        "to run without cloud access."
    ) from error


def numeric(values, prefix=""):
    result = {}
    for key, value in values.items():
        name = prefix + str(key)
        if isinstance(value, dict):
            result.update(numeric(value, name + "/"))
        elif (
            isinstance(value, Real)
            and not isinstance(value, bool)
            and math.isfinite(value)
        ):
            result[name] = float(value)
    return result


class ExperimentTracking:
    """Tracking failures after startup never invalidate a local training commit."""

    def __init__(self, root, options, *, job_type, group=None):
        self.root, self.settings = Path(root), resolved(options)
        if self.settings["mode"] not in {"online", "offline", "disabled"}:
            raise ValueError("tracking mode must be online, offline or disabled")
        self.job_type = job_type
        self.settings["group"] = (
            self.settings["group"]
            or group
            or (
                self.root.parent.parent.name
                if self.root.parent.name in {"experiments", "seeds", "added"}
                else self.root.name
            )
        )
        self.run = self.stream = self.tensorboard = None
        self.status = {"mode": self.settings["mode"], "sync_status": "not_started"}
        self.closed = False
        self.sequence = 0
        self.stop = threading.Event()
        self.monitor = None
        self.last_sync_progress = time.monotonic()

    def start(self, manifest, config):
        tracking = self.root / "tracking"
        tracking.mkdir(exist_ok=True)
        self.state_path = tracking / "status.json"
        previous = (
            json.loads(self.state_path.read_text()) if self.state_path.exists() else {}
        )
        self.sequence = previous.get("sequence", 0)
        mode = self.settings["mode"]
        if previous.get("online_run_id") and mode == "online":
            for field in ("project", "entity"):
                saved = (previous.get("online_identity") or {}).get(
                    field, previous.get(field)
                )
                explicit = self.settings[field]
                if explicit is not None and saved is not None and explicit != saved:
                    raise ValueError(
                        f"W&B resume {field} differs from saved run identity"
                    )
                self.settings[field] = saved or explicit
        self.segments = previous.get("segments", [])
        run_id = previous.get("online_run_id") if mode == "online" else None
        run_id = run_id or uuid4().hex
        self.status.update(
            experiment_id=previous.get("experiment_id", uuid4().hex),
            run_id=run_id,
            online_run_id=previous.get("online_run_id"),
            sequence=self.sequence,
            job_type=self.job_type,
            project=self.settings["project"],
            entity=self.settings["entity"],
            group=self.settings["group"],
            segments=self.segments,
            online_identity=previous.get("online_identity"),
        )
        self.stream = (tracking / "events.jsonl").open("a", encoding="utf-8")
        if self.settings["tensorboard"]:
            from smartsom.telemetry.tensorboard import TensorBoardRecorder

            try:
                with isolated_random_state():
                    self.tensorboard = TensorBoardRecorder(self.root / "tensorboard")
                self.status["tensorboard"] = "recording"
            except BaseException:
                self.stream.close()
                self.stream = None
                raise
        if mode == "disabled":
            self.status["sync_status"] = "disabled"
            self.publish()
            return self
        try:
            self._init(manifest, config, run_id, previous)
        except Exception as error:
            try:
                if mode != "online":
                    raise
                self.settings["mode"] = offline_choice(error)
                self.status.update(mode="offline", run_id=uuid4().hex)
                self._init(manifest, config, self.status["run_id"], {})
            except BaseException:
                self.stream.close()
                self.stream = None
                if self.tensorboard is not None:
                    self.tensorboard.close()
                raise
        self.publish()
        if self.status["mode"] == "online":
            self.monitor = threading.Thread(target=self._monitor, daemon=True)
            self.monitor.start()
        return self

    def _init(self, manifest, config, run_id, previous):
        with isolated_random_state():
            import wandb

            self.run = wandb.init(
                project=self.settings["project"],
                entity=self.settings["entity"],
                group=self.settings["group"],
                job_type=self.job_type,
                id=run_id,
                name=self.root.name,
                dir=str(self.root / "tracking"),
                mode=self.settings["mode"],
                resume="must" if previous.get("online_run_id") else None,
                reinit="create_new",
                config={
                    "experiment": config,
                    "source_commit": manifest["source"]["commit"],
                    "implementation_sha256": manifest["source"][
                        "implementation_sha256"
                    ],
                    "input_sha256": manifest["inputs"],
                    "scientific_sha256": manifest.get("scientific_sha256"),
                    "seed": manifest.get("seed"),
                    "checkpoint_sha256": manifest.get("checkpoint_sha256"),
                    "experiment_id": self.status["experiment_id"],
                },
                settings=wandb.Settings(
                    init_timeout=30,
                    finish_timeout=15,
                    finish_timeout_raises=True,
                    save_code=False,
                    disable_git=True,
                    console="off",
                ),
            )
            self.run.define_metric("event_sequence")
            self.run.define_metric("*", step_metric="event_sequence")
            self.run.define_metric("train/*", step_metric="train/physical_ticks")
            self.run.define_metric(
                "validation/*", step_metric="validation/training_progress/episodes"
            )
            self.run.define_metric("evaluation/*", step_metric="evaluation/case_index")
        online = self.settings["mode"] == "online"
        self.status.update(
            sync_status="pending" if online else "offline",
            online_run_id=run_id if online else self.status.get("online_run_id"),
            url=self.run.url if online else None,
            entity=self.run.entity,
            project=self.run.project,
            sync_directory=str(self.run.sync_dir),
            sync_command="wandb sync " + repr(str(self.run.sync_dir)),
        )
        if online:
            self.status["online_identity"] = {
                "project": self.run.project,
                "entity": self.run.entity,
            }
        self.segments.append(
            {
                "run_id": run_id,
                "mode": self.settings["mode"],
                "sync_directory": self.status["sync_directory"],
                "sync_command": self.status["sync_command"],
            }
        )
        manifest = dict(manifest)
        manifest["tracking"] = {
            key: self.status[key]
            for key in ("experiment_id", "project", "entity", "group")
        }
        write_json(self.root / "manifest.json", manifest)
        if online:
            print(f"W&B: {self.status['url']}", flush=True)

    def _monitor(self):
        previous = None
        while not self.stop.wait(10):
            try:
                # This SDK call may wait for its local service, so it stays off
                # the training thread. The monitor is a bounded-lifetime daemon.
                state = self.run.status()
                if self.stop.is_set():
                    return
                marker = state.sync_time
                if marker != previous or state.sync_items_pending == 0:
                    self.last_sync_progress = time.monotonic()
                previous = marker
                delayed = time.monotonic() - self.last_sync_progress >= 60
                if delayed and self.status.get("sync_status") != "sync_delayed":
                    warnings.warn(
                        "W&B synchronization is delayed; local training continues.",
                        stacklevel=2,
                    )
                self.status.update(
                    sync_items_pending=state.sync_items_pending,
                    sync_status="sync_delayed"
                    if delayed
                    else ("pending" if state.sync_items_pending else "synced"),
                )
            except Exception as error:
                if self.stop.is_set():
                    return
                self.status.update(
                    sync_status="sync_unknown", sync_error=type(error).__name__
                )

    def publish(self):
        write_json(self.state_path, self.status)

    def log(self, event):
        self.sequence += 1
        import psutil

        try:
            process = psutil.Process()
            event = {
                **event,
                "resources": {
                    "rss_bytes": process.memory_info().rss,
                    "cpu_seconds": sum(process.cpu_times()[:2]),
                },
            }
        except psutil.Error:
            pass
        row = {**event, "event_sequence": self.sequence}
        self.stream.write(json.dumps(row, allow_nan=False, default=str) + "\n")
        self.stream.flush()
        self.status["sequence"] = self.sequence
        phase = row.get("phase", "progress")
        series = phase
        if phase in {"validation", "evaluation"}:
            series += (
                "/"
                + str(row.get("method", "configured"))
                + "/"
                + str(row.get("selection", "sample"))
            )
        values = {
            "event_sequence": self.sequence,
            **numeric(
                {
                    k: v
                    for k, v in row.items()
                    if k not in {"tracking", "event_sequence"}
                },
                series + "/",
            ),
        }
        if phase == "validation":
            values["validation/training_progress/episodes"] = (
                row.get("training_progress") or {}
            ).get("episodes", 0)
        elif phase == "evaluation":
            values["evaluation/case_index"] = row.get("case_index", self.sequence)
        if self.tensorboard is not None:
            try:
                step = (
                    row.get("physical_ticks", self.sequence)
                    if phase == "train"
                    else (row.get("training_progress") or {}).get(
                        "episodes", row.get("case_index", self.sequence)
                    )
                )
                with isolated_random_state():
                    self.tensorboard.log(values, step)
            except Exception as error:
                self.status.update(
                    tensorboard="write_error", tensorboard_error=type(error).__name__
                )
                warnings.warn(
                    "TensorBoard write failed; local event journal is intact",
                    stacklevel=2,
                )
        if self.run is not None:
            try:
                with isolated_random_state():
                    self.run.log(values)
            except Exception as error:
                self.status.update(
                    sync_status="upload_error", sync_error=type(error).__name__
                )
                warnings.warn(
                    "W&B upload failed; training and local recording continue.",
                    stacklevel=2,
                )
        self.publish()
        with isolated_random_state():
            from smartsom.telemetry.runtime import emit

            emit(
                self.root.name,
                {**row, "tracking": dict(self.status)},
                total=row.get("tick_budget"),
                unit="physical ticks",
            )

    def close(self, *, failed=False):
        if self.closed:
            return
        self.closed = True
        self.stop.set()
        self.status["training_status"] = (
            "failed_or_interrupted" if failed else "finished"
        )
        if self.run is not None:
            try:
                with isolated_random_state():
                    self.run.finish(exit_code=int(failed))
                self.status["sync_status"] = (
                    "synced" if self.status["mode"] == "online" else "offline"
                )
            except Exception as error:
                self.status.update(
                    sync_status="incomplete", sync_error=type(error).__name__
                )
                warnings.warn(
                    "W&B sync incomplete; local checkpoints are intact. "
                    + self.status.get("sync_command", ""),
                    stacklevel=2,
                )
        if self.tensorboard is not None:
            try:
                with isolated_random_state():
                    self.tensorboard.close()
            except Exception as error:
                self.status.update(
                    tensorboard="close_error", tensorboard_error=type(error).__name__
                )
        if self.stream is not None:
            self.stream.close()
            self.publish()
            progress_path = self.root / "progress.json"
            if progress_path.exists():
                progress = json.loads(progress_path.read_text())
                progress["tracking"] = dict(self.status)
                write_json(progress_path, progress)
