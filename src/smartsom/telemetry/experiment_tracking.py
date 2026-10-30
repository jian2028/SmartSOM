"""One local metric journal and optional W&B sink for native experiments."""

import json
import math
import os
import shlex
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
        "tags": list(options.tags),
        "variant": options.variant,
        "upload_interval": options.upload_interval,
    }


def offline_choice(error, *, interactive=None):
    warnings.warn(f"W&B startup failed: {type(error).__name__}: {error}", stacklevel=2)
    if interactive is None and os.environ.get("SMARTSOM_FOREGROUND_PROMPTS"):
        from smartsom.experiments.foreground import offline_choice as terminal_choice

        if terminal_choice(error):
            return "offline"
        interactive = False
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


CLOUD_SCALARS = frozenset(
    """
physical_ticks committed_physical_ticks inflight_episode_tick tick tick_budget episode_budget
physical_tick_budget episodes episode updates wall_seconds deadline_unix completed total_orders
unfinished mean_tardiness passing_rate final_passing_rate joint_cost observed_return observed_overdue
failed decisions learner_update_seconds ppo_update_seconds reward case_index cases total last_arrival
order_delivery_fraction case_drain_rate qualified_delivery_coverage throughput_per_1000_ticks
qualified_throughput_per_1000_ticks common_window_throughput common_window_qualified_throughput
common_window_end_tick tardiness_p50 tardiness_p95 tardiness_p99 mean_flow_time on_time_coverage
conditional_on_time_rate actual_seeds planned_seeds sample_count stddev mean robustness sig car
""".split()
)


def cloud_fields(row):
    """Bound cardinality: entity IDs, traces and arbitrary nested evidence stay local."""
    result = {key: value for key, value in row.items() if key in CLOUD_SCALARS}
    for key in ("resources", "training_progress", "losses", "diagnostics"):
        if isinstance(row.get(key), dict):
            result[key] = row[key]
    for key in ("summary", "metrics"):
        if isinstance(row.get(key), dict):
            result[key] = {k: v for k, v in row[key].items() if k in CLOUD_SCALARS}
    return result


def representative_case(row):
    identity = row.get("evaluation_identity", {})
    world = identity.get("world_index", row.get("world_index"))
    if world is not None:
        return (
            world == 0
            and row.get("action_seed", 0) == 0
            and row.get("condition") == "normal"
        )
    return row.get("case_index") == 0


def entity_values(kind, values):
    keys = {
        "machines": ("processing", "idle", "blocked", "down"),
        "vehicles": (
            "loaded",
            "empty",
            "moving",
            "stationary",
            "service_at_start",
            "no_service_at_start",
        ),
        "buffers": ("resident_ticks", "observed_ticks"),
    }[kind]
    result = {key: values.get(key, 0) for key in keys}
    if kind == "machines":
        elapsed = sum(
            values.get(key, 0) for key in ("processing", "blocked", "idle", "down")
        )
        result["observed_ticks"] = elapsed
        result["available_ticks"] = elapsed - values.get("down", 0)
    elif kind == "vehicles":
        result["observed_ticks"] = values.get("loaded", 0) + values.get("empty", 0)
    return result


class ExperimentTracking:
    """Tracking failures after startup never invalidate a local training commit."""

    def __init__(self, root, options, *, job_type, group=None):
        self.root, self.settings = Path(root), resolved(options)
        if self.settings["mode"] not in {"online", "offline", "disabled"}:
            raise ValueError("tracking mode must be online, offline or disabled")
        self.job_type = job_type
        self.settings["group"] = self.settings["group"] or group
        self.last_upload = float("-inf")
        self.pending_values = None
        self.last_phase = None
        self.summary_rows = {}
        self.case_rows = {}
        self.entity_rows = {}
        self.entity_series_rows = {}
        self.table_epoch = None
        self.tables = {}
        self.pending_tables = {}
        self.run = self.stream = self.tensorboard = None
        self.status = {"mode": self.settings["mode"], "sync_status": "not_started"}
        self.closed = False
        self.sequence = 0
        self.stop = threading.Event()
        self.monitor = None
        self.sync_lock = threading.Lock()
        self.last_sync_progress = time.monotonic()

    def start(self, manifest, config):
        tracking = self.root / "tracking"
        tracking.mkdir(exist_ok=True)
        self.state_path = tracking / "status.json"
        previous = (
            json.loads(self.state_path.read_text()) if self.state_path.exists() else {}
        )
        from smartsom.telemetry.tracking_metadata import launch_group, metadata

        self.settings["group"] = (
            previous.get("group")
            or self.settings["group"]
            or manifest.get("tracking_group")
            or launch_group(self.root.name, manifest)
        )
        self.labels = metadata(
            self.root,
            config,
            seed=manifest.get("training_seed", manifest.get("seed")),
            variant=self.settings["variant"],
            tags=self.settings["tags"],
            short_name=manifest.get("experiment_short_name"),
        )
        self.training_seed = manifest.get("training_seed", manifest.get("seed"))
        self.planned_seeds = len(config.get("train_seeds", [])) or 1
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
            name=self.labels["name"],
            tags=self.labels["tags"],
            comparison=self.labels["comparison"],
            segments=self.segments,
            online_identity=previous.get("online_identity"),
            upload_errors=previous.get("upload_errors", 0),
            last_upload_error=previous.get("last_upload_error"),
            last_failed_event_sequence=previous.get("last_failed_event_sequence"),
        )
        journal = tracking / "events.jsonl"
        if journal.exists():
            for line in journal.read_text().splitlines():
                self._remember_tables(json.loads(line))
        self.stream = journal.open("a", encoding="utf-8")
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
        labels = self.labels
        with isolated_random_state():
            import wandb

            self.run = wandb.init(
                project=self.settings["project"],
                entity=self.settings["entity"],
                group=self.settings["group"],
                job_type=self.job_type,
                id=run_id,
                name=labels["name"],
                tags=labels["tags"],
                dir=str(self.root / "tracking"),
                mode=self.settings["mode"],
                resume="must" if previous.get("online_run_id") else None,
                reinit="create_new",
                config={
                    "experiment": config,
                    "resolved_inputs": labels["documents"],
                    "comparison": labels["comparison"],
                    "source_commit": manifest["source"]["commit"],
                    "implementation_sha256": manifest["source"][
                        "implementation_sha256"
                    ],
                    "input_sha256": manifest["inputs"],
                    "scientific_sha256": manifest.get("scientific_sha256"),
                    "seed": self.training_seed,
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
            sync_command="wandb sync "
            + ("--no-skip-online " if self.settings["mode"] == "online" else "")
            + shlex.quote(str(self.run.sync_dir)),
        )
        self._sync_status("pending" if online else "offline")
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
                self._sync_status(
                    "sync_delayed"
                    if delayed
                    else ("pending" if state.sync_items_pending else "synced"),
                    sync_items_pending=state.sync_items_pending,
                )
            except Exception as error:
                if self.stop.is_set():
                    return
                self._sync_status("sync_unknown", sync_error=type(error).__name__)

    def _sync_status(self, status, **details):
        # SDK queue/finish success cannot recover an event rejected by log().
        # Keep the loss visible across polling, later events and continuation.
        with self.sync_lock:
            self.status.update(details)
            self.status["sync_status"] = (
                "upload_error" if self.status.get("upload_errors") else status
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
        self.pending_tables = {}
        self._remember_tables(row)
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
                cloud_fields(row),
                series + "/",
            ),
        }
        if row.get("kind") == "summary":
            for method, conditions in row.get("methods", {}).items():
                for condition, summary in conditions.items():
                    prefix = f"{phase}/{method}/{row.get('selection', 'sample')}/{condition}/"
                    values.update(numeric(summary.get("metrics", {}), prefix))
                    values.update(
                        numeric(
                            {
                                k: summary[k]
                                for k in ("actual_cases", "planned_cases")
                                if k in summary
                            },
                            prefix,
                        )
                    )
                    if "complete_inventory" in summary:
                        values[prefix + "complete_inventory"] = int(
                            summary["complete_inventory"]
                        )
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
        immediate = (
            phase != "train"
            or phase != self.last_phase
            or row.get("kind", row.get("event_kind")) in {"summary", "stage", "final"}
            or row.get("status") in {"completed", "stopped", "paused", "failed"}
        )
        if self.run is not None and phase in {"validation", "evaluation"}:
            try:
                with isolated_random_state():
                    values.update(self._table_values(row))
            except Exception as error:
                self._sync_status(
                    "upload_error",
                    table_error=type(error).__name__,
                    upload_errors=self.status["upload_errors"] + 1,
                    last_failed_event_sequence=self.sequence,
                )
                warnings.warn(
                    "W&B table upload preparation failed; local evidence is intact",
                    stacklevel=2,
                )
        self.pending_values = {**(self.pending_values or {}), **values}
        if (
            immediate
            or time.monotonic() - self.last_upload >= self.settings["upload_interval"]
        ):
            self._flush_upload()
        self.last_phase = phase
        self.publish()
        if phase != "train":
            return
        with isolated_random_state():
            from smartsom.telemetry.current_progress import presentation
            from smartsom.telemetry.runtime import emit

            shown, total, unit = presentation({**row, "tracking": dict(self.status)})
            emit(
                self.root,
                shown,
                total=total,
                unit=unit,
            )

    def _remember_tables(self, row):
        phase = row.get("phase")
        if phase not in {"validation", "evaluation"}:
            return
        from smartsom.telemetry.evaluation_summary import case_values, hierarchical

        episode = (row.get("training_progress") or {}).get("episodes", 0)
        mode = row.get("selection", "sample")
        epoch = (phase, episode, mode)
        if epoch != self.table_epoch:
            self.case_rows = {}
            self.entity_rows = {}
            self.entity_series_rows = {}
            self.table_epoch = epoch
        if "scenario_seed" in row and "total_orders" in row:
            key = (
                row.get("method"),
                row["scenario_seed"],
                row.get("condition"),
                row.get("action_seed"),
            )
            compact = {
                k: v
                for k, v in row.items()
                if k in CLOUD_SCALARS
                or k
                in {
                    "status",
                    "termination_reason",
                    "condition",
                    "scenario_seed",
                    "action_seed",
                    "method",
                    "evaluation_identity",
                }
            }
            compact["research_metrics"] = {"summary": case_values(row)}
            self.case_rows[key] = compact
            detail = row.get("research_metrics", {})
            context = {
                "case": row.get("case_index"),
                "method": row.get("method"),
                "series": f"{row.get('method')}/{self.labels['comparison']['variant']} [{self.labels['comparison']['method_identity'][:8]}]",
                "condition": row.get("condition"),
                "world": str(row["scenario_seed"]),
                "episode": episode,
                "selection": mode,
                "seed": str(self.training_seed),
                "run": self.status["run_id"],
                "revision": self.sequence,
            }
            for kind in ("machines", "vehicles", "buffers"):
                source = "buffer_entities" if kind == "buffers" else kind
                for entity, values in detail.get(source, {}).items():
                    self.entity_rows[(key, kind, entity)] = {
                        **context,
                        "kind": kind,
                        "entity": entity,
                        **entity_values(kind, values),
                    }
            # Sequence upload is intentionally limited to the first recorded case.
            if representative_case(row):
                self.status.setdefault("entity_series_cases", {}).setdefault(mode, {})[
                    str(row.get("method"))
                ] = context
                for point in detail.get("entity_series", []):
                    for kind in ("machines", "vehicles", "buffers"):
                        for entity, values in point.get(kind, {}).items():
                            self.entity_series_rows[
                                (key, kind, entity, point["tick"])
                            ] = {
                                **context,
                                "kind": kind,
                                "entity": entity,
                                "tick": point["tick"],
                                **entity_values(kind, values),
                            }
            new_entities = [
                value
                for (identity, _, _), value in self.entity_rows.items()
                if identity == key
            ]
            self.pending_tables[f"research/{mode}/entities"] = new_entities
            if representative_case(row):
                self.pending_tables[f"research/{mode}/entity_series"] = list(
                    self.entity_series_rows.values()
                )
        if row.get("kind") != "summary":
            return
        for method, conditions in row.get("methods", {}).items():
            for condition, summary in conditions.items():
                self._summary_row(
                    phase, episode, mode, method, condition, "all", summary
                )
                cases = [
                    case
                    for case in self.case_rows.values()
                    if case.get("method") == method
                    and (condition == "all" or case.get("condition") == condition)
                ]
                for world in sorted({case["scenario_seed"] for case in cases}):
                    selected = [
                        case for case in cases if case["scenario_seed"] == world
                    ]
                    detail = hierarchical(selected)
                    detail["complete_inventory"] = summary.get(
                        "complete_inventory", False
                    )
                    self._summary_row(
                        phase, episode, mode, method, condition, str(world), detail
                    )

    def _panel_identity(self, method, condition, world):
        import hashlib

        cases = []
        for row in self.case_rows.values():
            if (
                row.get("method") != method
                or (condition != "all" and row["condition"] != condition)
                or (world != "all" and str(row["scenario_seed"]) != world)
            ):
                continue
            identity = row.get("evaluation_identity", {})
            comparison = identity.get("sl_comparison", {})
            progress = identity.get("training_progress") or {}
            if (
                comparison.get("status") != "supported"
                or not all(
                    identity.get(key)
                    for key in ("world_sha256", "source_sha256", "runtime_sha256")
                )
                or any(progress.get(key) is None for key in ("train_seed", "episodes"))
            ):
                return None
            cases.append(
                (
                    str(row["scenario_seed"]),
                    row["condition"],
                    str(row.get("action_seed")),
                    comparison["non_sl_sha256"],
                    identity["world_sha256"],
                    identity["source_sha256"],
                    identity["runtime_sha256"],
                    progress["train_seed"],
                    progress["episodes"],
                )
            )
        return (
            hashlib.sha256(json.dumps(sorted(cases)).encode()).hexdigest()
            if cases
            else None
        )

    def _summary_row(self, phase, episode, mode, method, condition, world, summary):
        from smartsom.telemetry.tracking_metadata import method_identity

        reference = method in {"rule", "random"}
        variant = method if reference else self.labels["comparison"]["variant"]
        key = (phase, episode, mode, method, condition, world)
        self.summary_rows[key] = {
            "phase": phase,
            "episode": episode,
            "selection": mode,
            "method": method,
            "variant": variant,
            "series": f"{method}/{variant} [{method_identity(self.labels['comparison'], method)[:8]}]",
            "seed": "reference"
            if reference
            else (str(self.training_seed) if self.training_seed is not None else None),
            "observation": method
            if reference
            else self.labels["comparison"]["observation"],
            "panel_id": self._panel_identity(method, condition, world),
            "world": world,
            "condition": condition,
            "planned_seeds": 1 if reference else self.planned_seeds,
            "actual_cases": summary.get("actual_cases"),
            "planned_cases": summary.get("planned_cases"),
            "complete_inventory": bool(summary.get("complete_inventory")),
            "revision": self.sequence,
            **summary.get("metrics", {}),
        }
        self.pending_tables.setdefault("research/summary_history", []).append(
            self.summary_rows[key]
        )

    def _table_values(self, row):
        import wandb

        result = {}
        for name, rows in self.pending_tables.items():
            if not rows:
                continue
            columns = sorted({key for item in rows for key in item})
            if "/entities" in name or "/entity_series" in name:
                columns = sorted(
                    set(columns)
                    | {
                        "processing",
                        "idle",
                        "blocked",
                        "down",
                        "loaded",
                        "empty",
                        "moving",
                        "stationary",
                        "service_at_start",
                        "no_service_at_start",
                        "resident_ticks",
                        "observed_ticks",
                        "available_ticks",
                    }
                )
            table = self.tables.get(name)
            if table is None:
                table = wandb.Table(columns=columns, log_mode="INCREMENTAL")
                self.tables[name] = table
            if set(columns) != set(table.columns):
                raise ValueError(f"tracking table schema changed: {name}")
            for item in rows:
                table.add_data(*(item.get(column) for column in table.columns))
            result[name] = table
        return result

    def _flush_upload(self):
        values = self.pending_values
        if values is None:
            return
        self.pending_values = None
        self.last_upload = time.monotonic()
        if self.run is not None:
            try:
                with isolated_random_state():
                    self.run.log(values)
            except Exception as error:
                self._sync_status(
                    "upload_error",
                    sync_error=type(error).__name__,
                    upload_errors=self.status["upload_errors"] + 1,
                    last_upload_error=type(error).__name__,
                    last_failed_event_sequence=self.sequence,
                )
                warnings.warn(
                    "W&B upload failed; training and local recording continue.",
                    stacklevel=2,
                )

    def close(self, *, failed=False):
        if self.closed:
            return
        self._flush_upload()
        self.closed = True
        self.stop.set()
        self.status["training_status"] = (
            "failed_or_interrupted" if failed else "finished"
        )
        if self.run is not None:
            try:
                with isolated_random_state():
                    self.run.finish(exit_code=int(failed))
                self._sync_status(
                    "synced" if self.status["mode"] == "online" else "offline"
                )
            except Exception as error:
                self._sync_status(
                    "incomplete",
                    sync_error=type(error).__name__,
                    finish_error=type(error).__name__,
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
