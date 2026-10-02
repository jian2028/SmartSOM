"""Presentation-only runtime sessions shared by CLI, API and read-only monitors."""

import inspect
import json
import logging
import math
import os
import re
import sys
import time
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass
from functools import wraps
from pathlib import Path

from rich import box
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn
from rich.table import Column, Table
from rich.text import Text

SCHEMA = "smartsom.runtime-progress/v1"
FINAL = {
    "recommended",
    "completed",
    "failed",
    "interrupted",
    "early_stopped",
    "pruned",
    "truncated",
    "finished_with_failures",
    "completed_with_failures",
    "ineligible",
    "incomplete",
    "stopped",
    "force_stopped",
}
FAILURES = {
    "failed",
    "interrupted",
    "finished_with_failures",
    "completed_with_failures",
    "ineligible",
}
CURRENT = ContextVar("smartsom_display", default=None)
OVERRIDES = ContextVar("smartsom_display_options", default={})
QUIET = ContextVar("smartsom_worker_display", default=False)
LABELS = {
    "sampled_steps": "Sampling decisions",
    "decisions": "Decisions",
    "decision_limit": "Decision limit",
    "physical_ticks": "Physical ticks",
    "agent_steps": "Agent steps",
    "physical_actions": "Physical actions",
    "ppo_updates": "PPO updates",
    "updates": "Update boundaries",
    "optimization_steps": "Optimizer steps (all groups)",
    "algorithm": "Algorithm",
    "runtime_mode": "Execution",
    "group_statistics": "Groups (decisions / samples / optimizer steps)",
    "validation_finished": "Validation ended episodes",
    "validation_requested": "Validation requested episodes",
    "validation_tick": "Validation case tick",
    "validation_tick_limit": "Validation case tick limit",
    "evaluation_tick": "Evaluation case tick",
    "evaluation_tick_limit": "Evaluation case tick limit",
    "evaluation_kind": "Evaluation kind",
    "bindings_summary": "Policy groups",
    "training_run_directory": "Training directory",
    "validation_case_active": "Validation case active",
    "evaluation_case_active": "Evaluation case active",
    "study_case": "Study condition",
    "workflow": "Workflow configuration",
    "validation_batches_finished": "Validation batches finished",
    "validation_round": "Current validation batch",
    "planned_work_completed": "Resolved planned work",
    "planned_work_total": "Total planned work",
    "learner_updates": "Learner updates",
    "completed_episodes": "Completed episodes",
    "failed_episodes": "Failed episodes",
    "qualified_demands": "Run qualified deliveries",
    "training_deliveries": "Training cumulative deliveries",
    "episode_return": "Latest episode return",
    "episode_makespan": "Latest episode makespan",
    "episode_deliveries": "Latest episode deliveries",
    "evaluation_completed": "Evaluation successful episodes",
    "evaluation_finished": "Evaluation finished episodes",
    "evaluation_requested": "Evaluation requested episodes",
}


@dataclass(frozen=True)
class DisplayOptions:
    title: str | None = None
    task_title: str | None = None
    verbose: bool = True
    debug: bool = False
    progress: str = "auto"
    format: str = "text"
    every_seconds: float = 30.0

    @classmethod
    def from_value(cls, value=None, **overrides):
        fields = cls.__dataclass_fields__
        if value is None:
            data = {}
        elif isinstance(value, dict):
            data = {k: v for k, v in value.items() if k in fields}
        else:
            data = {k: getattr(value, k) for k in fields if hasattr(value, k)}
        data.update(
            {k: v for k, v in overrides.items() if k in fields and v is not None}
        )
        result = cls(**data)
        from smartsom.telemetry.workflow import validate_title_template

        validate_title_template(result.task_title)
        if result.title is not None and (
            not result.title.strip() or any(c in result.title for c in "\n\r\t")
        ):
            raise ValueError("progress title must be nonempty single-line text")
        if result.every_seconds <= 0 or not math.isfinite(result.every_seconds):
            raise ValueError("summary interval must be positive and finite")
        if result.progress not in {"auto", "on", "off"} or result.format not in {
            "text",
            "json",
        }:
            raise ValueError("invalid display mode")
        return result


@contextmanager
def display_options(**options):
    """Session-only overrides; never rewrite a frozen experiment configuration."""
    token = OVERRIDES.set(
        {**OVERRIDES.get(), **{k: v for k, v in options.items() if v is not None}}
    )
    try:
        yield
    finally:
        OVERRIDES.reset(token)


def _options(arguments):
    config = arguments.get("config") or arguments.get("options")
    prepared = arguments.get("prepared")
    if prepared is not None and hasattr(prepared, "config_json"):
        config = json.loads(prepared.config_json).get("logging", {})
    if isinstance(config, (str, Path)):
        from smartsom.config.experiment import load_config

        config = load_config(config)
    if hasattr(config, "logging"):
        config = config.logging
    if config is None and arguments.get("configs"):
        config = arguments["configs"][0].logging
    if config is None:
        source = (
            arguments.get("source")
            or arguments.get("resume")
            or arguments.get("root")
            or arguments.get("directory")
        )
        if isinstance(source, (str, Path)):
            path = Path(source)
            for root in (path, *path.parents):
                saved = root / "config/experiment.json"
                if saved.is_file():
                    config = json.loads(saved.read_text()).get("logging", {})
                    break
                study_plan = root / "plan.json"
                if study_plan.is_file():
                    study = json.loads(study_plan.read_text())
                    if "recipe" in study:
                        config = study["recipe"].get("logging", {})
                        break
                plan_path = root / "config/plan.json"
                if plan_path.is_file():
                    plan = json.loads(plan_path.read_text())
                    base = plan.get("base")
                    if base is None and plan.get("trials"):
                        configs = plan["trials"][0].get("configs", [])
                        base = configs[0].get("config", {}) if configs else {}
                    config = (base or {}).get("logging", {})
                    break
    options = DisplayOptions.from_value(
        config, verbose=arguments.get("verbose"), debug=arguments.get("debug")
    )
    return DisplayOptions.from_value(options, **OVERRIDES.get())


def operation(kind):
    """Give the outermost runtime call ownership; nested calls borrow its session."""

    def decorate(function):
        signature = inspect.signature(function)

        @wraps(function)
        def wrapped(*args, **kwargs):
            override = kwargs.pop("display_options", None)
            with display_options(**(override or {})):
                arguments = signature.bind(*args, **kwargs).arguments
                current = CURRENT.get()
                owner = current is None
                from smartsom.experiments.control import CURRENT as CONTROL
                from smartsom.experiments.control import Scope

                control_owner = CONTROL.get() is None
                scope = Scope() if control_owner else CONTROL.get()
                control_token = CONTROL.set(scope)
                session = current or RuntimeDisplay(
                    _options(arguments), kind=kind, quiet=QUIET.get()
                )
                token = CURRENT.set(session)
                optuna_logger = logging.getLogger("optuna")
                previous_level = optuna_logger.level
                if owner and not session.options.debug:
                    optuna_logger.setLevel(logging.WARNING)
                try:
                    if owner:
                        session.start()
                    config = arguments.get("config")
                    if getattr(
                        getattr(config, "logging", None), "legacy_verbose", False
                    ) or OVERRIDES.get().get("legacy_verbose"):
                        session.legacy_warning()
                    if kind in {"training", "evaluation"}:
                        session.phase(kind)
                    result = function(*args, **kwargs)
                    if owner:
                        status = getattr(result, "status", None)
                        if status is None and isinstance(result, dict):
                            status = result.get("status")
                        if status is None and hasattr(result, "training"):
                            status = getattr(
                                result.evaluation or result.training, "status", None
                            )
                        if status is None and hasattr(result, "interrupted"):
                            status = (
                                "interrupted"
                                if result.interrupted
                                else "finished_with_failures"
                                if result.failed or result.pending
                                else "completed"
                            )
                        session.finish(
                            status or session.status_from_manifest() or "completed",
                            error=result.get("error")
                            if isinstance(result, dict)
                            else None,
                        )
                    return result
                except BaseException as exc:
                    if owner:
                        try:
                            session.finish(
                                "interrupted"
                                if isinstance(exc, KeyboardInterrupt)
                                else "failed",
                                error=str(exc),
                            )
                        except Exception as failure:
                            exc.add_note(f"display cleanup also failed: {failure}")
                    raise
                finally:
                    if control_owner:
                        scope.finish(session.status)
                    CONTROL.reset(control_token)
                    CURRENT.reset(token)
                    if owner:
                        optuna_logger.setLevel(previous_level)
                    if owner:
                        session.close()

        return wrapped

    return decorate


def worker_output(attempt_argument):
    """Workers own evidence files; only their coordinator owns the terminal."""

    def decorate(function):
        signature = inspect.signature(function)

        @wraps(function)
        def wrapped(*args, **kwargs):
            attempt = Path(signature.bind(*args, **kwargs).arguments[attempt_argument])
            attempt.mkdir(parents=True, exist_ok=True)
            quiet_token = QUIET.set(True)
            session_token = CURRENT.set(None)
            try:
                with (attempt / "worker.log").open("a", buffering=1) as stream:
                    with redirect_stdout(stream), redirect_stderr(stream):
                        return function(*args, **kwargs)
            finally:
                CURRENT.reset(session_token)
                QUIET.reset(quiet_token)

        return wrapped

    return decorate


class _PlainDiagnosticStream:
    """Strip terminal controls from framework diagnostics retained in files."""

    def __init__(self, stream, session=None):
        self.stream = stream
        self.session = session
        self.pending = ""

    def __getattr__(self, name):
        # Frameworks inspect standard text-stream attributes such as encoding.
        return getattr(self.stream, name)

    def write(self, value):
        clean = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", value).replace("\r", "")
        self.stream.write(clean)
        if self.session and self.session.options.debug and not self.session.quiet:
            self.pending += clean
            while "\n" in self.pending:
                line, self.pending = self.pending.split("\n", 1)
                if line:
                    message = (
                        json.dumps({"type": "backend", "message": line})
                        if self.session.options.format == "json"
                        else line
                    )
                    self.session.console.print(Text(message), soft_wrap=True)
        return len(value)

    def flush(self):
        self.stream.flush()

    def isatty(self):
        return False


@contextmanager
def backend_diagnostics(session=None):
    """Keep framework chatter in a file; the shared Console retains its own stream."""
    session = session or CURRENT.get()
    if session is None or session.root is None:
        yield
        return
    with (session.root / "logs/backend.log").open("a", buffering=1) as file:
        stream = _PlainDiagnosticStream(file, session)
        restored = []
        # Already imported libraries may hold their original stderr stream.
        loggers = [logging.getLogger(), *logging.Logger.manager.loggerDict.values()]
        for logger in loggers:
            if isinstance(logger, logging.Logger):
                for handler in logger.handlers:
                    if isinstance(
                        handler, logging.StreamHandler
                    ) and handler.stream in (sys.stdout, sys.stderr):
                        restored.append((handler, handler.stream))
                        handler.setStream(stream)
        try:
            with redirect_stdout(stream), redirect_stderr(stream):
                yield
        finally:
            for handler, original in restored:
                handler.setStream(original)
            # Handlers created by a framework must not retain a closed file.
            for logger in list(logging.Logger.manager.loggerDict.values()):
                if isinstance(logger, logging.Logger):
                    for handler in logger.handlers:
                        if (
                            isinstance(handler, logging.StreamHandler)
                            and handler.stream is stream
                        ):
                            handler.setStream(session.console.file)


def bind(root, name=None):
    from smartsom.experiments.control import bind as bind_control

    bind_control(root)
    session = CURRENT.get()
    if session:
        session.bind(root, name)
    return session


def configure_workflow(prepared=None, kind=None, *, metadata=None):
    """Attach detached presentation inputs without touching execution state."""
    session = CURRENT.get()
    if session:
        from smartsom.telemetry.workflow import describe_prepared

        session.configure(
            metadata
            if metadata is not None
            else describe_prepared(prepared, kind or session.kind)
        )


def emit(task, event, **kwargs):
    session = CURRENT.get()
    if session:
        session.update(str(task), event, **kwargs)


def shown(value):
    if value is None:
        return "N/A"
    return (
        f"{value:,}"
        if isinstance(value, int)
        else f"{value:.6g}"
        if isinstance(value, float)
        else str(value)
    )


class RuntimeDisplay:
    def __init__(
        self, options=None, *, kind="run", console=None, quiet=False, readonly=False
    ):
        self.options = options or DisplayOptions()
        self.console = console or Console(
            file=sys.stderr, highlight=False, markup=False
        )
        self.kind = self.stage = kind
        self.name = kind
        self.root = None
        self.tasks = {}
        self.status = "running"
        self.quiet, self.readonly = quiet, readonly
        self.live = None
        self.last_summary = self.last_snapshot = self.last_refresh = float("-inf")
        self.updated_at = time.time()
        self.notice = None
        self.total_tasks = None
        self.overview = None
        self.workflow = None
        self.tuning = None
        self.preflight = None
        self.controlling = False
        self.workflow_work = None
        self._legacy_warned = False
        self._batch_depth = 0
        self._pending_publish = None
        self._last_frame_state = None

    def start(self):
        if (
            not self.quiet
            and self.options.verbose
            and self.options.format == "text"
            and self.options.progress != "off"
            and self.console.is_terminal
            and getattr(self.console.file, "isatty", lambda: False)()
        ):
            self.live = Live(
                self.render(),
                console=self.console,
                auto_refresh=False,
                screen=self.kind
                in {
                    "study",
                    "training",
                    "evaluation",
                    "train-evaluate",
                    "run",
                    "tune",
                    "batch-directory",
                },
                vertical_overflow="crop",
            )
            self.live.start()

    def configure(self, metadata):
        from smartsom.telemetry.workflow import WorkflowWork

        previous = self.workflow or {}
        if self.kind == "study" and previous:
            return
        self.workflow = {**previous, **metadata}
        if self.kind in {"train-evaluate", "run"}:
            self.workflow["mode"] = self.kind
        if self.workflow_work is None:
            self.workflow_work = WorkflowWork()
        if self.root is not None:
            self.publish(snapshot_force=True)

    def configure_tuning(self, summary):
        """Accept presentation facts without controlling trials or resources."""
        from smartsom.telemetry.tuning_dashboard import clean_summary

        self.tuning = clean_summary(summary)
        stage = self.tuning.get("stage", self.stage)
        changed_stage = stage != self.stage
        self.stage = stage
        self.updated_at = time.time()
        if self.root is not None:
            self.publish(force=changed_stage, snapshot_force=True)

    def configure_preflight(self, state):
        self.preflight = deepcopy(state)
        self.stage = "preflight" if state.get("status") == "running" else self.stage
        self.updated_at = time.time()
        if self.root is not None:
            self.publish(force=True, snapshot_force=True)

    def from_snapshot(self, snapshot):
        """Restore recorded presentation state; never load model or scheduler state."""
        tuning = snapshot.get("tuning")
        if tuning is not None:
            from smartsom.telemetry.tuning_dashboard import clean_summary

            tuning = clean_summary(tuning)
        self.name = snapshot["name"]
        self.kind = snapshot["kind"]
        self.stage = snapshot["stage"]
        self.status = snapshot["status"]
        self.tasks = {row["id"]: deepcopy(row) for row in snapshot["tasks"]}
        self.total_tasks = snapshot.get("total_tasks")
        self.overview = deepcopy(snapshot.get("overview"))
        self.workflow = deepcopy(snapshot.get("workflow"))
        self.tuning = tuning
        self.preflight = deepcopy(snapshot.get("preflight"))
        self.updated_at = snapshot["updated_at"]
        self.notice = snapshot.get("notice")
        return self

    def bind(self, root, name=None):
        if self.root is None:
            self.root = Path(root)
            self.name = name or self.root.name
            if not self.readonly:
                (self.root / "logs").mkdir(parents=True, exist_ok=True)
            self.publish(force=True)

    def phase(self, stage):
        if stage != self.stage:
            self.stage = stage
            self.updated_at = time.time()
            if self.root is not None or self.tasks:
                self.publish(force=True)

    def status_from_manifest(self):
        if self.root:
            try:
                return json.loads((self.root / "run.json").read_text()).get("status")
            except (OSError, ValueError):
                pass
        return None

    def update(self, task, event, *, total=None, unit=None, phase=None, final=False):
        row = self.tasks.setdefault(
            task,
            {"id": task, "name": Path(task).name, "status": "running", "values": {}},
        )
        if event.get("display_run") and event["display_run"] != row.get("run"):
            row.update(run=event["display_run"], values={}, learner={})
            row.pop("completed", None)
            row["started_at"] = time.time()
        row.setdefault("started_at", time.time())
        if event.get("context"):
            row["context"] = str(event["context"])
        if event.get("display_name"):
            row["name"] = event["display_name"]
        if self.root and task == str(self.root):
            row["name"] = self.name
        previous = row["status"]
        previous_stage = row.get("stage")
        stage = str(
            event.get("stage", event.get("status", row.get("stage", self.stage)))
        )
        if not final and stage in FINAL:
            stage = "finalizing"
        if stage != previous_stage:
            row["stage_started_at"] = time.time()
        row.update(stage=stage, updated_at=time.time())
        # A sampling target or learner event is never proof that artifacts are saved.
        row["status"] = (
            event.get("status", "running")
            if final or event.get("status") in {"pending", "queued"}
            else "running"
        )
        if final:
            row["status"] = event.get("status", stage)
        if event.get("reason"):
            row["reason"] = str(event["reason"])
        elif not final:
            row.pop("reason", None)
        if phase:
            row["phase"] = phase
        if total is not None:
            row["total"] = total
        if unit is not None:
            row["unit"] = unit
        if row.get("unit") == "entries" and "completed" in event:
            row["completed"] = event["completed"]
        aliases = {"tick": "physical_ticks", "qualified_demands": "qualified_demands"}
        for key, value in {**event, **event.get("display_values", {})}.items():
            key = aliases.get(key, key)
            if key in LABELS:
                row["values"][key] = value
        for key in (
            "sampled_steps",
            "qualified_demands",
            "tick",
            "physical_ticks",
            "evaluation_finished",
        ):
            if key in event and (
                key == "sampled_steps"
                or row.get("unit")
                == {
                    "qualified_demands": "qualified deliveries",
                    "tick": "physical ticks",
                    "physical_ticks": "physical ticks",
                    "evaluation_finished": "evaluation episodes",
                }.get(key)
            ):
                row["completed"] = event[key]
        # Retain real role names and metric names, never infer updates from budget.
        selected = row.setdefault("learner", {})
        for key, value in event.get("metrics", {}).items():
            if not key.startswith("__") and key.rsplit("/", 1)[-1] in {
                "loss",
                "total_loss",
                "policy_loss",
                "value_loss",
                "entropy",
                "entropy_loss",
                "approx_kl",
                "mean_kl_loss",
            }:
                selected[key] = value
        self.updated_at = time.time()
        self.publish(
            force=final and row["status"] in FAILURES and previous != row["status"],
            snapshot_force=final or stage != previous_stage,
            error=final and row["status"] in FAILURES,
        )
        if self.options.debug:
            self.diagnostic(event)

    def diagnostic(self, event):
        text = json.dumps(event, ensure_ascii=False, default=str, allow_nan=False)
        if self.root and not self.readonly:
            with (self.root / "logs/debug.jsonl").open("a") as stream:
                stream.write(text + "\n")
        if not self.quiet:
            if self.options.format == "json":
                self.console.print(
                    json.dumps({"type": "debug", "event": event}, default=str),
                    soft_wrap=True,
                )
            else:
                self.console.print(Text(text), soft_wrap=True)

    def legacy_warning(self):
        if self._legacy_warned:
            return
        self._legacy_warned = True
        message = "verbose=2 is deprecated; use --debug / logging.debug instead"
        if not self.quiet:
            self.console.print(
                Text(
                    json.dumps({"type": "warning", "message": message})
                    if self.options.format == "json"
                    else message
                ),
                soft_wrap=True,
            )

    def snapshot(self):
        return {
            "schema": SCHEMA,
            "name": self.name,
            "kind": self.kind,
            "stage": self.stage,
            "status": self.status,
            "updated_at": self.updated_at,
            "tasks": list(self.tasks.values()),
            "total_tasks": self.total_tasks,
            "notice": self.notice,
            **({"overview": self.overview} if self.overview is not None else {}),
            **({"workflow": self.workflow} if self.workflow is not None else {}),
            **({"tuning": self.tuning} if self.tuning is not None else {}),
            **({"preflight": self.preflight} if self.preflight is not None else {}),
        }

    def text_summary(self):
        rows = [f"{self.name}: {self.stage} [{self.status}]"]
        if self.tuning is not None and (
            self.kind == "tune"
            or self.kind == "batch-directory"
            and self.tuning.get("batch_training_active")
        ):
            from smartsom.telemetry.tuning_dashboard import summary_lines

            rows.extend("  " + line for line in summary_lines(self.tuning))
        if self.overview:
            from smartsom.telemetry.study_progress import duration

            rows.append(
                f"  Overall planned work: {shown(self.overview['work_completed'])}/{shown(self.overview['work_total'])}; elapsed {duration(self.overview['elapsed_seconds'])}; ETA ~ {duration(self.overview['eta_seconds'])}"
            )
        if self.total_tasks is not None:
            finished, successful = self.task_counts()
            rows.append(
                f"  Tasks ended: {finished}/{self.total_tasks}; successful: {successful}"
            )
        for row in self.tasks.values():
            budget = (
                f" {shown(row.get('completed'))}/{shown(row.get('total'))} {row.get('unit', '')}"
                if "unit" in row
                else ""
            )
            values = " ".join(
                f"{key}={shown(value)}"
                for key, value in row["values"].items()
                if key != "workflow"
            )
            rows.append(
                f"  {row['name']}: {row.get('stage', '')} [{row['status']}]{budget} {values}".rstrip()
            )
            if row.get("reason"):
                rows.append(f"    Reason: {row['reason']}")
            if len(self.tasks) == 1:
                metrics = " ".join(
                    f"{k}={shown(v)}" for k, v in sorted(row.get("learner", {}).items())
                )
                if metrics:
                    rows.append(f"    Latest learner metrics: {metrics}")
        if self.notice:
            rows.append(str(self.notice).replace("\n", " "))
        return "\n".join(rows)

    @contextmanager
    def batch_updates(self):
        """Publish a complete polling cycle, never an intermediate worker state."""
        self._batch_depth += 1
        try:
            yield
        finally:
            self._batch_depth -= 1
            if self._batch_depth == 0 and self._pending_publish is not None:
                pending = self._pending_publish
                self._pending_publish = None
                self.publish(**pending)

    def _frame_state(self):
        # Poll timestamps are evidence, not changes to the visible task state.
        state = self.snapshot()
        state.pop("updated_at")
        state["tasks"] = [
            {key: value for key, value in row.items() if key != "updated_at"}
            for row in state["tasks"]
        ]
        return self.console.size, json.dumps(state, sort_keys=True)

    def publish(self, *, force=False, snapshot_force=False, error=False):
        if self._batch_depth:
            pending = self._pending_publish or {
                "force": False,
                "snapshot_force": False,
                "error": False,
            }
            for key, value in (
                ("force", force),
                ("snapshot_force", snapshot_force),
                ("error", error),
            ):
                pending[key] |= value
            self._pending_publish = pending
            return
        now = time.monotonic()
        if (
            self.workflow_work is not None
            and self.kind not in {"study", "tune"}
            and not self.readonly
        ):
            self.overview = self.workflow_work.overview(
                self.workflow, self.tasks, status=self.status, now=now
            )
        if (
            not self.readonly
            and self.root
            and (force or snapshot_force or now - self.last_snapshot >= 1)
        ):
            self.last_snapshot = now
            path = self.root / "logs/progress.json"
            temporary = path.with_suffix(".tmp")
            temporary.write_text(
                json.dumps(self.snapshot(), ensure_ascii=False, allow_nan=False) + "\n"
            )
            os.replace(temporary, path)
        if force or now - self.last_summary >= self.options.every_seconds:
            self.last_summary = now
            summary = self.text_summary()
            if not self.readonly and self.root:
                with (self.root / "logs/runtime.log").open("a") as stream:
                    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
                    stream.write(
                        "\n".join(f"{stamp} {line}" for line in summary.splitlines())
                        + "\n"
                    )
            if not self.quiet and (
                self.options.verbose or error or self.status in FAILURES
            ):
                if self.options.format == "json":
                    self.console.print(
                        Text(json.dumps(self.snapshot(), ensure_ascii=False)),
                        soft_wrap=True,
                    )
                elif not self.live:
                    self.console.print(Text(summary), soft_wrap=True)
        refresh_interval = (
            1.0 if self.workflow or self.kind in {"study", "tune"} else 0.25
        )
        if self.live and (force or now - self.last_refresh >= refresh_interval):
            state = self._frame_state()
            if state != self._last_frame_state:
                self.last_refresh = now
                self._last_frame_state = state
                self.live.update(self.render(), refresh=True)

    def task_counts(self):
        return (
            sum(row["status"] in FINAL for row in self.tasks.values()),
            sum(row["status"] == "completed" for row in self.tasks.values()),
        )

    def render(self):
        if self.preflight and self.preflight.get("status") == "running":
            from smartsom.telemetry.timeline import render as render_timeline

            return Panel(
                Group(
                    render_timeline(self),
                    Text(f"当前 {self.preflight.get('current') or '编译和检查输入'}"),
                    Text(
                        (
                            "p 切换烟测范围 · d 离开界面 · Ctrl+C×2 安全停止"
                            if self.controlling
                            else "d 离开界面 · Ctrl+C×2 关闭监控"
                        ),
                        style="dim",
                    ),
                ),
                title="SmartSOM · 预检",
                border_style="#b39aff",
            )
        if self.kind == "tune" or (
            self.kind == "batch-directory"
            and self.tuning is not None
            and self.tuning.get("batch_training_active")
        ):
            from smartsom.telemetry.tuning_dashboard import render

            return render(self)
        if self.workflow is not None:
            from smartsom.telemetry.dashboard import render

            return render(self)
        if self.kind == "study":
            return self._render_study()
        rows = list(self.tasks.values())
        active = [r for r in rows if r["status"] not in FINAL | {"pending", "queued"}]
        if len(rows) == 1 and len(active) == 1 and self.total_tasks is None:
            return self._render_detailed()
        evaluation = self.tasks.get("evaluation")
        items = [r for r in rows if r["id"] != "evaluation"]
        running = [r for r in active if r["id"] != "evaluation"]
        ended = sorted(
            [r for r in items if r["status"] in FINAL],
            key=lambda r: r.get("updated_at", 0),
            reverse=True,
        )
        waiting = sum(r["status"] in {"pending", "queued"} for r in items)
        if self.total_tasks is not None:
            waiting += max(0, self.total_tasks - len(items))
        width, height = self.console.width, self.console.height
        count = min(len(running), max(1, height // 3))
        recent = min(3, len(ended))
        units = {
            "sampling decisions": "sample dec",
            "environment steps": "env steps",
            "physical ticks": "ticks",
            "evaluation episodes": "eval episodes",
            "qualified deliveries": "deliveries",
            "adapter decisions": "adapter dec",
        }

        def budget(row):
            unit = units.get(row.get("unit"), row.get("unit", "progress"))
            result = f"{unit}: {shown(row.get('completed'))}/{shown(row.get('total'))}"
            values = row["values"]
            if "decisions" in values:
                result += f" · decisions {shown(values['decisions'])}/{shown(values.get('decision_limit'))}"
            if "ppo_updates" in values:
                result += f" · PPO {shown(values['ppo_updates'])}"
            if "optimization_steps" in values:
                result += f" · {values.get('algorithm', '')} optimizer {shown(values['optimization_steps'])}"
            prefix = row.get("stage")
            if f"{prefix}_requested" in values:
                result += f" · {values.get('evaluation_kind', prefix)} {shown(values.get(prefix + '_finished', 0))}/{shown(values[prefix + '_requested'])}"
                if prefix + "_tick" in values:
                    result += f" · case tick {shown(values[prefix + '_tick'])}/{shown(values.get(prefix + '_tick_limit'))}"
            return result

        def build():
            lines = []

            def line(value):
                lines.append(Text(str(value), no_wrap=True, overflow="ellipsis"))

            if self.total_tasks is not None:
                line(f"Tasks ended: {len(ended)}/{self.total_tasks}")
            line(f"Active {len(running)} · waiting {waiting}")
            successful = sum(r["status"] == "completed" for r in ended)
            if evaluation is None:
                line(f"Successful {successful} · other ended {len(ended) - successful}")
            else:
                for row in items:
                    if row.get("unit") in {
                        "sampling decisions",
                        "environment steps",
                        "adapter decisions",
                    }:
                        line(f"Training: {row['status']} · {row['name']}")
            failures = [r for r in ended if r["status"] == "failed"]
            incomplete = sum(
                r["status"]
                in {"truncated", "ineligible", "incomplete", "completed_with_failures"}
                for r in ended
            )
            line(f"Incomplete {incomplete} · failed {len(failures)}")
            if failures and failures[0] not in ended[:recent]:
                line(f"Latest failure: {failures[0].get('reason', 'See runtime.log')}")
            if evaluation is not None:
                values = evaluation["values"]
                line(
                    f"Evaluation ended {shown(values.get('evaluation_finished'))}/{shown(values.get('evaluation_requested'))}"
                )
                line(
                    f"Evaluation successes {shown(values.get('evaluation_completed'))}/{shown(values.get('evaluation_requested'))}"
                )
            bars = Progress(
                TextColumn(
                    "{task.description}",
                    markup=False,
                    table_column=Column(
                        max_width=max(8, min(32, width // 2)),
                        no_wrap=True,
                        overflow="ellipsis",
                    ),
                ),
                BarColumn(bar_width=None),
                TaskProgressColumn(),
                expand=True,
            )
            for row in running[:count]:
                line(f"{row['stage']} · {row['name']}")
                if row.get("context"):
                    line(row["context"])
                line(budget(row))
                if row["values"].get("group_statistics"):
                    line(
                        "Groups decisions/samples/optimizer: "
                        + row["values"]["group_statistics"]
                    )
                if (
                    row.get("total") is not None
                    and row.get("completed") is not None
                    and row["completed"] >= row["total"]
                ):
                    line(
                        "Sampling target reached; task still running"
                        if row.get("unit")
                        in {
                            "sampling decisions",
                            "environment steps",
                            "adapter decisions",
                        }
                        else "Target reached; finalizing"
                    )
                else:
                    unit = units.get(row.get("unit"), row.get("unit", "progress"))
                    bars.add_task(
                        f"{unit} · {row['name']}",
                        total=row.get("total")
                        if row.get("completed") is not None
                        else None,
                        completed=row.get("completed") or 0,
                    )
            if len(running) > count:
                line(f"+{len(running) - count} other active tasks")
            if recent:
                line("Recent results")
            for row in ended[:recent]:
                line(f"{row['status']} · {row['name']}")
                if row.get("reason"):
                    line(row["reason"])
                line(budget(row))
            if len(ended) > recent:
                line(f"+{len(ended) - recent} older results in runtime.log")
            if self.notice:
                line(self.notice)
            line(
                f"Updated {time.strftime('%H:%M:%S', time.localtime(self.updated_at))}"
            )
            if self.total_tasks is not None:
                bars.add_task(
                    "Finished tasks", total=self.total_tasks, completed=len(ended)
                )
            elif evaluation is not None:
                values = evaluation["values"]
                bars.add_task(
                    "Evaluation ended",
                    total=values.get("evaluation_requested"),
                    completed=values.get("evaluation_finished") or 0,
                )
            subtitle = (
                self.stage
                if self.stage == self.status
                else f"{self.stage} [{self.status}]"
            )
            return Group(
                Panel(
                    Group(*lines),
                    title=Text(
                        f"SmartSOM · {self.name}", no_wrap=True, overflow="ellipsis"
                    ),
                    subtitle=Text(subtitle),
                ),
                bars,
            )

        while True:
            result = build()
            used = len(
                self.console.render_lines(
                    result, self.console.options.update(height=None), pad=False
                )
            )
            if used <= height or (recent == 0 and count == 0):
                return result
            if count > 2:
                count -= 1
            elif recent:
                recent -= 1
            else:
                count -= 1

    def _render_study(self):
        """One labelled row per condition and a separate workflow overview."""
        from smartsom.telemetry.study_progress import duration

        width, height = self.console.width, self.console.height
        wide = width >= 120
        narrow = width < 85
        slots = max(0, (height - 16) // 2)
        rows = list(self.tasks.values())
        running = [r for r in rows if r["status"] not in FINAL | {"pending", "queued"}]
        ended = [r for r in rows if r["status"] in FINAL]
        visible = running[:slots]
        visible += list(reversed(ended))[: min(3, max(0, slots - len(visible)))]
        finished, successful = self.task_counts()
        failed = sum(r["status"] == "failed" for r in ended)
        waiting = sum(r["status"] in {"pending", "queued"} for r in rows)
        waiting += max(0, (self.total_tasks or len(rows)) - len(rows))

        def line(value, style=""):
            return Text(str(value), style=style, no_wrap=True, overflow="ellipsis")

        overview = self.overview or {}
        total, completed = overview.get("work_total"), overview.get("work_completed")
        overall = Progress(
            TextColumn("{task.description}", markup=False),
            BarColumn(bar_width=None, complete_style="cyan", finished_style="cyan"),
            TextColumn("{task.fields[percent]}", markup=False),
            expand=True,
        )
        overall.add_task(
            "Overall workflow (budget weighted)",
            total=total or 1,
            completed=completed or 0,
            percent=f"{completed / total:.1%}"
            if completed is not None and total
            else "N/A",
        )
        timing = line(
            f"Elapsed {duration(overview.get('elapsed_seconds'))}  |  ETA ~ {duration(overview.get('eta_seconds'))}",
            "bold cyan",
        )
        counts = line(
            f"Finished {finished}/{shown(self.total_tasks)}  |  Successful {successful}  |  Failed {failed}  |  Active {len(running)}  |  Queued {waiting}"
        )
        table = Table(
            box=box.SIMPLE,
            show_lines=True,
            expand=True,
            padding=(0, 1),
            header_style="bold cyan",
        )
        table.add_column(
            "Case",
            width=4 if wide else 8 if narrow else 22,
            no_wrap=True,
            overflow="ellipsis",
        )
        if wide:
            table.add_column("Algorithm", width=9, no_wrap=True)
            table.add_column("H / V", width=12, no_wrap=True)
            table.add_column("Travel", width=7, no_wrap=True)
        table.add_column(
            "Phase", width=11 if narrow else 16, no_wrap=True, overflow="ellipsis"
        )
        table.add_column(
            "Train %" if narrow else "Training ticks (%)",
            ratio=1,
            no_wrap=True,
            overflow="ellipsis",
        )
        table.add_column(
            "Case %" if narrow else "Cases ended; current tick",
            ratio=1,
            no_wrap=True,
            overflow="ellipsis",
        )
        if wide:
            table.add_column("Last activity", width=13, no_wrap=True)
        names = {
            "sampling": "Training",
            "optimizing": "Updating",
            "saving": "Saving",
            "initializing": "Initializing",
            "validation": "Validation",
            "evaluation": "Test",
            "waiting for shared control": "Control wait",
        }
        for row in visible:
            values = row["values"]
            case = values.get("study_case", {})
            number = rows.index(row) + 1
            phase = row["stage"] if row["status"] == "running" else row["status"]
            kind = values.get("evaluation_kind")
            state = names.get(phase, phase.capitalize())
            if phase == "evaluation" and kind in {
                "initial control",
                "rule control",
                "random control",
            }:
                state = kind.capitalize()
            style = (
                "yellow"
                if phase in {"validation", "evaluation"}
                else "red"
                if row["status"] in FAILURES
                else "green"
            )
            ticks, budget = row.get("completed"), row.get("total")
            percent = f"{ticks / budget:.1%}" if ticks is not None and budget else "N/A"
            training = (
                percent if narrow else f"{shown(ticks)}/{shown(budget)} ({percent})"
            )
            current = "—"
            if row["status"] not in FINAL and phase in {"validation", "evaluation"}:
                done = values.get(phase + "_finished", 0)
                requested = values.get(phase + "_requested")
                tick, limit = (
                    values.get(phase + "_tick"),
                    values.get(phase + "_tick_limit"),
                )
                current = (
                    f"{shown(done)}/{shown(requested)}; {shown(tick)}/{shown(limit)}"
                )
                if narrow:
                    current = (
                        f"{shown(done)}/{shown(requested)} {tick / limit:.0%}"
                        if tick is not None and limit
                        else f"{shown(done)}/{shown(requested)} N/A"
                    )
                if values.get(phase + "_case_active") is False:
                    current = (
                        f"{shown(done)}/{shown(requested)} ended"
                        if narrow
                        else f"{shown(done)}/{shown(requested)}; case ended"
                    )
            cells = [
                line(
                    f"{number:02d}"
                    if wide
                    else f"{number:02d} {case.get('algorithm', '?')}"
                    if narrow
                    else f"{number:02d} {case.get('algorithm', '?')} {case.get('H', '?')}/{case.get('V', '?')} {case.get('travel', '?')}"
                )
            ]
            if wide:
                cells += [
                    line(case.get("algorithm", "N/A")),
                    line(f"{case.get('H', 'N/A')} / {case.get('V', 'N/A')}"),
                    line(case.get("travel", "N/A")),
                ]
            cells += [
                line(state, style),
                line(training, "green"),
                line(current, "yellow"),
            ]
            if wide:
                activity = case.get("last_activity")
                cells.append(
                    line(
                        time.strftime("%H:%M:%S", time.localtime(activity))
                        if activity
                        else "N/A"
                    )
                )
            table.add_row(*cells)
        hidden = max(0, len(running) - slots)
        latest_failure = next((r for r in reversed(ended) if r.get("reason")), None)
        footer = line(
            f"Showing {min(len(running), slots)}/{len(running)} active | {hidden} hidden | Full counters/PIDs: logs/progress.json, runtime.log"
        )
        legend = line(
            "Overall includes training + validation + test + controls. ETA is approximate; training pauses during validation.",
            "dim",
        )
        detail = (
            line("Latest issue: " + latest_failure["reason"], "red")
            if latest_failure
            else line(self.notice or "Waiting for experiment workers", "dim")
        )
        return Panel(
            Group(overall, timing, counts, line(""), table, footer, legend, detail),
            title=line(f"SmartSOM · {self.name}"),
            subtitle=line(f"{self.stage} [{self.status}]"),
            height=max(3, height - 1),
        )

    def _render_detailed(self):
        width, height = self.console.width, self.console.height
        narrow = width < 85
        compact = narrow
        short_units = {
            "adapter decisions": "adapter dec",
            "environment steps": "env steps",
            "sampling decisions": "sample dec",
            "physical ticks": "ticks",
            "qualified deliveries": "deliveries",
            "evaluation episodes": "eval episodes",
        }
        table = Table(expand=True, box=None, padding=(0, 1))
        if compact:
            table.show_header = False
            table.add_column(no_wrap=True, overflow="ellipsis")
        else:
            table.add_column("Task", no_wrap=True, overflow="ellipsis", max_width=24)
            table.add_column("State", no_wrap=True, overflow="ellipsis", max_width=24)
            table.add_column("Progress", no_wrap=True, overflow="ellipsis")
            if not narrow:
                table.add_column("Optimizer steps", no_wrap=True)
        rows = list(self.tasks.values())
        evaluation = self.tasks.get("evaluation") if len(rows) > 1 else None
        # Each visible task occupies one table line and one progress-bar line.
        visible = max(
            1,
            (
                height
                - 9
                - int(evaluation is not None)
                - int(self.total_tasks is not None)
            )
            // (4 if compact else 2),
        )
        ordered = sorted(
            rows,
            key=lambda r: r["status"] in FINAL or r["status"] in {"pending", "queued"},
        )
        for row in ordered[:visible]:
            progress = f"{shown(row.get('completed'))}/{shown(row.get('total'))}"
            state = row["stage"] if row["status"] == "running" else row["status"]
            if compact:
                unit = short_units.get(row.get("unit"), row.get("unit", "progress"))
                table.add_row(Text(row["name"]))
                table.add_row(Text(state))
                table.add_row(Text(f"{unit}: {progress}"))
                continue
            cells = [
                Text(row["name"]),
                Text(state),
                Text(progress),
            ]
            if not narrow:
                cells.append(
                    Text(
                        shown(
                            row["values"].get(
                                "optimization_steps", row["values"].get("ppo_updates")
                            )
                        )
                    )
                )
            table.add_row(*cells)
        detail = Table.grid(padding=(0, 2))
        detail.add_column(
            no_wrap=True, overflow="ellipsis", max_width=max(12, width - 20)
        )
        detail.add_column(no_wrap=True, overflow="ellipsis")
        if self.total_tasks is not None:
            finished, successful = self.task_counts()
            detail.add_row(
                "Tasks ended", f"{finished}/{self.total_tasks}; successful {successful}"
            )
        if len(rows) == 1:
            row = rows[0]
            details = [
                (label, shown(row["values"][key]))
                for key, label in LABELS.items()
                if key in row["values"]
            ]
            details += [
                (LABELS[key], "N/A")
                for key in (
                    "training_deliveries",
                    "episode_return",
                    "evaluation_completed",
                )
                if key not in row["values"]
            ]
            details += [
                (key, shown(value))
                for key, value in sorted(row.get("learner", {}).items())
            ]
            for label, value in details[: max(0, height - 10)]:
                detail.add_row(Text(label), Text(value))
        if evaluation is not None:
            values = evaluation["values"]
            detail.add_row(
                "Evaluation successes",
                f"{shown(values.get('evaluation_completed'))}/{shown(values.get('evaluation_requested'))}",
            )
        bars = Progress(
            TextColumn(
                "{task.description}",
                markup=False,
                table_column=Column(
                    max_width=max(8, min(32, width // 2)),
                    no_wrap=True,
                    overflow="ellipsis",
                ),
            ),
            BarColumn(bar_width=None),
            TaskProgressColumn(),
            expand=True,
        )
        for row in ordered[:visible]:
            unit = short_units.get(row.get("unit"), row.get("unit", "unknown total"))
            bars.add_task(
                f"{unit} · {row['name']}",
                total=row.get("total") if row.get("completed") is not None else None,
                completed=row.get("completed") or 0,
            )
        if self.total_tasks is not None:
            bars.add_task(
                "Finished tasks",
                total=self.total_tasks,
                completed=self.task_counts()[0],
            )
        omitted = (
            f" · +{len(rows) - visible} other tasks" if len(rows) > visible else ""
        )
        footer = Text(
            f"Updated {time.strftime('%H:%M:%S', time.localtime(self.updated_at))}{omitted}"
            + (f" · {self.notice}" if self.notice else ""),
            no_wrap=True,
            overflow="ellipsis",
        )
        title = Text(f"SmartSOM · {self.name}")
        title.truncate(max(8, width - 12), overflow="ellipsis")
        return Group(
            Panel(
                Group(table, detail, footer),
                title=title,
                subtitle=Text(
                    self.stage
                    if self.stage == self.status
                    else f"{self.stage} [{self.status}]"
                ),
            ),
            bars,
        )

    def finish(self, status, error=None):
        for row in self.tasks.values():
            if row["status"] not in FINAL | {"pending", "queued"}:
                row["status"] = status
        self.status = status
        self.stage = status
        self.updated_at = time.time()
        if error:
            self.notice = error
        self.publish(force=True)

    def close(self):
        if self.live:
            self.live.update(self.render(), refresh=True)
            self.live.stop()
            self.live = None
            if (
                self.kind
                in {
                    "study",
                    "training",
                    "evaluation",
                    "train-evaluate",
                    "run",
                    "tune",
                    "batch-directory",
                }
                and self.options.verbose
            ):
                # Alternate-screen output disappears on exit; retain the outcome.
                self.console.print(Text(self.text_summary()), soft_wrap=True)
