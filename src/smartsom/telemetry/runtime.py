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
from dataclasses import dataclass
from functools import wraps
from pathlib import Path

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn
from rich.table import Column, Table
from rich.text import Text

SCHEMA = "smartsom.runtime-progress/v1"
FINAL = {
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
            arguments.get("source") or arguments.get("resume") or arguments.get("root")
        )
        if isinstance(source, (str, Path)):
            path = Path(source)
            for root in (path, *path.parents):
                saved = root / "config/experiment.json"
                if saved.is_file():
                    config = json.loads(saved.read_text()).get("logging", {})
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
                            status or session.status_from_manifest() or "completed"
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
    session = CURRENT.get()
    if session:
        session.bind(root, name)
    return session


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
        self._legacy_warned = False

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
                vertical_overflow="crop",
            )
            self.live.start()

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
        if event.get("display_name"):
            row["name"] = event["display_name"]
        if self.root and task == str(self.root):
            row["name"] = self.name
        previous = row["status"]
        stage = str(
            event.get("stage", event.get("status", row.get("stage", self.stage)))
        )
        if not final and stage in FINAL:
            stage = "finalizing"
        row.update(stage=stage, updated_at=time.time())
        # A sampling target or learner event is never proof that artifacts are saved.
        row["status"] = event.get("status", "running") if final else "running"
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
        aliases = {"tick": "physical_ticks", "qualified_demands": "qualified_demands"}
        for key, value in {**event, **event.get("display_values", {})}.items():
            key = aliases.get(key, key)
            if key in LABELS:
                row["values"][key] = value
        for key in (
            "sampled_steps",
            "qualified_demands",
            "tick",
            "evaluation_finished",
        ):
            if key in event and (
                key == "sampled_steps"
                or row.get("unit")
                == {
                    "qualified_demands": "qualified deliveries",
                    "tick": "physical ticks",
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
            snapshot_force=final,
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
        }

    def text_summary(self):
        rows = [f"{self.name}: {self.stage} [{self.status}]"]
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
                f"{key}={shown(value)}" for key, value in row["values"].items()
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

    def publish(self, *, force=False, snapshot_force=False, error=False):
        now = time.monotonic()
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
        if self.live and now - self.last_refresh >= 0.25:
            self.last_refresh = now
            self.live.update(self.render(), refresh=True)

    def task_counts(self):
        return (
            sum(row["status"] in FINAL for row in self.tasks.values()),
            sum(row["status"] == "completed" for row in self.tasks.values()),
        )

    def render(self):
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
                line(budget(row))
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
                table.add_column("PPO updates", no_wrap=True)
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
                cells.append(Text(shown(row["values"].get("ppo_updates"))))
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
