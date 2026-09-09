"""One metric event stream for terminal, TensorBoard and opt-in W&B."""

import importlib.util
import json
import random
import sys
import time
from contextlib import contextmanager
from numbers import Real
from pathlib import Path

from smartsom.config.codec import ConfigurationError, primitive


@contextmanager
def isolated_random_state():
    """Tracking clients must not consume the learner's existing global RNGs."""
    state = random.getstate()
    numpy = sys.modules.get("numpy")
    torch = sys.modules.get("torch")
    numpy_state = numpy.random.get_state() if numpy is not None else None
    torch_state = torch.get_rng_state() if torch is not None else None
    cuda_state = (
        torch.cuda.get_rng_state_all()
        if torch is not None and torch.cuda.is_initialized()
        else None
    )
    try:
        yield
    finally:
        random.setstate(state)
        if numpy_state is not None:
            numpy.random.set_state(numpy_state)
        if torch_state is not None:
            torch.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


class TrainingDisplay:
    @staticmethod
    def preflight(options):
        for enabled, package, extra in (
            (options.tensorboard, "tensorboard", "tensorboard"),
            (options.wandb, "wandb", "wandb"),
        ):
            if enabled and importlib.util.find_spec(package) is None:
                raise ConfigurationError(
                    f"{package} is enabled; install uv sync --locked --extra learning --extra cpu --extra {extra}"
                )
        if options.wandb and not options.wandb_project:
            raise ConfigurationError(
                "logging.wandb_project is required when W&B is enabled"
            )

    def __init__(self, root: Path, config, *, on_progress=None):
        self.root, self.config, self.options = root, config, config.logging
        self.callback = on_progress
        self.started = time.monotonic()
        self.last_display = 0.0
        self.writer = self.wandb_run = self.progress = self.log = None

    def __enter__(self):
        self.preflight(self.options)
        from rich.console import Console
        from rich.progress import (
            BarColumn,
            Progress,
            TaskProgressColumn,
            TimeElapsedColumn,
            TimeRemainingColumn,
        )

        self.console = Console(stderr=True)
        self.log = (self.root / "logs/events.jsonl").open("a", encoding="utf-8")
        try:
            with isolated_random_state():
                if self.options.tensorboard:
                    from torch.utils.tensorboard import SummaryWriter

                    self.writer = SummaryWriter(str(self.root / "logs/tensorboard"))
                if self.options.wandb:
                    import wandb

                    self.wandb_run = wandb.init(
                        project=self.options.wandb_project,
                        name=self.config.output.name,
                        dir=str(self.root / "logs"),
                        config=primitive(self.config),
                        mode=self.options.wandb_mode,
                    )
            show = self.options.progress == "on" or (
                self.options.progress == "auto" and self.console.is_terminal
            )
            if show and self.options.verbose and self.options.format == "text":
                self.progress = Progress(
                    "{task.description}",
                    BarColumn(),
                    TaskProgressColumn(),
                    TimeElapsedColumn(),
                    TimeRemainingColumn(),
                    console=self.console,
                )
                self.progress.start()
                self.task = self.progress.add_task(
                    self.config.output.name, total=self.config.training.total_steps
                )
            return self
        except BaseException:
            self.__exit__(*sys.exc_info())
            raise

    def __call__(self, event):
        with isolated_random_state():
            return self._emit(event)

    def _emit(self, event):
        event = primitive(event)
        elapsed = time.monotonic() - self.started
        steps = event.get("sampled_steps", 0)
        row = {
            **event,
            "elapsed_seconds": elapsed,
            "steps_per_second": steps / elapsed if elapsed else 0.0,
        }
        self.log.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        self.log.flush()
        metrics = {
            key: float(value)
            for key, value in row.items()
            if isinstance(value, Real) and not isinstance(value, bool)
        }
        metrics.update(
            {
                f"learner/{k}": float(v)
                for k, v in row.get("metrics", {}).items()
                if isinstance(v, Real) and not isinstance(v, bool)
            }
        )
        if self.writer:
            for key, value in metrics.items():
                self.writer.add_scalar(key, value, global_step=steps)
        if self.wandb_run:
            self.wandb_run.log({**metrics, "sampled_steps": steps})
        if self.progress:
            self.progress.update(
                self.task,
                completed=steps,
                description=str(row.get("stage", "training")),
            )
        now = time.monotonic()
        terminal = row.get("stage") in {
            "completed",
            "failed",
            "interrupted",
            "early_stopped",
        }
        if (
            terminal
            or self.options.verbose
            and now - self.last_display >= self.options.every_seconds
        ):
            self.last_display = now
            if self.options.format == "json":
                print(json.dumps(row, sort_keys=True), file=sys.stderr)
            else:
                from rich.table import Table

                table = Table(
                    title=f"{self.config.output.name}: {row.get('stage', 'training')}"
                )
                for column in (
                    "Steps",
                    "PPO updates",
                    "Completed",
                    "Failed",
                    "Steps/s",
                ):
                    table.add_column(column)
                updates = row.get(
                    "ppo_updates", steps // self.config.training.steps_per_update
                )
                table.add_row(
                    str(steps),
                    str(updates),
                    str(row.get("completed_episodes", 0)),
                    str(row.get("failed_episodes", 0)),
                    f"{metrics['steps_per_second']:.1f}",
                )
                self.console.print(table)
                if self.options.verbose == 2 and row.get("metrics"):
                    self.console.print(row["metrics"])
        if self.callback:
            return self.callback(row)

    def __exit__(self, exc_type, exc, tb):
        failures = []
        actions = [
            self.progress.stop if self.progress else None,
            self.writer.close if self.writer else None,
            (lambda: self.wandb_run.finish(exit_code=int(exc is not None)))
            if self.wandb_run
            else None,
            self.log.close if self.log else None,
        ]
        with isolated_random_state():
            for close in actions:
                if close is not None:
                    try:
                        close()
                    except Exception as failure:
                        failures.append(failure)
        if failures:
            if exc is not None:
                for failure in failures:
                    exc.add_note(f"tracking cleanup also failed: {failure!r}")
            else:
                raise ExceptionGroup("tracking cleanup failed", failures)
