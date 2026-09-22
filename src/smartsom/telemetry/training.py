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
from smartsom.telemetry.runtime import backend_diagnostics


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
        # Presentation only: never fill missing fields in the shared event stream.
        self.latest_progress = {}
        self.latest_metrics = {}
        self.resource_steps = False
        self.projection = None

    def __enter__(self):
        self.preflight(self.options)
        from smartsom.telemetry.runtime import CURRENT, DisplayOptions, RuntimeDisplay

        self.session = CURRENT.get()
        self.owns_session = self.session is None
        if self.owns_session:
            self.session = RuntimeDisplay(
                DisplayOptions.from_value(self.options), kind="training"
            )
            self.session.start()
        self.session.bind(self.root, self.config.output.name)
        self.console = self.session.console
        if getattr(self.options, "legacy_verbose", False):
            self.session.legacy_warning()
        self.log = (self.root / "logs/events.jsonl").open("a", encoding="utf-8")
        try:
            with isolated_random_state(), backend_diagnostics(self.session):
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
        self.latest_progress.update(
            {
                key: row[key]
                for key in (
                    "sampled_steps",
                    "ppo_updates",
                    "learner_updates",
                    "completed_episodes",
                    "failed_episodes",
                    "agent_steps",
                    "physical_actions",
                )
                if key in row
            }
        )
        self.latest_metrics.update(row.get("metrics", {}))
        self.resource_steps |= (
            "agent_steps" in row
            or "physical_actions" in row
            or any(
                key.split("/", 1)[0] in {"machine_policy", "agv_policy"}
                for key in self.latest_metrics
            )
        )
        # Evidence and callbacks are never throttled by terminal presentation.
        presentation = dict(row)
        if self.projection:
            presentation.update(self.projection())
        terminal = row.get("stage") in {
            "completed",
            "failed",
            "interrupted",
            "early_stopped",
            "pruned",
        }
        if terminal:
            presentation["stage"] = "finalizing"
        self.session.console = self.console
        self.session.update(
            str(self.root),
            presentation,
            total=self.config.training.total_steps,
            unit="adapter decisions" if self.resource_steps else "environment steps",
            phase="training",
        )
        if terminal and self.owns_session:
            self.session.tasks[str(self.root)]["stage"] = row["stage"]
            self.session.finish(row["stage"])
        if self.callback:
            return self.callback(row)

    def __exit__(self, exc_type, exc, tb):
        failures = []
        actions = [
            self.session.close if self.owns_session else None,
            self.writer.close if self.writer else None,
            (lambda: self.wandb_run.finish(exit_code=int(exc is not None)))
            if self.wandb_run
            else None,
            self.log.close if self.log else None,
        ]
        with isolated_random_state(), backend_diagnostics(self.session):
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
