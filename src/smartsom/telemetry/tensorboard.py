"""Optional scalar recording with a durable per-tag resume cursor."""

import importlib.util
import json
import math
from numbers import Real

from smartsom.config.codec import ConfigurationError
from smartsom.telemetry.training import isolated_random_state


class TensorBoardRecorder:
    def __init__(self, root, enabled):
        self.directory = root / "logs/tensorboard"
        self.enabled = enabled
        self.writer = None
        self.cursor = self.directory / "recorded-steps.json"
        self.steps = (
            json.loads(self.cursor.read_text())
            if enabled and self.cursor.exists()
            else {}
        )
        if enabled and importlib.util.find_spec("tensorboard") is None:
            raise ConfigurationError(
                "tensorboard is enabled; install uv sync --locked --extra learning --extra cpu --extra tensorboard"
            )

    def record(self, metrics, step):
        if not self.enabled:
            return
        pending = {
            tag: float(value)
            for tag, value in metrics.items()
            if isinstance(value, Real)
            and not isinstance(value, bool)
            and math.isfinite(value)
            and step > self.steps.get(tag, -1)
        }
        if not pending:
            return
        with isolated_random_state():
            if self.writer is None:
                from torch.utils.tensorboard import SummaryWriter

                self.writer = SummaryWriter(str(self.directory))
            for tag, value in pending.items():
                self.writer.add_scalar(tag, value, global_step=step)
            self.writer.flush()
        self.steps.update(dict.fromkeys(pending, step))
        temporary = self.cursor.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.steps, sort_keys=True) + "\n")
        temporary.replace(self.cursor)

    def close(self):
        if self.writer is not None:
            writer, self.writer = self.writer, None
            with isolated_random_state():
                writer.close()
