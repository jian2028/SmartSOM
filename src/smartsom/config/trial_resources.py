"""Static per-phase requests, authored only in an execution profile.

These are logical admission requests, never measurements or process limits.
Importing this module does not import Ray or a learning framework.
"""

from pydantic import Field

from smartsom.config.codec import ConfigurationError, StrictModel


class PhaseResources(StrictModel):
    cpus: int | None = Field(default=None, ge=1)
    gpus: float = Field(default=0, ge=0, allow_inf_nan=False)
    memory_gib: float | None = Field(default=None, gt=0, allow_inf_nan=False)


class TrialResources(StrictModel):
    training: PhaseResources = Field(default_factory=PhaseResources)
    evaluation: PhaseResources = Field(default_factory=PhaseResources)

    def bundles(self, phase, *, workers=0, threads=1):
        """One learner/coordinator bundle plus individually scheduled workers."""
        if phase not in {"training", "evaluation"}:
            raise ConfigurationError(f"unknown execution phase: {phase}")
        if type(workers) is not int or workers < 0:
            raise ConfigurationError("worker count must be a nonnegative integer")
        if type(threads) is not int or threads < 1:
            raise ConfigurationError("numerical threads must be a positive integer")
        settings = getattr(self, phase)
        minimum = threads + workers
        cpus = minimum if settings.cpus is None else settings.cpus
        if cpus < minimum:
            raise ConfigurationError(
                f"{phase} needs at least {minimum} CPUs "
                f"({threads} coordinator/learner plus {workers} workers)"
            )
        head = {"CPU": float(cpus - workers)}
        if settings.gpus:
            head["GPU"] = settings.gpus
        bundles = [head] + [{"CPU": 1.0} for _ in range(workers)]
        if settings.memory_gib is not None:
            # The profile states the entire phase's request. Splitting the
            # logical amount across bundles also accounts for child workers.
            memory = settings.memory_gib * 1024**3 / len(bundles)
            for bundle in bundles:
                bundle["memory"] = memory
        return bundles


def check_capacity(bundles, capacity):
    """Reject impossible declared requests before launching an owned runtime."""
    requested = {}
    for bundle in bundles:
        for name, amount in bundle.items():
            requested[name] = requested.get(name, 0) + amount
    for name, amount in requested.items():
        if amount > capacity.get(name, 0):
            raise ConfigurationError(
                f"trial requests {amount} {name}, exceeding total {capacity.get(name, 0)}"
            )
