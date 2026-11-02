"""Opt-in cooperative worker protection, separate from admission budgets."""

import os
import shutil
import time
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from pathlib import Path

CURRENT = ContextVar("smartsom_resource_guard", default=None)
GIB = 1024**3


class ResourceGuardExceeded(RuntimeError):
    """Keep committed evidence and require explicit recovery after diagnosis."""


def observe(root):
    import psutil

    path = Path(root)
    while not path.exists() and path != path.parent:
        path = path.parent
    try:
        return {
            "worker_rss": psutil.Process().memory_info().rss,
            "host_available": psutil.virtual_memory().available,
            "disk_free": shutil.disk_usage(path).free,
        }
    except psutil.Error as exc:
        raise OSError(str(exc)) from exc


class WorkerGuard:
    def __init__(self, root, settings):
        self.root = Path(root)
        self.limits = {
            "worker_rss": settings.max_worker_rss_gib * GIB,
            "host_available": settings.min_host_available_gib * GIB,
            "disk_free": settings.min_disk_free_gib * GIB,
        }
        self.last = None
        self.stage = "startup"
        self.failure = None

    def check(self, stage=None, *, force=False):
        if self.failure is not None:
            raise self.failure
        if stage is not None:
            self.stage = stage
        now = time.monotonic()
        if not force and self.last is not None and now - self.last < 1:
            return
        self.last = now
        try:
            actual = observe(self.root)
            reasons = [
                key
                for key, limit in self.limits.items()
                if (
                    actual[key] >= limit if key == "worker_rss" else actual[key] < limit
                )
            ]
        except OSError as exc:
            actual = {"observation_error": str(exc)}
            reasons = ["observation_unavailable"]
        if not reasons:
            return
        evidence = {
            "schema": "smartsom.resource-guard-failure/v1",
            "status": "failed",
            "reason": reasons,
            "stage": self.stage,
            "limits_bytes": self.limits,
            "observed_bytes": actual,
            "pid": os.getpid(),
            "time": time.time(),
            "recovery": "explicit retry from last committed checkpoint/case receipts",
        }
        self.failure = ResourceGuardExceeded(
            f"resource guard at {self.stage}: {reasons}; "
            f"observed={actual}; limits={self.limits}"
        )
        from smartsom.experiments.evidence import write_json

        try:
            self.root.mkdir(parents=True, exist_ok=True)
            write_json(self.root / "resource-guard-failure.json", evidence)
        except OSError as exc:
            self.failure.add_note(f"could not persist guard evidence: {exc}")
        raise self.failure


def check(stage=None, *, force=False):
    guard = CURRENT.get()
    if guard is not None:
        guard.check(stage, force=force)


@contextmanager
def scope(root, settings):
    # Nested final/recovery evaluation belongs to the same worker and limits.
    if CURRENT.get() is not None or settings is None:
        yield
        return
    guard = WorkerGuard(root, settings)
    token = CURRENT.set(guard)
    try:
        guard.check(force=True)
        yield
    finally:
        CURRENT.reset(token)


def guarded(function):
    @wraps(function)
    def execute(args, *positional, **kwargs):
        if CURRENT.get() is not None:
            return function(args, *positional, **kwargs)
        from smartsom.config.base_diagnostic import load_inputs

        prepared = getattr(args, "_prepared_inputs", None)
        ex = prepared[0][0] if prepared is not None else load_inputs(args.experiment)[0]
        if ex.resources.guard is not None and ex.run.tune != "off":
            raise ValueError("resources.guard requires native execution with tune=off")
        with scope(args.output, ex.resources.guard):
            return function(args, *positional, **kwargs)

    return execute
