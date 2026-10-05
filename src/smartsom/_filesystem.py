"""Atomic file publication with bounded Windows reader-sharing retries."""

import os
import time


def read_text(path, **kwargs):
    """Read a committed file despite a brief Windows replacement sharing race."""
    deadline = time.monotonic() + 2.0
    while True:
        try:
            return path.read_text(**kwargs)
        except PermissionError:
            if os.name != "nt" or time.monotonic() >= deadline:
                raise
            time.sleep(0.01)


def atomic_replace(source, destination, *, deadline=None):
    """Never remove the committed file before its replacement is ready."""
    retry_deadline = time.monotonic() + 2.0
    if deadline is not None:
        retry_deadline = min(retry_deadline, deadline)
    while True:
        try:
            os.replace(source, destination)
            return
        except PermissionError as exc:
            if (
                os.name != "nt"
                or getattr(exc, "winerror", None) not in {5, 32, 33}
                or time.monotonic() >= retry_deadline
            ):
                raise
            time.sleep(min(0.01, max(0.0, retry_deadline - time.monotonic())))
