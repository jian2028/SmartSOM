"""Replay debug system for the SO-MARL grid factory.

Trace format (``smartsom.somarl.replay.v1``) is documented in ``README.md``.
A trace directory holds ``manifest.json`` (static layout + episode metadata)
and ``replay.jsonl`` (one frame per tick). Everything else is derived.
"""

from .schema import SCHEMA_VERSION, load_trace, validate_trace, write_trace

__all__ = ["SCHEMA_VERSION", "load_trace", "validate_trace", "write_trace"]
