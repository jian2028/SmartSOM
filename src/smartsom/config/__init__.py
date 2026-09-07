"""File authoring and resolution, independent of simulation execution."""

from smartsom.config.codec import ConfigurationError
from smartsom.config.resolver import ResolvedRun, resolve_run

__all__ = ["ConfigurationError", "ResolvedRun", "resolve_run"]
