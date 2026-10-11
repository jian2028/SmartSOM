"""Observational capture settings; never part of model or learning semantics."""

from typing import Annotated

from pydantic import Field

from smartsom.config.codec import digest, primitive
from smartsom.config.experiment import EditableModel


class EventContextOptions(EditableModel):
    max_snippets: Annotated[int, Field(strict=True, ge=1, le=64)] = 8
    max_payload_bytes: Annotated[int, Field(strict=True, ge=4096, le=1048576)] = 262144
    lookback_boundaries: Annotated[int, Field(strict=True, ge=1, le=32)] = 4
    stagnation_ticks: Annotated[int, Field(strict=True, ge=1, le=2147483647)] = 128


class DiagnosticReportOptions(EditableModel):
    interval_updates: Annotated[int, Field(strict=True, ge=1, le=1048576)] = 1


class DiagnosticsOptions(EditableModel):
    event_context: EventContextOptions = Field(default_factory=EventContextOptions)
    reports: DiagnosticReportOptions = Field(default_factory=DiagnosticReportOptions)


def capture_identity(value=None):
    settings = primitive(DiagnosticsOptions.model_validate(value or {}))
    return {
        "schema": "smartsom.diagnostic-capture/v1",
        "settings": settings,
        "sha256": digest(settings),
        "scope": "observation and report retention; all diagnostics enabled",
    }
