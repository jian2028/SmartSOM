"""Restore embedded inputs, never follow the historical authoring references."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field

from smartsom.config.algorithm_binding import (
    bind_algorithm,
    validate_algorithm_references,
)
from smartsom.config.codec import (
    ConfigurationError,
    _UniqueLoader,
    canonical_json,
    digest,
)
from smartsom.config.models import StrictModel
from smartsom.config.resolver import ResolvedRun
from smartsom.config.seeds import derive_seeds
from smartsom.domain import validate_problem
from smartsom.modules.quality import prepare_quality


class _Snapshot(StrictModel):
    schema_id: Literal["smartsom.resolved-run/v1"] = Field(alias="schema")
    resolved: ResolvedRun


def validate_resolved(resolved: ResolvedRun) -> ResolvedRun:
    validate_problem(resolved.factory, resolved.workload)
    for name, value in (
        (
            "holding_buffer",
            resolved.factory.holding_buffer
            if resolved.holding_buffer_enabled
            else None,
        ),
        ("factory", resolved.factory),
        ("workload", resolved.workload),
        ("arrivals", resolved.arrivals),
        ("processing_times", resolved.processing_times),
        ("machine_events", resolved.machine_events),
        (
            "transport",
            resolved.factory.transport if resolved.transport_enabled else None,
        ),
        ("buffers", resolved.factory.buffers if resolved.buffers_enabled else None),
        ("quality_draws", resolved.quality.draws if resolved.quality else None),
        ("quality_modes", resolved.quality.modes if resolved.quality else None),
    ):
        expected = digest(value) if value is not None else None
        if getattr(resolved, f"{name}_sha256") != expected:
            raise ValueError(f"snapshot {name} digest mismatch")
    for field, flag in (
        ("transport", "transport_enabled"),
        ("buffers", "buffers_enabled"),
        ("holding_buffer", "holding_buffer_enabled"),
    ):
        if (getattr(resolved.scenario, field) is not None) != getattr(resolved, flag):
            raise ValueError(f"snapshot {field} enablement mismatch")
    for field, materialized in (
        ("arrivals", resolved.arrivals),
        ("processing_time", resolved.processing_times),
        ("machine_events", resolved.machine_events),
        ("quality", resolved.quality),
    ):
        if (getattr(resolved.scenario, field) is not None) != (
            materialized is not None
        ):
            raise ValueError(f"snapshot {field} enablement mismatch")
    if resolved.holding_buffer_enabled and (
        not resolved.transport_enabled or resolved.factory.holding_buffer is None
    ):
        raise ValueError("enabled holding buffer requires AGV and holding resources")
    if resolved.transport_enabled and resolved.factory.transport is None:
        raise ValueError("enabled transport requires factory transport")
    if resolved.arrivals is not None:
        resolved.arrivals.validate(resolved.workload)
    if resolved.processing_times is not None:
        resolved.processing_times.validate(resolved.workload)
    if resolved.machine_events is not None:
        resolved.machine_events.validate(resolved.factory)
    if (
        resolved.quality is not None
        and prepare_quality(
            resolved.factory,
            resolved.workload,
            resolved.quality.draws,
            processing_times=resolved.processing_times,
        )
        != resolved.quality
    ):
        raise ValueError("snapshot quality catalog does not match base inputs")
    bound = bind_algorithm(resolved.run, resolved.scenario, resolved.algorithm)
    if bound != resolved.run:
        raise ValueError("snapshot missing effective algorithm budget")
    validate_algorithm_references(
        resolved.algorithm,
        resolved.workload,
        resolved.factory,
        transport_enabled=resolved.transport_enabled,
        buffers_enabled=resolved.buffers_enabled,
        holding_buffer_enabled=resolved.holding_buffer_enabled,
        quality=resolved.quality,
    )
    origins = resolved.study_seed_origin
    if origins is not None:
        from smartsom.config.study import study_roots

        world, algorithm = study_roots(
            origins.study_seed,
            origins.case_id,
            origins.replication,
            origins.algorithm_id,
        )
        if (world, algorithm, origins.version) != (
            origins.world_seed,
            origins.algorithm_seed,
            "smartsom.study-seeds/v1",
        ) or resolved.run.seed != world:
            raise ValueError("snapshot study seed origin mismatch")
    else:
        algorithm = resolved.run.seed
    quality_seed = any(s.domain == "quality" for s in resolved.seeds)
    expected = {
        s.domain: s.value
        for s in derive_seeds(resolved.run.seed, generated=False, quality=quality_seed)
    }
    expected.update(
        {
            s.domain: s.value
            for s in derive_seeds(algorithm, generated=False)
            if s.domain in ("algorithm", "solver")
        }
    )
    if (
        len(resolved.seeds) != len(expected)
        or {s.domain: s.value for s in resolved.seeds} != expected
    ):
        raise ValueError("snapshot named seeds mismatch")
    if resolved.seed_version != "smartsom.seed/v1":
        raise ValueError("unsupported snapshot seed version")
    from smartsom.learning.checkpoint import validate_checkpoint

    validate_checkpoint(resolved)
    return resolved


def resolved_from_data(data: dict) -> ResolvedRun:
    if data.get("schema") in {
        "smartsom.prepared-grid-experiment/v1",
        "smartsom.production-run/v1",
    }:
        from smartsom.config.production import (
            prepared_from_data,
            prepared_from_run_record,
        )

        try:
            return (
                prepared_from_run_record(data)
                if data["schema"] == "smartsom.production-run/v1"
                else prepared_from_data(data)
            )
        except (ValueError, TypeError, KeyError) as exc:
            raise ConfigurationError(f"invalid resolved snapshot: {exc}") from exc
    payload = dict(data)
    schema = payload.pop("schema", None)
    try:
        envelope = _Snapshot.model_validate_json(
            canonical_json({"schema": schema, "resolved": payload})
        )
        return validate_resolved(envelope.resolved)
    except (ValueError, TypeError) as exc:
        raise ConfigurationError(f"invalid resolved snapshot: {exc}") from exc


def load_resolved_run(path: str | Path) -> ResolvedRun:
    try:
        data = yaml.load(Path(path).read_text(encoding="utf-8"), Loader=_UniqueLoader)
        if not isinstance(data, dict):
            raise ValueError("snapshot must be a mapping")
        return resolved_from_data(data)
    except (OSError, ValueError, TypeError, yaml.YAMLError) as exc:
        raise ConfigurationError(f"{path}: {exc}") from exc


def load_run_input(path: str | Path) -> ResolvedRun:
    from smartsom.config.resolver import resolve_run

    try:
        data = yaml.load(Path(path).read_text(encoding="utf-8"), Loader=_UniqueLoader)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        raise ConfigurationError(f"{path}: {exc}") from exc
    if isinstance(data, dict) and data.get("schema") in {
        "smartsom.resolved-run/v1",
        "smartsom.prepared-grid-experiment/v1",
        "smartsom.production-run/v1",
    }:
        return resolved_from_data(data)
    return resolve_run(path)
