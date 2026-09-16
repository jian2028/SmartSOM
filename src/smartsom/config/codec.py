"""Strict file decoding and canonical, location-independent domain encoding."""

import hashlib
import json
from dataclasses import fields, is_dataclass, replace
from decimal import Decimal
from pathlib import Path

import yaml
from pydantic import BaseModel

from smartsom.dispatch import DecisionContext
from smartsom.domain import ExecutionSchedule, FactorySpec, WorkloadInstance


class ConfigurationError(ValueError):
    """A file, field, reference or materialized input is invalid."""


class _UniqueLoader(yaml.SafeLoader):
    pass


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if not isinstance(key, str):
            raise ConfigurationError("mapping keys must be strings")
        if key in result:
            raise ConfigurationError(f"duplicate key: {key!r}")
        result[key] = value
    return result


def _yaml_mapping(loader, node):
    return _unique_pairs(
        (loader.construct_object(key), loader.construct_object(value))
        for key, value in node.value
    )


_UniqueLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _yaml_mapping
)


def primitive(value):
    """Produce detached JSON-compatible data, never mutable model internals."""
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json", by_alias=True)
    if is_dataclass(value):
        return {
            field.name: (
                dict(getattr(value, field.name))
                if field.name == "machine_nominal_ticks"
                else primitive(getattr(value, field.name))
            )
            for field in fields(value)
            if not (
                field.name in ("study_seed_origin", "holding_buffer", "buffer_id")
                and getattr(value, field.name) is None
            )
            if not (
                field.name == "holding_buffer_enabled"
                and not getattr(value, field.name)
                or field.name == "holding_buffer_sha256"
                and getattr(value, field.name) is None
            )
            if not (
                isinstance(value, FactorySpec)
                and (
                    (field.name == "transport" and value.transport is None)
                    or (field.name == "buffers" and not value.buffers)
                    or (field.name == "quality_speed" and value.quality_speed is None)
                )
            )
            and not (
                isinstance(value, DecisionContext)
                and field.name in ("quality_modes", "job_quality")
                and not getattr(value, field.name)
            )
            and not (
                isinstance(value, ExecutionSchedule)
                and value.version == 1
                and field.name in ("arrivals", "transfers", "action_order", "version")
            )
        }
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, dict):
        return {key: primitive(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [primitive(item) for item in value]
    return value


def canonical_json(value) -> str:
    return json.dumps(
        primitive(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def digest(value) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def read_model[T: BaseModel](path: Path, model: type[T]) -> tuple[T, str]:
    """Read once so the recorded byte digest describes the exact parsed input."""
    try:
        raw = path.read_bytes()
        if path.suffix.lower() == ".json":
            data = json.loads(raw, object_pairs_hook=_unique_pairs)
        elif path.suffix.lower() in {".yaml", ".yml"}:
            data = yaml.load(raw.decode("utf-8"), Loader=_UniqueLoader)
        else:
            raise ConfigurationError("expected a .yaml, .yml or .json file")
        # Strict JSON validation admits arrays as tuples, while rejecting scalar
        # coercion and propagating checks into stdlib domain dataclasses.
        parsed = model.model_validate_json(canonical_json(data))
        return parsed, hashlib.sha256(raw).hexdigest()
    except (OSError, ValueError, TypeError, yaml.YAMLError, RecursionError) as exc:
        raise ConfigurationError(f"{path}: {exc}") from exc


def normalize_factory(factory: FactorySpec) -> FactorySpec:
    return replace(
        factory,
        machines=tuple(sorted(factory.machines, key=lambda item: item.machine_id)),
    )


def normalize_workload(workload: WorkloadInstance) -> WorkloadInstance:
    return WorkloadInstance(
        tuple(
            replace(
                order,
                jobs=tuple(
                    replace(
                        job,
                        operations=tuple(
                            replace(
                                op,
                                modes=tuple(
                                    sorted(
                                        op.modes,
                                        key=lambda mode: mode.processing_mode_id,
                                    )
                                ),
                            )
                            for op in sorted(
                                job.operations, key=lambda item: item.operation_id
                            )
                        ),
                    )
                    for job in sorted(order.jobs, key=lambda item: item.job_id)
                ),
            )
            for order in sorted(workload.orders, key=lambda item: item.order_id)
        )
    )
