"""Strict v2 design files and optimistic, atomic saves, independent of Qt."""

import hashlib
import os
import stat
import tempfile
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field, ValidationError, model_validator

from smartsom.config.codec import (
    ConfigurationError,
    canonical_json,
    primitive,
    read_model,
)
from smartsom.config.models import StrictModel
from smartsom.domain.factory_design import FactoryDesign, validate_factory_design

FACTORY_DESIGN_SCHEMA = "smartsom.factory/v2"
_HEADER = """# SmartSOM factory design. Shared by Studio and production execution.
# Grid cells: origin at top left; x increases right, y increases down.
# Time uses ticks; energy uses abstract units. Capacity null means Unlimited.
# Inspection parallel_capacity: max uses all of that station's slots.
"""


class FactoryAuthoring(StrictModel):
    operation_catalog_mode: Literal["auto", "manual"] = "auto"


class FactoryDesignFile(StrictModel):
    schema_id: Literal["smartsom.factory/v2"] = Field(alias="schema")
    factory: FactoryDesign
    authoring: FactoryAuthoring = Field(default_factory=FactoryAuthoring)


class _FactoryDesignInput(StrictModel):
    schema_id: Literal["smartsom.factory/v2"] = Field(alias="schema")
    factory: dict[str, Any]
    authoring: FactoryAuthoring = Field(default_factory=FactoryAuthoring)

    @model_validator(mode="before")
    @classmethod
    def explicit_layout_migration(cls, data):
        if isinstance(data, dict) and data.get("schema") == "smartsom.factory/v1":
            raise ValueError(
                "matrix factories require migration to smartsom.factory/v2 with explicit grid, ports and capacities; "
                "the runtime does not infer a layout or implicit infinite buffers"
            )
        return data


def _legacy_catalog(value):
    if "operation_types" in value:
        return value
    machines = value.get("machines", [])
    if not (
        isinstance(machines, list)
        and all(
            isinstance(m, dict)
            and isinstance(m.get("operation_types", []), list)
            and all(isinstance(t, str) for t in m.get("operation_types", []))
            for m in machines
        )
    ):
        return value
    referenced = [t for m in machines for t in m.get("operation_types", [])]
    catalog = list(
        dict.fromkeys(
            [f"operation_{n}" for n in range(1, len(machines) + 1)] + referenced
        )
    )
    return {**value, "operation_types": catalog}


def _path(path: str | Path) -> Path:
    result = Path(path).expanduser()
    if result.suffix.lower() not in {".yaml", ".yml"}:
        raise ConfigurationError("Factory designs require a .yaml or .yml file")
    return result


def _valid(design: FactoryDesign) -> None:
    if not isinstance(design, FactoryDesign):
        raise ConfigurationError("Expected a FactoryDesign")
    errors = [
        issue for issue in validate_factory_design(design) if issue.severity == "error"
    ]
    if errors:
        details = "\n".join(
            f"{issue.entity_id or design.factory_id}: {issue.code}: {issue.message}"
            for issue in errors
        )
        raise ConfigurationError(details)


def load_factory_design(path: str | Path) -> tuple[FactoryDesign, str]:
    """Read one byte snapshot; return its validated design and SHA-256 digest.

    Incomplete designs with warnings are readable. Invalid geometry or references
    are rejected just as they are on save. Existing runtime v1 files are not migrated.
    """
    model, digest = load_factory_design_file(path)
    return model.factory, digest


def load_factory_design_file(path: str | Path) -> tuple[FactoryDesignFile, str]:
    """Read validated factory data and portable authoring preferences together."""
    source = _path(path)
    raw, digest = read_model(source, _FactoryDesignInput)
    data = primitive(raw)
    data["factory"] = _legacy_catalog(data["factory"])
    try:
        model = FactoryDesignFile.model_validate_json(canonical_json(data))
    except ValidationError as exc:
        raise ConfigurationError(f"{source}: {exc}") from exc
    _valid(model.factory)
    return model, digest


def _check_destination(path: Path, expected_digest: str | None) -> None:
    if not path.exists():
        if expected_digest is not None:
            raise ConfigurationError(
                f"{path}: file was removed externally; use Save As"
            )
        return
    if expected_digest is None:
        raise ConfigurationError(
            f"{path}: already exists; its expected_digest is required"
        )
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected_digest:
        raise ConfigurationError(
            f"{path}: file changed externally; reload or use Save As"
        )


def save_factory_design(
    path: str | Path,
    design: FactoryDesign,
    *,
    expected_digest: str | None = None,
) -> str:
    """Save a complete design and return the new file digest.

    Without a digest, create a new file exclusively. Replacing a file requires
    the byte digest obtained on load or the last successful save. Checks are
    optimistic; editors that do not share locks can still race after a check.
    A failed serialization, validation or replacement leaves the old file intact.
    """
    authoring = FactoryAuthoring()
    destination = _path(path).resolve()
    if expected_digest is not None:
        _check_destination(destination, expected_digest)
        envelope, _ = load_factory_design_file(destination)
        authoring = envelope.authoring
    return save_factory_design_file(
        path,
        FactoryDesignFile(
            schema=FACTORY_DESIGN_SCHEMA, factory=design, authoring=authoring
        ),
        expected_digest=expected_digest,
    )


def save_factory_design_file(
    path: str | Path, envelope: FactoryDesignFile, *, expected_digest: str | None = None
) -> str:
    """Atomically save the complete file, including authoring preferences."""
    if not isinstance(envelope, FactoryDesignFile):
        raise ConfigurationError("Expected a FactoryDesignFile")
    destination = _path(path).resolve()
    _valid(envelope.factory)
    payload = (
        _HEADER
        + yaml.safe_dump(
            primitive(envelope), allow_unicode=True, sort_keys=False, width=88
        )
    ).encode("utf-8")
    temporary: Path | None = None
    try:
        _check_destination(destination, expected_digest)
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            if destination.exists():
                os.chmod(temporary, stat.S_IMODE(destination.stat().st_mode))
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        _check_destination(destination, expected_digest)
        if expected_digest is None:
            # Linking publishes a complete file without replacing a concurrent create.
            os.link(temporary, destination)
        else:
            os.replace(temporary, destination)
        return hashlib.sha256(payload).hexdigest()
    except OSError as exc:
        raise ConfigurationError(f"Could not save {destination}: {exc}") from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
