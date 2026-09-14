"""Strict v2 design files and optimistic, atomic saves, independent of Qt."""

import hashlib
import os
import stat
import tempfile
from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field

from smartsom.config.codec import ConfigurationError, primitive, read_model
from smartsom.config.models import StrictModel
from smartsom.domain.factory_design import FactoryDesign, validate_factory_design

FACTORY_DESIGN_SCHEMA = "smartsom.factory/v2"
_HEADER = """# SmartSOM factory design. This v2 design is not a v1 runtime input.
# Grid cells: origin at top left; x increases right, y increases down.
# Time uses ticks; energy uses abstract units. Capacity null means Unlimited.
# Inspection parallel_capacity: max uses all of that station's slots.
"""


class FactoryDesignFile(StrictModel):
    schema_id: Literal["smartsom.factory/v2"] = Field(alias="schema")
    factory: FactoryDesign


def _path(path: str | Path) -> Path:
    result = Path(path).expanduser()
    if result.suffix.lower() not in {".yaml", ".yml"}:
        raise ConfigurationError("Factory designs require a .yaml or .yml file")
    return result


def _valid(design: FactoryDesign) -> None:
    if not isinstance(design, FactoryDesign):
        raise ConfigurationError("Expected a FactoryDesign, not a v1 FactorySpec")
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
    source = _path(path)
    model, digest = read_model(source, FactoryDesignFile)
    _valid(model.factory)
    return model.factory, digest


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
    destination = _path(path).resolve()
    _valid(design)
    envelope = FactoryDesignFile(schema=FACTORY_DESIGN_SCHEMA, factory=design)
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
