"""Portable, checksummed model/experiment ZIPs without rewriting source evidence.

Archives contain regular files, a verified inventory and a relocation map. Import
never executes model data and refuses traversal, duplicate members and symlinks.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import tempfile
from pathlib import Path, PurePosixPath
from zipfile import ZIP_DEFLATED, ZipFile

from smartsom.experiments.catalog import (
    EXPERIMENT_SCHEMA,
    artifact_paths,
    contained_path,
    read_json,
)
from smartsom.experiments.evidence import write_json

BUNDLE_SCHEMA = "smartsom.bundle/v1"


def _sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def model_locator(source: str | Path, checkpoint: str = "last") -> Path:
    """Locate an explicit checkpoint, named v2 selection, or legacy final model."""
    root = Path(source).resolve()
    if root.is_file():
        data = read_json(root)
        if "algorithm" in data and data["algorithm"].get("checkpoint"):
            return (root.parent / data["algorithm"]["checkpoint"]).resolve(strict=True)
        if root.name == "checkpoint.json":
            return root.parent
        raise ValueError(f"not a checkpoint descriptor: {root}")
    if (root / "checkpoint.json").is_file():
        return root
    if Path(checkpoint).name != checkpoint or checkpoint in {".", ".."}:
        raise ValueError("checkpoint selector must be a single name")
    selected = root / "checkpoints" / checkpoint
    if selected.is_dir() and (selected / "checkpoint.json").is_file():
        return selected.resolve()
    if checkpoint != "last":
        raise ValueError(f"checkpoint {checkpoint!r} is unavailable in {root}")
    if (root / "checkpoint_algorithm.json").is_file():
        return model_locator(root / "checkpoint_algorithm.json")
    current = root / "run.json"
    if current.is_file() and read_json(current).get("schema") == EXPERIMENT_SCHEMA:
        paths = artifact_paths(root)
        if "checkpoint" in paths:
            return model_locator(paths["checkpoint"])
        if "training" in paths:
            return model_locator(paths["training"])
    raise ValueError(f"no saved checkpoint in {root}")


def _member(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if (
        not name
        or "\\" in name
        or path.is_absolute()
        or any(part in {".", ".."} for part in name.split("/"))
        or ":" in path.parts[0]
        or path.as_posix() != name
    ):
        raise ValueError(f"unsafe archive member: {name!r}")
    return path


def _walk_files(root: Path):
    """Materialize internal links once per path; refuse escapes and cycles."""
    root = root.resolve()

    def walk(directory, relative, ancestors):
        real = directory.resolve(strict=True)
        if not real.is_relative_to(root) or real in ancestors:
            raise ValueError(f"escaping or cyclic artifact link: {directory}")
        for child in sorted(directory.iterdir()):
            if child.name in {".git", ".venv", "__pycache__"}:
                continue
            name = relative / child.name
            target = child.resolve(strict=True)
            if not target.is_relative_to(root):
                raise ValueError(f"artifact link escapes export root: {child}")
            if target.is_dir():
                yield from walk(child, name, ancestors | {real})
            elif target.is_file():
                yield name.as_posix(), target
            else:
                raise ValueError(f"not a regular artifact file: {child}")

    yield from walk(root, PurePosixPath(), set())


def _checkpoint_files(root: Path) -> dict:
    manifest = read_json(root / "checkpoint.json")
    inventory = manifest.get("files")
    if not isinstance(inventory, list) or not inventory:
        raise ValueError("checkpoint has no file inventory")
    seen = set()
    for entry in inventory:
        name = str(_member(entry["path"]))
        if name in seen:
            raise ValueError("duplicate checkpoint member")
        seen.add(name)
        target = contained_path(root, name)
        if not target.is_file() or _sha(target) != entry["sha256"]:
            raise ValueError(f"checkpoint file digest mismatch: {name}")
    return manifest


def _write_bundle(source, destination, kind, files, generated, relocations) -> Path:
    root = Path(source).resolve()
    destination = Path(destination).absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    if destination.resolve().is_relative_to(root):
        raise ValueError("export destination must be outside source evidence")
    destination.parent.mkdir(parents=True, exist_ok=True)
    names = [name for name, _ in files] + list(generated)
    if len(names) != len(set(names)) or "bundle.json" in names:
        raise ValueError("duplicate export member")
    inventory = []
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
        with ZipFile(
            temporary, "w", compression=ZIP_DEFLATED, allowZip64=True
        ) as archive:
            for name, path in files:
                _member(name)
                before = _sha(path)
                size = path.stat().st_size
                archive.write(path, name)
                if _sha(path) != before or path.stat().st_size != size:
                    raise ValueError(f"artifact changed while exporting: {path}")
                inventory.append({"path": name, "sha256": before, "size": size})
            for name, value in generated.items():
                _member(name)
                payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
                archive.writestr(name, payload)
                inventory.append(
                    {
                        "path": name,
                        "sha256": hashlib.sha256(payload).hexdigest(),
                        "size": len(payload),
                    }
                )
            archive.writestr(
                "bundle.json",
                json.dumps(
                    {
                        "schema": BUNDLE_SCHEMA,
                        "kind": kind,
                        "source": str(root),
                        "entrypoint": "payload",
                        "relocations": relocations,
                        "files": sorted(inventory, key=lambda row: row["path"]),
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
            )
        verify_bundle(temporary)
        # Hard-link publication refuses a destination created concurrently.
        os.link(temporary, destination)
        return destination
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def export_model(
    source: str | Path, destination: str | Path, *, checkpoint: str = "last"
) -> Path:
    root = model_locator(source, checkpoint)
    evidence_root = Path(source).resolve()
    if evidence_root.is_file():
        evidence_root = evidence_root.parent
    if Path(destination).resolve().is_relative_to(root):
        raise ValueError("export destination must be outside source evidence")
    manifest = _checkpoint_files(root)
    files = [("payload/checkpoint/" + name, path) for name, path in _walk_files(root)]
    algorithm = {
        "schema": "smartsom.algorithm/v1",
        "algorithm": {
            key: manifest[key] for key in ("provider", "projection", "parameters")
        }
        | {
            "checkpoint": "checkpoint",
            "checkpoint_sha256": _sha(root / "checkpoint.json"),
        },
    }
    return _write_bundle(
        evidence_root,
        destination,
        "model",
        files,
        {"payload/checkpoint_algorithm.json": algorithm},
        {str(root): "payload/checkpoint"},
    )


def export_experiment(source: str | Path, destination: str | Path) -> Path:
    root = Path(source).resolve()
    if not root.is_dir():
        raise ValueError(f"experiment directory does not exist: {root}")
    if not any(
        (root / name).is_file() for name in ("run.json", "manifest.json", "report.json")
    ):
        raise ValueError(f"unrecognized experiment evidence: {root}")
    files = [("payload/" + name, path) for name, path in _walk_files(root)]
    if not files:
        raise ValueError("empty experiment")
    # Check declared model members before exporting, without importing a framework.
    for name, path in files:
        if name.endswith("/checkpoint.json"):
            _checkpoint_files(path.parent)
    return _write_bundle(
        root, destination, "experiment", files, {}, {str(root): "payload"}
    )


def verify_bundle(
    path: str | Path, *, max_bytes: int = 20 * 1024**3, max_files: int = 100_000
) -> dict:
    """Verify every byte and member before import; checksum is not authentication."""
    with ZipFile(path) as archive:
        members = archive.infolist()
        names = [entry.filename for entry in members]
        if len(members) > max_files or len(names) != len(set(names)):
            raise ValueError("too many or duplicate archive members")
        if sum(entry.file_size for entry in members) > max_bytes:
            raise ValueError("archive exceeds uncompressed size limit")
        for entry in members:
            _member(entry.filename)
            mode = entry.external_attr >> 16
            if (
                entry.is_dir()
                or stat.S_ISLNK(mode)
                or (stat.S_IFMT(mode) not in (0, stat.S_IFREG))
            ):
                raise ValueError(
                    f"archive member is not a regular file: {entry.filename}"
                )
        if "bundle.json" not in names:
            raise ValueError("archive has no bundle manifest")
        if archive.getinfo("bundle.json").file_size > 16 * 1024**2:
            raise ValueError("bundle manifest exceeds size limit")
        manifest = json.loads(archive.read("bundle.json"))
        if manifest.get("schema") != BUNDLE_SCHEMA or manifest.get("kind") not in {
            "model",
            "experiment",
        }:
            raise ValueError("unsupported bundle format")
        if manifest.get("entrypoint") != "payload":
            raise ValueError("invalid bundle entrypoint")
        records = manifest.get("files", [])
        expected = {row["path"] for row in records}
        if len(records) != len(expected) or expected != set(names) - {"bundle.json"}:
            raise ValueError("bundle inventory differs from archive members")
        for row in records:
            if not row["path"].startswith("payload/"):
                raise ValueError("bundle data must be under payload")
            if archive.getinfo(row["path"]).file_size != row["size"]:
                raise ValueError(f"bundle member size mismatch: {row['path']}")
            with archive.open(row["path"]) as stream:
                actual = hashlib.file_digest(stream, "sha256").hexdigest()
            if actual != row["sha256"]:
                raise ValueError(f"bundle member digest mismatch: {row['path']}")
        relocations = manifest.get("relocations", {})
        if not isinstance(relocations, dict):
            raise ValueError("invalid relocation map")
        for original, relative in relocations.items():
            if not isinstance(original, str) or not Path(original).is_absolute():
                raise ValueError("invalid relocation origin")
            if str(_member(relative)).split("/")[0] != "payload":
                raise ValueError("invalid relocation destination")
    return manifest


def import_bundle(path: str | Path, destination: str | Path) -> Path:
    """Import into a new directory, returning its payload; preserve raw evidence."""
    destination = Path(destination).absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    manifest = verify_bundle(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    try:
        with ZipFile(path) as archive:
            for row in manifest["files"]:
                name = row["path"]
                output = staging / str(_member(name))
                output.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(name) as incoming, output.open("xb") as outgoing:
                    shutil.copyfileobj(incoming, outgoing, length=1024 * 1024)
        write_json(staging / "bundle.json", manifest)
        # Check the extracted bytes too; do not leave a partial successful import.
        for row in manifest["files"]:
            if _sha(staging / row["path"]) != row["sha256"]:
                raise ValueError("extracted artifact digest mismatch")
        destination.mkdir()  # Atomic refusal of a concurrent destination.
        for item in staging.iterdir():
            item.rename(destination / item.name)
        return destination / "payload"
    finally:
        shutil.rmtree(staging)


def relocate_reference(bundle: str | Path, original: str | Path) -> Path:
    """Resolve a historical absolute reference through an imported bundle map.

    This returns a location, not a changed snapshot. Readers must still validate the
    original model and evidence digests. Paths outside the export remain unavailable.
    """
    root = Path(bundle).resolve()
    if not (root / "bundle.json").is_file() and root.name == "payload":
        root = root.parent
    manifest = read_json(root / "bundle.json")
    value = Path(original)
    if not value.is_absolute():
        return contained_path(root / manifest["entrypoint"], str(value))
    for old, relative in sorted(
        manifest["relocations"].items(), key=lambda item: -len(item[0])
    ):
        if value.is_relative_to(old):
            result = contained_path(root, relative) / value.relative_to(old)
            if not result.resolve().is_relative_to(root):
                raise ValueError("relocated reference escapes bundle")
            return result
    raise ValueError(f"reference was not included in this bundle: {original}")
