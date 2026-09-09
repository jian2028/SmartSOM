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
    contained_path,
    read_json,
)
from smartsom.experiments.evidence import write_json

BUNDLE_SCHEMA = "smartsom.bundle/v1"


def _sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def model_locator(source: str | Path, checkpoint: str = "last") -> Path:
    """Locate inference files, preserving explicit best/last and bundle mappings.

    Update directories also contain training state; this helper deliberately
    returns their inference directory, not a promise of resumability.
    """
    if Path(checkpoint).name != checkpoint or checkpoint in {".", ".."}:
        raise ValueError("checkpoint selector must be a single name")
    seen = set()

    def locate(root):
        root = root.resolve(strict=True)
        if root in seen:
            raise ValueError(f"cyclic checkpoint reference: {root}")
        seen.add(root)
        if root.is_file():
            data = read_json(root)
            if root.name == "checkpoint.json":
                return root.parent
            algorithm = data.get("algorithm", data)
            if isinstance(algorithm, dict) and algorithm.get("checkpoint"):
                result = locate(locate_reference(root, algorithm["checkpoint"]))
                expected = algorithm.get("checkpoint_sha256")
                if expected and _sha(result / "checkpoint.json") != expected:
                    raise ValueError("checkpoint manifest digest mismatch")
                return result
            raise ValueError(f"not a checkpoint descriptor: {root}")
        if (root / "checkpoint.json").is_file():
            return root
        if (root / "inference/checkpoint.json").is_file():
            _update_files(root)
            return root / "inference"
        current = root / "run.json"
        if current.is_file() and read_json(current).get("schema") in {
            EXPERIMENT_SCHEMA,
            "smartsom.evaluation/v1",
        }:
            paths = read_json(current).get("paths", {})
            for key in ("training", "checkpoint"):
                if key in paths:
                    return locate(contained_path(root, paths[key]))
            identity = read_json(current).get("checkpoint")
            if isinstance(identity, dict) and identity.get("path"):
                return locate(locate_reference(current, identity["path"]))
        for selected in (
            root / "checkpoints" / f"{checkpoint}.json",
            root / "checkpoints" / checkpoint,
        ):
            if selected.exists():
                return locate(selected)
        if checkpoint != "last":
            raise ValueError(f"checkpoint {checkpoint!r} is unavailable in {root}")
        if (root / "checkpoint_algorithm.json").is_file():
            return locate(root / "checkpoint_algorithm.json")
        if (root / "checkpoint/checkpoint.json").is_file():
            return locate(root / "checkpoint")
        raise ValueError(f"no saved checkpoint in {root}")

    return locate(Path(source))


def locate_reference(owner: str | Path, value: str | Path) -> Path:
    """Resolve only a declared reference; imported evidence stays in its bundle."""
    owner = Path(owner).resolve()
    if not isinstance(value, (str, Path)) or not str(value):
        raise ValueError(f"invalid checkpoint reference in {owner}")
    original = Path(value)
    for parent in owner.parents:
        marker = parent / "bundle.json"
        if not marker.is_file():
            continue
        manifest = read_json(marker)
        if manifest.get("schema") != BUNDLE_SCHEMA:
            continue
        payload = contained_path(parent, manifest["entrypoint"])
        if not owner.is_relative_to(payload):
            continue
        if original.is_absolute():
            if original.resolve().is_relative_to(payload):
                # New attempts created after import already refer to this
                # package. Historical external paths still use the saved map.
                return original.resolve(strict=True)
            return relocate_reference(parent, original).resolve(strict=True)
        # A historical relative reference can leave its original evidence root.
        # Recover the owner's former location before resolving that reference;
        # the old bytes must not be rewritten to match the new dependencies tree.
        for old, relative in sorted(
            manifest["relocations"].items(), key=lambda item: -len(item[1])
        ):
            previous_location = contained_path(parent, relative)
            if owner.is_relative_to(previous_location):
                old_owner = Path(old) / owner.relative_to(previous_location)
                old_target = Path(os.path.normpath(old_owner.parent / original))
                return relocate_reference(parent, old_target).resolve(strict=True)
        target = (owner.parent / original).resolve(strict=True)
        if not target.is_relative_to(payload):
            raise ValueError("checkpoint reference escapes imported bundle")
        return target
    return (owner.parent / original).resolve(strict=True)


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


def _update_files(root: Path) -> None:
    manifest = read_json(root / "manifest.json")
    if (
        manifest.get("schema") != "smartsom.update-checkpoint/v1"
        or manifest.get("status") != "complete"
    ):
        raise ValueError(f"incomplete update checkpoint: {root}")
    actual = {
        name: _sha(path)
        for name, path in _walk_files(root)
        if name != "manifest.json" and not name.startswith("references/")
    }
    if manifest.get("files") != actual:
        raise ValueError(f"update checkpoint file digest mismatch: {root}")


def _training_snapshot(root: Path) -> Path | None:
    for candidate in (
        root / "resolved_training.json",
        root.parent / "resolved_training.json",
    ):
        if candidate.is_file():
            return candidate
    return None


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
        # Model contracts must still match after the archive was written, not
        # merely before export started. A changed source never gets published.
        for name, path in files:
            if name.endswith("/checkpoint.json"):
                _checkpoint_files(path.parent)
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
    relocations = {str(root): "payload/checkpoint"}
    snapshot = _training_snapshot(root)
    if snapshot is not None:
        files.append(("payload/resolved_training.json", snapshot))
        relocations[str(snapshot)] = "payload/resolved_training.json"
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
        relocations,
    )


def _declared_references(path: Path):
    """Read schema-defined model inputs, never arbitrary strings or source paths."""
    if path.name == "resolved_run.yaml":
        import yaml

        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    elif path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
    else:
        return
    if not isinstance(data, dict):
        return
    schema = data.get("schema")
    if schema == "smartsom.algorithm/v1":
        algorithm = data.get("algorithm", {})
    elif schema in {"smartsom.resolved-run/v1", "smartsom.resolved-training/v1"}:
        resolved = data.get("resolved", data)
        algorithm = resolved.get("algorithm", {}).get("algorithm", {})
    else:
        algorithm = {}
    if algorithm.get("checkpoint"):
        yield "checkpoint", algorithm["checkpoint"], algorithm.get("checkpoint_sha256")
    identities = []
    if schema in {EXPERIMENT_SCHEMA, "smartsom.evaluation/v1"}:
        identities.append(data.get("checkpoint"))
        identities.extend(row.get("checkpoint") for row in data.get("results", []))
    if path.name == "plan.json":
        identities.extend(row.get("checkpoint") for row in data.get("entries", []))
    for identity in identities:
        if not isinstance(identity, dict):
            continue
        if identity.get("path"):
            yield "checkpoint", identity["path"], identity.get("manifest_sha256")
        if identity.get("training_snapshot"):
            yield (
                "snapshot",
                identity["training_snapshot"],
                identity.get("training_snapshot_sha256"),
            )


def _dependency_closure(root: Path, files: list) -> dict[str, str]:
    relocations = {str(root): "payload"}
    # Re-export composes prior relocation aliases as well as direct references.
    # This preserves owner-relative references whose original parent has moved.
    for parent in root.parents:
        marker = parent / "bundle.json"
        if not marker.is_file():
            continue
        manifest = read_json(marker)
        if manifest.get("schema") != BUNDLE_SCHEMA:
            continue
        for original, relative in manifest["relocations"].items():
            target = contained_path(parent, relative)
            if target.is_relative_to(root):
                suffix = target.relative_to(root).as_posix()
                relocations[original] = (
                    "payload" if suffix == "." else f"payload/{suffix}"
                )
        break
    included = {path: name for name, path in files}
    checked = set()
    index = 0
    while index < len(files):
        _, owner = files[index]
        index += 1
        if owner in checked:
            continue
        checked.add(owner)
        for kind, raw, expected in _declared_references(owner):
            target = locate_reference(owner, raw)
            if kind == "checkpoint":
                model = model_locator(target)
                _checkpoint_files(model)
                if expected and _sha(model / "checkpoint.json") != expected:
                    raise ValueError(f"declared checkpoint digest mismatch: {owner}")
                # A reference denotes the exact model directory, not its parent
                # training tree. Never include unrelated external evidence.
                target = model
            elif not target.is_file():
                raise ValueError(f"declared training snapshot is not a file: {target}")
            elif expected and _sha(target) != expected:
                raise ValueError(f"declared training snapshot digest mismatch: {owner}")
            if target.is_relative_to(root):
                prefix = "payload/" + target.relative_to(root).as_posix()
            elif target in included:
                prefix = included[target]
            elif str(target) in relocations:
                prefix = relocations[str(target)]
            else:
                identity = hashlib.sha256(str(target).encode()).hexdigest()[:16]
                prefix = f"payload/_dependencies/{identity}/{target.name}"
                additions = (
                    [(prefix + "/" + name, path) for name, path in _walk_files(target)]
                    if target.is_dir()
                    else [(prefix, target)]
                )
                files.extend(additions)
                included.update((path, name) for name, path in additions)
            relocations[str(target)] = prefix
            if Path(raw).is_absolute():
                relocations[str(raw)] = prefix
    return relocations


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
    relocations = _dependency_closure(root, files)
    target = Path(destination).resolve()
    for original in relocations:
        dependency = Path(original)
        boundary = dependency.parent if dependency.is_file() else dependency
        if target.is_relative_to(boundary):
            raise ValueError("export destination must be outside source evidence")
    # Check declared model members before exporting, without importing a framework.
    for name, path in files:
        if name.endswith("/checkpoint.json"):
            _checkpoint_files(path.parent)
    return _write_bundle(root, destination, "experiment", files, {}, relocations)


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
