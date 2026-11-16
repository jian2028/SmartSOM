"""Self-contained Tune snapshots of native scientific evidence.

No Ray reporting checkpoint points at a separately garbage-collected generation.
Immutable model payloads can share storage; mutable metadata is always copied.
"""

import hashlib
import json
import os
import re
import shutil
from pathlib import Path

EXCLUDED = {"ray", "control", "workers", "wandb", "tensorboard", "logs"}


def _copy(source, target, *, immutable=False):
    source = Path(source)
    if immutable or source.suffix == ".pt":
        try:
            os.link(source, target)
            return str(target)
        except OSError:
            pass
    return shutil.copy2(source, target)


def _included(relative):
    parts = relative.parts
    return (
        parts[0] not in EXCLUDED
        and parts[:2] != ("tracking", "wandb")
        and not (len(parts) == 1 and relative.name.endswith(".lock"))
    )


def _owned_files(root):
    """Traverse evidence without following runtime/SDK directories or links."""
    if root.is_symlink():
        raise ValueError("checkpoint evidence must not contain symlinks")
    if not root.exists():
        return
    for folder, directories, files in os.walk(root):
        for name in list(directories):
            path = Path(folder) / name
            if not _included(path.relative_to(root)):
                directories.remove(name)
            elif path.is_symlink():
                raise ValueError("checkpoint evidence must not contain symlinks")
        for name in files:
            path = Path(folder) / name
            if not _included(path.relative_to(root)):
                continue
            if path.is_symlink():
                raise ValueError("checkpoint evidence must not contain symlinks")
            yield path


def _reject_links(root):
    if root.is_symlink() or any(path.is_symlink() for path in root.rglob("*")):
        raise ValueError("checkpoint evidence must not contain symlinks")


def save(root, destination, state, *, immutable_digests=None, execution=None):
    root, destination = Path(root), Path(destination)
    if destination.resolve().is_relative_to(root.resolve()) and not (
        destination.resolve().is_relative_to(root.resolve() / "ray")
    ):
        raise ValueError("checkpoint destination must be outside native evidence")
    payload = destination / "native"
    payload.mkdir()
    files, retained = {}, {}
    for source in _owned_files(root):
        relative = source.relative_to(root)
        # The driver publishes this status concurrently via os.replace. Its
        # staging file is not committed evidence and can disappear after walk.
        # Keep _included unchanged so historical snapshots remain readable.
        if relative.parts == ("tune-status.json.tmp",):
            continue
        name = str(relative)
        # Native generations are published atomically and never modified.
        # Alias pointers and mutable receipts are copied and hashed anew.
        immutable = (
            len(relative.parts) == 3
            and relative.parts[0] == "continuation"
            and re.fullmatch(r"[0-9]{8}-[0-9a-f]{32}", relative.parts[1])
            and relative.parts[2] in {"model.pt", "state.json"}
        )
        path = payload / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        _copy(source, path, immutable=immutable)
        if immutable and immutable_digests is not None:
            stat = (root / relative).stat()
            # Hardlink creation/removal changes ctime but not published bytes.
            # A restored/replaced generation has a different inode or mtime.
            identity = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
            previous = immutable_digests.get(name)
            value = (
                previous[1]
                if previous is not None and previous[0] == identity
                else hashlib.sha256(path.read_bytes()).hexdigest()
            )
            retained[name] = (identity, value)
        else:
            value = hashlib.sha256(path.read_bytes()).hexdigest()
        files[name] = value
    if immutable_digests is not None:
        immutable_digests.clear()
        immutable_digests.update(retained)
    metadata = json.dumps(
        {"state": state, "files": files, "execution": execution}
    ).encode()
    (destination / "phase.json").write_bytes(metadata)
    (destination / "phase.sha256").write_text(hashlib.sha256(metadata).hexdigest())
    return str(destination)


def read_metadata(source):
    source = Path(source)
    encoded = (source / "phase.json").read_bytes()
    if hashlib.sha256(encoded).hexdigest() != (source / "phase.sha256").read_text():
        raise ValueError("Tune checkpoint metadata digest mismatch")
    metadata = json.loads(encoded)
    if not isinstance(metadata.get("state"), dict) or not isinstance(
        metadata["state"].get("phase"), str
    ):
        raise ValueError("invalid Tune checkpoint phase state")
    return metadata


def restore(source, root, *, trial_id=None):
    source, root = Path(source), Path(root)
    metadata = read_metadata(source)
    if trial_id is not None:
        from smartsom.experiments.tune_provenance import validate

        validate(metadata.get("execution"), trial_id)
    payload = source / "native"
    _reject_links(payload)
    existing_files = list(_owned_files(root))
    actual_files = {
        str(path.relative_to(payload)) for path in payload.rglob("*") if path.is_file()
    }
    if actual_files != set(metadata["files"]):
        raise ValueError("Tune checkpoint payload inventory mismatch")
    # Verify every payload before changing the destination.
    for name, digest in metadata["files"].items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or not _included(relative):
            raise ValueError("invalid checkpoint payload path")
        path = payload / relative
        if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError("Tune checkpoint payload digest mismatch")
    frozen_plan = payload / "plan.json"
    current_plan = root / "plan.json"
    if current_plan.exists() and (
        not frozen_plan.is_file()
        or json.loads(current_plan.read_text())["run_id"]
        != json.loads(frozen_plan.read_text())["run_id"]
    ):
        raise ValueError("checkpoint belongs to a different run")
    if (root / "continuation/current.json").exists() and not (
        payload / "continuation/current.json"
    ).is_file():
        raise ValueError("trained run cannot restore a bootstrap-only checkpoint")
    current_execution = root / "execution.json"
    saved_execution = payload / "execution.json"
    if current_execution.exists() and saved_execution.exists():
        current_settings = json.loads(current_execution.read_text())["settings"]
        saved_settings = json.loads(saved_execution.read_text())["settings"]
        if any(
            current_settings.get(key) != saved_settings.get(key)
            for key in ("resources", "trial_resources", "concurrency")
        ):
            raise ValueError(
                "Tune checkpoint resource settings differ from frozen execution"
            )
    root.mkdir(parents=True, exist_ok=True)
    # A retry can roll back receipts as well as weights. Keeping receipts from
    # later work would skip evaluation of the restored model. Only this run's
    # snapshot-owned paths are mirrored; process control and logs stay local.
    for path in existing_files:
        relative = path.relative_to(root)
        if relative.parts[0] == "tracking":
            # Published telemetry cannot be rolled back at its external sinks.
            continue
        if str(relative) not in metadata["files"]:
            path.unlink()
    for name in metadata["files"]:
        target = root / name
        if target.exists() and (
            name == "execution.json" or Path(name).parts[0] == "tracking"
        ):
            # Operational settings and publication ledgers retain their latest
            # local history. A fresh destination still receives the snapshot.
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".restore.tmp")
        shutil.copy2(payload / name, temporary)
        os.replace(temporary, target)
    return metadata["state"]
