"""Opt-in immutable generations; one atomic pointer precedes owned garbage collection."""

import errno
import hashlib
import inspect
import json
import os
import shutil
import threading
import time
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from uuid import uuid4

from smartsom._filesystem import atomic_replace, native_path

CONTRACT = "smartsom.checkpoint-retention/v1"
_LOCKS = {}
_GUARD = threading.Lock()
_LOCAL = threading.local()


def leased(function):
    """Protect a retained source while a public reader uses its artifact paths."""
    signature = inspect.signature(function)

    @wraps(function)
    def wrapper(*args, **kwargs):
        arguments = signature.bind(*args, **kwargs).arguments
        source = arguments.get("source")
        if source is None and "selector" in arguments:
            source = arguments["selector"].source
        if source is not None:
            path = Path(source).resolve()
            for root in (path, *path.parents):
                if native_path(root / "checkpoints/retention-owner.json").is_file():
                    with lease(root):
                        return function(*args, **kwargs)
        return function(*args, **kwargs)

    return wrapper


@contextmanager
def lease(root):
    """Readers and publication share a process-safe, thread-reentrant lease."""
    root = Path(root).resolve()
    key = str(root)
    with _GUARD:
        lock = _LOCKS.setdefault(key, threading.RLock())
    with lock:
        depths = getattr(_LOCAL, "depths", {})
        _LOCAL.depths = depths
        if depths.get(key, 0):
            depths[key] += 1
            try:
                yield
            finally:
                depths[key] -= 1
            return
        directory = native_path(root / "checkpoints")
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / ".retention.lock").open("a+b") as stream:
            stream.seek(0)
            if os.name == "nt":
                import msvcrt

                if not stream.read(1):
                    stream.write(b"0")
                    stream.flush()
                while True:
                    try:
                        stream.seek(0)
                        msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                        break
                    except OSError as exc:
                        if exc.errno not in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
                            raise
                        time.sleep(0.05)
            else:
                import fcntl

                fcntl.flock(stream, fcntl.LOCK_EX)
            depths[key] = 1
            try:
                yield
            finally:
                del depths[key]
                if os.name == "nt":
                    stream.seek(0)
                    msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(stream, fcntl.LOCK_UN)


def enabled(session):
    return session.config.checkpointing.retention is not None


def configured(prepared):
    return (
        json.loads(prepared.config_json).get("checkpointing", {}).get("retention")
        is not None
    )


def _read(path):
    return json.loads(native_path(path).read_text(encoding="utf-8"))


def _sha(path):
    with native_path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _sync_directory(path):
    if os.name != "nt":
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def _write(path, value):
    path = Path(path)
    temporary = path.with_name(".pending-" + uuid4().hex)
    with native_path(temporary).open("w", encoding="utf-8") as stream:
        json.dump(value, stream, allow_nan=False, ensure_ascii=False)
        stream.flush()
        os.fsync(stream.fileno())
    atomic_replace(temporary, path)
    _sync_directory(path.parent)


def manifest(root):
    path = Path(root) / "checkpoints/retention.json"
    if not native_path(path).exists():
        return None
    value = _read(path)
    owner = _read(Path(root) / "checkpoints/retention-owner.json")
    return _verify(root, value, owner)


def _verify(root, value, owner):
    if value["schema"] != CONTRACT or value["owner"] != owner["owner"]:
        raise ValueError("checkpoint retention ownership changed")
    for name in ("latest_full", "selected_best", "initial"):
        item = value.get(name)
        if item is None:
            continue
        if not item["path"].startswith("update-") or not item["path"][7:].isdigit():
            raise ValueError("invalid retained checkpoint path")
        directory = Path(root) / "checkpoints" / item["path"]
        if native_path(directory).is_symlink():
            raise ValueError("retained checkpoint cannot be a symlink")
        if _sha(directory / "snapshot.json") != item["snapshot_sha256"]:
            raise ValueError("retained snapshot hash mismatch")
        metadata = _read(directory / "snapshot.json")
        if metadata.get("retention_owner") != owner["owner"]:
            raise ValueError("retained generation ownership changed")
        for file in native_path(directory).rglob("*"):
            if file.is_symlink():
                raise ValueError("retained payload cannot contain a symlink")
        for file in native_path(directory).rglob("model.json"):
            model = _read(file)
            for filename, key in (
                ("weights.pt", "weights_sha256"),
                ("encoder.json", "encoder_sha256"),
            ):
                if _sha(file.parent / filename) != model[key]:
                    raise ValueError("retained model payload hash mismatch")
        if name == "latest_full" and (
            metadata.get("inference_only")
            or _sha(directory / "continuation.pkl") != metadata["continuation_sha256"]
        ):
            raise ValueError("retained continuation hash mismatch")
        if name == "latest_full" and (
            metadata["update"] != value["updates"]
            or metadata["physical_ticks"] != value["physical_ticks"]
        ):
            raise ValueError("retained progress boundary mismatch")
        if name == "selected_best" and metadata["update"] != value["best_update"]:
            raise ValueError("retained best selection boundary mismatch")
    return value


def resolve(root, selection):
    value = manifest(root)
    if value is None or selection not in {"last", "best", "recovery", "initial"}:
        return None
    item = value[
        {
            "last": "latest_full",
            "recovery": "latest_full",
            "best": "selected_best",
            "initial": "initial",
        }[selection]
    ]
    if item is None:
        raise ValueError(f"{selection} checkpoint does not exist")
    return Path(root) / "checkpoints" / item["path"]


def _owner(session):
    path = session.root / "checkpoints/retention-owner.json"
    if native_path(path).exists():
        value = _read(path)
        if value["scientific_sha256"] != session.prepared.scientific_sha256:
            raise ValueError("retention scientific identity changed")
        return value["owner"]
    if list(native_path(session.root / "checkpoints").glob("update-*")):
        raise ValueError(
            "bounded retention cannot adopt historical checkpoint directories"
        )
    owner = uuid4().hex
    _write(
        path,
        {
            "schema": CONTRACT,
            "owner": owner,
            "scientific_sha256": session.prepared.scientific_sha256,
        },
    )
    return owner


def generation(session, *, full):
    owner = _owner(session)
    directory = session.root / "checkpoints" / ("update-" + str(time.time_ns()))
    staging = directory.with_name(".building-" + uuid4().hex)
    native_path(staging).mkdir()
    _write(
        staging / "retention-generation.json",
        {"schema": CONTRACT, "owner": owner, "staging": staging.name},
    )
    session._write_snapshot(staging, full=full, retention_owner=owner)
    for file in native_path(staging).rglob("*"):
        if file.is_file():
            with file.open("r+b" if os.name == "nt" else "rb") as stream:
                os.fsync(stream.fileno())
    _sync_directory(staging)
    native_path(staging).rename(native_path(directory))
    _sync_directory(directory.parent)
    return directory


def _item(path):
    return {"path": path.name, "snapshot_sha256": _sha(path / "snapshot.json")}


def _publish(session, context):
    previous = manifest(session.root)
    latest = generation(session, full=True)
    best = (
        _item(context["model"])
        if context["best"]
        else (previous["selected_best"] if previous else None)
    )
    initial = context.get("initial") or (previous.get("initial") if previous else None)
    value = {
        "schema": CONTRACT,
        "owner": _owner(session),
        "latest_full": _item(latest),
        "selected_best": best,
        "initial": initial,
        "updates": session.updates,
        "physical_ticks": session.ticks,
        "best_update": session.best_update,
        "record": session.record,
        "directory_sync": "unsupported_windows" if os.name == "nt" else "fsync",
    }
    _verify(
        session.root, value, _read(session.root / "checkpoints/retention-owner.json")
    )
    _write(session.root / "checkpoints/retention.json", value)
    # Everything before this point is disposable; this pointer is authoritative.
    for name, item in (
        ("last", value["latest_full"]),
        ("recovery", value["latest_full"]),
        ("best", best),
    ):
        if item is not None:
            selected = _read(
                session.root / "checkpoints" / item["path"] / "snapshot.json"
            )
            _write(
                session.root / "checkpoints" / (name + ".json"),
                {
                    "checkpoint": item["path"],
                    "physical_ticks": selected["physical_ticks"],
                    "actual_update": selected["update"],
                },
            )
    collect(session.root)
    return latest


def collect(root):
    value = manifest(root)
    keep = {
        item["path"]
        for key in ("latest_full", "selected_best", "initial")
        if (item := value.get(key))
    }
    for directory in native_path(Path(root) / "checkpoints").glob("update-*"):
        if directory.name in keep or directory.is_symlink():
            continue
        metadata = directory / "snapshot.json"
        if (
            metadata.is_file()
            and _read(metadata).get("retention_owner") == value["owner"]
        ):
            shutil.rmtree(directory)
    for directory in native_path(Path(root) / "checkpoints").glob(".building-*"):
        if directory.is_symlink():
            continue
        marker = directory / "retention-generation.json"
        if marker.is_file():
            ownership = _read(marker)
            if ownership == {
                "schema": CONTRACT,
                "owner": value["owner"],
                "staging": directory.name,
            }:
                shutil.rmtree(directory)


def save(session, *, best=False):
    context = getattr(session, "_retention_context", None)
    if context is None:
        if session.updates != 0:
            raise ValueError(
                "retained checkpoints require a complete-update transaction"
            )
        with lease(session.root):
            initial = generation(session, full=False)
            return _publish(
                session, {"model": initial, "best": best, "initial": _item(initial)}
            )
    if context["model"] is None or context["model_update"] != session.updates:
        context["model"] = generation(session, full=False)
        context["model_update"] = session.updates
        if session.updates == 0 and session.settings.record_initial:
            context["initial"] = _item(context["model"])
    context["best"] |= best
    return context["model"]


def step(session, on_progress):
    with lease(session.root):
        session._retention_context = {
            "model": None,
            "model_update": None,
            "best": False,
        }
        try:
            result = session._step_update(None)
            _publish(session, session._retention_context)
            if on_progress:
                on_progress(result)
            return result
        finally:
            session._retention_context = None
