"""Explicit POSIX background execution of an already frozen author plan.

The launcher never rereads authoring inputs. Execution metadata and startup
handshakes are independent of the immutable scientific plan.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

from smartsom.experiments import control
from smartsom.telemetry.runtime import DisplayOptions, bind, operation

REQUEST_SCHEMA = "smartsom.background-request/v1"
STARTUP_SCHEMA = "smartsom.background-startup/v1"
STARTUP_TIMEOUT = 20.0


def _read_json(path):
    return json.loads(path.read_text())


def _require_plan(root):
    if not root.is_dir() or not (root / "run.json").is_file():
        raise ValueError(
            "background execution requires an allocated frozen run directory"
        )
    plan = _read_json(root / "plan.json")
    if plan.get("schema") != "smartsom.author-plan/v1":
        raise ValueError("background execution requires a frozen author-plan/v1")


def _check_owner(root):
    owner = control.read(root)
    if owner is None:
        return
    if (
        owner.get("schema") != "smartsom.run-control/v1"
        or owner.get("root") != str(root)
        or owner.get("driver_root", str(root)) != str(root)
    ):
        raise ValueError("background run control registration has uncertain ownership")
    if owner.get("ownership_error"):
        raise ValueError("background run process ownership is incomplete")
    table = control.processes()
    if any(
        control.alive(identity, table)
        for identity in [owner["owner"], *owner.get("members", [])]
    ):
        raise ValueError("run still has a live control owner or owned worker")


@contextmanager
def _launch_lock(root):
    path = root / "control/background-launch.lock"
    identity = {"pid": os.getpid(), "created": control._birth(os.getpid())}
    if identity["created"] is None:
        raise ValueError("cannot verify background launcher process identity")
    lock = {**identity, "nonce": uuid4().hex}
    try:
        stream = path.open("x")
    except FileExistsError:
        previous = _read_json(path)
        if control._birth(previous["pid"]) == previous.get("created"):
            raise ValueError("another background launch is in progress") from None
        if _read_json(path) != previous:
            raise ValueError(
                "background launch lock changed during ownership verification"
            )
        path.unlink()
        stream = path.open("x")
    try:
        with stream:
            json.dump(lock, stream)
            stream.flush()
        yield
    finally:
        if path.is_file() and _read_json(path) == lock:
            path.unlink()


def _failure(
    root, nonce, error, *, status="failed", control_id=None, expected_owner=None
):
    """Retain inputs, stage ledger and results while recording the launch failure."""
    if not root.is_dir():
        return
    owner = control.read(root)
    own_control = owner and (
        owner.get("id") == control_id
        or expected_owner is not None
        and owner.get("owner") == expected_owner
    )
    foreign_live = bool(
        owner
        and not own_control
        and owner.get("status") in control.ACTIVE
        and control._birth(owner["owner"]["pid"]) == owner["owner"]["created"]
    )
    for name in ("run.json", "batch.json"):
        path = root / name
        if path.is_file() and not foreign_live:
            data = _read_json(path)
            data.update(status=status, error=str(error))
            control.write_json(path, data)
    payload = {
        "schema": STARTUP_SCHEMA,
        "nonce": nonce,
        "root": str(root),
        "status": "failed",
        "lifecycle_status": status,
        "error": str(error),
        "updated_at": time.time(),
    }
    if owner is not None and not foreign_live:
        payload.update(owner=owner["owner"], control_id=owner["id"])
    if foreign_live:
        payload["manifest_preserved_for_live_owner"] = True
    (root / "control").mkdir(exist_ok=True)
    control.write_json(root / "control/background-startup.json", payload)


def _request(root, nonce, resume):
    data = _read_json(root / "control/background-request.json")
    if (
        data.get("schema") != REQUEST_SCHEMA
        or data.get("root") != str(root)
        or data.get("nonce") != nonce
        or data.get("resume") is not resume
    ):
        raise ValueError("background startup request identity changed")
    if data.get("abort_requested"):
        raise ValueError("background startup was cancelled before execution")
    return data


@contextmanager
def _handshake_lock(root):
    """Make timeout cancellation and the child's ready commit mutually exclusive."""
    import fcntl

    with (root / "control/background-handshake.lock").open("a+b") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def _lifecycle_status(root, result=None):
    if isinstance(result, dict) and result.get("status"):
        return result["status"]
    if getattr(result, "status", None):
        return result.status
    path = root / "run.json"
    return _read_json(path).get("status", "running") if path.is_file() else "running"


@operation("run")
def _execute(root, *, max_concurrent=None):
    bind(root)
    # Frameworks, user modules and frozen scientific data are verified by the
    # ordinary driver. Startup success establishes process ownership only.
    from smartsom.experiments.author_driver import execute_saved

    return execute_saved(root, max_concurrent=max_concurrent)


def child(root, *, nonce, resume=False):
    """Private child entry. A terminal cannot provide its input or lifetime."""
    root = Path(root).expanduser().resolve()
    scope, token = None, None
    status = "failed"
    try:
        _require_plan(root)
        request = _request(root, nonce, resume)
        scope = control.Scope()
        token = control.CURRENT.set(scope)
        scope.bind(root)
        owner = control.read(root)
        with _handshake_lock(root):
            _request(root, nonce, resume)
            control.write_json(
                root / "control/background-startup.json",
                {
                    "schema": STARTUP_SCHEMA,
                    "nonce": nonce,
                    "root": str(root),
                    "status": "ready",
                    "lifecycle_status": "running",
                    "owner": owner["owner"],
                    "control_id": owner["id"],
                    "updated_at": time.time(),
                },
            )
        result = _execute(
            root,
            max_concurrent=request.get("max_concurrent"),
            display_options={**request["display_options"], "progress": "off"},
        )
        status = _lifecycle_status(root, result)
        return result
    except BaseException as exc:
        status = "stopped" if isinstance(exc, control.StopRequested) else "failed"
        _failure(root, nonce, exc, status=status, control_id=getattr(scope, "id", None))
        raise
    finally:
        if scope is not None:
            scope.finish(status)
        if token is not None:
            control.CURRENT.reset(token)


def _verify_ready(root, payload, process, created):
    owner = control.read(root)
    identity = payload.get("owner")
    if (
        payload.get("schema") != STARTUP_SCHEMA
        or payload.get("root") != str(root)
        or not owner
        or owner.get("root") != str(root)
        or owner.get("id") != payload.get("control_id")
        or identity != owner.get("owner")
        or identity.get("pid") != process.pid
        or identity.get("created") != created
    ):
        raise ValueError("background startup owner does not match the launched process")
    current_birth = control._birth(process.pid)
    if current_birth != created and not (
        process.poll() is not None and owner.get("status") not in control.ACTIVE
    ):
        raise ValueError(
            "background process identity changed before startup confirmation"
        )
    return owner


def _startup_result(root, nonce, process, created, stdout_path, stderr_path):
    path = root / "control/background-startup.json"
    payload = _read_json(path) if path.is_file() else None
    if payload is None or payload.get("nonce") != nonce:
        return None
    if payload.get("status") == "failed":
        raise ValueError(f"background startup failed: {payload.get('error')}")
    if payload.get("status") != "ready":
        raise ValueError("background child reported an invalid startup status")
    owner = _verify_ready(root, payload, process, created)
    status = owner["status"]
    if status in {"stopped", "force_stopped"} and process.poll() is None:
        status = "stopping"
    return {
        "status": status,
        "run_status": _lifecycle_status(root),
        "startup_status": "ready",
        "background": True,
        "directory": str(root),
        "owner": owner["owner"],
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
    }


def launch(root, *, resume=False, display_options=None, max_concurrent=None):
    """Return after the child registers a verified owner, never after mere spawn."""
    if os.name != "posix" or sys.platform not in {"darwin", "linux"}:
        raise ValueError("background execution is supported only on macOS and Linux")
    if type(resume) is not bool:
        raise ValueError("background resume must be boolean")
    if max_concurrent is not None and (
        type(max_concurrent) is not int or max_concurrent < 1
    ):
        raise ValueError("background concurrency must be a positive integer")
    root = Path(root).expanduser().resolve()
    _require_plan(root)
    display = asdict(DisplayOptions.from_value(display_options))
    display["progress"] = "off"
    (root / "control").mkdir(exist_ok=True)
    (root / "logs").mkdir(exist_ok=True)
    with _launch_lock(root):
        _check_owner(root)
        nonce = uuid4().hex
        request_path = root / "control/background-request.json"
        control.write_json(
            request_path,
            {
                "schema": REQUEST_SCHEMA,
                "nonce": nonce,
                "root": str(root),
                "resume": resume,
                "display_options": display,
                "max_concurrent": max_concurrent,
                "created_at": time.time(),
            },
        )
        command = [
            sys.executable,
            "-m",
            "smartsom.experiments.background",
            str(root),
            "--nonce",
            nonce,
        ]
        if resume:
            command.append("--resume")
        stdout_path, stderr_path = root / "logs/stdout.log", root / "logs/stderr.log"
        process, created = None, None
        try:
            with stdout_path.open("ab") as stdout, stderr_path.open("ab") as stderr:
                process = subprocess.Popen(
                    command,
                    stdin=subprocess.DEVNULL,
                    stdout=stdout,
                    stderr=stderr,
                    close_fds=True,
                    start_new_session=True,
                    env={
                        **os.environ,
                        "NO_COLOR": "1",
                        "TERM": "dumb",
                        "FORCE_COLOR": "0",
                        "CLICOLOR": "0",
                    },
                )
            created = control._birth(process.pid)
            if created is None:
                raise ValueError("cannot verify the launched background process birth")
            deadline = time.monotonic() + STARTUP_TIMEOUT
            while True:
                result = _startup_result(
                    root, nonce, process, created, stdout_path, stderr_path
                )
                if result is not None:
                    return result
                if process.poll() is not None:
                    raise ValueError(
                        "background child exited before verified startup; "
                        f"see {stderr_path}"
                    )
                if time.monotonic() >= deadline:
                    with _handshake_lock(root):
                        result = _startup_result(
                            root, nonce, process, created, stdout_path, stderr_path
                        )
                        if result is not None:
                            return result
                        request = _read_json(request_path)
                        request["abort_requested"] = True
                        control.write_json(request_path, request)
                    raise ValueError(
                        "background startup confirmation timed out; startup cancellation "
                        f"was requested without signalling unverified processes; see {stderr_path}"
                    )
                time.sleep(0.05)
        except BaseException as exc:
            request = _read_json(request_path)
            request["abort_requested"] = True
            control.write_json(request_path, request)
            previous = control.read(root)
            expected_owner = None
            if previous and process and created is not None:
                identity = previous.get("owner", {})
                if (
                    identity.get("pid") == process.pid
                    and identity.get("created") == created
                ):
                    expected_owner = identity
            _failure(root, nonce, exc, expected_owner=expected_owner)
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Execute a frozen SmartSOM background run"
    )
    parser.add_argument("root")
    parser.add_argument("--nonce", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    try:
        child(args.root, nonce=args.nonce, resume=args.resume)
    except BaseException as exc:
        print(f"SmartSOM background execution failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
