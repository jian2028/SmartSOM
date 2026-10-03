"""Local cooperative cancellation with verified process identities.

Control records are execution metadata, never frozen scientific inputs.
No signal is sent during ordinary stop: runners poll at their safe boundaries.
"""

import json
import math
import os
import signal
import subprocess
import sys
import threading
import time
from contextvars import ContextVar
from pathlib import Path
from uuid import uuid4

from smartsom._filesystem import atomic_replace, read_text

CURRENT = ContextVar("smartsom_control", default=None)
ACTIVE = {"running", "stop_requested", "stopping"}
_REQUEST_CACHE = {}


def write_json(path, value):
    temporary = path.with_name(path.name + "." + uuid4().hex + ".tmp")
    try:
        temporary.write_text(json.dumps(value, allow_nan=False) + "\n")
        atomic_replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class StopRequested(KeyboardInterrupt):
    """Cooperative stop, distinguishable from an execution failure."""


def _birth(pid):
    """Kernel start identity; ps's second-resolution timestamp is insufficient."""
    if os.name == "nt":
        from smartsom.experiments.windows_processes import birth

        return birth(pid)
    if sys.platform == "linux":
        try:
            fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
            return fields[19]  # field 22, starttime in boot ticks
        except (OSError, IndexError):
            return None
    if sys.platform == "darwin":
        import ctypes
        import struct

        # PROC_PIDTBSDINFO: twelve uint32 fields, comm/name (16/32 bytes),
        # six uint32 fields, and start timeval's two uint64 fields.
        library = ctypes.CDLL("/usr/lib/libproc.dylib")
        buffer = ctypes.create_string_buffer(136)
        size = library.proc_pidinfo(pid, 3, 0, buffer, len(buffer))
        if size == len(buffer):
            return ":".join(
                str(value) for value in struct.unpack_from("=QQ", buffer.raw, 120)
            )
        return None
    return None


def processes():
    if os.name == "nt":
        from smartsom.experiments.windows_processes import processes as windows_table

        return windows_table()
    if os.name != "posix":
        raise ValueError("local process control currently requires POSIX")
    result = subprocess.run(
        ["ps", "-axo", "pid=,ppid=,lstart=,stat="],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "LC_ALL": "C"},
    )
    rows = {}
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 8:
            born = _birth(int(parts[0]))
            if born is None:
                continue
            rows[int(parts[0])] = {
                "pid": int(parts[0]),
                "parent": int(parts[1]),
                "created": born,
                "started_at": " ".join(parts[2:7]),
                "state": parts[7],
            }
    return rows


def alive(identity, table):
    row = table.get(identity["pid"])
    return bool(
        row
        and row["created"] == identity["created"]
        and not row["state"].startswith("Z")
    )


def _parent_precedes_child(parent, child):
    """A stale parent PID must not attach an older process to a new owner."""
    try:
        return tuple(map(int, parent["created"].split(":"))) <= tuple(
            map(int, child["created"].split(":"))
        )
    except (ValueError, KeyError, AttributeError):
        return False


def read(root):
    path = Path(root) / "control/owner.json"
    return json.loads(read_text(path)) if path.is_file() else None


def requested(root=None):
    scope = CURRENT.get()
    roots = [Path(root)] if root is not None else list(scope.roots) if scope else []
    for directory in roots:
        directory = directory.resolve()
        cached = _REQUEST_CACHE.get(directory)
        if cached and time.monotonic() - cached[0] < 0.2:
            if cached[1]:
                return True
            continue
        found = False
        for parent in (directory.resolve(), *directory.resolve().parents):
            owner = read(parent)
            request = parent / "control/stop.json"
            if owner and owner["status"] in ACTIVE and request.is_file():
                if json.loads(request.read_text()).get("id") == owner["id"]:
                    found = True
                    break
        _REQUEST_CACHE[directory] = (time.monotonic(), found)
        if found:
            return True
    return False


def boundary(root=None):
    if requested(root):
        raise StopRequested("stop requested at a safe boundary")


def set_preflight_coverage(directory, coverage):
    """Change only optional smoke coverage for a verified live run."""
    if coverage not in {"representative", "skip"}:
        raise ValueError("coverage must be representative or skip")
    root = Path(directory).expanduser().resolve()
    owner = read(root)
    state_path = root / "preflight.json"
    if (
        not owner
        or owner.get("status") not in ACTIVE
        or not alive(owner["owner"], processes())
        or not state_path.is_file()
    ):
        raise ValueError("no verified active preflight owner")
    state = json.loads(state_path.read_text())
    if state.get("status") != "running" or state.get("level") != "full":
        raise ValueError("optional full preflight smoke is not running")
    pending_path = root / "control/preflight.json"
    pending = json.loads(pending_path.read_text()) if pending_path.is_file() else {}
    current = state.get("coverage")
    if pending.get("owner_id") == owner["id"]:
        current = pending.get("coverage", current)
    if current == "skip":
        raise ValueError("preflight smoke has already been skipped")
    if current == "representative" and coverage != "skip":
        raise ValueError("preflight coverage is already representative")
    write_json(
        root / "control/preflight.json",
        {
            "owner_id": owner["id"],
            "coverage": coverage,
            "requested_at": time.time(),
        },
    )
    return {"directory": str(root), "requested_coverage": coverage}


class Scope:
    def __init__(self):
        self.id = uuid4().hex
        self.roots = {}
        self.members = {}
        self.closed = threading.Event()
        self.thread = None
        self.guard = threading.RLock()
        self.excluded = None
        self.driver_root = None

    def inherit_driver(self, root):
        """A spawned worker inherits only its verified live ancestor driver."""
        root = Path(root).resolve()
        owner = read(root)
        table = processes()
        if (
            not owner
            or owner.get("status") not in ACTIVE
            or not alive(owner["owner"], table)
        ):
            raise ValueError("shared driver registration is unavailable")
        pid = os.getpid()
        ancestors = set()
        while pid in table and pid not in ancestors:
            ancestors.add(pid)
            child = table[pid]
            pid = child["parent"]
            if pid in table and not _parent_precedes_child(table[pid], child):
                break
        if owner["owner"]["pid"] not in ancestors:
            raise ValueError("worker is not a descendant of the registered driver")
        self.driver_root = root

    def bind(self, root):
        root = Path(root).resolve()
        with self.guard:
            if root in self.roots:
                return
            table = processes()
            identity = table.get(os.getpid())
            if identity is None:
                raise ValueError("cannot verify control process identity")
            if self.excluded is None:
                # Children already present before execution may belong to other work.
                self.excluded = {
                    (pid, row["created"])
                    for pid, row in table.items()
                    if pid != os.getpid()
                }
            previous = read(root)
            if (
                previous
                and previous["status"] in ACTIVE
                and alive(previous["owner"], table)
            ):
                raise ValueError("run already has a live control owner")
            (root / "control").mkdir(exist_ok=True)
            owner = {
                "schema": "smartsom.run-control/v1",
                "id": self.id,
                "root": str(root),
                "driver_root": str(self.driver_root or next(iter(self.roots), root)),
                "owner": identity,
                "members": [],
                "status": "running",
            }
            self.roots[root] = owner
            _REQUEST_CACHE.pop(root, None)
            write_json(root / "control/owner.json", owner)
            if self.thread is None:
                self.thread = threading.Thread(target=self.watch, daemon=True)
                self.thread.start()

    def watch(self):
        while not self.closed.wait(0.25):
            try:
                self.capture()
            except (OSError, ValueError, subprocess.SubprocessError) as exc:
                # Incomplete ownership must never authorize a forced cleanup.
                with self.guard:
                    for root, owner in self.roots.items():
                        owner["ownership_error"] = str(exc)
                        write_json(root / "control/owner.json", owner)

    def capture(self):
        table = processes()
        with self.guard:
            self.members = {
                pid: item for pid, item in self.members.items() if alive(item, table)
            }
            owned = {os.getpid()} | {
                pid for pid, item in self.members.items() if alive(item, table)
            }
            changed = True
            while changed:
                changed = False
                for pid, item in table.items():
                    if (
                        pid not in owned
                        and (pid, item["created"]) not in self.excluded
                        and item["parent"] in owned
                        and _parent_precedes_child(table[item["parent"]], item)
                    ):
                        self.members[pid] = item
                        owned.add(pid)
                        changed = True
            for root, owner in self.roots.items():
                owner["members"] = list(self.members.values())
                owner["updated_at"] = time.time()
                if requested(root):
                    owner["status"] = "stopping"
                write_json(root / "control/owner.json", owner)

    def finish(self, status):
        self.closed.set()
        if self.thread:
            self.thread.join()
        with self.guard:
            for root, owner in self.roots.items():
                owner["status"] = "stopping" if requested(root) else status
                owner["finished_at"] = time.time()
                write_json(root / "control/owner.json", owner)


def bind(root):
    scope = CURRENT.get()
    if scope is not None:
        scope.bind(root)


def stop(directory, *, timeout=60, force=False):
    if not math.isfinite(timeout) or timeout < 0:
        raise ValueError("stop timeout must be finite and nonnegative")
    root = Path(directory).expanduser().resolve()
    owner = read(root)
    if not owner or owner.get("schema") != "smartsom.run-control/v1":
        raise ValueError(
            "no verified control registration; legacy runs cannot be stopped safely"
        )
    if owner.get("root") != str(root):
        raise ValueError("control record belongs to a different run directory")
    if owner.get("driver_root", str(root)) != str(root):
        raise ValueError(
            f"this is a child of a shared driver; stop {owner['driver_root']} instead"
        )
    if owner.get("ownership_error"):
        raise ValueError(
            "process ownership tracking is incomplete; cannot verify stopping"
        )
    table = processes()
    live = [item for item in [owner["owner"], *owner["members"]] if alive(item, table)]
    if not live:
        owner["status"] = "stopped" if owner["status"] in ACTIVE else owner["status"]
        write_json(root / "control/owner.json", owner)
        return {"status": owner["status"], "directory": str(root), "remaining": []}
    if owner["status"] not in ACTIVE and not any(
        alive(item, table) for item in owner["members"]
    ):
        return {"status": owner["status"], "directory": str(root), "remaining": []}
    if owner["status"] not in ACTIVE or not alive(owner["owner"], table):
        raise ValueError(
            "control owner is unavailable; refusing to guess process ownership"
        )
    if owner["owner"]["pid"] == os.getpid():
        raise ValueError("cannot stop the calling control process")
    write_json(
        root / "control/stop.json", {"id": owner["id"], "requested_at": time.time()}
    )
    deadline = time.monotonic() + timeout
    while True:
        current = read(root)
        if current["id"] != owner["id"]:
            raise ValueError("run control identity changed while waiting")
        if current.get("ownership_error"):
            raise ValueError(
                "process ownership became incomplete; cannot certify stopped"
            )
        table = processes()
        live = [
            item
            for item in [current["owner"], *current["members"]]
            if alive(item, table)
        ]
        if not live or time.monotonic() >= deadline:
            break
        time.sleep(0.2)
    forced = bool(live and force)
    if forced:
        # Signal only identities registered by the driver, rechecking each PID.
        signals = (None,) if os.name == "nt" else (signal.SIGTERM, signal.SIGKILL)
        for signum in signals:
            current = read(root)
            if current["id"] != owner["id"] or current.get("ownership_error"):
                raise ValueError(
                    "control ownership changed or became incomplete; refusing force"
                )
            for item in reversed(live):
                if alive(item, processes()):
                    try:
                        if os.name == "nt":
                            from smartsom.experiments.windows_processes import terminate

                            terminate(item)
                        else:
                            os.kill(item["pid"], signum)
                    except ProcessLookupError:
                        pass
            until = time.monotonic() + 2
            while time.monotonic() < until:
                live = [item for item in live if alive(item, processes())]
                if not live:
                    break
                time.sleep(0.1)
    current = read(root)
    current["status"] = "stopping" if live else "force_stopped" if forced else "stopped"
    current["forced"] = forced
    if not live:
        write_json(root / "control/owner.json", current)
    return {
        "directory": str(root),
        "status": current["status"],
        "remaining": [item["pid"] for item in live],
        "recovery": "latest committed recovery state; unfinished work may be lost",
        "recovery_points": recovery_points(root),
    }


def recovery_points(root):
    roots = {"run": Path(root)}
    results = {}
    for name in ("batch.json", "study.json"):
        path = Path(root) / name
        if path.is_file():
            for key, row in json.loads(path.read_text()).get("entries", {}).items():
                if row.get("checkpoint"):
                    results[key] = row["checkpoint"]
                if row.get("run_dir"):
                    roots[key] = Path(row["run_dir"])
                for phase, stage in row.get("stages", {}).items():
                    if stage.get("run_dir"):
                        roots[key + ":" + phase] = Path(stage["run_dir"])
    for key, directory in roots.items():
        for name in ("adaptive-recovery.json", "recovery.json"):
            path = directory / "checkpoints" / name
            if path.is_file():
                pointer = json.loads(path.read_text())["checkpoint"]
                results[key] = str(path.parent / pointer)
                break
    return results
