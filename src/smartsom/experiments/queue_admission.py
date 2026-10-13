"""Atomic admission shared by a directory driver and its live queue.

Execution sidecars never edit frozen plans. Reservations span pending actors,
running learners and retiring children; only their owning scheduler releases them.
No stale-PID inference or crash recovery is performed here.
"""

import json
import time
from contextlib import contextmanager
from pathlib import Path

from smartsom.experiments.control import boundary, read
from smartsom.experiments.evidence import write_json
from smartsom.experiments.locking import acquire, release

SCHEMA = "smartsom.queue-admission/v1"


@contextmanager
def transaction(root):
    root = Path(root)
    control = root / "control"
    control.mkdir(exist_ok=True)
    with (control / "admission.lock").open("a+") as stream:
        while True:
            try:
                acquire(stream)
                break
            except BlockingIOError:
                time.sleep(0.01)
        try:
            path = control / "admission.json"
            state = json.loads(path.read_text()) if path.is_file() else None
            yield state
            if state is not None:
                write_json(path, state)
        finally:
            release(stream)


def initialize(root, *, supported):
    owner = read(root)
    if not owner:
        raise ValueError("shared admission requires a registered directory driver")
    with transaction(root) as state:
        if state and state.get("queue"):
            raise ValueError("an adopted directory driver cannot be restarted")
        write_json(
            Path(root) / "control/admission.json",
            dict(schema=SCHEMA, owner_id=owner["id"], supported=supported, slots={}),
        )
        # Avoid overwriting the new generation with the old transaction value.
        if state is not None:
            state.update(
                schema=SCHEMA, owner_id=owner["id"], supported=supported, slots={}
            )


def _validate(state, owner_id):
    if not state or state.get("schema") != SCHEMA or state["owner_id"] != owner_id:
        raise ValueError("shared admission owner changed or is unavailable")


def claim(root, queue, *, limit, owner_id):
    with transaction(root) as state:
        _validate(state, owner_id)
        owner = read(root)
        if not owner or owner["id"] != owner_id:
            raise ValueError("directory control identity changed before adoption")
        if not state["supported"]:
            raise ValueError("active adoption requires off/online calibration")
        if state.get("queue"):
            raise ValueError("directory batch already belongs to a live queue")
        state.update(queue=str(queue), limit=limit, stopping=False)


def configure(root, queue, *, limit, stopping=False):
    with transaction(root) as state:
        if not state or state.get("queue") != str(queue):
            raise ValueError("shared admission belongs to another queue")
        state.update(limit=limit, stopping=stopping)


def detach(root, queue):
    """Release a drained queue claim so the original batch can resume normally."""
    with transaction(root) as state:
        if not state or state.get("queue") != str(queue) or state["slots"]:
            raise ValueError("queue admission is not drained")
        for key in ("queue", "limit", "stopping"):
            state.pop(key, None)


def reserve(root, key, *, owner_id, status="pending"):
    with transaction(root) as state:
        _validate(state, owner_id)
        if key in state["slots"]:
            return True
        if state.get("stopping") or (
            state.get("queue") and len(state["slots"]) >= state["limit"]
        ):
            return False
        state["slots"][key] = dict(status=status)
        return True


def retire(root, key, *, owner_id):
    with transaction(root) as state:
        _validate(state, owner_id)
        state["slots"].pop(key, None)


def source_exited(root, *, owner_id):
    """Called only after verified directory owner and all children have exited."""
    with transaction(root) as state:
        _validate(state, owner_id)
        state["slots"] = {
            key: value
            for key, value in state["slots"].items()
            if not key.startswith("source:")
        }


def synchronize(root, leases, *, owner_id):
    """Publish allocation phases, retaining every local reservation."""
    with transaction(root) as state:
        _validate(state, owner_id)
        for key, lease in leases.items():
            slot = state["slots"].get("source:" + key)
            if slot is not None:
                slot["status"] = lease.state


def snapshot(root):
    with transaction(root) as state:
        return dict(state or {})


@contextmanager
def smoke_slot(root, identity):
    owner_id = read(root)["id"]
    key = "source:" + identity
    while True:
        boundary(root, fresh=True)
        if reserve(root, key, owner_id=owner_id, status="preflight"):
            break
        time.sleep(0.1)
    try:
        yield
    finally:
        retire(root, key, owner_id=owner_id)
