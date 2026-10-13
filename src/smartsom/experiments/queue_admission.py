"""Atomic admission shared by a directory driver and its live queue.

Execution sidecars never edit frozen plans. Reservations span pending actors,
running learners and retiring children; only their owning scheduler releases them.
No stale-PID inference or crash recovery is performed here.
"""

import json
import time
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

from smartsom.experiments.control import boundary, read
from smartsom.experiments.evidence import write_json
from smartsom.experiments.locking import acquire, release

SCHEMA = "smartsom.queue-admission/v2"


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
            state.clear()
            state.update(
                schema=SCHEMA, owner_id=owner["id"], supported=supported, slots={}
            )


def completed_source(root, *, owner_id):
    """Provision a quiescent source journal without changing its frozen inputs."""
    from smartsom.experiments import control

    with transaction(root) as state:
        owner = read(root)
        if (owner["id"] if owner else None) != owner_id:
            raise ValueError("completed source ownership changed")
        if owner:
            table = control.processes()
            if (
                owner.get("ownership_error")
                or (
                    owner["status"] in control.ACTIVE
                    and control.alive(owner["owner"], table)
                )
                or any(control.alive(item, table) for item in owner["members"])
            ):
                raise ValueError("completed source still has owned execution")
        if state and state.get("queue"):
            raise ValueError("directory batch already belongs to a live queue")
        if state and any(key.startswith("native:") for key in state.get("slots", {})):
            raise ValueError("completed source has unresolved native allocations")
        identity = owner_id or uuid4().hex
        fresh = dict(
            schema=SCHEMA, owner_id=identity, supported=True, slots={}, quiescent=True
        )
        if state is None:
            write_json(Path(root) / "control/admission.json", fresh)
        else:
            state.clear()
            state.update(fresh)
        return identity


def _validate(state, owner_id):
    if not state or state.get("schema") != SCHEMA or state["owner_id"] != owner_id:
        raise ValueError("shared admission owner changed or is unavailable")


def claim(root, queue, *, limit, owner_id):
    with transaction(root) as state:
        _validate(state, owner_id)
        owner = read(root)
        if (
            owner
            and owner["id"] != owner_id
            or not owner
            and not state.get("quiescent")
        ):
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


@contextmanager
def registration(root):
    """Reject a source rebind atomically, before changing its control owner."""
    root = Path(root)
    plan_path = root / "plan.json"
    plan = json.loads(plan_path.read_text()) if plan_path.is_file() else {}
    queue = plan.get("schema") == "smartsom.live-queue-plan/v1"
    source = plan.get("schema") == "smartsom.author-batch-plan/v1"
    if not queue and not source and not (root / "control/admission.json").is_file():
        yield
        return
    with transaction(root) as state:
        if queue:
            if "admission_root" not in plan:
                raise ValueError("legacy live queue lacks shared admission")
            admission_root = Path(plan["admission_root"])
            if admission_root == root:
                _validate(state, plan["admission_owner_id"])
                if state.get("queue") != str(root):
                    raise ValueError("queue startup claim changed")
                yield
            else:
                # Serialize registration and failed-spawn rollback in the same
                # lock order: coordinator journal, then source journal.
                with transaction(admission_root) as shared:
                    _validate(shared, plan["admission_owner_id"])
                    if shared.get("queue") != str(root):
                        raise ValueError("queue startup claim changed")
                    yield
            return
        if state and state.get("queue") and state["queue"] != str(root):
            raise ValueError("an adopted directory driver cannot be restarted")
        yield


def check_registration(root):
    with registration(root):
        pass


def rollback_unstarted(root, queue, *, owner_id):
    """Only a launcher that created no process may withdraw its own claim."""

    def withdraw(state):
        from smartsom.experiments.control import _birth

        launcher = Path(queue) / "control/background-launch.lock"
        if launcher.is_file():
            lock = json.loads(launcher.read_text())
            if _birth(lock["pid"]) == lock.get("created"):
                raise ValueError("startup has another active background launcher")
        _validate(state, owner_id)
        if state.get("queue") != str(queue) or read(queue) is not None:
            raise ValueError("startup has another claim or a registered coordinator")
        if any(key.startswith("native:") for key in state["slots"]):
            raise ValueError("startup claim already has native allocations")
        for key in ("queue", "limit", "stopping"):
            state.pop(key, None)

    with transaction(queue) as local:
        if Path(root) == Path(queue):
            withdraw(local)
        else:
            with transaction(root) as shared:
                withdraw(shared)


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
def smoke_slot(root, identity, *, control_root=None):
    owner_id = read(root)["id"]
    key = "source:" + identity
    while True:
        boundary(control_root or root, fresh=True)
        if reserve(root, key, owner_id=owner_id, status="preflight"):
            break
        time.sleep(0.1)
    try:
        yield
    finally:
        retire(root, key, owner_id=owner_id)
