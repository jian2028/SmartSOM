"""Atomic admission shared by a directory driver and its live queue.

Execution sidecars never edit frozen plans. Reservations span pending actors,
running learners and retiring children; only their owning scheduler releases them.
No stale-PID inference or crash recovery is performed here.
"""

import json
import math
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


def _resources(cpus, memory):
    cpus = 1 if cpus is None else cpus
    memory = 320 * 1024**2 if memory is None else memory
    if isinstance(cpus, bool) or not math.isfinite(cpus) or cpus <= 0:
        raise ValueError("CPU reservation must be positive and finite")
    if type(memory) is not int or memory < 0:
        raise ValueError("memory reservation must be nonnegative bytes")
    return dict(cpus=cpus, memory=memory)


def _held(key, runtime):
    from smartsom.experiments.runtime_control import matches

    identity = key.split(":", 1)[-1]
    return runtime and (
        runtime.get("tasks", {}).get(identity, {}).get("status")
        in {
            "paused",
            "pausing",
            "stopped",
            "stopping",
            "removed",
            "removing",
            "completed",
            "failed",
        }
        or runtime["paused"]
        or runtime["stopping"]
        or matches(
            identity,
            runtime["paused_tasks"]
            + runtime["stopped_tasks"]
            + runtime.get("removed_tasks", []),
        )
    )


def _host_capacity(state, host, *, cpu_reserve=0.25, memory_reserve=None):
    """Recover only verified live usage, never reservations or future capacity."""
    # Every caller also passes the shared node guard. Its policy can add
    # headroom, but cannot claim capacity that actual reserve() will refuse.
    cpu_reserve = max(0.25, cpu_reserve)
    memory_reserve = max(2 * 1024**3, int(host.memory_total * 0.2), memory_reserve or 0)
    actual = {"memory": 0, "cpus": 0.0}
    seen = set()
    for slot in state["slots"].values():
        observation = slot.get("observed_usage", {})
        identities = {tuple(identity) for identity in observation.get("processes", ())}
        if not identities or identities & seen:
            continue
        if not 0 <= time.monotonic() - observation.get("at", -math.inf) <= 5:
            continue
        import psutil

        try:
            verified = all(
                psutil.Process(pid).create_time() == born
                and psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
                for pid, born in identities
            )
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            verified = False
        if verified:
            seen.update(identities)
            for name in actual:
                actual[name] += observation.get(name) or 0
    return {
        "cpus": max(
            0,
            min(host.cpus, host.cpus * (1 - host.cpu_load / 100) + actual["cpus"])
            - cpu_reserve,
        ),
        "memory": max(
            0,
            min(host.memory_total, host.memory_available + actual["memory"])
            - memory_reserve,
        ),
    }


def envelope_capacity(root, *, owner_id, host):
    """Read the current shared envelope, without treating reservations as usage.

    This is total capacity, not free slots or permission to launch. Callers must
    still use reserve() and their own resource guards at admission.
    """
    from smartsom.experiments.runtime_control import snapshot

    with transaction(root) as state:
        _validate(state, owner_id)
        host_limits = _host_capacity(state, host)
        budget = snapshot(root).get("budget", {})
        return {
            key: min(value, budget.get(key, math.inf))
            for key, value in host_limits.items()
        }


def note_wait(
    root,
    key,
    *,
    owner_id,
    cpus,
    memory,
    feasible=True,
    host=None,
    cpu_reserve=0.25,
    memory_reserve=None,
):
    """Age feasible requests across native and Tune schedulers, not per frontend."""
    charge = _resources(cpus, memory)
    from smartsom.experiments.runtime_control import snapshot

    with transaction(root) as state:
        _validate(state, owner_id)
        waiters = state.setdefault("waiters", {})
        runtime = snapshot(root)
        budget = runtime.get("budget", {})
        capacity = (
            _host_capacity(
                state, host, cpu_reserve=cpu_reserve, memory_reserve=memory_reserve
            )
            if host is not None
            else {}
        )
        feasible = feasible and all(
            charge[k] <= min(budget.get(k, math.inf), capacity.get(k, math.inf))
            for k in charge
        )
        if _held(key, runtime) or key in state["slots"]:
            waiters.pop(key, None)
            return
        previous = waiters.get(key, {})
        waiters[key] = dict(
            charge,
            feasible=bool(feasible),
            cpu_reserve=cpu_reserve,
            memory_reserve=memory_reserve,
            since=previous.get("since", time.monotonic())
            if previous.get("feasible") and feasible
            else time.monotonic(),
        )


def clear_wait(root, key, *, owner_id):
    with transaction(root) as state:
        _validate(state, owner_id)
        state.setdefault("waiters", {}).pop(key, None)


def update_resources(
    root,
    key,
    *,
    owner_id,
    memory,
    cpus=None,
    resource_estimate=None,
    observed_memory=None,
    observed_cpus=None,
    observed_processes=(),
):
    """Charge measured high-water growth without pretending it can be refused."""
    _resources(1 if cpus is None else cpus, memory)
    with transaction(root) as state:
        _validate(state, owner_id)
        slot = state["slots"].get(key)
        if slot is not None:
            slot["memory"] = max(memory, slot.get("memory", 320 * 1024**2))
            if "memory_floor" in slot:
                slot["memory_floor"] = max(memory, slot["memory_floor"])
            if cpus is not None:
                slot["cpus"] = max(cpus, slot.get("cpus", 1))
            if resource_estimate is not None:
                slot["resource_estimate"] = resource_estimate
            if observed_memory is not None:
                if (
                    not math.isfinite(observed_memory)
                    or observed_memory < 0
                    or (
                        observed_cpus is not None
                        and (not math.isfinite(observed_cpus) or observed_cpus < 0)
                    )
                ):
                    raise ValueError("invalid observed resource usage")
                slot["observed_usage"] = {
                    "memory": observed_memory,
                    "cpus": observed_cpus,
                    "at": time.monotonic(),
                    "processes": list(observed_processes),
                }


def reserve(
    root,
    key,
    *,
    owner_id,
    status="pending",
    cpus=None,
    memory=None,
    resource_model=None,
    host=None,
    temporary_memory=False,
    memory_floor=None,
):
    charge = _resources(cpus, memory)
    floor = _resources(cpus, memory if memory_floor is None else memory_floor)["memory"]
    charge["memory"] = max(charge["memory"], floor)
    with transaction(root) as state:
        _validate(state, owner_id)
        prediction = None
        if resource_model is not None:
            from smartsom.experiments.runtime_resources import estimate

            prediction = estimate(
                root,
                resource_model["descriptor"],
                cpus=resource_model["cpus"],
                memory=resource_model["memory"],
            )
            charge["cpus"] = max(charge["cpus"], prediction["cpus"])
            charge["memory"] = max(
                charge["memory"],
                math.ceil(
                    prediction["memory"] * resource_model.get("peak_factor", 1.25)
                ),
            )
        if key in state["slots"]:
            return True
        from smartsom.experiments import runtime_control

        runtime = runtime_control.snapshot(root)
        budget = runtime.get("budget", {})
        host_limits = {}
        if host is not None:
            host_limits = _host_capacity(state, host)
            state["node_envelope_capacity"] = host_limits
        for resource, requested in charge.items():
            used = sum(
                slot.get(resource, _resources(None, None)[resource])
                for slot in state["slots"].values()
            )
            if used + requested > min(
                budget.get(resource, math.inf), host_limits.get(resource, math.inf)
            ):
                return False
        waiters = state.setdefault("waiters", {})
        rank = runtime_control.priority_rank(runtime, key.split(":", 1)[-1])
        for waiting, request in sorted(
            waiters.items(),
            key=lambda item: (
                runtime_control.priority_rank(runtime, item[0].split(":", 1)[-1]),
                item[1]["since"],
                item[0],
            ),
        ):
            if waiting == key:
                break
            waiting_rank = runtime_control.priority_rank(
                runtime, waiting.split(":", 1)[-1]
            )
            if waiting_rank > rank:
                continue
            if _held(waiting, runtime):
                waiters.pop(waiting, None)
                continue
            if host is not None:
                capacity = _host_capacity(
                    state,
                    host,
                    cpu_reserve=request.get("cpu_reserve", 0.25),
                    memory_reserve=request.get("memory_reserve"),
                )
                if any(request[k] > capacity[k] for k in charge):
                    request.update(feasible=False, since=time.monotonic())
                    continue
            if (
                request["feasible"]
                and (waiting_rank < rank or time.monotonic() - request["since"] >= 30)
                and all(request[k] <= budget.get(k, math.inf) for k in charge)
            ):
                return False
        identity = key.split(":", 1)[-1]
        if runtime and (
            runtime["paused"]
            or runtime["stopping"]
            or runtime_control.matches(identity, runtime["paused_tasks"])
            or runtime_control.matches(identity, runtime["stopped_tasks"])
            or runtime_control.matches(identity, runtime.get("removed_tasks", []))
            or len(state["slots"]) >= runtime["limit"]
        ):
            return False
        if state.get("stopping") or (
            state.get("queue") and len(state["slots"]) >= state["limit"]
        ):
            return False
        state["slots"][key] = dict(status=status, **charge)
        if temporary_memory:
            if prediction is not None:
                floor = max(
                    floor,
                    math.ceil(
                        prediction["memory"] * resource_model.get("peak_factor", 1.25)
                    ),
                )
            state["slots"][key].update(
                temporary_memory=charge["memory"], memory_floor=floor
            )
        if prediction is not None:
            state["slots"][key]["resource_estimate"] = prediction
        waiters.pop(key, None)
        return True


def settle_memory_placeholder(root, key, *, owner_id, memory):
    """Retire only an explicitly recorded temporary envelope after a real commit.

    Persistent resource growth may have arrived since the caller's observation;
    retain that floor under the same lock rather than allowing a general decrease.
    """
    _resources(None, memory)
    with transaction(root) as state:
        _validate(state, owner_id)
        slot = state["slots"].get(key)
        if slot is None or "temporary_memory" not in slot:
            return None
        slot["memory_floor"] = max(slot["memory_floor"], memory)
        slot["memory"] = slot["memory_floor"]
        slot["settled_temporary_memory"] = slot.pop("temporary_memory")
        return slot["memory"]


def retire(root, key, *, owner_id):
    with transaction(root) as state:
        _validate(state, owner_id)
        state["slots"].pop(key, None)
        state.setdefault("waiters", {}).pop(key, None)


def source_exited(root, *, owner_id):
    """Called only after verified directory owner and all children have exited."""
    with transaction(root) as state:
        _validate(state, owner_id)
        state["slots"] = {
            key: value
            for key, value in state["slots"].items()
            if not key.startswith("source:")
        }


def synchronize(root, leases, *, owner_id, peak_factor=1.25):
    """Publish allocation phases, retaining every local reservation."""
    with transaction(root) as state:
        _validate(state, owner_id)
        for key, lease in leases.items():
            slot = state["slots"].get("source:" + key)
            if slot is not None:
                slot["status"] = lease.state
                if hasattr(lease, "request"):
                    slot["cpus"] = max(slot.get("cpus", 0), lease.request.cpus)
                    slot["memory"] = max(
                        slot.get("memory", 0),
                        math.ceil(lease.request.memory * peak_factor),
                    )


def snapshot(root):
    with transaction(root) as state:
        return dict(state or {})


@contextmanager
def smoke_slot(root, identity, *, control_root=None):
    owner_id = read(root)["id"]
    key = "source:" + identity
    while True:
        from smartsom.experiments import runtime_control
        from smartsom.experiments.control import StopRequested

        runtime = runtime_control.apply(root)
        if runtime.get("stopping"):
            raise StopRequested("batch stopped during preflight")
        boundary(control_root or root, fresh=True)
        if reserve(root, key, owner_id=owner_id, status="preflight"):
            break
        time.sleep(0.1)
    try:
        yield
    finally:
        retire(root, key, owner_id=owner_id)
