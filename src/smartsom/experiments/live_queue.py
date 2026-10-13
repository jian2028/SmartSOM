"""Append-only manual queue; adopted frozen runs keep their original drivers."""

import copy
import json
import multiprocessing
import time
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from smartsom.config.codec import digest
from smartsom.experiments import queue_admission
from smartsom.experiments.batch import exclusive_lock
from smartsom.experiments.composable import implementation_identity
from smartsom.experiments.control import requested
from smartsom.experiments.evidence import source_identity, write_json
from smartsom.telemetry.execution_metrics import RollingThroughput as Throughput
from smartsom.telemetry.runtime import CURRENT, bind, operation

PLAN = "smartsom.live-queue-plan/v1"
STATE = "smartsom.live-queue-state/v1"
RUN = "smartsom.live-queue-run/v1"
FINAL = {"completed", "failed", "stopped", "force_stopped", "interrupted"}


def read(path):
    return json.loads(Path(path).read_text())


def positive_limit(value):
    if type(value) is not int or not 1 <= value <= 1024:
        raise ValueError("concurrency must be an integer between 1 and 1024")
    return value


def load(root):
    root = Path(root).expanduser().resolve()
    plan = read(root / "plan.json")
    state = read(root / "batch.json")
    if plan.get("schema") != PLAN or state.get("schema") != STATE:
        raise ValueError("this command requires a live queue")
    if digest(plan) != state["plan_sha256"]:
        raise ValueError("live queue manifest changed")
    return root, plan, state


def adopt(source, *, limit=3):
    """Join a cooperating driver without migrating learners or frozen inputs."""
    from smartsom.experiments import control

    limit = positive_limit(limit)
    source = Path(source).expanduser().resolve()
    old = read(source / "plan.json")
    if old.get("schema") != "smartsom.author-batch-plan/v1":
        raise ValueError("adoption requires an existing V4 directory batch")
    if any(row.get("requires") or row.get("gate") for row in old["files"]):
        raise ValueError("adoption of gated/dependent stages is not supported")
    if any(len(row["entry_ids"]) != 1 for row in old["files"]):
        raise ValueError("adoption currently requires one entry per Experiment")
    if any(row["task"] != "train-evaluate" for row in old["files"]):
        raise ValueError("adoption currently requires train-evaluate entries")
    owner = control.read(source)
    oldstate = read(source / "batch.json")
    finished = oldstate["status"] == "completed" and all(
        row["status"] == "completed" for row in oldstate["files"].values()
    )
    live = _source_live(source, owner["id"] if owner else None)
    if live or not finished:
        admission = queue_admission.snapshot(source)
        if (
            not owner
            or not live
            or owner.get("ownership_error")
            or admission.get("schema") != queue_admission.SCHEMA
            or admission.get("owner_id") != owner["id"]
            or not admission.get("supported")
        ):
            raise ValueError(
                "active adoption requires a cooperating live directory driver "
                "with off/online calibration; legacy active batches cannot share admission"
            )
        admission_owner = owner["id"]
    else:
        admission_owner = queue_admission.completed_source(
            source, owner_id=owner["id"] if owner else None
        )
    root = source.parent / (
        datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-live-queue-" + uuid4().hex[:10]
    )
    root.mkdir()
    (root / "control/requests").mkdir(parents=True)
    (root / "jobs").mkdir()
    plan = dict(
        schema=PLAN,
        adopted_root=str(source),
        adopted_plan_sha256=digest(old),
        source=source_identity(),
        implementation_sha256=implementation_identity(),
        adopted_owner_id=owner["id"] if owner else None,
        admission_root=str(source),
        admission_owner_id=admission_owner,
    )
    write_json(root / "plan.json", plan)
    jobs = {}
    for row in old["files"]:
        jobs[row["id"]] = dict(
            id=row["id"],
            kind="adopted",
            root=str(source),
            entry_ids=[f"{row['id']}__{entry}" for entry in row["entry_ids"]],
            status="queued",
            source=old["source"],
        )
    state = dict(
        schema=STATE,
        plan_sha256=digest(plan),
        status="prepared",
        accepting=True,
        limit=limit,
        admission_root=plan["admission_root"],
        jobs=jobs,
        requests={},
        updated_at=time.time(),
    )
    write_json(root / "batch.json", state)
    write_json(
        root / "run.json",
        dict(
            schema=RUN,
            kind="live-queue",
            status="prepared",
            source=plan["source"],
            adopted_root=str(source),
        ),
    )
    queue_admission.claim(source, root, limit=limit, owner_id=admission_owner)
    return root


def _source_live(source, owner_id):
    """Terminal file status cannot establish verified driver/child exit."""
    from smartsom.experiments import control

    owner = control.read(source)
    if not owner:
        if owner_id is not None:
            raise ValueError("adopted control owner disappeared")
        return False
    if owner["id"] != owner_id or owner.get("ownership_error"):
        raise ValueError("adopted control ownership changed or is incomplete")
    table = control.processes()
    return (
        owner["status"] in control.ACTIVE and control.alive(owner["owner"], table)
    ) or any(control.alive(item, table) for item in owner["members"])


def _stop_source(source, plan):
    from smartsom.experiments import control

    if not _source_live(source, plan["adopted_owner_id"]):
        return False
    # stop validates generation, PID birth and descendants. Never force here.
    result = control.stop(source, timeout=0, expected_owner_id=plan["adopted_owner_id"])
    return bool(result["remaining"])


def _ensure_accepting(root):
    root, plan, state = load(root)
    if not state["accepting"] or state["status"] in FINAL:
        raise ValueError("queue is closed; create a new queue")
    if plan["implementation_sha256"] != implementation_identity():
        raise ValueError("queue source changed; use its original checkout")
    return root, plan, state


def _request(root, payload):
    identity = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f") + "-" + uuid4().hex
    write_json(
        root / "control/requests" / (identity + ".json"),
        {"id": identity, "created_at": time.time(), **payload},
    )
    return dict(status="queued", request_id=identity, directory=str(root))


def set_limit(root, limit):
    root, _, _ = _ensure_accepting(root)
    with exclusive_lock(root / "control/append.lock"):
        _ensure_accepting(root)
        return _request(root, dict(kind="limit", limit=positive_limit(limit)))


def close(root):
    root, _, _ = _ensure_accepting(root)
    with exclusive_lock(root / "control/append.lock"):
        return _request(root, dict(kind="close"))


def append(root, directory, *, ids=None, tags=None):
    """Freeze new scientific inputs before atomically publishing an add request."""
    from smartsom.config.experiment_v4 import compile_experiment
    from smartsom.experiments.author_batch import compile_directory
    from smartsom.experiments.author_driver import allocate

    root, _, _ = _ensure_accepting(root)
    checked = compile_directory(
        directory, require_dependencies=True, ids=ids, tags=tags
    )
    prepared = []
    for row in checked.files:
        if row["stage"] != 0 or row.get("requires") or row.get("gate"):
            raise ValueError("live queue additions must be independent stage-0 tasks")
        native = compile_experiment(
            row["path"],
            require_dependencies=True,
            sets=[
                "experiment.execution.executor=native",
                'experiment.execution.tuning="off"',
                'experiment.execution.calibration_level="off"',
                "experiment.execution.scheduling=fixed",
                "experiment.execution.max_concurrent=1",
            ],
        )
        if [e.prepared.scientific_sha256 for e in native.entries] != [
            e.prepared.scientific_sha256 for e in row["compiled"].entries
        ]:
            raise ValueError("manual execution override changed scientific identity")
        if len(native.entries) != 1:
            raise ValueError("live queue currently accepts one entry per Experiment")
        prepared.append((row["id"], native))
    with exclusive_lock(root / "control/append.lock"):
        _, _, state = _ensure_accepting(root)
        known = set(state["jobs"])
        for request in (root / "control/requests").glob("*.json"):
            payload = read(request)
            if payload["kind"] == "close":
                raise ValueError("queue close already requested")
            known.update(job["id"] for job in payload.get("jobs", []))
        names = [name for name, _ in prepared]
        if len(names) != len(set(names)) or known.intersection(names):
            raise ValueError(
                "duplicate live queue task ID; use a distinct Experiment file stem"
            )
        jobs = []
        for name, compiled in prepared:
            child = allocate(compiled, root=root / "jobs" / name)
            child_plan = read(child / "plan.json")
            jobs.append(
                dict(
                    id=name,
                    kind="native",
                    root=str(child),
                    entry_ids=[e.id for e in compiled.entries],
                    status="queued",
                    child_plan_sha256=digest(child_plan),
                    source=child_plan["source"],
                )
            )
        return _request(root, dict(kind="add", jobs=jobs))


def apply_requests(root, state):
    for path in sorted((root / "control/requests").glob("*.json")):
        request = read(path)
        identity = request["id"]
        if identity in state["requests"]:
            continue
        if request["kind"] == "limit":
            state["limit"] = positive_limit(request["limit"])
        elif request["kind"] == "close":
            state["accepting"] = False
        elif request["kind"] == "add":
            for job in request["jobs"]:
                if job["id"] in state["jobs"]:
                    raise ValueError("duplicate task in queue journal")
                child = Path(job["root"])
                if not child.resolve().is_relative_to(root / "jobs"):
                    raise ValueError("queue job escapes its frozen directory")
                if digest(read(child / "plan.json")) != job["child_plan_sha256"]:
                    raise ValueError("queued scientific plan changed")
                state["jobs"][job["id"]] = job
        else:
            raise ValueError("unknown queue command")
        state["requests"][identity] = dict(status="applied", applied_at=time.time())
    return state


def collect(state):
    """Project original progress without modifying adopted manifests or checkpoints."""
    snapshots, tasks, entries = {}, [], []
    shared = (
        queue_admission.snapshot(state["admission_root"])
        if state.get("admission_root")
        else {}
    )
    slots = shared.get("slots", {})
    if shared:
        state["reserved"] = len(slots)
        state["active"] = sum(slot["status"] == "running" for slot in slots.values())
    for job in state["jobs"].values():
        child = Path(job["root"])
        if job["root"] not in snapshots:
            try:
                snapshots[job["root"]] = read(child / "logs/progress.json")
            except (OSError, ValueError):
                snapshots[job["root"]] = {}
        snapshot = snapshots[job["root"]]
        allocations = {
            t["experiment_id"]: t for t in snapshot.get("tuning", {}).get("entries", [])
        }
        candidates = [
            t for t in snapshot.get("tasks", []) if t["id"] in job["entry_ids"]
        ]
        if not candidates and job["kind"] == "adopted":
            # Rules-only directory files publish the child driver aggregate;
            # learning files publish their individual Tune entries instead.
            candidates = [
                t
                for t in snapshot.get("tasks", [])
                if t["id"] == job["id"] and t.get("execution")
            ]
        if job["kind"] == "adopted":
            oldstate = read(child / "batch.json")
            phases = [
                slots["source:" + e]["status"]
                for e in job["entry_ids"]
                if "source:" + e in slots
            ]
            allocated = [allocations.get(e, {}).get("status") for e in job["entry_ids"]]
            ledger = oldstate["files"][job["id"]]["status"]
            terminal = next(
                (t.get("status") for t in candidates if t.get("status") in FINAL), None
            )
            if phases:
                job["status"] = "running" if "running" in phases else phases[0]
            elif shared:
                job["status"] = terminal or (ledger if ledger in FINAL else "queued")
            else:
                job["status"] = terminal or (
                    "running"
                    if "running" in allocated
                    else ledger
                    if ledger in FINAL
                    else "queued"
                )
            if oldstate["status"] in FINAL and job["status"] not in FINAL:
                if not phases:
                    job["status"] = oldstate["status"]
        if not candidates:
            candidates = [
                dict(
                    id=job["entry_ids"][0],
                    name=job["id"],
                    status=job["status"],
                    stage=job["status"],
                    values={},
                    learner={},
                )
            ]
        for original in candidates:
            row = copy.deepcopy(original)
            row["id"] = job["id"] + "::" + original["id"]
            row["status"] = job["status"]
            tasks.append(row)
            allocation = copy.deepcopy(allocations.get(original["id"], {}))
            allocation.update(experiment_id=row["id"], status=job["status"])
            entries.append(allocation)
    return tasks, entries


def admission_jobs(state, *, memory_available, stopping=False):
    """A lower limit drains naturally; running work is never preempted."""
    if stopping or memory_available <= 2 * 1024**3:
        return []
    active = state.get(
        "reserved",
        sum(
            job["status"] in {"running", "pending", "stopping"}
            for job in state["jobs"].values()
        ),
    )
    slots = max(0, state["limit"] - active)
    return [
        job
        for job in state["jobs"].values()
        if job["kind"] == "native" and job["status"] == "queued"
    ][:slots]


def _worker(root, parent):
    from smartsom.experiments.author_batch import _quiet_worker

    _quiet_worker(str(root), str(parent))


def _save(root, state):
    state["updated_at"] = time.time()
    write_json(root / "batch.json", state)
    manifest = read(root / "run.json")
    manifest["status"] = state["status"]
    write_json(root / "run.json", manifest)


@operation("live-queue")
def execute_saved(root):
    from smartsom.experiments.tuning_resources import ResourceMonitor
    from smartsom.telemetry.execution_metrics import resource_fields

    root, plan, state = load(root)
    if plan["implementation_sha256"] != implementation_identity():
        raise ValueError("live queue implementation changed")
    source = Path(plan["adopted_root"])
    if digest(read(source / "plan.json")) != plan["adopted_plan_sha256"]:
        raise ValueError("adopted frozen batch changed")
    if "admission_root" not in plan:
        raise ValueError("legacy live queue lacks shared admission; create a new queue")
    _source_live(source, plan["adopted_owner_id"])
    state["admission_root"] = plan["admission_root"]
    bind(root)
    display = CURRENT.get()
    display.kind = "batch-directory"
    source_plan = read(source / "plan.json")
    display.configure_batch(
        [
            dict(
                id=row["id"],
                stage=row.get("stage", 0),
                barrier_group=str(source),
                requires=row.get("requires", []),
            )
            for row in source_plan.get("files", [])
        ]
    )
    observer, rates = ResourceMonitor(), Throughput()
    processes = {}
    context = multiprocessing.get_context("spawn")
    state["status"] = "running"
    with exclusive_lock(root / "driver.lock"):
        try:
            while True:
                apply_requests(root, state)
                stopping = requested(root)
                queue_admission.configure(
                    plan["admission_root"],
                    root,
                    limit=state["limit"],
                    stopping=stopping,
                )
                if stopping:
                    state.update(status="stopping", accepting=False)
                    _stop_source(source, plan)
                    # Native workers inherit this queue's stop request; their
                    # child registrations must not be stopped independently.
                source_live = _source_live(source, plan["adopted_owner_id"])
                if not source_live:
                    queue_admission.source_exited(
                        plan["admission_root"], owner_id=plan["admission_owner_id"]
                    )
                for key, process in list(processes.items()):
                    if not process.is_alive():
                        process.join()
                        job = state["jobs"][key]
                        childstatus = read(Path(job["root"]) / "batch.json")["status"]
                        job["status"] = (
                            childstatus if childstatus in FINAL else "failed"
                        )
                        if process.exitcode and job["status"] == "completed":
                            job["status"] = "failed"
                        del processes[key]
                        queue_admission.retire(
                            plan["admission_root"],
                            "native:" + key,
                            owner_id=plan["admission_owner_id"],
                        )
                tasks, entries = collect(state)
                snapshot = observer.snapshot()
                active = state.get("active", 0)
                for job in admission_jobs(
                    state, memory_available=snapshot.memory_available, stopping=stopping
                ):
                    if not queue_admission.reserve(
                        plan["admission_root"],
                        "native:" + job["id"],
                        owner_id=plan["admission_owner_id"],
                        status="running",
                    ):
                        break
                    process = context.Process(
                        target=_worker, args=(job["root"], str(root))
                    )
                    try:
                        process.start()
                    except BaseException:
                        queue_admission.retire(
                            plan["admission_root"],
                            "native:" + job["id"],
                            owner_id=plan["admission_owner_id"],
                        )
                        raise
                    processes[job["id"]] = process
                    job["status"] = "running"
                    active += 1
                tasks, entries = collect(state)
                active = state["active"]
                display.tasks = {t["id"]: t for t in tasks}
                display.total_tasks = len(tasks)
                display.stage = "training"
                display.status = state["status"]
                display.updated_at = time.time()
                display.configure_tuning(
                    dict(
                        stage="training",
                        manual=True,
                        queue_open=state["accepting"],
                        concurrency_limit=state["limit"],
                        active=active,
                        batch_training_active=True,
                        calibration_status="skipped",
                        calibration=dict(
                            level="off", calibrated=True, schedule_status="manual"
                        ),
                        entries=entries,
                        total_ticks_per_second=rates.sample(
                            tasks, time.monotonic(), wall_time=time.time()
                        ),
                        throughput=rates.payload,
                        training_ticks_per_second=rates.training_rate,
                        resources=dict(
                            **resource_fields(snapshot, roots=[root, source]),
                            cpus_available=snapshot.cpus
                            * (1 - snapshot.cpu_load / 100),
                            memory_available=snapshot.memory_available,
                            mode="manual",
                        ),
                    )
                )
                _save(root, state)
                display.publish(force=True)
                unfinished = any(
                    j["status"] not in FINAL for j in state["jobs"].values()
                )
                if stopping and not processes and not source_live:
                    for job in state["jobs"].values():
                        if job["status"] not in FINAL:
                            job["status"] = "stopped"
                    state["status"] = "stopped"
                    break
                if (
                    not state["accepting"]
                    and not unfinished
                    and not processes
                    and not source_live
                ):
                    state["status"] = (
                        "failed"
                        if any(j["status"] == "failed" for j in state["jobs"].values())
                        else "completed"
                    )
                    break
                time.sleep(1)
        except BaseException as exc:
            state.update(status="stopping", accepting=False)
            from smartsom.experiments.author_batch import _request_owned_stop

            _request_owned_stop(root)
            queue_admission.configure(
                plan["admission_root"], root, limit=state["limit"], stopping=True
            )
            _save(root, state)
            while True:
                live = _stop_source(source, plan)
                for process in processes.values():
                    process.join(timeout=0.2)
                if not live and not any(p.is_alive() for p in processes.values()):
                    break
                time.sleep(0.2)
            for job in state["jobs"].values():
                if job["status"] not in FINAL:
                    job["status"] = "stopped"
            queue_admission.source_exited(
                plan["admission_root"], owner_id=plan["admission_owner_id"]
            )
            for key in processes:
                queue_admission.retire(
                    plan["admission_root"],
                    "native:" + key,
                    owner_id=plan["admission_owner_id"],
                )
            state["status"] = (
                "stopped" if isinstance(exc, KeyboardInterrupt) else "failed"
            )
            raise
        finally:
            if state["status"] in FINAL:
                queue_admission.detach(plan["admission_root"], root)
            _save(root, state)
    return dict(status=state["status"], directory=str(root))
