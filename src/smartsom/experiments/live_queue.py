"""Append-only manual queue; adopted frozen runs keep their original drivers."""

import copy
import json
import multiprocessing
import time
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from smartsom.config.codec import digest
from smartsom.experiments.batch import exclusive_lock
from smartsom.experiments.composable import implementation_identity
from smartsom.experiments.control import requested
from smartsom.experiments.evidence import source_identity, write_json
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
    """Register an existing batch read-only; no source/learner migration occurs."""
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
        limit=positive_limit(limit),
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
    return root


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


class Throughput:
    """Rolling training throughput from per-task monotone physical tick deltas."""

    def __init__(self):
        self.previous = {}
        self.samples = []
        self.last = None

    def sample(self, tasks, now):
        current = {
            t["id"]: (t.get("values", {}).get("physical_ticks") or 0) for t in tasks
        }
        delta = sum(
            max(0, value - self.previous.get(key, value))
            for key, value in current.items()
        )
        if self.last is not None:
            self.samples.append((now, delta, now - self.last))
        self.previous, self.last = current, now
        self.samples = [s for s in self.samples if now - s[0] <= 30]
        elapsed = sum(s[2] for s in self.samples)
        return sum(s[1] for s in self.samples) / elapsed if elapsed else None


def collect(state):
    """Project original progress without modifying adopted manifests or checkpoints."""
    snapshots, tasks, entries = {}, [], []
    for job in state["jobs"].values():
        child = Path(job["root"])
        if job["root"] not in snapshots:
            try:
                snapshots[job["root"]] = read(child / "logs/progress.json")
            except (OSError, ValueError):
                snapshots[job["root"]] = {}
        snapshot = snapshots[job["root"]]
        candidates = [
            t for t in snapshot.get("tasks", []) if t["id"] in job["entry_ids"]
        ]
        if job["kind"] == "adopted":
            oldstate = read(child / "batch.json")
            job["status"] = oldstate["files"][job["id"]]["status"]
            allocations = {
                t["experiment_id"]: t
                for t in snapshot.get("tuning", {}).get("entries", [])
            }
            if any(
                allocations.get(e, {}).get("status") in {"running", "pending"}
                for e in job["entry_ids"]
            ):
                job["status"] = "running"
            if oldstate["status"] in FINAL and job["status"] not in FINAL:
                job["status"] = oldstate["status"]
        elif job["status"] == "running":
            childstate = read(child / "batch.json")
            if childstate["status"] in FINAL:
                job["status"] = childstate["status"]
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
            entries.append(
                dict(
                    experiment_id=row["id"],
                    status=job["status"],
                    threads=None,
                    actual_cpus=None,
                )
            )
    return tasks, entries


def admission_jobs(state, *, memory_available, stopping=False):
    """A lower limit drains naturally; running work is never preempted."""
    if stopping or memory_available <= 2 * 1024**3:
        return []
    active = sum(job["status"] == "running" for job in state["jobs"].values())
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
    from smartsom.experiments.control import stop
    from smartsom.experiments.tuning_resources import ResourceMonitor

    root, plan, state = load(root)
    if plan["implementation_sha256"] != implementation_identity():
        raise ValueError("live queue implementation changed")
    source = Path(plan["adopted_root"])
    if digest(read(source / "plan.json")) != plan["adopted_plan_sha256"]:
        raise ValueError("adopted frozen batch changed")
    bind(root)
    display = CURRENT.get()
    display.kind = "batch-directory"
    observer, rates = ResourceMonitor(), Throughput()
    processes = {}
    context = multiprocessing.get_context("spawn")
    state["status"] = "running"
    with exclusive_lock(root / "driver.lock"):
        try:
            while True:
                apply_requests(root, state)
                stopping = requested(root)
                if stopping:
                    state.update(status="stopping", accepting=False)
                    if any(
                        job["kind"] == "adopted" and job["status"] == "running"
                        for job in state["jobs"].values()
                    ):
                        stop(source, timeout=0)
                    # Native workers inherit this queue's stop request; their
                    # child registrations must not be stopped independently.
                tasks, entries = collect(state)
                for key, process in list(processes.items()):
                    if not process.is_alive():
                        process.join()
                        if state["jobs"][key]["status"] not in FINAL:
                            state["jobs"][key]["status"] = (
                                "failed" if process.exitcode else "completed"
                            )
                        del processes[key]
                snapshot = observer.snapshot()
                active = sum(
                    job["status"] == "running" for job in state["jobs"].values()
                )
                for job in admission_jobs(
                    state, memory_available=snapshot.memory_available, stopping=stopping
                ):
                    process = context.Process(
                        target=_worker, args=(job["root"], str(root))
                    )
                    process.start()
                    processes[job["id"]] = process
                    job["status"] = "running"
                    active += 1
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
                        total_ticks_per_second=rates.sample(tasks, time.monotonic()),
                        resources=dict(
                            cpus_total=snapshot.cpus,
                            cpus_used=(
                                None
                                if "CPU load first sample unavailable"
                                in snapshot.unavailable
                                else snapshot.cpus * snapshot.cpu_load / 100
                            ),
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
                if (
                    stopping
                    and not processes
                    and not any(
                        j["kind"] == "adopted" and j["status"] == "running"
                        for j in state["jobs"].values()
                    )
                ):
                    state["status"] = "stopped"
                    break
                if not state["accepting"] and not unfinished and not processes:
                    state["status"] = (
                        "failed"
                        if any(j["status"] == "failed" for j in state["jobs"].values())
                        else "completed"
                    )
                    break
                time.sleep(1)
        except BaseException:
            state["status"] = "failed"
            from smartsom.experiments.author_batch import _request_owned_stop

            _request_owned_stop(root)
            for process in processes.values():
                process.join()
            raise
        finally:
            _save(root, state)
    return dict(status=state["status"], directory=str(root))
