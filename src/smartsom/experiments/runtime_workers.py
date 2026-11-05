"""Cooperative native worker lifecycle under the shared single-node control API."""

import json
import math
import multiprocessing
import time
from pathlib import Path

from smartsom.experiments import control, queue_admission
from smartsom.experiments import runtime_control as runtime
from smartsom.experiments import runtime_resources as resources


class NativeScheduler:
    """One driver owns these children; controllers only submit durable intents.

    A supplied worker is useful for lightweight process acceptance, but follows
    the same ownership, checkpoint, exit, admission and journal path as learners.
    """

    def __init__(self, root, jobs=None, *, worker=None, observer=None):
        from smartsom.experiments.author_batch import _quiet_worker
        from smartsom.experiments.tuning_resources import ResourceMonitor

        self.root = Path(root)
        self.jobs = jobs if jobs is not None else {}
        self.worker = worker or _quiet_worker
        self.observer = observer or ResourceMonitor()
        self.context = multiprocessing.get_context("spawn")
        self.processes = {}
        self.pause_sent = set()
        self.stop_sent = set()
        self.peak_memory = {}
        self.samples = 0
        self.resource_sampler = resources.TreeSampler()
        self._retry_reconciled = False
        self._retry_conflicts = set()

    def _requirements(self, job):
        if "authored_cpus" in job:
            return
        plan_path = Path(job["root"]) / "plan.json"
        cpus = job.get("cpus", 1)
        if plan_path.exists():
            plan = json.loads(plan_path.read_text())
            for entry in plan.get("entries", []):
                prepared_path = (
                    Path(job["root"]) / entry["snapshot"] / "config/prepared.json"
                )
                if not job.get("own_slots") and prepared_path.exists():
                    job["resource_descriptor"] = resources.describe(
                        json.loads(prepared_path.read_text()), task=entry.get("task")
                    )
                path = Path(job["root"]) / entry["snapshot"] / "config/experiment.json"
                if path.exists():
                    config = json.loads(path.read_text())
                    settings = config.get("runtime", {})
                    cpus = max(
                        cpus,
                        settings.get("numerical_threads", 1)
                        * (1 + settings.get("sampling_processes", 0)),
                    )
        job["cpus"] = cpus
        job["authored_cpus"] = cpus

    def _resource_estimate(self, key, job):
        descriptor = job.get("resource_descriptor")
        if not descriptor:
            return
        resources.ingest(self.root, descriptor, job["root"])
        prediction = resources.estimate(
            self.root,
            descriptor,
            cpus=job["authored_cpus"],
            memory=max(256 * 1024**2, self.peak_memory.get(key, 0)),
        )
        job.update(
            cpus=prediction["cpus"],
            resource_memory=prediction["memory"],
            resource_estimate=prediction,
        )

    def _memory(self, key):
        return max(
            256 * 1024**2,
            self.peak_memory.get(key, 0),
            self.jobs[key].get("resource_memory", 0),
        )

    def _persist(self, state, reason=None):
        shared = queue_admission.snapshot(self.root)
        active = len(shared.get("slots", {})) if shared else len(self.processes)
        protected = set(getattr(self, "_retry_conflicts", set()))
        with runtime.transaction(self.root) as latest:
            for key, job in self.jobs.items():
                if key in protected:
                    continue
                records = (
                    latest.get("tasks", {}).get(key),
                    latest.get("additions", {}).get(key),
                )
                if any(
                    isinstance(record, dict)
                    and record.get("retry_pending")
                    and record.get("retry_attempt") != job.get("retry_attempt")
                    for record in records
                ):
                    # A retry was committed but this driver's in-memory job has
                    # not yet observed it. Never overwrite that durable attempt.
                    protected.add(key)
                    continue
                if key in latest.get("additions", {}):
                    latest["additions"][key].update(job)

        safe_jobs = {key: job for key, job in self.jobs.items() if key not in protected}
        from smartsom.experiments.batch_append import synchronize

        synchronize(self.root, safe_jobs, protected=protected)

        def held_job(key, job):
            if job["status"] in runtime.TERMINAL | {"paused"}:
                return True
            children = [
                task
                for task in state.get("tasks", {}).values()
                if task.get("file_id") == key
            ]
            return (
                job.get("own_slots")
                and bool(children)
                and all(
                    task["status"] in runtime.TERMINAL | {"paused"} for task in children
                )
            )

        held = (
            not active
            and self.unfinished
            and all(held_job(key, job) for key, job in self.jobs.items())
        )
        return runtime.publish(
            self.root,
            active=active,
            status="paused" if held else None,
            blocking_reason="tasks manually paused" if held else reason,
            tasks={key: dict(job) for key, job in safe_jobs.items()},
        )

    def _restore_queued_retries(self, state):
        """Reconstruct only journaled retries which were never marked starting."""
        import copy

        from smartsom.experiments.batch_retry import inspect_target, task_view
        from smartsom.experiments.locking import exclusive_lock

        if not runtime._RETRY_ENABLED:
            return
        for key, job in list(self.jobs.items()):
            raw = (state.get("tasks", {}).get(key), state.get("additions", {}).get(key))
            if not any(
                isinstance(item, dict)
                and (
                    item.get("retry_pending")
                    or item.get("retry_once_active")
                    or item.get("retry_recovery_required")
                )
                for item in raw
            ):
                continue
            try:
                saved = task_view(state, key)
            except ValueError as exc:
                # Preserve both conflicting journal views; do not overwrite
                # either with stale scheduler memory or admit new work.
                job["retry_recovery_required"] = str(exc)
                conflicts = getattr(self, "_retry_conflicts", set())
                conflicts.add(key)
                self._retry_conflicts = conflicts
                continue
            if not saved:
                continue
            paused_retry = (
                saved.get("retry_once_active")
                and not saved.get("retry_pending")
                and saved.get("status") in {"paused", "queued"}
            )
            if not saved.get("retry_pending") and not paused_retry:
                continue
            reason = None
            try:
                if saved.get("status") not in {"queued", "paused"}:
                    raise ValueError(
                        "retry may have started before driver exit; verify worker ownership"
                    )
                history = saved["retry_history"][-1]
                receipt = state["commands"][history["command_id"]]
                if receipt.get("status") != "applied" or receipt.get(
                    "retry_attempt"
                ) != saved.get("retry_attempt"):
                    raise ValueError("retry journal identity mismatch")
                if paused_retry:
                    # This attempt was safely paused after it started. Preserve
                    # its once-only policy for the eventual normal resume.
                    job.update(saved)
                    continue
                check_state = copy.deepcopy(state)
                for collection in ("tasks", "additions"):
                    if key in check_state.get(collection, {}):
                        check_state[collection][key].update(history["failure_attempt"])
                child = Path(saved["root"])
                with exclusive_lock(child / "trial.lock"):
                    info = inspect_target(
                        self.root, key, check_state, allow_paused=True
                    )
                if info["pointer_sha256"] != history["pointer_sha256"]:
                    raise ValueError("retry continuation changed after acceptance")
                if info.get("owner_claim_id") != history.get("owner_claim_id"):
                    raise ValueError("retry owner claim changed after acceptance")
                job.update(saved)
            except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
                reason = str(exc)
            if reason is not None:
                job.update(
                    status="failed",
                    retry_pending=False,
                    retry_recovery_required=reason,
                    blocking_reason=reason,
                )
                with runtime.transaction(self.root) as latest:
                    for collection in ("tasks", "additions"):
                        if key in latest.get(collection, {}):
                            latest[collection][key].update(job)

    def _consume_retries(self, state):
        """Validate received retry requests and transition this driver's jobs."""
        if not runtime._RETRY_ENABLED:
            return
        from uuid import uuid4

        from smartsom.experiments.batch_retry import inspect_target, task_view
        from smartsom.experiments.locking import exclusive_lock

        for receipt in list(state.get("commands", {}).values()):
            if (
                receipt.get("status") != "received"
                or receipt.get("payload", {}).get("kind") != "retry"
            ):
                continue
            command_id = receipt["id"]
            key = receipt["payload"]["task_id"]
            job = self.jobs.get(key)
            try:
                if (
                    not job
                    or job.get("status") not in {"failed", "paused"}
                    or key in self.processes
                ):
                    raise ValueError(
                        "retry task is not owned and joined by this scheduler stage"
                    )
                child = Path(job["root"])
                with exclusive_lock(child / "trial.lock"):
                    observed = runtime.snapshot(self.root)
                    info = inspect_target(self.root, key, observed, allow_paused=True)
                    requested = receipt["payload"].get("value")
                    if (
                        requested is not None
                        and requested.get("pointer_sha256") != info["pointer_sha256"]
                    ):
                        raise ValueError("continuation changed since retry preview")
                    with runtime.transaction(self.root, wait=False) as latest:
                        current = latest.get("commands", {}).get(command_id, {})
                        current_job = task_view(latest, key)
                        if (
                            current.get("status") != "received"
                            or not current_job
                            or current_job.get("status") not in {"failed", "paused"}
                        ):
                            raise ValueError(
                                "failed attempt changed before retry application"
                            )
                        if {
                            name: current_job.get(name)
                            for name in ("status", "pid", "exit_code", "retry_attempt")
                        } != receipt.get("failure_attempt"):
                            raise ValueError(
                                "failed attempt identity changed before retry application"
                            )
                        if any(
                            later.get("revision", 0) > current.get("revision", 0)
                            and later.get("status") in {"received", "applied"}
                            and later.get("payload", {}).get("kind")
                            in {"pause", "stop", "remove"}
                            and later.get("payload", {}).get("task_id") in {None, key}
                            for later in latest.get("commands", {}).values()
                        ):
                            raise ValueError(
                                "a newer pause, stop or removal superseded retry"
                            )
                        if (
                            latest.get("cleanup_active")
                            or latest.get("stopping")
                            or any(
                                runtime.matches(key, latest.get(field, []))
                                for field in (
                                    "stopped_tasks",
                                    "removed_tasks",
                                )
                            )
                        ):
                            raise ValueError("cleanup or task hold prevents retry")
                        if info["owner_root"] != str(self.root.resolve()):
                            claim = queue_admission.snapshot(Path(info["owner_root"]))
                            if (
                                claim.get("queue") != str(self.root.resolve())
                                or claim.get("owner_id") != info["owner_claim_id"]
                            ):
                                raise ValueError(
                                    "external owner claim changed before retry"
                                )
                        import hashlib

                        pointer = child / "continuation/current.json"
                        if (
                            hashlib.sha256(pointer.read_bytes()).hexdigest()
                            != info["pointer_sha256"]
                        ):
                            raise ValueError(
                                "continuation pointer changed before retry"
                            )
                        # Only the explicitly selected task's pause is cleared;
                        # a global pause remains authoritative.
                        latest["paused_tasks"] = [
                            held
                            for held in latest.get("paused_tasks", [])
                            if held != key
                        ]
                        new_job = dict(job)
                        history = list(new_job.get("retry_history", []))
                        history.append(
                            {
                                "command_id": command_id,
                                "failure_attempt": receipt["failure_attempt"],
                                "generation": info["generation"],
                                "pointer_sha256": info["pointer_sha256"],
                                "owner_claim_id": info.get("owner_claim_id"),
                                "requested_at": receipt["received_at"],
                            }
                        )
                        new_job.update(
                            status="queued",
                            pid=None,
                            exit_code=None,
                            retry_attempt=uuid4().hex,
                            retry_history=history,
                            retry_pending=True,
                            retry_once_active=True,
                        )
                        for collection in ("tasks", "additions"):
                            if key in latest.get(collection, {}):
                                latest[collection][key].update(new_job)
                        current.update(
                            status="applied",
                            applied_at=time.time(),
                            result="validated retry queued",
                            retry_attempt=new_job["retry_attempt"],
                        )
                    self.jobs[key].update(new_job)
            except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
                with runtime.transaction(self.root) as latest:
                    current = latest.get("commands", {}).get(command_id, {})
                    if current.get("status") == "received":
                        current.update(
                            status="rejected", reason=str(exc), applied_at=time.time()
                        )

    def tick(self):
        state = runtime.apply(self.root)
        if not state:
            raise ValueError("native scheduling requires initialized runtime control")
        for key, job in state["additions"].items():
            self.jobs.setdefault(key, dict(job))
        if not getattr(self, "_retry_reconciled", False):
            self._restore_queued_retries(state)
            self._retry_reconciled = True
        if runtime._RETRY_ENABLED:
            self._consume_retries(state)
            state = runtime.snapshot(self.root)
        for key, job in self.jobs.items():
            self._requirements(job)
            self.peak_memory.setdefault(key, job.get("peak_memory", 0))
            self._resource_estimate(key, job)
            prior = job.get("resources", {})
            job["resources"] = dict(
                cpu_cores=None,
                memory=None,
                memory_peak=max(
                    self.peak_memory.get(key, 0), prior.get("memory_peak") or 0
                )
                or None,
                observed_at=prior.get("observed_at"),
                status="unavailable" if key in self.processes else "inactive",
            )
        shared = queue_admission.snapshot(self.root)
        owner_id = shared.get("owner_id")
        if state["stopping"]:
            from smartsom.experiments.author_batch import _request_owned_stop

            _request_owned_stop(self.root)
        reason = None
        owned_rss = 0
        # Retain reservation until the actual process has exited and been joined.
        for key, process in list(self.processes.items()):
            job = self.jobs[key]
            if process.is_alive():
                import psutil

                try:
                    tree = [psutil.Process(process.pid)]
                    tree.extend(tree[0].children(recursive=True))
                    rss = sum(p.memory_info().rss for p in tree)
                    owned_rss += rss
                    self.peak_memory[key] = max(rss, self.peak_memory.get(key, 0))
                    job["peak_memory"] = self.peak_memory[key]
                    measured_memory, measured_cpu = None, None
                    if not job.get("own_slots"):
                        stage = resources.stage_at(job["root"])
                        measured_memory, measured_cpu = self.resource_sampler.sample(
                            process.pid, stage
                        )
                    job["resources"] = dict(
                        cpu_cores=measured_cpu,
                        memory=measured_memory if measured_memory is not None else rss,
                        memory_peak=max(
                            self.peak_memory[key],
                            measured_memory or 0,
                            job["resources"].get("memory_peak") or 0,
                        ),
                        observed_at=time.time(),
                        status="current",
                    )
                    if not job.get("own_slots") and job.get("resource_descriptor"):
                        resources.observe(
                            self.root,
                            job["resource_descriptor"],
                            stage,
                            memory=measured_memory,
                            cpus=measured_cpu,
                        )
                        self._resource_estimate(key, job)
                    if owner_id and not job.get("own_slots"):
                        queue_admission.update_resources(
                            self.root,
                            "native:" + key,
                            owner_id=owner_id,
                            cpus=job["cpus"],
                            resource_estimate=job.get("resource_estimate"),
                            memory=math.ceil(self._memory(key) * 1.25),
                            observed_memory=measured_memory,
                            observed_cpus=measured_cpu,
                            observed_processes=self.resource_sampler.identities.get(
                                process.pid, ()
                            ),
                        )
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
                removing = runtime.matches(key, state.get("removed_tasks", []))
                stopping = (
                    state["stopping"]
                    or runtime.matches(key, state["stopped_tasks"])
                    or removing
                )
                pausing = state["paused"] or runtime.matches(key, state["paused_tasks"])
                if stopping and key not in self.stop_sent:
                    owner = control.read(job["root"])
                    if owner:
                        control.write_json(
                            Path(job["root"]) / "control/stop.json", {"id": owner["id"]}
                        )
                        self.stop_sent.add(key)
                        job["status"] = "removing" if removing else "stopping"
                elif (
                    pausing and key not in self.pause_sent and key not in self.stop_sent
                ):
                    try:
                        control.request_pause(job["root"])
                    except ValueError:
                        # Spawned child has not registered its verified owner yet.
                        pass
                    else:
                        self.pause_sent.add(key)
                        job["status"] = "pausing"
                continue
            process.join()
            path = Path(job["root"]) / "batch.json"
            result = json.loads(path.read_text()) if path.exists() else {}
            status = result.get("status", "failed")
            if process.exitcode or status not in {
                "completed",
                "paused",
                "stopped",
                "partial_removed",
                "partial_stopped",
            }:
                status = "failed"
            elif runtime.matches(key, state.get("removed_tasks", [])):
                status = "removed"
            elif status == "stopped" and not state["stopping"]:
                if not runtime.matches(key, state["stopped_tasks"]):
                    # An interrupted worker ends this attempt, but does not
                    # invent a durable per-task operator stop intent.
                    with runtime.transaction(self.root) as latest:
                        latest["stopping"] = True
                    state["stopping"] = True
            job.update(status=status, exit_code=process.exitcode)
            if job.get("retry_pending"):
                job["retry_pending"] = False
            if status in {"completed", "failed", "stopped", "removed"}:
                job["retry_once_active"] = False
            if "native_status" in result:
                job["native_status"] = result["native_status"]
            job["resources"].update(cpu_cores=None, memory=None, status="inactive")
            del self.processes[key]
            self.pause_sent.discard(key)
            self.stop_sent.discard(key)
            if owner_id:
                queue_admission.retire(self.root, "native:" + key, owner_id=owner_id)
        for key, job in self.jobs.items():
            if (
                job.get("retry_pending")
                and job.get("status") == "starting"
                and key not in self.processes
            ):
                # Conflicting durable views must remain untouched for manual
                # reconciliation; do not replace either with an inferred state.
                if key in getattr(self, "_retry_conflicts", set()):
                    continue
                # The driver may have been interrupted after start() returned
                # but before it recorded the process handle. Never call it stopped.
                job.update(
                    status="failed",
                    retry_pending=False,
                    retry_recovery_required="retry started but worker handle is unverified",
                    blocking_reason="retry started but worker handle is unverified",
                )
                with runtime.transaction(self.root) as latest:
                    for collection in ("tasks", "additions"):
                        if key in latest.get(collection, {}):
                            latest[collection][key].update(job)
                continue
            if key in self.processes or job["status"] in {
                "completed",
                "failed",
                "removed",
            }:
                continue
            if runtime.matches(key, state.get("removed_tasks", [])):
                job["status"] = "removed"
            elif runtime.matches(key, state["stopped_tasks"]) or state["stopping"]:
                job["status"] = "stopped"
            elif runtime.matches(key, state["paused_tasks"]) or state["paused"]:
                job["status"] = "paused"
            elif job["status"] in {"paused", "stopped"}:
                # Only explicit task resume removes its stopped intent.
                job["status"] = "queued"
            elif job["status"] in {"partial_stopped", "partial_removed"}:
                # A matrix can exit with held leaves. Resume reopens only a
                # stopped leaf whose exact scoped intent has been cleared.
                result = json.loads((Path(job["root"]) / "batch.json").read_text())
                holds = (
                    state["stopped_tasks"]
                    + state["paused_tasks"]
                    + state.get("removed_tasks", [])
                )
                if any(
                    row["status"] == "stopped"
                    and not runtime.matches(key + "__" + leaf, holds)
                    for leaf, row in result.get("entries", {}).items()
                ):
                    job["status"] = "queued"
        observed = self.observer.snapshot()
        self.samples += 1
        memory_reserve = max(2 * 1024**3, int(observed.memory_total * 0.2))
        cpu_free = observed.cpus * (1 - observed.cpu_load / 100)
        budget = state.get("budget", {})
        cpu_used = sum(self.jobs[key].get("cpus", 1) for key in self.processes)
        memory_used = sum(self._memory(key) for key in self.processes)
        cpu_capacity = min(
            max(0, observed.cpus - 0.25),
            max(0, cpu_free + cpu_used - 0.25),
            budget.get("cpus", math.inf),
        )
        memory_capacity = min(
            max(0, observed.memory_total - memory_reserve),
            max(0, observed.memory_available + owned_rss - memory_reserve),
            budget.get("memory", math.inf),
        )
        effective_limit = state["limit"]
        if state["mode"] == "adaptive" and self.jobs:
            eligible = [
                key
                for key, job in self.jobs.items()
                if job["status"] == "queued" or key in self.processes
            ]
            typical_cpu = min(
                (self.jobs[key].get("cpus", 1) for key in eligible), default=1
            )
            typical_memory = min(
                (math.ceil(self._memory(key) * 1.25) for key in eligible),
                default=320 * 1024**2,
            )
            effective_limit = min(
                effective_limit,
                math.floor(cpu_capacity / typical_cpu),
                math.floor(memory_capacity / typical_memory),
            )
        with runtime.transaction(self.root) as latest:
            latest.update(
                native_effective_limit=effective_limit,
                native_adaptation="resource_headroom"
                if state["mode"] == "adaptive"
                else "manual_target",
            )
        pending = runtime.priority_order(
            state,
            [
                key
                for key, job in self.jobs.items()
                if job["status"] == "queued"
                and (runtime._RETRY_ENABLED or not job.get("retry_pending"))
            ],
        )
        uncertain_retry = any(
            job.get("retry_recovery_required") for job in self.jobs.values()
        )
        if uncertain_retry:
            reason = "retry worker ownership is uncertain; new admission is blocked pending verification"
        elif state["stopping"] or state["paused"]:
            reason = (
                "batch stopping"
                if state["stopping"]
                else "batch paused; admission closed"
            )
        else:
            for key in pending:
                own_slots = self.jobs[key].get("own_slots", False)
                memory_charge = math.ceil(self._memory(key) * 1.25)
                if owner_id and not own_slots:
                    queue_admission.note_wait(
                        self.root,
                        "native:" + key,
                        owner_id=owner_id,
                        cpus=self.jobs[key].get("cpus", 1),
                        memory=memory_charge,
                        host=observed,
                    )
                if self.jobs[key].get("cpus", 1) > min(
                    observed.cpus - 0.25, budget.get("cpus", math.inf)
                ) or memory_charge > min(
                    observed.memory_total - memory_reserve,
                    budget.get("memory", math.inf),
                ):
                    reason = "task cannot fit current node CPU/memory budget"
                    self.jobs[key]["blocking_reason"] = reason
                    continue
                active = (
                    len(queue_admission.snapshot(self.root).get("slots", {}))
                    if owner_id
                    else len(self.processes)
                )
                if active >= effective_limit:
                    reason = (
                        "waiting for native resource-adaptive ceiling"
                        if state["mode"] == "adaptive"
                        else "waiting for target concurrency; existing workers drain naturally"
                    )
                    break
                # Actual host observations protect manual and adaptive admission.
                # This is a logical budget, not OS isolation. Unknown workers
                # start serially until the next fresh observation.
                if (
                    self.samples < 2
                    or cpu_free < self.jobs[key].get("cpus", 1) + 0.25
                    or cpu_used + self.jobs[key].get("cpus", 1)
                    > budget.get("cpus", float("inf"))
                    or memory_used + self._memory(key) * 1.25
                    > budget.get("memory", float("inf"))
                    or observed.memory_available < memory_reserve + self._memory(key)
                ):
                    reason = "waiting for CPU/memory headroom and a fresh resource observation"
                    self.jobs[key]["blocking_reason"] = reason
                    continue
                if (
                    owner_id
                    and not own_slots
                    and not queue_admission.reserve(
                        self.root,
                        "native:" + key,
                        owner_id=owner_id,
                        status="running",
                        cpus=self.jobs[key].get("cpus", 1),
                        memory=memory_charge,
                        host=observed,
                        resource_model=(
                            dict(
                                descriptor=self.jobs[key]["resource_descriptor"],
                                cpus=self.jobs[key]["authored_cpus"],
                                memory=max(256 * 1024**2, self.peak_memory.get(key, 0)),
                            )
                            if self.jobs[key].get("resource_descriptor")
                            else None
                        ),
                    )
                ):
                    reason = "waiting for shared concurrency/resource budget or older feasible task"
                    self.jobs[key]["blocking_reason"] = reason
                    continue
                job = self.jobs[key]
                if job.get("retry_once_active"):
                    job["retry_pending"] = True
                    # Persist starting BEFORE spawn: after a hard crash an
                    # unregistered live child must never be mistaken for queued.
                    with runtime.transaction(self.root) as latest:
                        for collection in ("tasks", "additions"):
                            if key in latest.get(collection, {}):
                                latest[collection][key].update(
                                    status="starting", retry_pending=True
                                )
                    job["status"] = "starting"
                worker_args = (job["root"], str(self.root))
                if job.get("retry_once_active"):
                    worker_args += (True,)
                process = self.context.Process(target=self.worker, args=worker_args)
                try:
                    process.start()
                except BaseException as exc:
                    if job.get("retry_pending"):
                        # start() can fail after creating a child: retain the
                        # slot and persist uncertainty even on interruption.
                        job.update(
                            status="failed",
                            retry_pending=False,
                            retry_recovery_required="retry spawn outcome must be verified",
                            blocking_reason=str(exc),
                        )
                        with runtime.transaction(self.root) as latest:
                            for collection in ("tasks", "additions"):
                                if key in latest.get(collection, {}):
                                    latest[collection][key].update(job)
                        reason = job["retry_recovery_required"]
                        if not isinstance(exc, Exception):
                            raise
                        break
                    if owner_id:
                        queue_admission.retire(
                            self.root, "native:" + key, owner_id=owner_id
                        )
                    raise
                self.processes[key] = process
                job.update(status="running", pid=process.pid)
                job["resources"]["status"] = "unavailable"
                job.pop("blocking_reason", None)
                # Never start a burst from a single stale resource observation.
                break
        return self._persist(state, reason)

    @property
    def unfinished(self):
        return bool(self.processes) or any(
            job.get("retry_recovery_required") or job["status"] not in runtime.TERMINAL
            for job in self.jobs.values()
        )

    def close(self):
        """Cooperative cleanup only; no timeout-to-kill escalation."""
        from smartsom.experiments.author_batch import _request_owned_stop

        with runtime.transaction(self.root) as state:
            state["stopping"] = True
        _request_owned_stop(self.root)
        for process in self.processes.values():
            process.join()
        self.tick()
