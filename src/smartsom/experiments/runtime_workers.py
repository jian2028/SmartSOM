"""Cooperative native worker lifecycle under the shared single-node control API."""

import json
import math
import multiprocessing
from pathlib import Path

from smartsom.experiments import control, queue_admission
from smartsom.experiments import runtime_control as runtime


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

    def _requirements(self, job):
        if "cpus" in job:
            return
        plan_path = Path(job["root"]) / "plan.json"
        cpus = 1
        if plan_path.exists():
            plan = json.loads(plan_path.read_text())
            for entry in plan.get("entries", []):
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

    def _persist(self, state, reason=None):
        shared = queue_admission.snapshot(self.root)
        active = len(shared.get("slots", {})) if shared else len(self.processes)
        with runtime.transaction(self.root) as latest:
            for key, job in self.jobs.items():
                if key in latest.get("additions", {}):
                    latest["additions"][key].update(job)

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
            tasks={key: dict(job) for key, job in self.jobs.items()},
        )

    def tick(self):
        state = runtime.apply(self.root)
        if not state:
            raise ValueError("native scheduling requires initialized runtime control")
        for key, job in state["additions"].items():
            self.jobs.setdefault(key, dict(job))
        for key, job in self.jobs.items():
            self._requirements(job)
            self.peak_memory.setdefault(key, job.get("peak_memory", 0))
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
                    if owner_id and not job.get("own_slots"):
                        queue_admission.update_resources(
                            self.root,
                            "native:" + key,
                            owner_id=owner_id,
                            memory=math.ceil(
                                max(256 * 1024**2, self.peak_memory[key]) * 1.25
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
            del self.processes[key]
            self.pause_sent.discard(key)
            self.stop_sent.discard(key)
            if owner_id:
                queue_admission.retire(self.root, "native:" + key, owner_id=owner_id)
        for key, job in self.jobs.items():
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
        observed = self.observer.snapshot()
        self.samples += 1
        memory_reserve = max(2 * 1024**3, int(observed.memory_total * 0.2))
        cpu_free = observed.cpus * (1 - observed.cpu_load / 100)
        budget = state.get("budget", {})
        cpu_used = sum(self.jobs[key].get("cpus", 1) for key in self.processes)
        memory_used = sum(
            max(256 * 1024**2, self.peak_memory.get(key, 0)) for key in self.processes
        )
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
                (
                    math.ceil(max(256 * 1024**2, self.peak_memory.get(key, 0)) * 1.25)
                    for key in eligible
                ),
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
        pending = [key for key, job in self.jobs.items() if job["status"] == "queued"]
        if state["stopping"] or state["paused"]:
            reason = (
                "batch stopping"
                if state["stopping"]
                else "batch paused; admission closed"
            )
        else:
            for key in pending:
                own_slots = self.jobs[key].get("own_slots", False)
                memory_charge = math.ceil(
                    max(256 * 1024**2, self.peak_memory.get(key, 0)) * 1.25
                )
                if owner_id and not own_slots:
                    queue_admission.note_wait(
                        self.root,
                        "native:" + key,
                        owner_id=owner_id,
                        cpus=self.jobs[key].get("cpus", 1),
                        memory=memory_charge,
                        feasible=(
                            self.jobs[key].get("cpus", 1) <= cpu_capacity
                            and memory_charge <= memory_capacity
                        ),
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
                    or memory_used
                    + max(256 * 1024**2, self.peak_memory.get(key, 0)) * 1.25
                    > budget.get("memory", float("inf"))
                    or observed.memory_available
                    < memory_reserve + max(256 * 1024**2, self.peak_memory.get(key, 0))
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
                    )
                ):
                    reason = "waiting for shared concurrency/resource budget or older feasible task"
                    self.jobs[key]["blocking_reason"] = reason
                    continue
                job = self.jobs[key]
                process = self.context.Process(
                    target=self.worker, args=(job["root"], str(self.root))
                )
                try:
                    process.start()
                except BaseException:
                    if owner_id:
                        queue_admission.retire(
                            self.root, "native:" + key, owner_id=owner_id
                        )
                    raise
                self.processes[key] = process
                job.update(status="running", pid=process.pid)
                job.pop("blocking_reason", None)
                # Never start a burst from a single stale resource observation.
                break
        return self._persist(state, reason)

    @property
    def unfinished(self):
        return bool(self.processes) or any(
            job["status"] not in runtime.TERMINAL for job in self.jobs.values()
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
