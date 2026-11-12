"""Native evaluation leases; science and case ordering remain immutable."""

import math
import time
from contextvars import ContextVar
from pathlib import Path

from smartsom.experiments import queue_admission, runtime_control

BATCH = ContextVar("native_batch", default=None)
MEMORY = 256 * 1024**2


class EvaluationLease:
    """Resize an existing experiment slot, never consume another experiment slot.

    The pool reports live children before every resize. Reductions cannot release
    their CPU charge until they have exited and been joined. A baseline CPU stays
    reserved, allowing all evaluators to progress without a resource deadlock.
    """

    def __init__(self, workers, *, budget=None, cpus=None, baseline_cpus=1):
        self.baseline_cpus = baseline_cpus
        self.sampler_memory = 0
        if baseline_cpus > 1:
            import psutil

            for child in psutil.Process().children(recursive=True):
                try:
                    self.sampler_memory += child.memory_info().rss
                except psutil.NoSuchProcess:
                    pass
        self.workers = workers
        self.cpus = cpus
        self.budget = budget
        self.binding = BATCH.get()
        self.key = None
        self.sample_at = -math.inf
        self.cached = None
        if self.binding:
            self.root, child = self.binding
            state = runtime_control.snapshot(self.root)
            for key, job in {
                **state.get("tasks", {}),
                **state.get("additions", {}),
            }.items():
                if job.get("root") and Path(job["root"]).resolve() == child:
                    self.key = "native:" + key
                    break
            if self.key is None:
                raise ValueError("evaluation has no owning native task")
            self.owner_id = queue_admission.snapshot(self.root)["owner_id"]
            from smartsom.experiments.tuning_resources import ResourceMonitor

            self.monitor = ResourceMonitor()

    def target(self, live):
        if not self.binding:
            return min(self.workers, self.cpus or self.workers)
        state = runtime_control.snapshot(self.root)
        identity = (live, state.get("applied_revision"))
        if (
            self.cached
            and self.cached[0] == identity
            and time.monotonic() - self.sample_at < 0.5
        ):
            return self.cached[1]
        import psutil

        parent_memory = max(MEMORY, psutil.Process().memory_info().rss)
        baseline_memory = parent_memory + self.sampler_memory
        host = self.monitor.snapshot()
        self.sample_at = time.monotonic()
        with queue_admission.transaction(self.root) as shared:
            queue_admission._validate(shared, self.owner_id)
            slot = shared["slots"].get(self.key)
            if slot is None:
                raise ValueError("evaluation lost its native resource lease")
            state = runtime_control.snapshot(self.root)
            settings = state.get("evaluation", {})
            requested = {
                **settings.get("*", {}),
                **settings.get(self.key.removeprefix("native:"), {}),
            }
            desired = min(
                requested.get("processes", self.workers),
                requested.get(
                    "cpus", self.cpus or requested.get("processes", self.workers)
                ),
            )
            policy = runtime_control.memory_policy(state)
            factor = 1 + policy["memory_headroom_ratio"]
            queue_admission.reprice_memory(shared, policy["memory_headroom_ratio"])
            capacity = queue_admission._host_capacity(
                shared,
                host,
                host_memory_reserve_ratio=policy["host_memory_reserve_ratio"],
            )
            budget = state.get("budget", {})
            others = [v for k, v in shared["slots"].items() if k != self.key]
            free_cpu = min(capacity["cpus"], budget.get("cpus", math.inf)) - sum(
                v.get("cpus", 1) for v in others
            )
            free_memory = min(capacity["memory"], budget.get("memory", math.inf)) - sum(
                v.get("memory", MEMORY) for v in others
            )
            if self.budget is not None:
                if self.budget.cpus is not None:
                    free_cpu = min(free_cpu, self.budget.cpus)
                if self.budget.memory_gib is not None:
                    free_memory = min(free_memory, self.budget.memory_gib * 1024**3)
            for key, request in shared.get("waiters", {}).items():
                if (
                    request.get("feasible")
                    and time.monotonic() - request["since"] >= 30
                    and not queue_admission._held(key, state)
                ):
                    free_cpu -= request["cpus"]
                    free_memory -= request["memory"]
                    break
            # The already admitted baseline can always finish a case. Extra
            # workers require both an aggregate lease and observed host headroom.
            available = max(
                1,
                min(
                    math.floor(free_cpu),
                    math.floor(
                        (free_memory - baseline_memory * factor)
                        / (parent_memory * factor)
                    ),
                ),
            )
            target = min(desired, available)
            charged = max(self.baseline_cpus, live, target)
            slot.update(
                cpus=charged,
                memory_base=max(
                    slot.get("memory_base", MEMORY),
                    baseline_memory
                    + (
                        max(live, target) * parent_memory
                        if max(live, target) > 1
                        else 0
                    ),
                ),
                memory_factor=factor,
                phase="evaluation",
                evaluation_requested=desired,
                evaluation_effective=target,
                evaluation_live=live,
            )
            slot["memory"] = math.ceil(slot["memory_base"] * factor)
            self.cached = (identity, target)
            return target

    def close(self):
        if self.binding:
            with queue_admission.transaction(self.root) as state:
                queue_admission._validate(state, self.owner_id)
                slot = state["slots"].get(self.key)
                if slot is not None:
                    import psutil

                    raw = (
                        max(MEMORY, psutil.Process().memory_info().rss)
                        + self.sampler_memory
                    )
                    factor = (
                        1
                        + runtime_control.memory_policy(
                            runtime_control.snapshot(self.root)
                        )["memory_headroom_ratio"]
                    )
                    slot.update(
                        cpus=self.baseline_cpus,
                        memory=math.ceil(raw * factor),
                        memory_base=raw,
                        memory_factor=factor,
                        phase="training",
                        evaluation_live=0,
                        evaluation_effective=0,
                    )
