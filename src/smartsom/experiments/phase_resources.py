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

    def __init__(self, workers, *, budget=None):
        self.workers = workers
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
            return self.workers
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
        host = self.monitor.snapshot()
        self.sample_at = time.monotonic()
        with queue_admission.transaction(self.root) as shared:
            queue_admission._validate(shared, self.owner_id)
            slot = shared["slots"].get(self.key)
            if slot is None:
                raise ValueError("evaluation lost its native resource lease")
            state = runtime_control.snapshot(self.root)
            settings = state.get("evaluation", {})
            requested = settings.get(
                self.key.removeprefix("native:"), settings.get("*", {})
            )
            desired = min(
                requested.get("processes", self.workers),
                requested.get("cpus", self.workers),
            )
            capacity = queue_admission._host_capacity(shared, host)
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
                    math.floor(free_memory / (parent_memory * 1.25)) - 1,
                ),
            )
            target = min(desired, available)
            charged = max(1, live, target)
            slot.update(
                cpus=charged,
                memory=max(
                    slot.get("memory", MEMORY),
                    math.ceil(
                        (charged + 1 if charged > 1 else 1) * parent_memory * 1.25
                    ),
                ),
                phase="evaluation",
                evaluation_requested=desired,
                evaluation_effective=target,
                evaluation_live=live,
            )
            self.cached = (identity, target)
            return target

    def close(self):
        if self.binding:
            with queue_admission.transaction(self.root) as state:
                queue_admission._validate(state, self.owner_id)
                slot = state["slots"].get(self.key)
                if slot is not None:
                    import psutil

                    slot.update(
                        cpus=1,
                        memory=math.ceil(
                            max(MEMORY, psutil.Process().memory_info().rss) * 1.25
                        ),
                        phase="training",
                        evaluation_live=0,
                        evaluation_effective=0,
                    )
