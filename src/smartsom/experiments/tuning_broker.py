"""Measured admission with observed actor, child-process and allocation lifetimes.

Reservations are accounting, not OS enforcement. Actor death does not establish
sampling-child exit. Requested resources become applied only after a matching
epoch, live PID identity and worker acknowledgement are observed.
"""

import copy
import json
import math
import os
import pickle
import threading
import time
from dataclasses import asdict, replace
from pathlib import Path

from smartsom._filesystem import native_path
from smartsom.experiments.tuning_resources import (
    ResourceBroker,
    ResourceLease,
    ResourceRequest,
)

RESOURCE_KEYS = ("CPU", "GPU", "memory")
INACTIVE_PHASES = {"starting", "stopping", "closed"}


def _resources(value):
    if set(value) != set(RESOURCE_KEYS):
        raise ValueError("allocation requires exactly CPU, GPU and memory")
    result = {}
    for key in RESOURCE_KEYS:
        number = value[key]
        if (
            isinstance(number, bool)
            or not isinstance(number, (int, float))
            or not math.isfinite(number)
            or number != int(number)
        ):
            raise ValueError("allocations use whole CPUs, GPUs and byte counts")
        result[key] = int(number)
    if result["CPU"] < 1 or result["GPU"] not in {0, 1} or result["memory"] < 1:
        raise ValueError("allocation requires positive CPU/RAM and zero or one GPU")
    return result


def _process_identity(value):
    if not isinstance(value, dict):
        raise ValueError("process identity must contain pid and create_time")
    pid, created = value.get("pid"), value.get("create_time")
    if type(pid) is not int or pid < 1 or isinstance(created, bool):
        raise ValueError("process identity is invalid")
    if (
        not isinstance(created, (int, float))
        or not math.isfinite(created)
        or created <= 0
    ):
        raise ValueError("process creation time is unavailable")
    return pid, float(created)


class AdaptiveBroker:
    """Single-driver reservations; serialized state contains no locks or monitors.

    A worker writes tuning-runtime.json: experiment_id, allocation_epoch,
    threads, phase, pid, pid_create_time, child_processes (pid/create_time) and
    children_complete. Sampling ownership must be registered before failed
    setup can be retired. Unavailable observations never establish death.
    """

    def __init__(
        self,
        entries,
        profiles,
        *,
        mode="balanced",
        execution="adaptive",
        clock=time.monotonic,
        global_limit=None,
        file_limits=None,
        incompatible_group_pairs=(),
        uncalibrated_groups=(),
        online_context=None,
        online_root=None,
        online_output=None,
    ):
        if mode not in {
            "office",
            "throughput",
            "balanced",
            "performance",
        } or execution not in {
            "adaptive",
            "fixed",
        }:
            raise ValueError("invalid resource admission mode")
        self.entries = {e["experiment_id"]: copy.deepcopy(e) for e in entries}
        if len(self.entries) != len(entries) or set(profiles) != set(self.entries):
            raise ValueError("resource profiles require unique matching experiment IDs")
        self.profiles = copy.deepcopy(profiles)
        self.mode, self.execution, self.clock = mode, execution, clock
        self.global_limit = (
            max(1, len(entries)) if global_limit is None else global_limit
        )
        self.file_limits = dict(file_limits or {})
        self.incompatible_group_pairs = {
            frozenset(pair) for pair in incompatible_group_pairs
        }
        self.uncalibrated_groups = set(uncalibrated_groups)
        self._uncalibrated_peak = {}
        self._ramp_limit = 1 if self.uncalibrated_groups else self.global_limit
        self._online = {}
        self._online_context = dict(online_context or {})
        self._online_root = str(online_root) if online_root else None
        self._online_output = str(online_output) if online_output else None
        self._online_finished = set()
        if type(self.global_limit) is not int or self.global_limit < 1:
            raise ValueError("global concurrency limit must be positive")
        self._lock = threading.RLock()
        self._broker, self._lease_state = None, {}
        self._last_sample, self._snapshot, self._ema = 0.0, None, None
        self._actors, self._retiring, self._resources = {}, {}, {}
        self._children, self._owners, self._ownership_complete = {}, {}, {}
        self._last_change, self._pending, self._ack = {}, {}, {}
        self._expected_epoch, self._overhead, self._sampling = {}, {}, {}
        self._failures, self._reasons = {}, {}
        self._memory_peaks = {}
        self._protected_pids = set()
        self._last_reason = "initial calibrated allocation"
        for identity, entry in self.entries.items():
            group = entry.get("calibration_group")
            if group is not None and (not isinstance(group, str) or not group):
                raise ValueError("calibration group must be a nonempty identity")
            contract = entry.get("execution_contract", {})
            overhead, epoch = (
                contract.get("cpu_overhead", 0),
                contract.get("allocation_epoch", 0),
            )
            if type(overhead) is not int or overhead < 0:
                raise ValueError("sampling CPU overhead must be a frozen integer")
            if type(epoch) is not int or epoch < 0:
                raise ValueError("allocation epoch must be a nonnegative integer")
            config = json.loads(entry.get("prepared", {}).get("config_json", "{}"))
            runtime = config.get("runtime", {})
            sampling = runtime.get("sampling_processes", 0)
            if type(sampling) is not int or sampling < 0:
                raise ValueError("sampling child count must be a frozen integer")
            self._overhead[identity], self._sampling[identity] = overhead, sampling
            self._expected_epoch[identity] = epoch
            child_threads = contract.get(
                "sampling_child_threads", runtime.get("numerical_threads", 1)
            )
            if (
                "numerical_threads" in runtime or "sampling_child_threads" in contract
            ) and overhead != sampling * child_threads:
                raise ValueError("sampling overhead disagrees with frozen runtime")
            if not self.profiles[identity]:
                raise ValueError("each experiment needs a measured resource profile")
            for profile in self.profiles[identity]:
                resources = _resources({key: profile[key] for key in RESOURCE_KEYS})
                if resources["CPU"] <= overhead:
                    raise ValueError(
                        "allocation omits learner CPU or sampling overhead"
                    )
                if "device" in runtime and resources["GPU"] != int(
                    runtime["device"] == "cuda"
                ):
                    raise ValueError("measured profile changes frozen device cohort")
                if (
                    type(profile["concurrency"]) is not int
                    or profile["concurrency"] < 1
                ):
                    raise ValueError("measured concurrency must be a positive integer")
                for key in ("throughput", "update_seconds", "restart_seconds"):
                    number = profile[key]
                    if (
                        isinstance(number, bool)
                        or not isinstance(number, (int, float))
                        or not math.isfinite(number)
                        or number < 0
                    ):
                        raise ValueError(
                            "measured costs must be finite and nonnegative"
                        )
        if self._online_context:
            from smartsom.experiments.online_concurrency import OnlineConcurrency

            saved = self._online_report()
            for group, context in self._online_context.items():
                members = [
                    key
                    for key, entry in self.entries.items()
                    if entry.get("calibration_group") == group
                ]
                if not members:
                    continue
                ceiling = min(
                    self.global_limit,
                    max(self.profiles[key][0]["concurrency"] for key in members),
                )
                self._online[group] = OnlineConcurrency(
                    ceiling,
                    hint=(context.get("hint") or {}).get("concurrency", 1),
                    history=saved.get("groups", {})
                    .get(group, {})
                    .get("measurements", ()),
                )
            self._ramp_limit = self.global_limit

    def _online_report(self):
        if self._online_root is None:
            return {}
        try:
            path = Path(self._online_root) / "online-performance.json"
            report = json.loads(path.read_text())
            from smartsom.experiments.performance_profiles import ONLINE_SCHEMA

            return report if report.get("schema") == ONLINE_SCHEMA else {}
        except (OSError, ValueError):
            return {}

    def _save_online(self):
        if self._online_root is None:
            return
        from smartsom.experiments.evidence import write_json
        from smartsom.experiments.performance_profiles import (
            ONLINE_SCHEMA,
            store_online,
        )

        report = self._online_report()
        groups = report.get("groups", {})
        profiles = {
            (row["hardware"], row["shape"]): row for row in report.get("profiles", ())
        }
        path = Path(self._online_root) / "online-performance.json"
        records = []
        for group, controller in self._online.items():
            groups[group] = controller.summary()
            accepted = [row for row in controller.history if row["accepted"]]
            if not accepted:
                continue
            best = accepted[-1]
            context = self._online_context[group]
            row = {
                "hardware": context["hardware"],
                "shape": context["shape"],
                "concurrency": best["concurrency"],
                "throughput": best["throughput"],
                "at": time.time(),
                "report": str(path.resolve()),
                "evidence": best,
            }
            profiles[(row["hardware"], row["shape"])] = row
            records.append(row)
        write_json(
            path,
            {
                "schema": ONLINE_SCHEMA,
                "groups": groups,
                "profiles": list(profiles.values()),
            },
        )
        if self._online_output:
            store_online(self._online_output, records)

    def __getstate__(self):
        with self._lock:
            data = dict(self.__dict__)
            if self._broker is not None:
                data["_lease_state"] = {
                    key: asdict(lease) for key, lease in self._broker.leases.items()
                }
            data.pop("_lock", None)
            data["_broker"], data["_snapshot"] = None, None
            data["_last_sample"], data["clock"] = 0.0, time.monotonic
            return data

    def __setstate__(self, data):
        self.__dict__.update(data)
        self._memory_peaks = data.get("_memory_peaks", {})
        self._uncalibrated_peak = data.get("_uncalibrated_peak", {})
        self._lock = threading.RLock()

    def _manager(self):
        if self._broker is None:
            self._broker = ResourceBroker(mode=self.mode)
            self._broker.leases.update(
                {
                    key: ResourceLease(
                        key,
                        ResourceRequest(**value["request"]),
                        value["state"],
                        tuple(value["pids"]),
                    )
                    for key, value in self._lease_state.items()
                }
            )
            self._lease_state = {}
        return self._broker

    def _reason(self, identity, message):
        self._reasons[identity] = self._last_reason = message

    def _actor_state(self, actor):
        if not actor:
            return {}
        import ray

        try:
            return ray._private.state.state.actor_table(actor)
        except (RuntimeError, ValueError, KeyError, ray.exceptions.RayError):
            return {}

    def _process_state(self, pid, created):
        """Return live/dead/unknown; zombies and PID reuse cannot hold a lease."""
        import psutil

        try:
            process = psutil.Process(pid)
            if process.create_time() != created or not process.is_running():
                return "dead"
            return "dead" if process.status() == psutil.STATUS_ZOMBIE else "live"
        except psutil.NoSuchProcess:
            return "dead"
        except (OSError, psutil.Error):
            return "unknown"

    def _descendants(self, pid, created):
        import psutil

        try:
            parent = psutil.Process(pid)
            if parent.create_time() != created:
                return ()
            return tuple(
                (p.pid, p.create_time()) for p in parent.children(recursive=True)
            )
        except (OSError, psutil.Error):
            return ()

    def notify_allocation(self, identity, trial):
        """Called after RCS assigns resources, before starting the next actor."""
        contract = trial.config.get("execution_contract", {})
        epoch = contract.get("allocation_epoch", 0)
        if contract.get("cpu_overhead", 0) != self._overhead[identity]:
            raise ValueError("adaptive resize changed frozen sampling overhead")
        if type(epoch) is not int or epoch < self._expected_epoch[identity]:
            raise ValueError("allocation epoch regressed")
        self._expected_epoch[identity] = epoch

    def note_actor(self, identity, trial):
        with self._lock:
            self.notify_allocation(identity, trial)
            actor = getattr(trial.temporary_state, "ray_actor", None)
            actor_id = getattr(actor, "_actor_id", None)
            if actor_id is not None:
                self._actors[identity] = actor_id.hex()
                manager = self._manager()
                if identity in manager.leases and identity not in self._retiring:
                    manager.transition(identity, "staged")
            self._observe_ack(identity)

    @staticmethod
    def _failure_actor(trial):
        try:
            from ray.exceptions import RayActorError
        except ImportError:
            return None

        try:
            error = trial.get_pickled_error()
        except (OSError, ValueError, EOFError, pickle.PickleError):
            return None
        if not isinstance(error, RayActorError) or not error.actor_id:
            return None
        actor = error.actor_id
        return actor.hex() if hasattr(actor, "hex") else str(actor)

    def actor_failed(self, identity, trial):
        """Recover startup actor ID from Ray's saved error, never from ERROR status."""
        with self._lock:
            self.notify_allocation(identity, trial)
            actor = self._failure_actor(trial) or self._actors.get(identity)
            self._failures[identity], self._retiring[identity] = True, actor
            if actor:
                self._actors[identity] = actor
            self._observe_ack(identity)
            if identity in self._manager().leases:
                self._manager().transition(identity, "stopping")
            self._reason(
                identity,
                "waiting for failed actor and owned children to exit"
                if actor
                else "actor failure identity unresolved; reservation retained",
            )

    def unresolved_failures(self):
        """Failed actors whose reservations cannot be safely released.

        An unknown actor identity cannot establish authoritative death. A known
        DEAD actor with incomplete sampling-child ownership is also unsafe.
        The driver must abort instead of leaving later trials queued forever.
        """
        self.refresh()
        with self._lock:
            return tuple(
                sorted(
                    identity
                    for identity in self._manager().leases
                    if self._failures.get(identity)
                    and identity in self._retiring
                    and (
                        self._retiring[identity] is None
                        or (
                            self._sampling[identity] > 0
                            and not self._ownership_complete.get(identity)
                            and self._actor_state(self._retiring[identity]).get("State")
                            == "DEAD"
                        )
                    )
                )
            )

    def _observe_ack(self, identity):
        path = Path(self.entries[identity]["run_dir"]) / "tuning-runtime.json"
        try:
            data = json.loads(native_path(path).read_text(encoding="utf-8"))
            owner = _process_identity(
                {"pid": data["pid"], "create_time": data["pid_create_time"]}
            )
            epoch, threads = data["allocation_epoch"], data["threads"]
            if (
                data["experiment_id"] != identity
                or type(epoch) is not int
                or epoch != self._expected_epoch[identity]
                or type(threads) is not int
                or threads < 1
                or not isinstance(data["phase"], str)
            ):
                return
            actor = self._actor_state(self._actors.get(identity))
            if actor.get("Pid") != owner[0]:
                return
            children = {_process_identity(p) for p in data.get("child_processes", [])}
            children.update(self._descendants(*owner))
            children.discard(owner)
            self._children.setdefault(identity, set()).update(children)
            self._owners[identity] = owner
            self._ownership_complete[identity] = data.get("children_complete") is True
            self._ack[identity] = data
            manager = self._manager()
            if identity in manager.leases:
                manager.transition(
                    identity,
                    manager.leases[identity].state,
                    pids=(owner[0], *[pid for pid, _ in self._children[identity]]),
                )
                if (
                    self._applied_ack(identity, actor=actor)
                    and identity not in self._retiring
                ):
                    manager.transition(identity, "running")
                    self._pending.pop(identity, None)
        except (OSError, ValueError, KeyError, TypeError):
            return

    def _applied_ack(self, identity, *, actor=None):
        data, resources = self._ack.get(identity), self._resources.get(identity)
        if not data or not resources or identity not in self._manager().leases:
            return None
        actor = (
            self._actor_state(self._actors.get(identity)) if actor is None else actor
        )
        if (
            actor.get("State") != "ALIVE"
            or actor.get("Pid") != data["pid"]
            or data["allocation_epoch"] != self._expected_epoch[identity]
            or data["threads"] + self._overhead[identity] != resources["CPU"]
            or data["phase"] in INACTIVE_PHASES
            or self._process_state(data["pid"], data["pid_create_time"]) != "live"
        ):
            return None
        return data

    def _children_dead(self, identity):
        if self._sampling[identity] and not self._ownership_complete.get(identity):
            self._reason(
                identity, "sampling process ownership unresolved; reservation retained"
            )
            return False
        return all(
            self._process_state(pid, created) == "dead"
            for pid, created in self._children.get(identity, ())
        )

    def _live_owned_pids(self, identity):
        pids = set()
        owner = self._owners.get(identity)
        if (
            owner
            and self._actor_state(self._actors.get(identity)).get("State") == "ALIVE"
            and self._process_state(*owner) == "live"
        ):
            pids.add(owner[0])
        pids.update(
            pid
            for pid, created in self._children.get(identity, ())
            if self._process_state(pid, created) == "live"
        )
        return pids

    def _budget_snapshot(self, snapshot):
        # Excluding the whole Ray tree is appropriate for external CPU pressure.
        # RAM add-back is narrower: only observed processes in an existing trial
        # reservation qualify. Ray services, driver and idle workers stay charged
        # by actual available RAM, without being offered as closeable programs.
        self._protected_pids = {p.pid for p in snapshot.processes if p.excluded}
        trial_pids = set()
        manager = self._manager()
        for identity, lease in tuple(manager.leases.items()):
            pids = self._live_owned_pids(identity)
            trial_pids.update(pids)
            rss = sum(p.rss for p in snapshot.processes if p.pid in pids)
            # A short probe cannot predict growing replay buffers. Charge the
            # observed process tree, retaining its high-water mark across
            # checkpoint restarts which may restore the same replay state.
            self._memory_peaks[identity] = max(self._memory_peaks.get(identity, 0), rss)
            manager.leases[identity] = replace(
                lease,
                request=replace(
                    lease.request,
                    memory=max(
                        self._resources[identity]["memory"],
                        self._memory_peaks[identity],
                    ),
                ),
            )
        return replace(
            snapshot,
            processes=tuple(
                replace(p, excluded=p.pid in trial_pids) for p in snapshot.processes
            ),
        )

    def refresh(self, *, force=False):
        with self._lock:
            now = self.clock()
            if not force and self._snapshot is not None and now - self._last_sample < 5:
                return
            manager = self._manager()
            for identity in self.entries:
                self._observe_ack(identity)
            # Accumulate children before releasing; an empty tree after parent
            # death must not erase identities of children reparented elsewhere.
            for identity, actor in list(self._actors.items()):
                if self._actor_state(actor).get(
                    "State"
                ) == "DEAD" and self._children_dead(identity):
                    if identity in manager.leases:
                        manager.release(identity)
                    self._retiring.pop(identity, None)
                    self._actors.pop(identity, None)
                    self._ack.pop(identity, None)
                    self._owners.pop(identity, None)
                    self._children.pop(identity, None)
                    self._ownership_complete.pop(identity, None)
            # Do not exclude a historical PID now reused by another program.
            for identity, lease in tuple(manager.leases.items()):
                manager.transition(
                    identity, lease.state, pids=tuple(self._live_owned_pids(identity))
                )
            snapshot = manager.snapshot(exclude_pids=(os.getpid(),))
            external = snapshot.external_cpu_load
            if external is not None:
                alpha = 1 - math.exp(-max(0.0, now - self._last_sample) / 30)
                self._ema = (
                    external
                    if self._ema is None
                    else self._ema + alpha * (external - self._ema)
                )
                snapshot = replace(snapshot, external_cpu_load=max(external, self._ema))
            self._snapshot, self._last_sample = self._budget_snapshot(snapshot), now

    def _profile(self, identity, resources):
        resources = _resources(resources)
        for profile in self.profiles[identity]:
            if _resources({key: profile[key] for key in RESOURCE_KEYS}) == resources:
                return profile
        raise ValueError("requested allocation was not measured for this experiment")

    def _request(self, identity, resources):
        profile = self._profile(identity, resources)
        gpu, gpu_memory = (
            None,
            int(profile.get("gpu_memory", 0)) if resources["GPU"] else 0,
        )
        if resources["GPU"]:
            manager = self._manager()
            if gpu_memory <= 0:
                self._reason(identity, "measured GPU peak memory unavailable")
                return None
            available = manager.capacity(self._snapshot).gpu_memory
            # Ray independently assigns a GPU. Budget the worst visible device
            # until placement is reported; never guess a favorable physical ID.
            if len(available) != len(self._snapshot.gpus) or not available:
                self._reason(identity, "visible GPU capacity unavailable or ambiguous")
                return None
            if min(available.values()) < math.ceil(gpu_memory * manager.peak_factor):
                self._reason(identity, "a possible Ray-assigned GPU lacks peak memory")
                return None
            occupied = {
                lease.request.gpu
                for key, lease in manager.leases.items()
                if key != identity
            }
            gpu = next((g for g in available if g not in occupied), None)
            if gpu is None:
                self._reason(identity, "waiting for an unoccupied visible GPU")
                return None
        memory = max(resources["memory"], self._memory_peaks.get(identity, 0))
        group = self.entries[identity].get("calibration_group")
        if group in self.uncalibrated_groups:
            if group in self._uncalibrated_peak:
                memory = max(memory, self._uncalibrated_peak[group])
            else:
                # One unknown worker owns the live available budget until its
                # first committed update supplies an observed process peak.
                capacity = self._manager().capacity(self._snapshot)
                memory = max(memory, int(capacity.memory / self._manager().peak_factor))
        return ResourceRequest(
            resources["CPU"],
            memory,
            gpu,
            gpu_memory,
        )

    def try_acquire(self, identity, resources):
        resources = _resources(resources)
        profile = self._profile(identity, resources)
        self.refresh()
        with self._lock:
            manager = self._manager()
            if identity in self._failures or identity in self._retiring:
                return False
            if identity in manager.leases:
                # RCS asks twice before staging; a reserved resize need not have
                # an applied acknowledgement to permit its own actor to start.
                return self._resources.get(identity) == resources
            if len(manager.leases) >= min(self.global_limit, self._ramp_limit):
                self._reason(identity, "waiting for stage concurrency or measured ramp")
                return False
            file_id = self.entries[identity].get("file_id", identity.split("__", 1)[0])
            if (
                file_id in self.file_limits
                and sum(
                    self.entries[key].get("file_id", key.split("__", 1)[0]) == file_id
                    for key in manager.leases
                )
                >= self.file_limits[file_id]
            ):
                self._reason(identity, "waiting for Experiment concurrency limit")
                return False
            group = self.entries[identity].get("calibration_group")
            if group in self._online:
                count = sum(
                    self.entries[key].get("calibration_group") == group
                    for key in manager.leases
                )
                if count >= self._online[group].limit:
                    self._reason(identity, self._online[group].reason)
                    return False
            if any(
                frozenset((group, self.entries[key].get("calibration_group")))
                in self.incompatible_group_pairs
                for key in manager.leases
            ):
                self._reason(
                    identity,
                    "waiting for a comparable online task group"
                    if self._online
                    else "mixed pair measured slower; waiting for separate execution",
                )
                return False
            members = [
                key
                for key in manager.leases
                if self.entries[key].get("calibration_group") == group
            ]
            limits = [
                profile["concurrency"],
                *[
                    self._profile(key, self._resources[key])["concurrency"]
                    for key in members
                ],
            ]
            if len(members) >= min(limits):
                self._reason(identity, "waiting for calibrated concurrency capacity")
                return False
            request = self._request(identity, resources)
            if request is None:
                return False
            admission = manager.admit(request, self._snapshot)
            if (
                not admission.allowed
                or "CPU load first sample unavailable" in self._snapshot.unavailable
            ):
                self._reason(
                    identity, admission.reason or "waiting for second CPU observation"
                )
                return False
            manager.reserve(identity, request, snapshot=self._snapshot)
            self._resources[identity] = resources
            self._last_change[identity] = self.clock()
            return True

    def observe_formal_update(self, identity, result=None):
        """Increase fallback admission only after a committed update and RSS sample."""
        self.refresh(force=True)
        with self._lock:
            group = self.entries[identity].get("calibration_group")
            peak = self._memory_peaks.get(identity, 0)
            if group in self.uncalibrated_groups and peak > 0:
                self._uncalibrated_peak[group] = max(
                    peak, self._uncalibrated_peak.get(group, 0)
                )
                if group not in self._online:
                    self._ramp_limit = min(self.global_limit, self._ramp_limit + 1)
            if group not in self._online or result is None:
                return
            controller = self._online[group]
            if result.get("phase") != "training":
                self._online_finished.add(identity)
                controller.suspend()
                return
            members = [
                key
                for key, lease in self._manager().leases.items()
                if self.entries[key].get("calibration_group") == group
            ]
            if (
                peak <= 0
                or self._pressure()[0]
                or any(
                    key in self._online_finished
                    or self._manager().leases[key].state != "running"
                    or self._ack.get(key, {}).get("phase")
                    not in {"training", "committed"}
                    for key in members
                )
            ):
                controller.suspend()
                return
            decision = controller.observe(
                members,
                identity,
                ticks=result.get("physical_ticks"),
                updates=result.get("updates"),
                now=self.clock(),
            )
            if decision is not None:
                self._reason(identity, decision["reason"])
                self._save_online()

    def defer_resize(self, identity, current_resources, requested_resources):
        current, requested = (
            _resources(current_resources),
            _resources(requested_resources),
        )
        self._profile(identity, requested)
        with self._lock:
            if self._resources.get(identity) != current:
                raise ValueError("resize current resources disagree with reservation")
            if current["GPU"] != requested["GPU"]:
                raise ValueError("resize cannot change the frozen device cohort")
            self._pending[identity] = requested
            self._last_change[identity] = self.clock()
            self._retiring[identity] = self._actors.get(identity)
            self._manager().transition(identity, "stopping")
            if self._retiring[identity] is None:
                self._reason(
                    identity, "resize actor identity unresolved; reservation retained"
                )

    def release(self, identity):
        """Only searcher's rollback before returning a trial uses this method."""
        with self._lock:
            manager = self._manager()
            if (
                identity in manager.leases
                and manager.leases[identity].state == "pending"
                and identity not in self._actors
                and identity not in self._retiring
                and identity not in self._ack
            ):
                manager.release(identity)

    def _pressure(self):
        manager = self._manager()
        cap = manager.capacity(self._snapshot)
        cpu = sum(lease.request.cpus for lease in manager.leases.values())
        ram = sum(
            math.ceil(lease.request.memory * manager.peak_factor)
            for lease in manager.leases.values()
        )
        return cpu > cap.cpus or ram > cap.memory, cap, cpu

    def _victim(self):
        leases = self._manager().leases
        # One stopping lease already responds to pressure. Await its observed
        # exit before pausing additional trials on the same pressure sample.
        if any(lease.state == "stopping" for lease in leases.values()):
            return None
        running = sorted(
            key for key, lease in leases.items() if lease.state == "running"
        )
        return running[-1] if running else None

    def should_pause(self, identity):
        if self.execution == "fixed":
            return False
        self.refresh()
        with self._lock:
            if self._pressure()[0] and identity == self._victim():
                return True
            group = self.entries[identity].get("calibration_group")
            if group not in self._online:
                return False
            leases = self._manager().leases
            members = [
                key
                for key, lease in leases.items()
                if self.entries[key].get("calibration_group") == group
            ]
            running = sorted(key for key in members if leases[key].state == "running")
            return (
                len(members) > self._online[group].limit
                and not any(leases[key].state == "stopping" for key in members)
                and bool(running)
                and identity == running[-1]
            )

    def allocation(self, controller, trial, result, scheduler):
        identity = trial.config["experiment_id"]
        self.note_actor(identity, trial)
        self.refresh()
        if self.execution == "fixed" or identity in self._retiring:
            return None
        current = dict(trial.placement_group_factory.required_resources)
        # Ray PlacementGroupFactory removes zero-valued GPU bundles on CPU.
        current.setdefault("GPU", 0)
        current = _resources(current)
        profiles = self.profiles[identity]
        with self._lock:
            pressure, cap, used = self._pressure()
            if pressure:
                if identity != self._victim():
                    return None
                smaller = [
                    p
                    for p in profiles
                    if p["CPU"] <= current["CPU"]
                    and p["memory"] <= current["memory"]
                    and (p["CPU"] < current["CPU"] or p["memory"] < current["memory"])
                ]
                if smaller:
                    target = max(smaller, key=lambda p: p["throughput"])
                    self._reason(
                        identity,
                        "external pressure; measured smaller allocation at completed update",
                    )
                    return _resources({key: target[key] for key in RESOURCE_KEYS})
                return None
            if self.clock() - self._last_change.get(identity, self.clock()) < 60:
                return None
            remaining = max(
                0,
                self.entries[identity]["remaining_updates"] - result.get("updates", 0),
            )
            baseline = self._profile(identity, current)
            for target in sorted(profiles, key=lambda p: -p["throughput"]):
                if target["CPU"] <= current["CPU"]:
                    continue
                resources = _resources({key: target[key] for key in RESOURCE_KEYS})
                if used - current["CPU"] + target["CPU"] > cap.cpus:
                    continue
                request = self._request(identity, resources)
                if request is None:
                    continue
                manager = self._manager()
                ram = sum(
                    math.ceil(lease.request.memory * manager.peak_factor)
                    for key, lease in manager.leases.items()
                    if key != identity
                )
                if ram + math.ceil(request.memory * manager.peak_factor) > cap.memory:
                    continue
                saved = remaining * (
                    baseline["update_seconds"] - target["update_seconds"]
                )
                if (
                    target["throughput"] > baseline["throughput"] * 1.05
                    and saved > target["restart_seconds"] * 2
                ):
                    self._reason(identity, "measured gain amortizes checkpoint restart")
                    return resources
        return None

    def summary(self):
        self.refresh()
        with self._lock:
            cap, manager = self._manager().capacity(self._snapshot), self._manager()
            entries = []
            for identity in self.entries:
                ack, lease = self._applied_ack(identity), manager.leases.get(identity)
                status = lease.state if lease else "waiting_resources"
                if identity in self._failures:
                    status = "failed" if lease is None else "stopping"
                    if self._retiring.get(identity) is None and lease is not None:
                        status = "unresolved"
                entries.append(
                    {
                        "experiment_id": identity,
                        "calibration_group": self.entries[identity].get(
                            "calibration_group"
                        ),
                        "status": status,
                        "threads": ack["threads"] if ack else None,
                        "requested_cpus": self._pending.get(
                            identity, self._resources.get(identity, {})
                        ).get("CPU"),
                        "actual_cpus": ack["threads"] + self._overhead[identity]
                        if ack
                        else None,
                        "allocation_epoch": ack["allocation_epoch"] if ack else None,
                        "pending_resize": identity in self._pending,
                        "resource_change_reason": self._reasons.get(
                            identity, self._last_reason
                        ),
                    }
                )
            return {
                "online": {
                    key: {
                        "active": sum(
                            lease.state == "running"
                            and self.entries[identity].get("calibration_group") == key
                            for identity, lease in manager.leases.items()
                        ),
                        **{
                            k: v
                            for k, v in controller.summary().items()
                            if k != "measurements"
                        },
                    }
                    for key, controller in self._online.items()
                },
                "resources": {
                    "cpus_available": cap.cpus,
                    "memory_available": cap.memory,
                    "external_cpu_load": self._snapshot.external_cpu_load,
                    "mode": self.mode,
                    "processes": [
                        {
                            "pid": p.pid,
                            "name": p.name,
                            "cpu_cores": p.cpu_cores,
                            "rss": p.rss,
                            "protected": p.pid in self._protected_pids
                            or p.excluded
                            or "python" in p.name.lower()
                            or "smartsom" in p.name.lower(),
                        }
                        for p in sorted(
                            self._snapshot.processes, key=lambda p: p.rss, reverse=True
                        )[:8]
                    ],
                },
                "entries": entries,
            }
