"""Bounded active calibration; callers provide isolated, terminable real probes.

This module schedules measurements, never manufactures learning throughput or
imports a training framework. Waiting for external load consumes the total
wall-clock deadline. A supervisor must actually terminate
its owned process when the supplied remaining budget expires.
"""

import math
import os
import statistics
import time
from dataclasses import asdict, dataclass, field, replace
from typing import Callable, Literal, Protocol

from smartsom.experiments.tuning_resources import (
    ExecutionProfile,
    ResourceBroker,
    ResourceRequest,
)


@dataclass(frozen=True)
class CandidateMeasurement:
    profile: ExecutionProfile
    throughput: float
    peak_memory: int
    stages: dict[str, float] = field(default_factory=dict)
    valid: bool = True
    reason: str | None = None
    group: str = ""
    peak_gpu_memory: int = 0
    elapsed_seconds: float = 0.0
    termination: Literal["deadline", "cancelled"] | None = None

    def __post_init__(self):
        if self.termination not in {None, "deadline", "cancelled"}:
            raise ValueError(
                "measurement termination must be deadline/cancelled or null"
            )

    @property
    def eligible(self):
        return (
            self.valid
            and self.termination is None
            and math.isfinite(self.throughput)
            and self.throughput > 0
            and type(self.peak_memory) is int
            and self.peak_memory >= 0
            and type(self.peak_gpu_memory) is int
            and self.peak_gpu_memory >= 0
            and math.isfinite(self.elapsed_seconds)
            and self.elapsed_seconds >= 0
            and all(math.isfinite(v) and v >= 0 for v in self.stages.values())
        )


@dataclass(frozen=True)
class CalibrationReport:
    measurements: tuple[CandidateMeasurement, ...]
    recommendations: dict[str, ExecutionProfile]
    baselines: dict[str, CandidateMeasurement]
    active_seconds: float
    waiting_seconds: float
    status: str
    missing_groups: tuple[str, ...] = ()
    reason: str | None = None
    converged_groups: tuple[str, ...] = ()
    unstable_groups: tuple[str, ...] = ()
    wall_seconds: float = 0.0

    @property
    def ready(self):
        return (
            self.status in {"completed", "deadline"}
            and not self.missing_groups
            and bool(self.recommendations)
        )


class ProbeSupervisor(Protocol):
    def run(
        self,
        probe: Callable,
        group: object,
        profile: ExecutionProfile,
        remaining_seconds: float,
        cancelled: Callable[[], bool],
    ) -> CandidateMeasurement:
        """Run and join an owned process, terminating on deadline/cancellation."""


def integer_candidates(maximum):
    """Include one, powers of two and the allocation boundary exactly once."""
    maximum = math.floor(maximum)
    if maximum < 1:
        return ()
    values, value = {1, maximum}, 2
    while value < maximum:
        values.add(value)
        value *= 2
    return tuple(sorted(values))


def generate_candidates(snapshot, *, mode="balanced"):
    """Explore CPU learner and sampler layouts before the formal run starts."""
    capacity = ResourceBroker(mode=mode).capacity(snapshot)
    values = integer_candidates(capacity.cpus)
    devices = [
        "cpu",
        *(
            f"cuda:{g.index}"
            for g in snapshot.gpus
            if g.reason is None and g.memory_available is not None
        ),
    ]
    environment_values = tuple(v for v in values if v <= 8)
    profiles = {
        ExecutionProfile(threads, concurrency, device, envs, processes)
        for device in devices
        for threads in values
        for concurrency in values
        for envs in environment_values
        for processes in (0, *[v for v in environment_values if v <= envs and envs > 1])
        if (threads + processes) * concurrency <= capacity.cpus
        and (device == "cpu" or (envs == 1 and processes == 0))
        and (device == "cpu" or concurrency == 1)
    }
    return tuple(
        sorted(
            profiles,
            key=lambda p: (
                (p.threads + p.sampling_processes) * p.concurrency,
                p.sampling_processes == 0,
                p.num_envs,
                p.device != "cpu",
                p.threads,
                p.concurrency,
                p.device,
            ),
        )
    )


def measurement_request(measurement, *, cpu_overhead=0.0):
    device = measurement.profile.device
    gpu = (
        None if device == "cpu" else device.split(":", 1)[-1] if ":" in device else "0"
    )
    return ResourceRequest(
        (
            measurement.profile.threads
            + measurement.profile.sampling_processes
            + cpu_overhead
        )
        * measurement.profile.concurrency,
        measurement.peak_memory,
        gpu,
        measurement.peak_gpu_memory,
    )


def rank_measurements(
    measurements, *, snapshot=None, mode="balanced", cpu_overhead=0.0
):
    """Rank valid batch throughput after peak-budget constraints, with stable ties."""
    broker = ResourceBroker(mode=mode)
    valid = [m for m in measurements if m.eligible]
    if snapshot is not None:
        valid = [
            m
            for m in valid
            if broker.admit(
                measurement_request(m, cpu_overhead=cpu_overhead), snapshot
            ).allowed
        ]
    return tuple(
        sorted(
            valid,
            key=lambda m: (
                -m.throughput,
                m.peak_memory,
                (m.profile.threads + m.profile.sampling_processes)
                * m.profile.concurrency,
                m.profile.device,
            ),
        )
    )


class CalibrationController:
    def __init__(
        self,
        probe,
        monitor,
        *,
        mode="balanced",
        active_limit=600.0,
        clock=time.monotonic,
        sleeper=time.sleep,
        supervisor=None,
        wait_seconds=1.0,
        bootstrap_memory=256 * 1024**2,
        profile=None,
        exclude_pids=None,
        fair_baselines=False,
    ):
        if not math.isfinite(active_limit) or active_limit <= 0:
            raise ValueError("active calibration limit must be positive and finite")
        if not math.isfinite(wait_seconds) or wait_seconds <= 0:
            raise ValueError("resource wait interval must be positive and finite")
        self.probe, self.monitor, self.mode = probe, monitor, mode
        self.active_limit, self.clock, self.sleeper = active_limit, clock, sleeper
        self.supervisor, self.wait_seconds = supervisor, wait_seconds
        self.bootstrap_memory, self.profile = bootstrap_memory, profile
        self.fair_baselines = fair_baselines
        self.exclude_pids = (
            (os.getpid(),) if exclude_pids is None else tuple(exclude_pids)
        )
        self.broker = ResourceBroker(monitor, mode=mode)

    def _snapshot(self):
        return self.monitor.snapshot(exclude_pids=self.exclude_pids)

    def run(
        self,
        groups,
        candidates=None,
        *,
        cancelled=lambda: False,
        on_wait=None,
        baseline_profiles=None,
        on_measure=None,
    ):
        """Measure actual candidates within one wall-clock budget."""
        if not groups:
            raise ValueError("calibration requires at least one workload group")
        baseline = self.profile or ExecutionProfile(1, 1, "cpu")
        baseline_profiles = baseline_profiles or {}
        if set(baseline_profiles) - set(groups):
            raise ValueError("baseline profile references an unknown workload group")
        group_baselines = {
            name: baseline_profiles.get(name, baseline) for name in groups
        }
        if any(
            not isinstance(profile, ExecutionProfile)
            for profile in group_baselines.values()
        ):
            raise ValueError("baseline profiles must be ExecutionProfile values")
        started_calibration = self.clock()
        deadline = started_calibration + self.active_limit
        measurements, baselines, recommendations = [], {}, {}
        active, waiting, status, reason = 0.0, 0.0, "completed", None
        stable = {name: 0 for name in groups}
        stagnant = {name: 0 for name in groups}
        explored = {name: 0 for name in groups}
        unstable, blocked = set(), set()
        overheads = {}
        for name, group in groups.items():
            overhead = (
                group.get("cpu_overhead", 0.0) if isinstance(group, dict) else 0.0
            )
            if (
                not isinstance(overhead, (int, float))
                or not math.isfinite(overhead)
                or overhead < 0
            ):
                raise ValueError("group CPU overhead must be nonnegative and finite")
            overheads[name] = overhead
        initial = self._snapshot()
        proposed = generate_candidates(initial, mode=self.mode)
        per_group = {
            name: tuple(dict.fromkeys((candidates or {}).get(name, proposed)))
            for name in groups
        }
        candidate_count = 3 * len(groups) + sum(
            sum(profile != group_baselines[name] for profile in profiles)
            for name, profiles in per_group.items()
        )

        def progress(name, profile, stage, reason=None):
            if on_measure:
                on_measure(
                    {
                        "group": name,
                        "profile": asdict(profile),
                        "active_seconds": active,
                        "wall_seconds": max(0.0, self.clock() - started_calibration),
                        "remaining_seconds": max(0.0, deadline - self.clock()),
                        "waiting_seconds": waiting,
                        "measured": len(measurements),
                        "candidates": max(
                            candidate_count, len(measurements) + (stage == "measuring")
                        ),
                        "reason": reason,
                        "stage": stage,
                    }
                )

        def idle(snapshot):
            return replace(
                snapshot,
                memory_available=snapshot.memory_total,
                external_cpu_load=0.0,
                cpu_load=0.0,
                processes=(),
                gpus=tuple(
                    replace(gpu, memory_available=gpu.memory_total)
                    if gpu.memory_total is not None
                    else gpu
                    for gpu in snapshot.gpus
                ),
            )

        def leaders(name, snapshot=None):
            # Repeated observations use median throughput and worst observed peaks.
            # A single fast outlier must not remain the winner indefinitely.
            by_profile = {}
            for measurement in measurements:
                if (
                    measurement.group == name
                    and measurement.eligible
                    and (name, measurement.profile) not in blocked
                ):
                    by_profile.setdefault(measurement.profile, []).append(measurement)
            aggregated = [
                replace(
                    items[0],
                    throughput=statistics.median(m.throughput for m in items),
                    peak_memory=max(m.peak_memory for m in items),
                    peak_gpu_memory=max(m.peak_gpu_memory for m in items),
                    elapsed_seconds=max(m.elapsed_seconds for m in items),
                )
                for items in by_profile.values()
            ]
            return rank_measurements(
                aggregated,
                snapshot=snapshot,
                mode=self.mode,
                cpu_overhead=overheads[name],
            )

        def observed_cost(name):
            return max(
                (m.elapsed_seconds for m in measurements if m.group == name),
                default=0.0,
            )

        def measure(name, profile, is_baseline=False, budget=None):
            nonlocal active, waiting, status
            if cancelled():
                status = "cancelled"
                return None
            if self.clock() >= deadline:
                status = "deadline"
                return None
            group_deadline = min(
                deadline,
                self.clock() + budget if budget is not None else deadline,
            )
            known = baselines.get(name)
            memory = (
                known.peak_memory * profile.concurrency
                if known
                else self.bootstrap_memory
            )
            gpu_memory = known.peak_gpu_memory * profile.concurrency if known else 0
            gpu = (
                None
                if profile.device == "cpu"
                else profile.device.split(":", 1)[-1]
                if ":" in profile.device
                else "0"
            )
            if gpu is not None and profile.concurrency != 1:
                result = CandidateMeasurement(
                    profile,
                    0.0,
                    memory,
                    valid=False,
                    reason="GPU calibration currently supports concurrency=1 only",
                    group=name,
                )
                measurements.append(result)
                blocked.add((name, profile))
                return result
            request = ResourceRequest(
                (profile.threads + profile.sampling_processes + overheads[name])
                * profile.concurrency,
                memory,
                gpu,
                gpu_memory,
            )
            while True:
                if cancelled():
                    status = "cancelled"
                    return None
                if self.clock() >= deadline:
                    status = "deadline"
                    return None
                if self.clock() >= group_deadline:
                    result = CandidateMeasurement(
                        profile,
                        0.0,
                        memory,
                        valid=False,
                        reason="group probe budget exhausted waiting for resources",
                        group=name,
                        termination="deadline",
                    )
                    measurements.append(result)
                    progress(name, profile, "measured", result.reason)
                    return result
                snapshot = self._snapshot()
                if (
                    not is_baseline
                    and not self.broker.admit(request, idle(snapshot)).allowed
                ):
                    result = CandidateMeasurement(
                        profile,
                        0.0,
                        memory,
                        valid=False,
                        reason="candidate exceeds static resource allocation",
                        group=name,
                    )
                    measurements.append(result)
                    blocked.add((name, profile))
                    return result
                decision = self.broker.admit(request, snapshot)
                if (
                    not decision.allowed
                    and not is_baseline
                    and profile != group_baselines[name]
                ):
                    result = CandidateMeasurement(
                        profile,
                        0.0,
                        memory,
                        valid=False,
                        reason="current resource load exceeds candidate budget; skipped",
                        group=name,
                    )
                    measurements.append(result)
                    blocked.add((name, profile))
                    progress(name, profile, "skipped", result.reason)
                    return result
                if (
                    decision.allowed
                    and "CPU load first sample unavailable" not in snapshot.unavailable
                ):
                    break
                if on_wait:
                    on_wait(
                        {
                            "group": name,
                            "profile": profile,
                            "reason": decision.reason or "waiting for CPU load sample",
                            "active_seconds": active,
                            "wall_seconds": max(
                                0.0, self.clock() - started_calibration
                            ),
                            "remaining_seconds": max(0.0, deadline - self.clock()),
                            "waiting_seconds": waiting,
                        }
                    )
                before = self.clock()
                self.sleeper(self.wait_seconds)
                waiting += max(0.0, self.clock() - before)
            started = self.clock()
            remaining = max(0.0, group_deadline - started)
            if remaining <= 0:
                result = CandidateMeasurement(
                    profile,
                    0.0,
                    memory,
                    valid=False,
                    reason="group probe budget exhausted before measurement",
                    group=name,
                    termination="deadline",
                )
                measurements.append(result)
                progress(name, profile, "measured", result.reason)
                return result
            progress(name, profile, "measuring")
            try:
                if self.supervisor:
                    result = self.supervisor.run(
                        self.probe, groups[name], profile, remaining, cancelled
                    )
                else:
                    # The callback contract requires its own terminable isolation.
                    result = self.probe(groups[name], profile, remaining)
                if not isinstance(result, CandidateMeasurement):
                    raise TypeError("probe must return CandidateMeasurement")
                if result.profile != profile:
                    raise ValueError("probe measured a different execution profile")
            except Exception as exc:
                result = CandidateMeasurement(
                    profile, 0.0, 0, valid=False, reason=f"{type(exc).__name__}: {exc}"
                )
            spent = max(0.0, self.clock() - started)
            active += spent
            if spent > remaining + 0.05:
                result = replace(
                    result,
                    valid=False,
                    reason="probe exceeded active deadline; supervisor isolation required",
                    termination=None,
                )
            elif cancelled():
                status = "cancelled"
                result = replace(
                    result,
                    valid=False,
                    reason="calibration cancelled during probe",
                    termination="cancelled",
                )
            elif result.termination == "cancelled":
                status = "cancelled"
            elif result.termination == "deadline" and budget is None:
                # The supervisor reserves shutdown time inside its budget. A
                # joined deadline outcome can therefore consume slightly less
                # wall time than ``remaining`` while still exhausting this probe.
                # Expansion has a separate sub-budget; its deadline leaves the
                # reserved leader-repeat budget available.
                status = "deadline"
            result = replace(result, group=name, elapsed_seconds=spent)
            constrained = rank_measurements(
                [result],
                snapshot=snapshot,
                mode=self.mode,
                cpu_overhead=overheads[name],
            )
            if not constrained and result.eligible:
                result = replace(
                    result,
                    valid=False,
                    reason="measured candidate peak exceeds resource budget",
                )
            measurements.append(result)
            if not result.eligible:
                if result.termination is None:
                    blocked.add((name, profile))
            elif is_baseline:
                baselines[name] = result
            if self.clock() >= deadline:
                status = "deadline"
            progress(name, profile, "measured", result.reason)
            return result

        # Every group gets its baseline before any expansion or repeated leader.
        for index, name in enumerate(groups):
            # Directory batches have one shared deadline. Bound each group's
            # first probe so one slow backend cannot starve every other group.
            budget = None
            if self.fair_baselines:
                budget = max(
                    0.001,
                    (deadline - self.clock()) * 0.9 / (len(groups) - index),
                )
            measure(name, group_baselines[name], True, budget=budget)
            if status != "completed":
                break
        if status == "completed":
            # Round robin expansion; observed costs reserve two confirmation rounds.
            stop_expansion = False
            for index in range(max((len(p) for p in per_group.values()), default=0)):
                for name, profiles in per_group.items():
                    if (
                        index >= len(profiles)
                        or profiles[index] == group_baselines[name]
                        or (explored[name] >= 6 and stagnant[name] >= 5)
                    ):
                        continue
                    reserve = 2 * sum(observed_cost(group) for group in groups)
                    available = deadline - self.clock() - reserve
                    if available <= 0 or observed_cost(name) > available:
                        stop_expansion = True
                        break
                    previous = leaders(name)
                    measured = measure(name, profiles[index], budget=available)
                    if measured is not None and measured.eligible:
                        explored[name] += 1
                        current = leaders(name)
                        if current and (
                            not previous
                            or current[0].throughput > previous[0].throughput * 1.05
                        ):
                            stagnant[name] = 0
                        else:
                            stagnant[name] += 1
                    if status != "completed":
                        break
                if stop_expansion or status != "completed":
                    break

        while status == "completed" and any(rounds < 2 for rounds in stable.values()):
            before_round = active
            blocked_before_round = set(blocked)
            for name in groups:
                if stable[name] >= 2:
                    continue
                ranked = leaders(name)
                if not ranked:
                    status, reason = (
                        "failed",
                        "leader repeats failed; no confirmed feasible candidate",
                    )
                    break
                previous = ranked[0]
                # Use the selected leader's valid observed cost, not a failed
                # expansion's cost. A hopeless final repeat can otherwise run
                # into the deadline and turn valid calibration into an error.
                leader_cost = max(
                    m.elapsed_seconds
                    for m in measurements
                    if m.group == name and m.profile == previous.profile and m.eligible
                )
                if leader_cost > deadline - self.clock():
                    status = "deadline"
                    break
                latest = measure(name, previous.profile)
                if status == "cancelled" or latest is None:
                    break
                current = leaders(name)
                if latest is None or not latest.eligible or not current:
                    stable[name] = 0
                    unstable.add(name)
                    continue
                # Both directions matter: a slowdown is instability, not proof
                # that the previous fast observation is an excellent setting.
                changed = (
                    abs(latest.throughput / previous.throughput - 1) > 0.05 + 1e-12
                )
                improved = current[0].throughput > previous.throughput * 1.05 + 1e-12
                if changed or improved or current[0].profile != previous.profile:
                    stable[name] = 0
                    unstable.add(name)
                else:
                    stable[name] += 1
                    if stable[name] >= 2:
                        unstable.discard(name)
                if status != "completed":
                    break
            # Blocking a leader is progress even without probe time; retry
            # the next feasible candidate before declaring a stalled clock.
            if (
                status == "completed"
                and active == before_round
                and blocked == blocked_before_round
                and unstable
            ):
                status, reason = (
                    "failed",
                    "probe clock did not advance; cannot enforce active calibration deadline",
                )

        final = self._snapshot()
        converged = tuple(name for name in groups if stable[name] >= 2)
        for name in groups:
            ranked = leaders(name, final)
            if ranked:
                recommendations[name] = ranked[0].profile
        missing = tuple(name for name in groups if name not in recommendations)
        if missing and status not in {"cancelled", "deadline"}:
            status, reason = (
                "failed",
                "every workload group needs a valid constrained baseline",
            )
        if status == "deadline":
            reason = (
                "wall-clock calibration deadline; no valid measured candidate for all groups; starting layouts remain uncalibrated"
                if missing
                else "wall-clock calibration deadline reached after all leader repeats converged"
                if len(converged) == len(groups)
                else "wall-clock calibration deadline; best valid measured candidates selected without convergence"
            )
        elif status == "cancelled":
            reason = "calibration cancelled; no automatic launch"
        elif status == "completed":
            reason = "leaders stable within 5% for two consecutive repeat rounds"
        return CalibrationReport(
            tuple(measurements),
            recommendations,
            baselines,
            active,
            waiting,
            status,
            missing,
            reason,
            converged,
            tuple(name for name in groups if name in unstable),
            max(0.0, self.clock() - started_calibration),
        )
