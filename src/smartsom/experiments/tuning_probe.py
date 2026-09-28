"""Terminable real-work calibration in isolated, disposable process sessions.

No formal model is selected or continued here. Frozen worlds, seeds, algorithms,
networks and sampling streams stay fixed. Only numerical threads and physical
GPU visibility are execution controls. Worker and resource-observer processes
are owned by this supervisor; no timeout leaves a background Python thread.
"""

from __future__ import annotations

import argparse
import copy
import csv
import inspect
import json
import math
import os
import pickle
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass, is_dataclass, replace
from pathlib import Path


@dataclass(frozen=True)
class ProbeWork:
    physical_ticks: int
    updates: int
    stages: dict[str, float]
    valid: bool = True
    reason: str | None = None
    peak_gpu_memory: int = 0


@dataclass(frozen=True)
class _WorkerProfile:
    threads: int
    concurrency: int
    device: str
    num_envs: int = 1
    sampling_processes: int = 0


def _json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, allow_nan=False) + "\n")
    temporary.replace(path)


def _read(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {}


def _payload(group):
    value = group.get("prepared", group) if isinstance(group, Mapping) else group
    return asdict(value) if is_dataclass(value) else dict(value)


def sampling_cpu_overhead(group, profile=None):
    """Reserve one CPU per selected persistent sampling child."""
    if profile is not None and hasattr(profile, "sampling_processes"):
        return profile.sampling_processes
    config = json.loads(_payload(group)["config_json"])
    runtime = config.get("runtime", {})
    processes = runtime.get("sampling_processes", 0)
    threads = runtime.get("numerical_threads", 1)
    if (
        type(processes) is not int
        or processes < 0
        or type(threads) is not int
        or threads < 1
    ):
        raise ValueError("invalid frozen sampling-process/thread configuration")
    required = processes * threads
    declared = (
        group.get("cpu_overhead", required) if isinstance(group, Mapping) else required
    )
    if type(declared) is not int or declared < required:
        raise ValueError("cpu_overhead omits frozen sampling child reservations")
    return declared


def _gpu_token(group, profile):
    frozen = (
        json.loads(_payload(group)["config_json"])
        .get("runtime", {})
        .get("device", "cpu")
    )
    if profile.device == "cpu":
        if frozen != "cpu":
            raise ValueError("CPU calibration cannot change a frozen CUDA experiment")
        return ""
    if not (
        profile.device == "cuda"
        or profile.device.startswith("cuda:")
        and profile.device.split(":", 1)[1].isdigit()
    ):
        raise ValueError("probe device must be CPU or an explicitly mapped CUDA GPU")
    if frozen != "cuda":
        raise ValueError("CUDA calibration cannot change a frozen CPU experiment")
    if profile.concurrency != 1:
        raise ValueError(
            "multi-GPU calibration requires a separately verified aggregate contract"
        )
    visible = group.get("visible_gpus") if isinstance(group, Mapping) else None
    index = int(profile.device.split(":", 1)[1]) if ":" in profile.device else 0
    if not isinstance(visible, (list, tuple)) or index >= len(visible):
        raise ValueError("CUDA calibration requires an explicit visible GPU mapping")
    token = visible[index]
    if not isinstance(token, str) or not token or "," in token:
        raise ValueError("each probe must receive exactly one explicit GPU token")
    return token


class ProbeSupervisor:
    """Run the calibration protocol with a hard, cancellable process watchdog.

    run(probe, group, profile, remaining_seconds, cancelled) matches
    tuning_calibration.ProbeSupervisor. ``probe`` is a pickleable callable taking
    (group, a per-worker profile with concurrency=1, remaining_seconds), returning
    ProbeWork or its field mapping. Default production work is run_training_probe.
    group carries ``prepared`` and, for CUDA, explicit ``visible_gpus`` tokens.
    Peak RAM is the observed concurrent sum of worker/owned-child RSS; per-worker
    peaks and observer availability are recorded separately in numeric stages.
    """

    def __init__(
        self,
        *,
        work_root=None,
        poll_seconds=0.02,
        shutdown_reserve=0.25,
        keep_artifacts=False,
        measurement_factory=None,
        monitor_factory=None,
        on_poll=None,
    ):
        if not math.isfinite(poll_seconds) or poll_seconds <= 0:
            raise ValueError("probe polling interval must be positive and finite")
        if not math.isfinite(shutdown_reserve) or shutdown_reserve <= 0:
            raise ValueError("probe shutdown reserve must be positive and finite")
        self.work_root = Path(work_root) if work_root else None
        self.poll_seconds, self.shutdown_reserve = poll_seconds, shutdown_reserve
        self.keep_artifacts = keep_artifacts
        self.measurement_factory, self.monitor_factory = (
            measurement_factory,
            monitor_factory,
        )
        self.last_pids = ()
        self.on_poll = on_poll

    def _measurement(self, profile, **values):
        factory = self.measurement_factory
        if factory is None:
            from smartsom.experiments.tuning_calibration import CandidateMeasurement

            factory = CandidateMeasurement
        return factory(profile=profile, **values)

    def run(self, probe, group, profile, remaining_seconds, cancelled=lambda: False):
        started = time.monotonic()
        self.last_pids = ()
        if not math.isfinite(remaining_seconds) or remaining_seconds <= 0:
            raise ValueError("remaining probe deadline must be positive and finite")
        deadline = started + remaining_seconds
        reserve = min(self.shutdown_reserve, remaining_seconds / 4)
        work_deadline = deadline - reserve
        workers, observer, logs, directory = [], None, [], None
        reason, stats, results, termination = None, {}, [], None
        overhead = 0
        try:
            if cancelled():
                reason = "calibration cancelled"
                return self._measurement(
                    profile,
                    throughput=0.0,
                    peak_memory=0,
                    valid=False,
                    reason=reason,
                    termination="cancelled",
                )
            if (
                type(profile.threads) is not int
                or profile.threads < 1
                or type(profile.concurrency) is not int
                or profile.concurrency < 1
            ):
                raise ValueError(
                    "probe threads and concurrency must be positive integers"
                )
            overhead = sampling_cpu_overhead(group, profile)
            token = _gpu_token(group, profile)
            if self.work_root:
                self.work_root.mkdir(parents=True, exist_ok=True)
            directory = Path(
                tempfile.mkdtemp(prefix="smartsom-probe-", dir=self.work_root)
            )
            blob = directory / "probe.pkl"
            # This is our own trusted callback/input file, never a downloaded pickle.
            blob.write_bytes(pickle.dumps({"probe": probe, "group": group}, protocol=5))
            monitor_blob = directory / "monitor.pkl"
            monitor_blob.write_bytes(pickle.dumps(self.monitor_factory, protocol=5))
            environment = dict(os.environ)
            environment["PYTHONPATH"] = os.pathsep.join(str(p) for p in sys.path if p)
            environment["CUDA_VISIBLE_DEVICES"] = token
            for index in range(profile.concurrency):
                cancelled_now = cancelled()
                if time.monotonic() >= work_deadline or cancelled_now:
                    termination = "cancelled" if cancelled_now else "deadline"
                    reason = (
                        "calibration cancelled"
                        if cancelled_now
                        else "probe deadline expired"
                    )
                    break
                target = directory / f"worker-{index}"
                target.mkdir()
                _json(
                    target / "request.json",
                    {
                        "blob": str(blob),
                        "deadline": work_deadline,
                        "profile": {
                            "threads": profile.threads,
                            "concurrency": 1,
                            "device": profile.device,
                            "num_envs": getattr(profile, "num_envs", 1),
                            "sampling_processes": getattr(
                                profile, "sampling_processes", 0
                            ),
                        },
                    },
                )
                stream = (target / "worker.log").open("wb")
                logs.append(stream)
                workers.append(
                    subprocess.Popen(
                        [sys.executable, "-m", __name__, "worker", str(target)],
                        env=environment,
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                        start_new_session=os.name == "posix",
                    )
                )
            self.last_pids = tuple(p.pid for p in workers)
            if len(workers) != profile.concurrency:
                reason = reason or "probe did not start its full concurrency"
            if workers and reason is None:
                _json(
                    directory / "observer-request.json",
                    {
                        "roots": list(self.last_pids),
                        "deadline": work_deadline,
                        "poll": self.poll_seconds,
                        "gpu": profile.device != "cpu",
                        "monitor_blob": str(monitor_blob),
                    },
                )
                stream = (directory / "observer.log").open("wb")
                logs.append(stream)
                observer = subprocess.Popen(
                    [sys.executable, "-m", __name__, "observe", str(directory)],
                    env=environment,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    start_new_session=os.name == "posix",
                )
                next_notification = 0.0
                while any(p.poll() is None for p in workers):
                    stats = _read(directory / "observer.json") or stats
                    now = time.monotonic()
                    if self.on_poll and now >= next_notification:
                        self.on_poll(
                            {
                                "elapsed_seconds": now - started,
                                "profile": profile,
                                "group": copy.deepcopy(group),
                                "stage": "measuring",
                                "remaining_seconds": max(0.0, deadline - now),
                                "peak_memory": int(stats.get("peak_memory", 0)),
                                "pids": self.last_pids,
                                "phase": next(
                                    (
                                        phase["phase"]
                                        for i in range(len(workers))
                                        if (
                                            phase := _read(
                                                directory / f"worker-{i}" / "phase.json"
                                            )
                                        )
                                        and phase.get("phase")
                                        not in {"completed", "finish"}
                                    ),
                                    "starting",
                                ),
                            }
                        )
                        next_notification = now + 1.0
                    if cancelled():
                        reason = "calibration cancelled"
                        termination = "cancelled"
                        break
                    remaining = work_deadline - time.monotonic()
                    if remaining <= 0:
                        reason = "probe deadline expired"
                        termination = "deadline"
                        break
                    time.sleep(min(self.poll_seconds, remaining))
                for index, process in enumerate(workers):
                    result = _read(directory / f"worker-{index}" / "result.json")
                    if not result or process.poll() not in (None, 0):
                        reason = (
                            reason
                            or result.get("reason")
                            or "probe worker failed or did not finish"
                        )
                    results.append(result)
                stats = _read(directory / "observer.json") or stats
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            termination = None
        finally:
            if directory:
                _json(directory / "observer-stop.json", {"stop": True})
            try:
                self._reap(
                    [*workers, *([observer] if observer else [])], stats, deadline
                )
            except Exception as exc:
                cleanup_reason = (
                    f"owned process cleanup failed: {type(exc).__name__}: {exc}"
                )
                reason = f"{reason}; {cleanup_reason}" if reason else cleanup_reason
                termination = None
            for stream in logs:
                stream.close()
            if directory:
                stats = _read(directory / "observer.json") or stats
                if not self.keep_artifacts:
                    shutil.rmtree(directory)
        elapsed = time.monotonic() - started
        stages = {
            "cpu_request": float((profile.threads + overhead) * profile.concurrency),
            "sampling_cpu_overhead": float(overhead),
            "physical_ticks": float(sum(r.get("physical_ticks", 0) for r in results)),
            "updates": float(sum(r.get("updates", 0) for r in results)),
            "resource_samples": float(stats.get("samples", 0)),
            "per_trial_peak_memory": float(
                max(stats.get("per_trial_peaks", [0]), default=0)
            ),
            "gpu_utilization_peak": float(stats.get("gpu_utilization_peak", 0.0)),
        }
        for result in results:
            for name, value in result.get("stages", {}).items():
                stages[name] = max(stages.get(name, 0.0), float(value))
        if results and all("started" in r and "ended" in r for r in results):
            stages["overlap_seconds"] = max(
                0.0,
                min(r["ended"] for r in results) - max(r["started"] for r in results),
            )
        failures = [
            r.get("reason") or "invalid worker measurement"
            for r in results
            if not r.get("valid", False)
        ]
        # A completed worker error is not a budget exhaustion, even if another
        # worker later reaches the watchdog deadline. Missing results at that
        # deadline are expected and cannot invalidate a previous full baseline.
        completed_failure = next(
            (r for r in results if r and not r.get("valid", False)), None
        )
        if termination == "deadline" and completed_failure:
            termination = None
            reason = completed_failure.get("reason") or "invalid worker measurement"
        reason = reason or (failures[0] if failures else None)
        if (
            not stats.get("samples")
            or not stats.get("peak_memory")
            or stats.get("error")
        ):
            reason = (
                reason or stats.get("error") or "owned process RSS was not observed"
            )
            if stats.get("error"):
                termination = None
        gpu_peak = max(
            int(stats.get("peak_gpu_memory", 0)),
            max((int(r.get("peak_gpu_memory", 0)) for r in results), default=0),
        )
        if profile.device != "cpu" and not stats.get("gpu_metrics_available"):
            reason = reason or "owned GPU memory observation is unavailable"
        if elapsed > remaining_seconds + 0.05:
            reason = reason or "owned process cleanup exceeded the hard probe deadline"
            termination = None
        return self._measurement(
            profile,
            throughput=stages["physical_ticks"] / elapsed if elapsed else 0.0,
            peak_memory=int(stats.get("peak_memory", 0)),
            stages=stages,
            valid=reason is None,
            reason=reason,
            peak_gpu_memory=gpu_peak,
            elapsed_seconds=elapsed,
            termination=termination,
        )

    @staticmethod
    def _reap(processes, stats, deadline):
        """Signal owned sessions only; join handles and kill before the deadline."""

        def signal_session(process, signum, *, force=False):
            try:
                if os.name == "posix":
                    try:
                        os.killpg(process.pid, signum)
                    except PermissionError:
                        # Some macOS process-group views reject an empty/stopping
                        # session. The direct child handle remains authoritative.
                        if process.poll() is None:
                            process.send_signal(signum)
                elif process.poll() is None:
                    (process.kill if force else process.terminate)()
            except ProcessLookupError:
                pass

        for process in processes:
            signal_session(process, signal.SIGTERM)
        grace_deadline = min(deadline, time.monotonic() + 0.05)
        while any(process.poll() is None for process in processes):
            remaining = grace_deadline - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(0.005, remaining))
        for process in processes:
            signal_session(
                process, getattr(signal, "SIGKILL", signal.SIGTERM), force=True
            )
        # Detached descendants observed inside an owned tree are identity-checked.
        # Give parents their grace interval to reap their own children first.
        try:
            import psutil

            for member in stats.get("known_processes", ()):
                try:
                    process = psutil.Process(member["pid"])
                    if process.create_time() == member["created"]:
                        process.kill()
                except (psutil.Error, OSError):
                    pass
        except ImportError:
            pass
        for process in processes:
            process.wait(timeout=max(0.001, deadline - time.monotonic()))


def _worker(directory):
    directory = Path(directory)
    request = _read(directory / "request.json")
    started = time.monotonic()
    os.environ["SMARTSOM_PROBE_DIRECTORY"] = str(directory)
    if os.name == "posix":

        def interrupted(signum, frame):
            raise KeyboardInterrupt("calibration probe stopped")

        signal.signal(signal.SIGTERM, interrupted)
    try:
        with Path(request["blob"]).open("rb") as stream:
            data = pickle.load(stream)
        profile = _WorkerProfile(**request["profile"])
        result = data["probe"](
            data["group"], profile, max(0.0, request["deadline"] - time.monotonic())
        )
        result = asdict(result) if is_dataclass(result) else dict(result)
        result.update(started=started, ended=time.monotonic(), pid=os.getpid())
        _json(directory / "result.json", result)
    except BaseException as exc:
        _json(
            directory / "result.json",
            {
                "valid": False,
                "reason": f"{type(exc).__name__}: {exc}",
                "pid": os.getpid(),
            },
        )
        raise


def _observe(directory):
    directory = Path(directory)
    request = _read(directory / "observer-request.json")
    stats = {
        "samples": 0,
        "peak_memory": 0,
        "per_trial_peaks": [0] * len(request["roots"]),
        "known_processes": [],
        "peak_gpu_memory": 0,
    }
    try:
        import psutil

        with Path(request["monitor_blob"]).open("rb") as stream:
            monitor_factory = pickle.load(stream)
        if monitor_factory is None and request["gpu"]:
            from smartsom.experiments.tuning_resources import ResourceMonitor

            monitor_factory = ResourceMonitor
        monitor = monitor_factory() if monitor_factory else None
        known = {}
        while time.monotonic() < request["deadline"]:
            trial_rss, owned = [], {}
            for root in request["roots"]:
                members = dict(known.get(root, {}))
                try:
                    parent = psutil.Process(root)
                    created = parent.create_time()
                    if root not in members or members[root] == created:
                        for process in [parent, *parent.children(recursive=True)]:
                            members[process.pid] = process.create_time()
                except psutil.NoSuchProcess:
                    pass
                total = 0
                for pid, created in members.items():
                    try:
                        process = psutil.Process(pid)
                        if process.create_time() == created:
                            total += process.memory_info().rss
                            owned[pid] = created
                    except psutil.NoSuchProcess:
                        pass
                known[root] = members
                trial_rss.append(total)
            stats["samples"] += 1
            stats["peak_memory"] = max(stats["peak_memory"], sum(trial_rss))
            stats["per_trial_peaks"] = [
                max(old, new)
                for old, new in zip(stats["per_trial_peaks"], trial_rss, strict=True)
            ]
            stats["known_processes"] = [
                {"pid": pid, "created": created}
                for members in known.values()
                for pid, created in members.items()
            ]
            if request["gpu"]:
                timeout = max(0.01, min(0.2, request["deadline"] - time.monotonic()))
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-compute-apps=pid,used_memory",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    check=True,
                )
                values = list(
                    csv.reader(result.stdout.splitlines(), skipinitialspace=True)
                )
                gpu_memory = sum(
                    int(float(memory) * 1024**2)
                    for pid, memory in values
                    if int(pid) in owned
                )
                stats["gpu_metrics_available"] = True
                stats["peak_gpu_memory"] = max(stats["peak_gpu_memory"], gpu_memory)
                if monitor:
                    snapshot = monitor.snapshot(exclude_pids=tuple(owned))
                    stats["gpu_utilization_peak"] = max(
                        stats.get("gpu_utilization_peak", 0.0),
                        max((g.utilization or 0.0 for g in snapshot.gpus), default=0.0),
                    )
            _json(directory / "observer.json", stats)
            if (directory / "observer-stop.json").exists():
                break
            time.sleep(request["poll"])
    except BaseException as exc:
        stats["error"] = f"{type(exc).__name__}: {exc}"
        _json(directory / "observer.json", stats)


def run_training_probe(group, profile, remaining_seconds):
    """One real update, its checkpoint, and the original validation cases only."""
    started = time.monotonic()
    from smartsom.config.experiment_v3 import PreparedComposition
    from smartsom.experiments.composable import (
        TrainingSession,
        archive_inputs,
        evaluate_cases,
        evaluation_recipe,
        implementation_identity,
    )
    from smartsom.experiments.evidence import source_identity
    from smartsom.experiments.tuning_ray import _set_threads
    from smartsom.telemetry.runtime import CURRENT, DisplayOptions, RuntimeDisplay

    original = PreparedComposition(**_payload(group))
    config = json.loads(original.config_json)
    runtime = config.setdefault("runtime", {})
    original_threads = runtime.get("numerical_threads", 1)
    child_options = {}
    selected_processes = getattr(
        profile, "sampling_processes", runtime.get("sampling_processes", 0)
    )
    if selected_processes:
        if (
            "sampling_numerical_threads"
            in inspect.signature(TrainingSession).parameters
        ):
            child_options["sampling_numerical_threads"] = (
                1 if hasattr(profile, "sampling_processes") else original_threads
            )
        elif profile.threads != original_threads:
            raise ValueError(
                "sampling child thread initialization is not supported by this source"
            )
    runtime["numerical_threads"] = profile.threads
    runtime["num_envs"] = getattr(profile, "num_envs", runtime.get("num_envs", 1))
    runtime["sampling_processes"] = getattr(
        profile, "sampling_processes", runtime.get("sampling_processes", 0)
    )
    prepared = replace(original, config_json=json.dumps(config, sort_keys=True))
    root = Path(os.environ["SMARTSOM_PROBE_DIRECTORY"]) / "native"
    root.mkdir()
    for folder in (
        "config",
        "checkpoints",
        "evidence",
        "evaluation",
        "logs",
        "reports",
    ):
        (root / folder).mkdir()
    prepared = archive_inputs(root, prepared)
    record = {
        "schema": "smartsom.calibration-probe/v1",
        "kind": "calibration",
        "status": "running",
        "scientific_sha256": original.scientific_sha256,
        "source": source_identity(),
        "implementation_sha256": implementation_identity(),
        "physical_ticks": 0,
        "updates": 0,
        "formal_evidence": False,
    }
    _json(root / "config/original-prepared.json", asdict(original))
    display = RuntimeDisplay(
        DisplayOptions(progress="off", verbose=False), kind="training", quiet=True
    )
    display.bind(root)
    token = CURRENT.set(display)
    limits, post_limits, session = None, None, None
    stages, stage, previous = {}, "cold_start", started

    def phase(name):
        nonlocal stage, previous
        now = time.monotonic()
        stages[stage] = stages.get(stage, 0.0) + now - previous
        stage, previous = name, now
        _json(
            root.parent / "phase.json", {"phase": name, "pid": os.getpid(), "at": now}
        )

    try:
        limits = _set_threads(profile.threads)
        if profile.device != "cpu":
            import torch

            torch.cuda.reset_peak_memory_stats()
        session = TrainingSession(prepared, root, record, **child_options)
        post_limits = _set_threads(profile.threads)
        report = session.report_progress

        def progress(name):
            phase(name)
            report(name)

        session.report_progress = progress
        # step_update owns the native initial-save contract as well as the full
        # post-validation boundary; do not add a second cold checkpoint here.
        phase(
            "initial_checkpoint"
            if session.settings.record_initial
            else "training_update"
        )
        session.step_update()
        checkpoint = root / "checkpoints" / f"update-{session.updates:06d}"
        phase("saving")
        if not (checkpoint / "continuation.pkl").is_file():
            checkpoint = session.save()
        cases = json.loads(prepared.validation_json)
        logged = root / "logs" / f"validation-{session.updates:06d}.json"
        if cases:
            phase("validation")
            rows = (
                _read(logged)
                if logged.is_file()
                else evaluate_cases(
                    evaluation_recipe(prepared, checkpoint), cases, validation=True
                )
            )
            if any(row.get("engineering_failure") for row in rows):
                raise RuntimeError(
                    "original validation cases have engineering failures"
                )
        if session.training_done:
            phase("finish")
            session.finish()
        else:
            record.update(status="interrupted", calibration_status="measured")
            _json(root / "run.json", record)
        if session.updates < 1 or not (checkpoint / "continuation.pkl").is_file():
            raise RuntimeError("probe did not produce a complete update and checkpoint")
        gpu_peak = 0
        if profile.device != "cpu":
            gpu_peak = int(torch.cuda.max_memory_reserved())
        ticks, updates = session.ticks, session.updates
        phase("closing")
        session.close()
        session = None
        phase("completed")
        stages["validation_cases"] = float(len(cases))
        return ProbeWork(ticks, updates, stages, peak_gpu_memory=gpu_peak)
    finally:
        if session is not None:
            session.close()
        if post_limits is not None:
            post_limits.restore_original_limits()
        if limits is not None:
            limits.restore_original_limits()
        CURRENT.reset(token)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("worker", "observe"))
    parser.add_argument("directory", type=Path)
    args = parser.parse_args(argv)
    (_worker if args.mode == "worker" else _observe)(args.directory)


if __name__ == "__main__":
    main()
