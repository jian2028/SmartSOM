"""Owned-process calibration watchdog and frozen native-work contracts."""

import copy
import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import pytest

from smartsom.experiments.tuning_probe import (
    ProbeSupervisor,
    ProbeWork,
    sampling_cpu_overhead,
)


@dataclass(frozen=True)
class Profile:
    threads: int = 2
    concurrency: int = 1
    device: str = "cpu"


@dataclass(frozen=True)
class Measurement:
    profile: Profile
    throughput: float
    peak_memory: int
    stages: dict | None = None
    valid: bool = True
    reason: str | None = None
    peak_gpu_memory: int = 0
    elapsed_seconds: float = 0.0
    termination: str | None = None


def group(*, device="cpu", sampling_processes=1):
    return {
        "prepared": {
            "config_json": json.dumps(
                {
                    "runtime": {
                        "device": device,
                        "num_envs": 4,
                        "sampling_processes": sampling_processes,
                        "numerical_threads": 3,
                    },
                    "training": {"seed": 91, "algorithm": "ppo", "budget": 10000},
                    "world": {"factory": "unchanged"},
                }
            )
        }
    }


def memory_work(group, profile, remaining_seconds):
    assert profile.concurrency == 1
    assert os.environ["CUDA_VISIBLE_DEVICES"] == ""
    allocation = bytearray(10 * 1024**2)
    allocation[0] = 1
    time.sleep(0.4)
    return ProbeWork(120, 1, {"validation": 0.02, "saving": 0.01})


def work_with_child(group, profile, remaining_seconds):
    child = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import time; data=bytearray(16*1024**2); time.sleep(30)",
        ]
    )
    Path(group["child_marker"]).write_text(str(child.pid))
    try:
        time.sleep(30)
        return ProbeWork(0, 0, {})
    finally:
        child.terminate()
        child.wait(timeout=0.1)


def invalid_work(group, profile, remaining_seconds):
    return ProbeWork(120, 1, {}, valid=False, reason="frozen validation failed")


def mixed_failure_work(group, profile, remaining_seconds):
    if Path(os.environ["SMARTSOM_PROBE_DIRECTORY"]).name == "worker-0":
        return ProbeWork(0, 0, {}, valid=False, reason="native worker error")
    time.sleep(30)
    return ProbeWork(120, 1, {})


def supervisor(tmp_path, **kwargs):
    return ProbeSupervisor(
        work_root=tmp_path,
        keep_artifacts=True,
        measurement_factory=Measurement,
        **kwargs,
    )


def assert_stopped(pids):
    psutil = pytest.importorskip("psutil")
    for pid in pids:
        if psutil.pid_exists(pid):
            process = psutil.Process(pid)
            pytest.fail(f"owned process {pid} was not joined: {process.status()}")


def test_actual_parallel_work_and_concurrent_rss(tmp_path):
    pytest.importorskip("psutil")
    frozen = group()
    original = copy.deepcopy(frozen)
    polls = []

    def poll(value):
        polls.append(value)
        value["group"]["prepared"]["config_json"] = "display cannot mutate input"

    runner = supervisor(tmp_path, on_poll=poll)
    result = runner.run(memory_work, frozen, Profile(concurrency=2), 4.0)
    assert result.valid, result.reason
    assert result.termination is None
    assert result.stages["physical_ticks"] == 240
    assert result.stages["updates"] == 2
    assert result.stages["overlap_seconds"] > 0.2
    assert result.stages["resource_samples"] > 1
    assert result.peak_memory > result.stages["per_trial_peak_memory"] > 10 * 1024**2
    assert result.stages["cpu_request"] == 10  # (2 + original 1 * 3) * 2
    assert result.stages["validation"] == 0.02
    assert result.throughput == pytest.approx(240 / result.elapsed_seconds)
    assert frozen == original
    assert polls and all(value["stage"] == "measuring" for value in polls)
    assert polls[0]["profile"] == Profile(concurrency=2)
    assert polls[0]["pids"] == runner.last_pids
    assert_stopped(runner.last_pids)


def test_timeout_reaps_only_owned_processes_and_children(tmp_path):
    pytest.importorskip("psutil")
    external = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    frozen = group(sampling_processes=0)
    frozen["child_marker"] = str(tmp_path / "child.pid")
    runner = supervisor(tmp_path)
    try:
        result = runner.run(work_with_child, frozen, Profile(), 1.4)
        assert not result.valid
        assert result.termination == "deadline"
        assert "deadline" in result.reason
        assert "cleanup failed" not in result.reason
        assert result.elapsed_seconds <= 1.45
        assert external.poll() is None
        child = int(Path(frozen["child_marker"]).read_text())
        assert_stopped((*runner.last_pids, child))
        assert result.stages["per_trial_peak_memory"] > 16 * 1024**2
        stats_path = next(tmp_path.glob("smartsom-probe-*/observer.json"))
        stats = json.loads(stats_path.read_text())
        assert child in {member["pid"] for member in stats["known_processes"]}
    finally:
        external.terminate()
        external.wait(timeout=2)


def test_cancel_joins_processes_without_timeout_threads(tmp_path):
    pytest.importorskip("psutil")
    frozen = group(sampling_processes=0)
    frozen["child_marker"] = str(tmp_path / "child.pid")
    existing = {thread.ident for thread in threading.enumerate()}
    runner = supervisor(tmp_path)
    started = time.monotonic()
    result = runner.run(
        work_with_child,
        frozen,
        Profile(),
        4,
        cancelled=lambda: time.monotonic() - started > 0.6,
    )
    assert not result.valid
    assert result.termination == "cancelled"
    assert "cancelled" in result.reason
    assert "cleanup failed" not in result.reason
    assert result.elapsed_seconds < 1
    assert {thread.ident for thread in threading.enumerate()} == existing
    assert_stopped(runner.last_pids)
    if Path(frozen["child_marker"]).exists():
        assert_stopped([int(Path(frozen["child_marker"]).read_text())])


def test_invalid_native_work_cannot_rank_as_success(tmp_path):
    pytest.importorskip("psutil")
    result = supervisor(tmp_path).run(invalid_work, group(), Profile(), 4)
    assert not result.valid
    assert result.reason == "frozen validation failed"
    assert result.termination is None


def test_worker_failure_is_not_hidden_by_another_workers_deadline(tmp_path):
    pytest.importorskip("psutil")
    runner = supervisor(tmp_path)
    result = runner.run(
        mixed_failure_work, group(sampling_processes=0), Profile(concurrency=2), 1.4
    )
    assert not result.valid and result.termination is None
    assert result.reason == "native worker error"
    assert_stopped(runner.last_pids)


def test_cleanup_failure_does_not_claim_a_joined_deadline(tmp_path, monkeypatch):
    pytest.importorskip("psutil")
    runner = supervisor(tmp_path)
    reap = runner._reap

    def cleanup_error(processes, stats, deadline):
        reap(processes, stats, deadline)
        raise RuntimeError("cleanup ownership could not be confirmed")

    monkeypatch.setattr(runner, "_reap", cleanup_error)
    result = runner.run(memory_work, group(sampling_processes=0), Profile(), 0.3)
    assert not result.valid and result.termination is None
    assert "cleanup failed" in result.reason
    assert_stopped(runner.last_pids)


def test_display_callback_failure_still_reaps_its_workers(tmp_path):
    pytest.importorskip("psutil")

    def broken_display(value):
        raise RuntimeError("display unavailable")

    runner = supervisor(tmp_path, on_poll=broken_display)
    result = runner.run(memory_work, group(), Profile(), 4)
    assert not result.valid
    assert "display unavailable" in result.reason
    assert_stopped(runner.last_pids)


def test_cancel_before_start_and_finite_deadline(tmp_path):
    runner = supervisor(tmp_path)
    result = runner.run(memory_work, group(), Profile(), 4, cancelled=lambda: True)
    assert not result.valid
    assert result.reason == "calibration cancelled"
    assert result.termination == "cancelled"
    assert runner.last_pids == ()
    assert not list(tmp_path.iterdir())
    with pytest.raises(ValueError, match="deadline"):
        runner.run(memory_work, group(), Profile(), float("inf"))


@pytest.mark.parametrize(
    ("frozen", "profile", "message"),
    [
        (group(device="cuda"), Profile(), "frozen CUDA"),
        (group(), Profile(device="cuda"), "frozen CPU"),
        (group(device="cuda"), Profile(device="cuda"), "explicit visible GPU"),
        (
            group(device="cuda"),
            Profile(device="cuda", concurrency=2),
            "multi-GPU",
        ),
    ],
)
def test_invalid_device_mapping_never_starts_worker(tmp_path, frozen, profile, message):
    runner = supervisor(tmp_path)
    result = runner.run(memory_work, frozen, profile, 4)
    assert not result.valid
    assert message in result.reason
    assert runner.last_pids == ()


def test_fixed_sampling_children_are_reserved():
    frozen = group(sampling_processes=2)
    assert sampling_cpu_overhead(frozen) == 6
    frozen["cpu_overhead"] = 8
    assert sampling_cpu_overhead(frozen) == 8
    frozen["cpu_overhead"] = 5
    with pytest.raises(ValueError, match="omits"):
        sampling_cpu_overhead(frozen)
