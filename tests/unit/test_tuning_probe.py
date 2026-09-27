"""Owned-process calibration watchdog and frozen native-work contracts."""

import contextvars
import copy
import json
import os
import subprocess
import sys
import threading
import time
import types
from dataclasses import dataclass
from pathlib import Path

import pytest

from smartsom.experiments.tuning_probe import (
    ProbeSupervisor,
    ProbeWork,
    run_training_probe,
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


@pytest.mark.parametrize("validation_logged", [False, True])
def test_native_probe_preserves_science_and_never_uses_heldout(
    tmp_path, monkeypatch, validation_logged
):
    @dataclass(frozen=True)
    class Prepared:
        config_json: str
        validation_json: str
        scientific_sha256: str
        evaluation_json: str

    frozen = group(sampling_processes=2)
    frozen["prepared"].update(
        validation_json=json.dumps([{"case_id": "val", "seed": 99}]),
        scientific_sha256="immutable-scientific-identity",
        evaluation_json=json.dumps([{"case_id": "heldout", "seed": 101}]),
    )
    original = copy.deepcopy(frozen)
    events, sessions, limits = [], [], []

    class Native:
        def __init__(self, prepared, root, record, *, sampling_numerical_threads=None):
            self.prepared, self.root, self.record = prepared, root, record
            self.settings = types.SimpleNamespace(record_initial=True)
            self.sampling_threads = sampling_numerical_threads
            self.updates = self.ticks = 0
            self.training_done = False
            sessions.append(self)

        def report_progress(self, name):
            events.append(name)

        def step_update(self):
            if self.settings.record_initial:
                self.save()
            self.report_progress("sampling")
            self.updates, self.ticks = 1, 128
            self.save()
            if validation_logged:
                (self.root / "logs/validation-000001.json").write_text(
                    json.dumps([{"case_id": "val", "status": "completed"}])
                )

        def save(self):
            events.append(f"saved-{self.updates}")
            folder = self.root / f"checkpoints/update-{self.updates:06d}"
            folder.mkdir(exist_ok=True)
            (folder / "continuation.pkl").write_bytes(b"full-state")
            return folder

        def finish(self):
            pytest.fail("partial probe must not finish the original training budget")

        def close(self):
            events.append("closed")

    def evaluate(recipe, cases, *, validation):
        assert validation is True
        assert cases == [{"case_id": "val", "seed": 99}]
        assert recipe[1].name == "update-000001"
        events.append("original-validation")
        return [{"status": "completed"}]

    def set_threads(value):
        assert value == 2
        handle = types.SimpleNamespace(
            restore_original_limits=lambda: limits.append("restored")
        )
        limits.append(handle)
        return handle

    class Display:
        def __init__(self, *args, **kwargs):
            pass

        def bind(self, path):
            pass

    modules = {
        "smartsom.config.experiment_v3": types.SimpleNamespace(
            PreparedComposition=Prepared
        ),
        "smartsom.experiments.composable": types.SimpleNamespace(
            TrainingSession=Native,
            archive_inputs=lambda root, prepared: prepared,
            evaluate_cases=evaluate,
            evaluation_recipe=lambda prepared, checkpoint: (prepared, checkpoint),
            implementation_identity=lambda: "implementation-hash",
        ),
        "smartsom.experiments.evidence": types.SimpleNamespace(
            source_identity=lambda: {"commit": "source-hash"}
        ),
        "smartsom.experiments.tuning_ray": types.SimpleNamespace(
            _set_threads=set_threads
        ),
        "smartsom.telemetry.runtime": types.SimpleNamespace(
            CURRENT=contextvars.ContextVar("probe-test"),
            DisplayOptions=lambda **kwargs: kwargs,
            RuntimeDisplay=Display,
        ),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setenv("SMARTSOM_PROBE_DIRECTORY", str(tmp_path))
    result = run_training_probe(frozen, Profile(), 10)
    session = sessions[0]
    actual = json.loads(session.prepared.config_json)
    expected = json.loads(original["prepared"]["config_json"])
    expected["runtime"]["numerical_threads"] = 2
    assert actual == expected
    assert session.prepared.scientific_sha256 == "immutable-scientific-identity"
    assert session.sampling_threads == 3
    assert frozen == original
    assert result.physical_ticks == 128 and result.updates == 1
    assert result.stages["validation_cases"] == 1
    assert ("original-validation" in events) is not validation_logged
    assert events.count("closed") == 1
    assert events.count("saved-0") == 1
    assert limits.count("restored") == 2
    assert (tmp_path / "native/checkpoints/update-000001/continuation.pkl").is_file()
    saved = json.loads((tmp_path / "native/config/original-prepared.json").read_text())
    assert saved == original["prepared"]
