"""Reader sharing may delay publication, never deadlines or owned cleanup."""

import json
import os
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from smartsom import _filesystem
from smartsom.experiments import tuning_probe


class Clock:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


def sharing_error(code=5):
    error = PermissionError(13, "reader holds committed file")
    error.winerror = code
    return error


def publication_setup(monkeypatch, tmp_path, *, platform="nt", release=None, code=5):
    clock = Clock()
    target = tmp_path / "observer.json"
    target.write_text('{"old": true}')
    replace = os.replace
    attempts = []

    def blocked(source, destination):
        attempts.append(clock.now)
        assert target.read_text() == '{"old": true}'
        assert json.loads(Path(source).read_text()) == {"new": True}
        if release is None or clock.now < release:
            raise sharing_error(code)
        replace(source, destination)

    injected_os = SimpleNamespace(**vars(os))
    injected_os.name, injected_os.replace = platform, blocked
    monkeypatch.setattr(_filesystem, "os", injected_os)
    monkeypatch.setattr(_filesystem, "time", clock)
    monkeypatch.setattr(tuning_probe, "time", clock)
    return clock, target, attempts


@pytest.mark.parametrize("code", [5, 32, 33])
def test_reader_release_publishes_same_closed_temporary_file(
    monkeypatch, tmp_path, code
):
    clock, target, attempts = publication_setup(
        monkeypatch, tmp_path, release=0.03, code=code
    )
    tuning_probe._json(target, {"new": True})
    assert len(attempts) > 1 and clock.now <= 0.1
    assert json.loads(target.read_text()) == {"new": True}
    assert not target.with_suffix(".json.tmp").exists()


@pytest.mark.parametrize("deadline", [None, 0.025, 0.0])
def test_persistent_sharing_fails_at_short_cap_or_remaining_deadline(
    monkeypatch, tmp_path, deadline
):
    clock, target, _ = publication_setup(monkeypatch, tmp_path)
    with pytest.raises(PermissionError):
        tuning_probe._json(target, {"new": True}, deadline=deadline)
    assert clock.now <= (0.1 if deadline is None else deadline)
    assert target.read_text() == '{"old": true}'


@pytest.mark.parametrize("platform,code", [("posix", 5), ("nt", 87)])
def test_non_windows_or_unrelated_denial_is_immediate(
    monkeypatch, tmp_path, platform, code
):
    clock, target, attempts = publication_setup(
        monkeypatch, tmp_path, platform=platform, code=code
    )
    with pytest.raises(PermissionError):
        tuning_probe._json(target, {"new": True})
    assert len(attempts) == 1 and not clock.sleeps


@pytest.mark.skipif(os.name != "nt", reason="native Windows file sharing")
def test_native_windows_reader_release_allows_publication(tmp_path):
    target = tmp_path / "observer.json"
    target.write_text('{"old": true}')
    started = threading.Event()
    done = threading.Event()
    errors = []

    def publish():
        started.set()
        try:
            tuning_probe._json(target, {"new": True})
        except Exception as error:
            errors.append(error)
        finally:
            done.set()

    thread = threading.Thread(target=publish)
    try:
        with target.open() as reader:
            thread.start()
            assert started.wait(1)
            assert not done.wait(0.02)
            assert json.loads(reader.read()) == {"old": True}
    finally:
        if thread.ident is not None:
            thread.join(1)
    assert not thread.is_alive() and done.is_set() and not errors
    assert json.loads(target.read_text()) == {"new": True}


@pytest.mark.skipif(os.name != "nt", reason="native Windows file sharing")
def test_native_windows_held_reader_fails_within_remaining_deadline(tmp_path):
    target = tmp_path / "observer.json"
    target.write_text('{"old": true}')
    with target.open() as reader:
        started = time.monotonic()
        with pytest.raises(PermissionError):
            tuning_probe._json(target, {"new": True}, deadline=started + 0.04)
        assert time.monotonic() - started < 0.2
        assert json.loads(reader.read()) == {"old": True}


class ObserverFailsAfterGoodStats:
    """Inject a real observer's late publication failure, not a fake result."""

    def __init__(self):
        main = sys.modules["__main__"]
        original = main._json
        calls = 0

        def publish(path, value, **kwargs):
            nonlocal calls
            if Path(path).name == "observer.json":
                calls += 1
                if calls > 1:
                    raise sharing_error()
            return original(path, value, **kwargs)

        main._json = publish


class ObserverFailsDuringStop:
    def __init__(self):
        main = sys.modules["__main__"]
        original = main._json

        def publish(path, value, **kwargs):
            if (Path(path).parent / "fail-observer").exists():
                raise sharing_error()
            return original(path, value, **kwargs)

        main._json = publish


@pytest.mark.parametrize("boundary", ["stop", "reap"])
def test_observer_failure_during_stop_publication_invalidates_good_stats(
    tmp_path, monkeypatch, boundary
):
    pytest.importorskip("psutil")
    from test_tuning_probe import (
        Profile,
        assert_stopped,
        group,
        memory_work,
        supervisor,
    )

    processes = []
    original_spawn = tuning_probe.subprocess.Popen
    original_publish = tuning_probe._json

    def spawn(*args, **kwargs):
        process = original_spawn(*args, **kwargs)
        processes.append(process)
        return process

    def fail_observer():
        observer = processes[-1]
        assert observer.poll() is None
        directory = Path(observer.args[-1])
        (directory / "fail-observer").touch()
        assert observer.wait(timeout=1) != 0

    def publish(path, value, **kwargs):
        if Path(path).name == "observer-stop.json":
            if boundary == "stop":
                fail_observer()
            else:
                # Hold STOP visibility so the observer remains alive until the
                # barrier immediately before the actual cleanup signal check.
                return
        return original_publish(path, value, **kwargs)

    monkeypatch.setattr(tuning_probe.subprocess, "Popen", spawn)
    monkeypatch.setattr(tuning_probe, "_json", publish)
    runner = supervisor(tmp_path, monitor_factory=ObserverFailsDuringStop)
    original_reap = runner._reap

    def reap(*args):
        if boundary == "reap":
            fail_observer()
        return original_reap(*args)

    monkeypatch.setattr(runner, "_reap", reap)
    result = runner.run(memory_work, group(sampling_processes=0), Profile(), 4)
    stats = json.loads(
        next(tmp_path.glob("smartsom-probe-*/observer.json")).read_text()
    )
    assert stats["samples"] > 0 and stats["peak_memory"] > 0
    assert "error" not in stats
    assert not result.valid and "observer exited with error" in result.reason
    assert result.termination is None
    assert_stopped([process.pid for process in processes])


def test_windows_intentional_termination_code_one_is_not_natural_failure(monkeypatch):
    class Process:
        pid = 999999
        code = None

        def poll(self):
            return self.code

        def terminate(self):
            self.code = 1

        kill = terminate

        def wait(self, timeout):
            return self.code

    monkeypatch.setattr(tuning_probe, "os", SimpleNamespace(name="nt"))
    process = Process()
    natural_exits = tuning_probe.ProbeSupervisor._reap(
        [process], {}, time.monotonic() + 1
    )
    assert process.code == 1 and natural_exits == {}


def test_late_observer_failure_cannot_reuse_good_stats_as_success(
    tmp_path, monkeypatch
):
    pytest.importorskip("psutil")
    from test_tuning_probe import (
        Profile,
        assert_stopped,
        group,
        memory_work,
        supervisor,
    )

    runner = supervisor(tmp_path, monitor_factory=ObserverFailsAfterGoodStats)
    reaped = []
    original = runner._reap

    def reap(processes, stats, deadline):
        reaped.extend(p.pid for p in processes)
        return original(processes, stats, deadline)

    monkeypatch.setattr(runner, "_reap", reap)
    result = runner.run(memory_work, group(sampling_processes=0), Profile(), 4)
    stats = json.loads(
        next(tmp_path.glob("smartsom-probe-*/observer.json")).read_text()
    )
    assert stats["samples"] >= 1 and stats["peak_memory"] > 0
    assert "error" not in stats
    assert not result.valid and "observer exited with error" in result.reason
    assert result.elapsed_seconds <= 4.05
    assert_stopped(reaped)


def test_stop_publication_failure_still_reaps_and_preserves_both_errors(
    tmp_path, monkeypatch
):
    pytest.importorskip("psutil")
    from test_tuning_probe import (
        Profile,
        assert_stopped,
        group,
        memory_work,
        supervisor,
    )

    original_publish = tuning_probe._json

    def publish(path, value, **kwargs):
        if Path(path).name == "observer-stop.json":
            raise sharing_error()
        return original_publish(path, value, **kwargs)

    def display_failure(value):
        raise RuntimeError("primary display failure")

    runner = supervisor(tmp_path, on_poll=display_failure)
    original_reap = runner._reap
    reaped = []

    def reap(processes, stats, deadline):
        reaped.extend(p.pid for p in processes)
        original_reap(processes, stats, deadline)
        raise RuntimeError("cleanup failure marker")

    monkeypatch.setattr(tuning_probe, "_json", publish)
    monkeypatch.setattr(runner, "_reap", reap)
    result = runner.run(memory_work, group(sampling_processes=0), Profile(), 0.8)
    assert not result.valid and result.termination is None
    assert "primary display failure" in result.reason
    assert "observer stop publication failed" in result.reason
    assert "cleanup failure marker" in result.reason
    assert result.elapsed_seconds <= 0.85
    assert reaped
    assert_stopped(reaped)
