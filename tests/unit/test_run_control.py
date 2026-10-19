"""Cancellation never confuses a reused PID or old request with a live run."""

import json
import os
import subprocess
import sys
import time

import pytest

from smartsom.experiments import control
from smartsom.experiments.evidence import write_json


def test_pid_reuse_is_not_owned():
    identity = {"pid": 123, "created": "old"}
    assert not control.alive(identity, {123: {"created": "new", "state": "R"}})
    assert not control.alive(identity, {123: {"created": "old", "state": "Z"}})


def test_legacy_run_is_refused(tmp_path):
    with pytest.raises(ValueError, match="legacy"):
        control.stop(tmp_path, force=True)


def test_previous_request_does_not_stop_a_new_generation(tmp_path):
    (tmp_path / "control").mkdir()
    write_json(tmp_path / "control/owner.json", {"id": "new", "status": "running"})
    write_json(tmp_path / "control/stop.json", {"id": "old"})
    assert not control.requested(tmp_path)


@pytest.mark.parametrize("timeout", [-1, float("inf"), float("nan")])
def test_bad_timeout_is_rejected(tmp_path, timeout):
    with pytest.raises(ValueError, match="timeout"):
        control.stop(tmp_path, timeout=timeout)


def launch(tmp_path, *, stubborn=False):
    script = """
import sys, time, subprocess
from pathlib import Path
from smartsom.telemetry.runtime import operation, bind
from smartsom.experiments.control import boundary
@operation("training")
def run():
    root = Path(sys.argv[1])
    bind(root)
    if sys.argv[2] == "stubborn":
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        (root / "child").write_text(str(child.pid))
        time.sleep(.5)
    (root / "ready").write_text("ready")
    while True:
        time.sleep(.05)
        if sys.argv[2] == "graceful":
            (root / "committed").write_text("saved boundary")
            boundary(root)
run()
"""
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            script,
            str(tmp_path),
            "stubborn" if stubborn else "graceful",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        env={**os.environ, "PYTHONPATH": "src"},
    )
    deadline = time.monotonic() + 10
    while not (tmp_path / "ready").exists():
        if process.poll() is not None or time.monotonic() > deadline:
            process.kill()
            raise AssertionError(process.communicate()[1].decode())
        time.sleep(0.05)
    return process


def test_real_cooperative_stop_and_repeat(tmp_path):
    process = launch(tmp_path)
    try:
        result = control.stop(tmp_path, timeout=5)
        assert result["status"] == "stopped"
        assert (tmp_path / "committed").read_text() == "saved boundary"
        assert control.stop(tmp_path)["remaining"] == []
    finally:
        if process.poll() is None:
            process.kill()
        process.communicate()


def test_timeout_does_not_kill_and_force_preserves_unrelated_process(tmp_path):
    process = launch(tmp_path, stubborn=True)
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        assert control.stop(tmp_path, timeout=0)["remaining"]
        assert process.poll() is None
        result = control.stop(tmp_path, timeout=0.1, force=True)
        assert result["status"] == "force_stopped"
        child_pid = int((tmp_path / "child").read_text())
        table = control.processes()
        assert child_pid not in table or table[child_pid]["state"].startswith("Z")
        assert unrelated.poll() is None
        assert json.loads((tmp_path / "control/owner.json").read_text())["forced"]
    finally:
        for child in (process, unrelated):
            if child.poll() is None:
                child.kill()
            child.communicate()


def test_monitor_resolves_finished_stop_without_second_stop_call(tmp_path, monkeypatch):
    from smartsom.telemetry.monitor import read_snapshot
    from smartsom.telemetry.runtime import DisplayOptions, RuntimeDisplay

    view = RuntimeDisplay(
        DisplayOptions(progress="off", verbose=False), kind="batch-directory"
    )
    view.bind(tmp_path)
    view.finish("stopped")
    (tmp_path / "control").mkdir()
    identity = {"pid": 101, "created": "birth"}
    write_json(
        tmp_path / "control/owner.json",
        {
            "schema": "smartsom.run-control/v1",
            "id": "batch",
            "status": "stopping",
            "owner": identity,
            "members": [],
        },
    )
    write_json(tmp_path / "control/stop.json", {"id": "batch"})
    monkeypatch.setattr(control, "processes", lambda: {})
    assert read_snapshot(tmp_path)["status"] == "stopped"
    progress = json.loads((tmp_path / "logs/progress.json").read_text())
    progress.update(status="running", stage="running")
    write_json(tmp_path / "logs/progress.json", progress)
    write_json(tmp_path / "batch.json", {"status": "failed"})
    assert read_snapshot(tmp_path)["status"] == "failed"
    monkeypatch.setattr(
        control,
        "processes",
        lambda: {101: {"pid": 101, "created": "birth", "state": "R"}},
    )
    assert read_snapshot(tmp_path)["status"] == "stopping"


def test_fresh_boundary_bypasses_negative_cache_without_changing_polling(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(control.time, "monotonic", lambda: 100.0)
    monkeypatch.setattr(control, "_REQUEST_CACHE", {})
    (tmp_path / "control").mkdir()
    write_json(tmp_path / "control/owner.json", {"id": "run", "status": "running"})
    assert control.requested(tmp_path) is False
    write_json(tmp_path / "control/stop.json", {"id": "run"})
    assert control.requested(tmp_path) is False
    with pytest.raises(control.StopRequested):
        control.boundary(tmp_path, fresh=True)
    assert control.requested(tmp_path) is True


def test_fresh_boundary_preserves_ancestor_and_generation_checks(tmp_path, monkeypatch):
    monkeypatch.setattr(control.time, "monotonic", lambda: 100.0)
    monkeypatch.setattr(control, "_REQUEST_CACHE", {})
    child = tmp_path / "worker"
    child.mkdir()
    (tmp_path / "control").mkdir()
    write_json(tmp_path / "control/owner.json", {"id": "old", "status": "running"})
    write_json(tmp_path / "control/stop.json", {"id": "old"})
    with pytest.raises(control.StopRequested):
        control.boundary(child, fresh=True)
    write_json(tmp_path / "control/owner.json", {"id": "new", "status": "running"})
    control.boundary(child, fresh=True)
    assert control.requested(child) is False


def test_fresh_boundary_ignores_stopped_owner_and_absent_request(tmp_path, monkeypatch):
    monkeypatch.setattr(control, "_REQUEST_CACHE", {})
    (tmp_path / "control").mkdir()
    write_json(tmp_path / "control/owner.json", {"id": "run", "status": "running"})
    control.boundary(tmp_path, fresh=True)
    write_json(tmp_path / "control/stop.json", {"id": "run"})
    write_json(tmp_path / "control/owner.json", {"id": "run", "status": "stopped"})
    control.boundary(tmp_path, fresh=True)


def test_finished_owner_waits_for_registered_child_exit_without_signals(
    tmp_path, monkeypatch
):
    script = r"""
import subprocess, sys
from pathlib import Path
from smartsom.experiments.control import Scope
root = Path(sys.argv[1])
scope = Scope()
scope.bind(root)
subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(1)'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
scope.capture()
scope.finish('completed')
(root / 'finished').touch()
"""
    process = subprocess.Popen(
        [sys.executable, "-c", script, str(tmp_path)],
        env={**os.environ, "PYTHONPATH": "src"},
    )
    try:
        deadline = time.monotonic() + 10
        while not (tmp_path / "finished").exists():
            assert process.poll() is None and time.monotonic() < deadline
            time.sleep(0.01)
        process.wait(timeout=5)
        owner = control.read(tmp_path)
        assert owner["finished_at"] and owner["members"]
        monkeypatch.setattr(
            control.os,
            "kill",
            lambda *_: pytest.fail("no signals after a finished scope"),
        )
        result = control.stop(tmp_path, timeout=5)
        assert result["status"] == "completed" and not result["remaining"]
        assert not (tmp_path / "control/stop.json").exists()
        assert not any(
            control.alive(item, control.processes()) for item in owner["members"]
        )
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=5)


def test_pause_request_is_generation_scoped_and_stop_has_priority(tmp_path):
    (tmp_path / "control").mkdir()
    write_json(tmp_path / "control/owner.json", {"id": "new", "status": "running"})
    write_json(tmp_path / "control/pause.json", {"id": "old"})
    control.boundary(tmp_path)
    write_json(tmp_path / "control/pause.json", {"id": "new"})
    assert not control.requested(tmp_path, fresh=True)
    with pytest.raises(control.PauseRequested):
        control.boundary(tmp_path)
    write_json(tmp_path / "control/stop.json", {"id": "new"})
    with pytest.raises(control.StopRequested) as raised:
        control.boundary(tmp_path, fresh=True)
    assert not isinstance(raised.value, control.PauseRequested)
    control.clear_pause(tmp_path)
    assert not control.pause_requested(tmp_path)


def test_real_pause_exits_worker_after_saved_boundary(tmp_path):
    process = launch(tmp_path)
    try:
        receipt = control.request_pause(tmp_path)
        assert receipt["status"] == "pause_requested"
        process.wait(timeout=10)
        assert (tmp_path / "committed").read_text() == "saved boundary"
        assert b"PauseRequested" in process.communicate()[1]
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
