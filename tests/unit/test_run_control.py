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
