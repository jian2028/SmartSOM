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


def deletable_run(root):
    root.mkdir(parents=True, exist_ok=True)
    (root / "control").mkdir(exist_ok=True)
    write_json(
        root / "plan.json",
        {
            "schema": "smartsom.reservation-plan/v1",
            "runtime_contract": "smartsom.current-runtime/v1",
            "run_id": "delete-fixture",
        },
    )
    write_json(
        root / "control/owner.json",
        {
            "schema": "smartsom.run-control/v1",
            "id": "old",
            "root": str(root),
            "driver_root": str(root),
            "owner": {"pid": 99999999, "created": "old"},
            "members": [],
            "status": "completed",
            "finished_at": 1,
        },
    )
    (root / "last.pt").write_bytes(b"test checkpoint")
    return root


@pytest.mark.parametrize("force", [False, True])
def test_delete_after_real_exit_keeps_configs_and_external_link_target(tmp_path, force):
    config = tmp_path / "experiments/experiment.yaml"
    config.parent.mkdir()
    config.write_text("authoring stays")
    external = tmp_path / "external"
    external.mkdir()
    (external / "cloud-record.txt").write_text("external stays")
    root = deletable_run(tmp_path / "runs/trial")
    (root / "linked").symlink_to(external, target_is_directory=True)
    process = launch(root, stubborn=force)
    try:
        result = control.stop(
            root, timeout=0.1 if force else 10, force=force, delete=True
        )
        process.wait(timeout=10)
        assert result["status"] == "deleted"
        assert result["deleted"] and result["remaining"] == []
        assert not root.exists()
        assert config.read_text() == "authoring stays"
        assert (external / "cloud-record.txt").read_text() == "external stays"
        assert not list(root.parent.glob(".*.deleting-*"))
    finally:
        if process.poll() is None:
            process.kill()
        process.communicate()


def test_delete_timeout_retains_entire_bundle(tmp_path):
    root = deletable_run(tmp_path / "run")
    process = launch(root, stubborn=True)
    try:
        result = control.stop(root, timeout=0, delete=True)
        assert result["deleted"] is False
        assert result["remaining"]
        assert (root / "last.pt").read_bytes() == b"test checkpoint"
        assert (root / "plan.json").is_file()
    finally:
        control.stop(root, timeout=0, force=True)
        process.communicate(timeout=10)


def test_delete_cli_by_id_and_already_completed_run(tmp_path, monkeypatch, capsys):
    from smartsom.experiments.cli import main

    root = deletable_run(tmp_path / "run")
    monkeypatch.setattr(control, "processes", lambda: {})
    assert main(["stop", "delete-fixt", "--root", str(tmp_path), "--delete"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "deleted"
    assert not root.exists()


def test_delete_cli_timeout_is_nonzero_and_keeps_files(tmp_path, monkeypatch, capsys):
    from smartsom.experiments.cli import main

    root = deletable_run(tmp_path / "run")
    owner = control.read(root)
    owner.update(status="running")
    owner.pop("finished_at")
    write_json(root / "control/owner.json", owner)
    monkeypatch.setattr(
        control,
        "processes",
        lambda: {99999999: {"pid": 99999999, "created": "old", "state": "S"}},
    )
    assert main(["stop", str(root), "--delete", "--timeout", "0"]) == 1
    assert json.loads(capsys.readouterr().out)["deleted"] is False
    assert root.exists()


@pytest.mark.parametrize(
    "unsafe", ["child", "launcher", "ownership", "queue", "source", "symlink"]
)
def test_delete_refuses_unverified_or_protected_results(tmp_path, monkeypatch, unsafe):
    root = deletable_run(tmp_path / "run")
    table = {}
    target = root
    if unsafe == "child":
        child = deletable_run(root / "trials/child")
        owner = control.read(child)
        owner["driver_root"] = str(root)
        owner["owner"] = {"pid": 222, "created": "child"}
        write_json(child / "control/owner.json", owner)
        table[222] = {"pid": 222, "created": "child", "state": "S"}
    elif unsafe == "launcher":
        write_json(
            root / "control/background-launch.lock", {"pid": 222, "created": "launcher"}
        )
        table[222] = {"pid": 222, "created": "launcher", "state": "S"}
    elif unsafe == "ownership":
        owner = control.read(root)
        owner["ownership_error"] = "lost tracking"
        write_json(root / "control/owner.json", owner)
    elif unsafe == "queue":
        write_json(root / "control/admission.json", {"queue": str(tmp_path / "queue")})
    elif unsafe == "source":
        (root / "original.yaml").write_text("retain configuration")
        write_json(
            root / "authoring.json",
            {"source_files": {"experiment": {"path": str(root / "original.yaml")}}},
        )
    else:
        target = tmp_path / "alias"
        target.symlink_to(root, target_is_directory=True)
    monkeypatch.setattr(control, "processes", lambda: table)
    with pytest.raises(ValueError):
        control.stop(target, timeout=0, delete=True)
    assert (root / "last.pt").is_file()


def test_delete_rechecks_generation_after_stop(tmp_path, monkeypatch):
    root = deletable_run(tmp_path / "run")
    monkeypatch.setattr(control, "processes", lambda: {})
    stop = control._stop

    def rebound(*args, **kwargs):
        result = stop(*args, **kwargs)
        owner = control.read(root)
        owner["id"] = "new"
        write_json(root / "control/owner.json", owner)
        return result

    monkeypatch.setattr(control, "_stop", rebound)
    with pytest.raises(ValueError, match="identity changed"):
        control.stop(root, delete=True)
    assert (root / "last.pt").is_file()


def test_delete_and_registration_are_exclusive(tmp_path, monkeypatch):
    root = deletable_run(tmp_path / "run")
    monkeypatch.setattr(control, "processes", lambda: {})
    with control.registration_lock(root):
        with pytest.raises(ValueError, match="still active"):
            control.stop(root, delete=True)
    assert root.exists()
    remove = control.shutil.rmtree

    def during_remove(detached):
        assert not root.exists()
        with pytest.raises(FileNotFoundError):
            control.Scope().bind(root)
        assert not root.exists()
        remove(detached)

    monkeypatch.setattr(control.shutil, "rmtree", during_remove)
    assert control.stop(root, delete=True)["deleted"]


def test_delete_batch_and_block_unregistered_child_start(tmp_path, monkeypatch):
    root = deletable_run(tmp_path / "batch")
    child = deletable_run(root / "trials/child")
    owner = control.read(child)
    owner["driver_root"] = str(root)
    write_json(child / "control/owner.json", owner)
    monkeypatch.setattr(control, "processes", lambda: {})
    with pytest.raises(ValueError, match="shared driver"):
        control.stop(child, delete=True)
    with control.registration_lock(root):
        with pytest.raises(ValueError, match="still active"):
            with control.registration_lock(child):
                pytest.fail("child registration bypassed the parent lock")
    assert control.stop(root, delete=True)["deleted"]
    assert not root.exists()


def test_delete_refuses_nested_pending_launcher_without_owner(tmp_path, monkeypatch):
    root = deletable_run(tmp_path / "batch")
    child = root / "trials/pending"
    (child / "control").mkdir(parents=True)
    write_json(
        child / "control/background-launch.lock", {"pid": 222, "created": "pending"}
    )
    monkeypatch.setattr(
        control, "processes", lambda: {222: {"created": "pending", "state": "S"}}
    )
    with pytest.raises(ValueError, match="ownership"):
        control.stop(root, delete=True)
    assert root.exists()


@pytest.mark.skipif(os.name != "posix", reason="POSIX foreground terminal")
@pytest.mark.parametrize("choice", [b"1", b"2", b"d"])
def test_foreground_terminal_stop_and_delete_real_processes(tmp_path, choice):
    import pty
    import select
    import signal

    root = tmp_path / "output"
    root.mkdir()
    write_json(
        root / "plan.json",
        {
            "schema": "smartsom.reservation-plan/v1",
            "runtime_contract": "smartsom.current-runtime/v1",
        },
    )
    write_json(root / "batch.json", {"status": "allocated"})
    config = tmp_path / "experiment.yaml"
    config.write_text("authoring stays")
    driver = r"""
import sys, time
from pathlib import Path
from smartsom.experiments import background, control
from smartsom.telemetry.runtime import operation, bind
@operation("run")
def execute(root, **kwargs):
    bind(root)
    (root / "ready").touch()
    while True:
        (root / "last.pt").write_text("fixture safe state")
        control.boundary(root, fresh=True)
        time.sleep(.02)
background._execute = execute
background.main(sys.argv[1:])
"""
    supervisor = r"""
import json, sys
from pathlib import Path
from smartsom.experiments import background, foreground
from smartsom.telemetry import monitor
original_key = monitor._key
def key(fd, timeout):
    (Path(sys.argv[1]) / "terminal-ready").touch()
    return original_key(fd, timeout)
monitor._key = key
original = background.subprocess.Popen
def spawn(command, **kwargs):
    if len(command) > 2 and command[2] == "smartsom.experiments.background":
        command = [sys.executable, "-c", sys.argv[2], *command[3:]]
    return original(command, **kwargs)
background.subprocess.Popen = spawn
result = foreground.run(Path(sys.argv[1]), display_options={"progress": "on"})
print("RESULT=" + json.dumps(result), flush=True)
"""
    master, slave = pty.openpty()
    process = subprocess.Popen(
        [sys.executable, "-c", supervisor, str(root), driver],
        stdin=slave,
        stdout=slave,
        stderr=slave,
        start_new_session=True,
        env={**os.environ, "PYTHONPATH": "src", "TERM": "xterm-256color"},
    )
    os.close(slave)
    output = bytearray()

    def read_until(predicate, seconds=15):
        deadline = time.monotonic() + seconds
        while not predicate():
            assert time.monotonic() < deadline, output.decode(errors="replace")
            assert process.poll() is None, output.decode(errors="replace")
            if select.select([master], [], [], 0.05)[0]:
                try:
                    output.extend(os.read(master, 65536))
                except OSError:
                    assert predicate(), output.decode(errors="replace")

    try:
        read_until(
            lambda: (root / "ready").exists() and (root / "terminal-ready").exists()
        )
        owner = control.read(root)
        if choice == b"2":
            os.kill(process.pid, signal.SIGINT)
        elif choice != b"d":
            os.write(master, b"q")
        if choice != b"d":
            read_until(lambda: "停止并保留".encode() in output)
        os.write(master, choice)
        read_until(lambda: b"RESULT=" in output, seconds=20)
        process.wait(timeout=5)
        assert process.returncode == 0, output.decode(errors="replace")
        assert control.alive(owner["owner"], control.processes()) == (choice == b"d")
        assert config.read_text() == "authoring stays"
        if choice == b"2":
            assert not root.exists()
            assert b'"deleted": true' in output
        elif choice == b"1":
            assert (root / "last.pt").read_text() == "fixture safe state"
            assert b'"status": "stopped"' in output
        else:
            assert b'"detached": true' in output
            assert root.exists()
    finally:
        if process.poll() is None:
            process.kill()
        process.wait()
        if root.exists() and control.read(root):
            control.stop(root, timeout=2, force=True)
        os.close(master)
