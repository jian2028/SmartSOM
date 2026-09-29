"""Background startup identity, durable failures and detached process behavior."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from smartsom.experiments import background, control


@pytest.fixture
def root(tmp_path):
    root = tmp_path / "run with spaces"
    root.mkdir()
    (root / "control").mkdir()
    (root / "plan.json").write_text(
        json.dumps({"schema": "smartsom.author-plan/v1", "scientific_sha256": "frozen"})
    )
    (root / "run.json").write_text(json.dumps({"status": "prepared", "preserved": 17}))
    (root / "batch.json").write_text(json.dumps({"status": "prepared", "entries": [1]}))
    return root


def owner(root, pid=112233, created="child-birth", status="running"):
    return {
        "schema": "smartsom.run-control/v1",
        "id": "generation-a",
        "root": str(root),
        "driver_root": str(root),
        "owner": {"pid": pid, "created": created, "parent": 2, "state": "S"},
        "status": status,
        "members": [],
    }


def fake_spawn(root, monkeypatch, *, startup=True, status="running", returncode=None):
    calls = []
    monkeypatch.setattr(
        control,
        "_birth",
        lambda pid: "child-birth" if pid == 112233 else "parent-birth",
    )

    def spawn(command, **options):
        calls.append((command, options))
        if startup:
            request = background._read_json(root / "control/background-request.json")
            record = owner(root, status=status)
            control.write_json(root / "control/owner.json", record)
            control.write_json(
                root / "control/background-startup.json",
                {
                    "schema": background.STARTUP_SCHEMA,
                    "root": str(root),
                    "nonce": request["nonce"],
                    "status": "ready",
                    "control_id": record["id"],
                    "owner": record["owner"],
                },
            )
        return SimpleNamespace(pid=112233, poll=lambda: returncode)

    monkeypatch.setattr(background.subprocess, "Popen", spawn)
    return calls


def test_explicit_spawn_parameters_and_only_execution_metadata(root, monkeypatch):
    original_plan = (root / "plan.json").read_bytes()
    calls = fake_spawn(root, monkeypatch)
    result = background.launch(
        root, resume=True, display_options={"progress": "on"}, max_concurrent=2
    )
    assert result["startup_status"] == "ready"
    assert result["status"] == "running"
    assert result["run_status"] == "prepared"
    command, options = calls[0]
    assert command[:4] == [
        sys.executable,
        "-m",
        "smartsom.experiments.background",
        str(root),
    ]
    assert command[-1] == "--resume"
    assert options["start_new_session"] is True
    assert options["stdin"] == subprocess.DEVNULL
    assert options["close_fds"] is True
    assert "shell" not in options
    assert Path(options["stdout"].name) == root / "logs/stdout.log"
    assert Path(options["stderr"].name) == root / "logs/stderr.log"
    assert options["env"]["NO_COLOR"] == "1"
    request = background._read_json(root / "control/background-request.json")
    assert request["display_options"]["progress"] == "off"
    assert request["max_concurrent"] == 2
    assert (root / "plan.json").read_bytes() == original_plan
    assert not (root / "control/background-launch.lock").exists()


def test_directory_background_resume_passes_retry_failed_as_execution_metadata(
    root, monkeypatch
):
    (root / "plan.json").write_text(
        json.dumps({"schema": "smartsom.author-batch-plan/v1"})
    )
    fake_spawn(root, monkeypatch)
    result = background.launch(root, resume=True, retry_failed=True)
    assert result["startup_status"] == "ready"
    request = background._read_json(root / "control/background-request.json")
    assert request["retry_failed"] is True
    with pytest.raises(ValueError, match="retry-failed requires"):
        background.launch(root, retry_failed=True)


def test_directory_resume_forwards_failed_trial_retry(root, monkeypatch):
    from smartsom.experiments import author_batch, commands

    (root / "plan.json").write_text(
        json.dumps({"schema": "smartsom.author-batch-plan/v1"})
    )
    monkeypatch.setattr(author_batch, "load", lambda path: (path, {}, {}))
    calls = []
    monkeypatch.setattr(
        author_batch,
        "execute_saved",
        lambda path, *, retry_failed: (
            calls.append((path, retry_failed)) or {"status": "completed"}
        ),
    )
    args = SimpleNamespace(
        source=root,
        retry_failed=True,
        max_concurrent=None,
        extension_module=[],
        background=False,
    )
    assert commands.resume(args)["status"] == "completed"
    assert calls == [(root, True)]


def test_refuse_existing_owner_and_members_without_starting(root, monkeypatch):
    record = owner(root)
    control.write_json(root / "control/owner.json", record)
    monkeypatch.setattr(control, "processes", lambda: {112233: record["owner"]})
    called = []
    monkeypatch.setattr(
        background.subprocess, "Popen", lambda *a, **k: called.append(1)
    )
    with pytest.raises(ValueError, match="live control owner"):
        background.launch(root)
    assert not called
    assert background._read_json(root / "run.json")["status"] == "prepared"
    record["status"] = "completed"
    record["owner"]["pid"] = 44
    record["members"] = [{"pid": 112233, "created": "child-birth", "state": "S"}]
    control.write_json(root / "control/owner.json", record)
    with pytest.raises(ValueError, match="owned worker"):
        background.launch(root)


def test_pid_reuse_is_not_a_live_old_owner(root, monkeypatch):
    record = owner(root)
    control.write_json(root / "control/owner.json", record)
    monkeypatch.setattr(
        control,
        "processes",
        lambda: {112233: {"pid": 112233, "created": "new-birth", "state": "S"}},
    )
    assert background._check_owner(root) is None


def test_completed_child_owner_is_verified_after_exit(root, monkeypatch):
    fake_spawn(root, monkeypatch, status="completed", returncode=0)
    count = 0

    def birth(pid):
        nonlocal count
        if pid != 112233:
            return "parent-birth"
        count += 1
        return "child-birth" if count == 1 else None

    monkeypatch.setattr(control, "_birth", birth)
    assert background.launch(root)["status"] == "completed"


def test_unverified_pid_change_rejected_without_signals(root, monkeypatch):
    fake_spawn(root, monkeypatch)
    count = 0

    def birth(pid):
        nonlocal count
        if pid != 112233:
            return "parent-birth"
        count += 1
        return "child-birth" if count == 1 else "reused-birth"

    monkeypatch.setattr(control, "_birth", birth)
    with pytest.raises(ValueError, match="identity changed"):
        background.launch(root)
    assert background._read_json(root / "run.json")["status"] == "failed"


@pytest.mark.parametrize("returncode", [None, 1])
def test_missing_handshake_timeout_or_exit_durable_failure(
    root, monkeypatch, returncode
):
    original_plan = (root / "plan.json").read_bytes()
    fake_spawn(root, monkeypatch, startup=False, returncode=returncode)
    monkeypatch.setattr(background, "STARTUP_TIMEOUT", 0.0)
    with pytest.raises(ValueError, match="timed out|exited before verified"):
        background.launch(root)
    request = background._read_json(root / "control/background-request.json")
    assert request["abort_requested"] is True
    assert background._read_json(root / "run.json")["preserved"] == 17
    assert background._read_json(root / "batch.json")["entries"] == [1]
    assert background._read_json(root / "batch.json")["status"] == "failed"
    assert (root / "plan.json").read_bytes() == original_plan


def test_spawn_exception_is_durable(root, monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("cannot spawn")

    monkeypatch.setattr(background.subprocess, "Popen", fail)
    with pytest.raises(OSError, match="cannot spawn"):
        background.launch(root)
    assert background._read_json(root / "run.json")["status"] == "failed"


def test_windows_and_invalid_inputs_before_spawn(root, monkeypatch):
    monkeypatch.setattr(background.sys, "platform", "win32")
    with pytest.raises(ValueError, match="only on macOS and Linux"):
        background.launch(root)
    monkeypatch.setattr(background.sys, "platform", "darwin")
    with pytest.raises(ValueError, match="positive integer"):
        background.launch(root, max_concurrent=True)
    with pytest.raises(ValueError, match="resume must be boolean"):
        background.launch(root, resume="yes")
    (root / "plan.json").write_text(json.dumps({"schema": "old"}))
    with pytest.raises(ValueError, match="frozen author, directory or Tune plan"):
        background.launch(root)


def test_child_binds_before_ready_and_driver_then_finishes_scope(root, monkeypatch):
    control.write_json(
        root / "control/background-request.json",
        {
            "schema": background.REQUEST_SCHEMA,
            "root": str(root),
            "nonce": "expected",
            "resume": True,
            "display_options": {"progress": "on"},
            "max_concurrent": 2,
        },
    )
    events = []

    class Scope:
        def bind(self, path):
            events.append("bind")
            control.write_json(path / "control/owner.json", owner(path))

        def finish(self, status):
            events.append(("finish", status))

    def execute(path, **kwargs):
        events.append("execute")
        assert control.CURRENT.get() is not None
        assert (
            background._read_json(path / "control/background-startup.json")["status"]
            == "ready"
        )
        assert kwargs == {
            "max_concurrent": 2,
            "display_options": {"progress": "off"},
            "recommend_only": False,
            "retry_failed": False,
        }
        return {"status": "completed"}

    monkeypatch.setattr(control, "Scope", Scope)
    monkeypatch.setattr(background, "_execute", execute)
    assert background.child(root, nonce="expected", resume=True) == {
        "status": "completed"
    }
    assert events == ["bind", "execute", ("finish", "completed")]
    assert control.CURRENT.get() is None


def test_child_rejects_cancelled_or_drifted_request(root, monkeypatch):
    control.write_json(
        root / "control/background-request.json",
        {
            "schema": background.REQUEST_SCHEMA,
            "root": str(root),
            "nonce": "expected",
            "resume": False,
            "display_options": {},
            "abort_requested": True,
        },
    )
    with pytest.raises(ValueError, match="cancelled before execution"):
        background.child(root, nonce="expected")
    assert background._read_json(root / "run.json")["status"] == "failed"
    with pytest.raises(ValueError, match="request identity changed"):
        background.child(root, nonce="wrong")


def test_startup_race_does_not_mark_another_live_owner_failed(root, monkeypatch):
    control.write_json(
        root / "control/background-request.json",
        {
            "schema": background.REQUEST_SCHEMA,
            "root": str(root),
            "nonce": "expected",
            "resume": False,
            "display_options": {},
        },
    )
    foreign = owner(root, pid=os.getpid(), created=control._birth(os.getpid()))
    control.write_json(root / "control/owner.json", foreign)
    original = (root / "control/owner.json").read_bytes()

    class Scope:
        id = "new-attempt"

        def bind(self, path):
            raise ValueError("run already has a live control owner")

        def finish(self, status):
            assert status == "failed"

    monkeypatch.setattr(control, "Scope", Scope)
    with pytest.raises(ValueError, match="live control owner"):
        background.child(root, nonce="expected")
    assert background._read_json(root / "run.json")["status"] == "prepared"
    assert background._read_json(root / "batch.json")["status"] == "prepared"
    assert (root / "control/owner.json").read_bytes() == original
    assert (
        background._read_json(root / "control/background-startup.json")[
            "manifest_preserved_for_live_owner"
        ]
        is True
    )


@pytest.mark.skipif(
    sys.platform not in {"darwin", "linux"}, reason="POSIX background only"
)
def test_real_detached_child_continues_after_launcher_exits(root, tmp_path):
    """Mock driver/ownership inventory; exercise real PID birth/session/stdio."""
    source = Path(__file__).resolve().parents[2] / "src/smartsom"
    overlay = tmp_path / "overlay"
    package = overlay / "smartsom"
    experiments = package / "experiments"
    experiments.mkdir(parents=True)
    (package / "__init__.py").write_text(f"__path__.append({str(source)!r})\n")
    (experiments / "__init__.py").write_text(
        f"__path__.append({str(source / 'experiments')!r})\n"
        "from smartsom.experiments import control\n"
        "import os\n"
        "def inventory():\n"
        "    pid = os.getpid()\n"
        "    return {pid: {'pid': pid, 'created': control._birth(pid), 'parent': os.getppid(), 'state': 'R'}}\n"
        "control.processes = inventory\n"
    )
    (experiments / "author_driver.py").write_text(
        "import json, os, sys, time\n"
        "from smartsom.experiments.control import write_json\n"
        "from smartsom.telemetry.runtime import CURRENT\n"
        "def execute_saved(root, max_concurrent=None):\n"
        "    write_json(root / 'probe.json', {'pid': os.getpid(), 'session': os.getsid(0),\n"
        "        'stdin': sys.stdin.read(), 'stdout_tty': sys.stdout.isatty(),\n"
        "        'stderr_tty': sys.stderr.isatty(), 'progress': CURRENT.get().options.progress})\n"
        "    print('engineering child started', flush=True)\n"
        "    time.sleep(0.6)\n"
        "    write_json(root / 'run.json', {'status': 'completed'})\n"
        "    print('completed after launcher returned', flush=True)\n"
        "    return {'status': 'completed'}\n"
    )
    program = (
        "import json\nfrom smartsom.experiments.background import launch\n"
        f"print(json.dumps(launch({str(root)!r})))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", program],
        env={**os.environ, "PYTHONPATH": str(overlay)},
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    started = json.loads(result.stdout)
    assert started["startup_status"] == "ready"
    deadline = time.monotonic() + 5
    while background._read_json(root / "run.json")["status"] != "completed":
        assert time.monotonic() < deadline
        time.sleep(0.02)
    probe = background._read_json(root / "probe.json")
    assert probe["session"] == probe["pid"] == started["owner"]["pid"]
    assert probe["stdin"] == ""
    assert probe["stdout_tty"] is False
    assert probe["stderr_tty"] is False
    assert probe["progress"] == "off"
    logs = (root / "logs/stdout.log").read_text()
    assert "completed after launcher returned" in logs
    assert "\x1b[" not in logs
    while control.read(root)["status"] == "running":
        assert time.monotonic() < deadline
        time.sleep(0.02)
    assert control.read(root)["status"] == "completed"
