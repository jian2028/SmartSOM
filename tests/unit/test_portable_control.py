"""Real local advisory locks and kernel process identities."""

import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from smartsom.experiments.batch import exclusive_lock
from smartsom.experiments.control import _birth, _parent_precedes_child, processes


def test_atomic_publication_survives_a_concurrent_reader(tmp_path):
    from smartsom.experiments.control import write_json

    path = tmp_path / "owner.json"
    write_json(path, {"generation": 1})
    with ThreadPoolExecutor(max_workers=1) as executor:
        with path.open() as reader:
            future = executor.submit(write_json, path, {"generation": 2})
            time.sleep(0.05)
            assert '"generation": 1' in reader.read()
        future.result(timeout=5)
    assert '"generation": 2' in path.read_text()


def test_control_reads_remain_complete_during_repeated_replacement(tmp_path):
    from smartsom.experiments.control import read, write_json

    folder = tmp_path / "control"
    folder.mkdir()
    path = folder / "owner.json"
    write_json(path, {"generation": 0})

    def publish():
        for generation in range(1, 201):
            write_json(path, {"generation": generation})

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(publish)
        while not future.done():
            assert 0 <= read(tmp_path)["generation"] <= 200
        future.result()
    assert read(tmp_path) == {"generation": 200}


def test_lock_conflicts_across_processes_and_releases(tmp_path):
    path = tmp_path / "共享.lock"
    script = """
import sys
from pathlib import Path
from smartsom.experiments.batch import exclusive_lock
try:
    with exclusive_lock(Path(sys.argv[1])):
        pass
except RuntimeError:
    sys.exit(23)
"""

    def attempt():
        return subprocess.run([sys.executable, "-c", script, str(path)], check=False)

    with exclusive_lock(path):
        assert attempt().returncode == 23
    assert path.exists()
    assert attempt().returncode == 0


def test_real_identity_and_parent():
    table = processes()
    own = table[os.getpid()]
    assert own["created"] == _birth(os.getpid())
    assert own["parent"] == os.getppid()


@pytest.mark.parametrize(
    "parent,child,valid",
    [
        ("100", "101", True),
        ("101", "100", False),
        ("123:900", "124:100", True),
        ("124:100", "123:900", False),
        ("unknown", "123", False),
    ],
)
def test_parent_generation_order(parent, child, valid):
    assert _parent_precedes_child({"created": parent}, {"created": child}) is valid


@pytest.mark.skipif(os.name != "nt", reason="Windows console API")
def test_windows_console_keys_and_interrupt(monkeypatch):
    import msvcrt

    from smartsom.telemetry.monitor import _key

    monkeypatch.setattr(msvcrt, "kbhit", lambda: True)
    monkeypatch.setattr(msvcrt, "getwch", lambda: "d")
    assert _key("windows-console", 0) == "d"
    monkeypatch.setattr(msvcrt, "getwch", lambda: "\x03")
    with pytest.raises(KeyboardInterrupt):
        _key("windows-console", 0)
    monkeypatch.setattr(msvcrt, "kbhit", lambda: False)
    assert _key("windows-console", 0) is None


@pytest.mark.skipif(os.name != "nt", reason="Windows handle identity regression")
def test_windows_force_does_not_kill_mismatched_birth():
    from smartsom.experiments.windows_processes import terminate

    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        identity = processes()[child.pid]
        terminate({**identity, "created": "wrong generation"})
        assert child.poll() is None
        terminate(identity)
        child.wait(timeout=5)
    finally:
        if child.poll() is None:
            child.kill()
        child.wait()
