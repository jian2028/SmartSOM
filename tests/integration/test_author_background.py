"""Actual four-file native background lifecycle on tiny engineering cases."""

import io
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from uuid import uuid4

import pytest
import yaml
from rich.console import Console

from smartsom.config.codec import primitive
from smartsom.config.experiment_v4 import compile_experiment
from smartsom.config.factory_design import load_factory_design_file
from smartsom.experiments import author_driver, background, control
from smartsom.telemetry.monitor import read_snapshot
from smartsom.telemetry.runtime import DisplayOptions, RuntimeDisplay

SOURCE = Path(author_driver.__file__).resolve().parents[3]
pytestmark = pytest.mark.skipif(
    sys.platform not in {"darwin", "linux"}, reason="POSIX background lifecycle"
)


@pytest.fixture
def author_case(tmp_path):
    source = SOURCE
    factory, _ = load_factory_design_file(
        source / "configs/test/factories/factory_hand.yaml"
    )
    folder = tmp_path / "author"
    folder.mkdir()
    files = {
        "factory": primitive(factory),
        "workload": {
            "schema": "smartsom.workload/v2",
            "demands": [
                {
                    "demand_id": f"job-{i:03d}",
                    "steps": [
                        {
                            "operation_id": "op-1",
                            "operation_type": "operation_1",
                            "nominal_ticks": 1,
                        }
                    ],
                    "release_at": 0,
                    "due_at": 900,
                }
                for i in range(12)
            ],
        },
        "algorithm": {
            "schema": "smartsom.algorithm/v2",
            "mode": "rules",
            "agents": {
                role: {"default": {"kind": "rule", "name": name}}
                for role, name in {
                    "machine": "spt",
                    "buffer": "edd",
                    "dispatcher": "nearest",
                    "mover": "shortest_path",
                }.items()
            },
        },
        "experiment": {
            "schema": "smartsom.experiment-config/v4",
            "task": "evaluate",
            "factory": "factory.yaml",
            "workload": "workload.yaml",
            "algorithm": "algorithm.yaml",
            "data_seed": 901,
            "runtime": {"environment": {"tick_limit": 1000}},
            "evaluation": {"replications": 1, "full_replay": False},
            "output": {
                "root": str(tmp_path / "results"),
                "name": "background-engineering",
            },
            "logging": {"progress": "off", "verbose": False},
        },
    }
    for name, data in files.items():
        (folder / f"{name}.yaml").write_text(yaml.safe_dump(data, sort_keys=False))
    return folder, files


def wait_for(predicate, *, timeout=20):
    deadline = time.monotonic() + timeout
    while True:
        result = predicate()
        if result:
            return result
        if time.monotonic() >= deadline:
            raise AssertionError("timed out waiting for native engineering lifecycle")
        time.sleep(0.03)


def save(folder, name, value):
    (folder / f"{name}.yaml").write_text(yaml.safe_dump(value, sort_keys=False))


def control_exited(root):
    owner = control.read(root)
    table = control.processes()
    return owner and not any(
        control.alive(identity, table)
        for identity in [owner["owner"], *owner.get("members", [])]
    )


def test_real_background_rule_evaluation_monitor_and_frozen_sources(
    author_case, monkeypatch
):
    folder, files = author_case
    source = SOURCE / "src"
    monkeypatch.setenv("PYTHONPATH", str(source))
    before = {path: path.read_bytes() for path in folder.glob("*.yaml")}
    plan = compile_experiment(folder / "experiment.yaml")
    root = author_driver.allocate(plan)
    # Author edits after allocation cannot change this run's frozen inputs.
    (folder / "workload.yaml").write_text("not a valid workload anymore\n")
    started = background.launch(
        root, display_options={"progress": "on", "verbose": False}
    )
    try:
        assert started["startup_status"] == "ready"
        assert started["owner"]["pid"] != os.getpid()
        state = wait_for(
            lambda: (
                (data if data["status"] in {"completed", "failed"} else None)
                if (data := json.loads((root / "batch.json").read_text()))
                else None
            )
        )
        assert state["status"] == "completed", state
        wait_for(lambda: control_exited(root))
        entry = plan.entries[0]
        ledger_path = root / "entries" / entry.id / "stages.json"
        ledger = json.loads(ledger_path.read_text())
        assert set(ledger["stages"]) == {"evaluation"}
        assert ledger["stages"]["evaluation"]["status"] == "completed"
        assert len(ledger["stages"]["evaluation"]["attempts"]) == 1
        snapshot = read_snapshot(root)
        assert snapshot["status"] == "completed"
        assert snapshot["total_tasks"] == 1
        assert all(row.get("learner", {}) == {} for row in snapshot["tasks"])
        for width in (40, 52, 120, 180):
            stream = io.StringIO()
            display = RuntimeDisplay(
                DisplayOptions(verbose=True, progress="off"),
                kind=snapshot["kind"],
                readonly=True,
                console=Console(
                    file=stream, width=width, height=35, force_terminal=False
                ),
            ).from_snapshot(snapshot)
            display.console.print(display.render())
            assert "completed" in stream.getvalue().lower()
            assert "\x1b[" not in stream.getvalue()
        monitored = subprocess.run(
            [
                sys.executable,
                "-m",
                "smartsom.experiments.cli",
                "monitor",
                str(root),
                "--once",
                "--log-format",
                "json",
                "--progress",
                "off",
            ],
            env=os.environ,
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        assert monitored.returncode == 0, monitored.stderr
        lines = [
            json.loads(line) for line in monitored.stderr.splitlines() if line.strip()
        ]
        assert lines and any(line.get("status") == "completed" for line in lines)
        assert monitored.stdout == ""
        assert "\x1b[" not in monitored.stderr
        assert "\x1b[" not in (root / "logs/stdout.log").read_text()
        saved = ledger_path.read_bytes()
        # Resuming a completed plan does not repeat the completed evaluation.
        second = background.launch(
            root, resume=True, display_options={"verbose": False}
        )
        assert second["startup_status"] == "ready"
        wait_for(lambda: control_exited(root))
        assert ledger_path.read_bytes() == saved
        for path, value in before.items():
            if path.name != "workload.yaml":
                assert path.read_bytes() == value
    finally:
        if control.read(root) and not control_exited(root):
            control.stop(root, timeout=20, force=True)


def test_real_background_stop_then_resume_interrupted_rule_phase(
    author_case, tmp_path, monkeypatch
):
    folder, files = author_case
    module_name = "smartsom_slow_engineering_" + uuid4().hex
    module = tmp_path / (module_name + ".py")
    rule_name = "engineering.slow_" + uuid4().hex
    module.write_text(
        "import time\n"
        "from smartsom.algorithms.rule_registry import register_rule\n"
        "class SlowRule:\n"
        "    def __init__(self, parameters, seed): self.delay = parameters['delay']\n"
        "    def reset(self): self.calls = 0\n"
        "    def state_dict(self): return {'calls': self.calls}\n"
        "    def load_state_dict(self, state): self.calls = state['calls']\n"
        "    def choose(self, request):\n"
        "        time.sleep(self.delay)\n"
        "        self.calls += 1\n"
        "        return min(request.candidates, key=lambda c: c.feature('processing_time_scaled')).action\n"
        f"register_rule({rule_name!r}, '1', SlowRule, roles=('machine',), stateful=True)\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv("PYTHONPATH", str(tmp_path) + os.pathsep + str(SOURCE / "src"))
    files["algorithm"]["agents"]["machine"]["default"] = {
        "kind": "rule",
        "name": rule_name,
        "version": "1",
        "parameters": {"delay": 0.03},
    }
    save(folder, "algorithm", files["algorithm"])
    plan = compile_experiment(
        folder / "experiment.yaml", extension_modules=(module_name,)
    )
    root = author_driver.allocate(plan)
    entry = plan.entries[0]
    background.launch(root, display_options={"verbose": False})
    try:
        ledger_path = root / "entries" / entry.id / "stages.json"
        wait_for(lambda: ledger_path.is_file())
        wait_for(
            lambda: list(
                (root / "entries" / entry.id / "runs").glob("*/evidence/*/trace.jsonl")
            )
        )
        stopped = control.stop(root, timeout=20)
        assert stopped["status"] == "stopped", stopped
        assert stopped["remaining"] == []
        assert control_exited(root)
        ledger = json.loads(ledger_path.read_text())
        assert ledger["status"] == "stopped"
        assert set(ledger["stages"]) == {"evaluation"}
        assert ledger["stages"]["evaluation"]["status"] == "stopped"
        partial_attempts = list(ledger["stages"]["evaluation"]["attempts"])
        assert partial_attempts
        assert read_snapshot(root)["status"] == "stopped"
        assert control.stop(root, timeout=0)["status"] == "stopped"
        resumed = background.launch(
            root, resume=True, display_options={"verbose": False}
        )
        assert resumed["startup_status"] == "ready"
        state = wait_for(
            lambda: (
                (data if data["status"] in {"completed", "failed"} else None)
                if (data := json.loads((root / "batch.json").read_text()))
                else None
            )
        )
        wait_for(lambda: control_exited(root))
        assert state["status"] == "completed", state
        ledger = json.loads(ledger_path.read_text())
        attempts = ledger["stages"]["evaluation"]["attempts"]
        assert attempts[: len(partial_attempts)] == partial_attempts
        assert len(attempts) == len(partial_attempts) + 1
        assert ledger["stages"]["evaluation"]["status"] == "completed"
        assert set(ledger["stages"]) == {"evaluation"}
    finally:
        if control.read(root) and not control_exited(root):
            control.stop(root, timeout=20, force=True)


def test_background_invalid_frozen_plan_failure_is_durable(author_case, monkeypatch):
    folder, _ = author_case
    monkeypatch.setenv("PYTHONPATH", str(SOURCE / "src"))
    root = author_driver.allocate(compile_experiment(folder / "experiment.yaml"))
    frozen = json.loads((root / "plan.json").read_text())
    frozen["implementation_sha256"] = "0" * 64
    (root / "plan.json").write_text(json.dumps(frozen))
    try:
        try:
            background.launch(root, display_options={"verbose": False})
        except ValueError as exc:
            assert "background startup failed" in str(exc)
        wait_for(
            lambda: json.loads((root / "run.json").read_text())["status"] == "failed"
        )
        wait_for(lambda: control_exited(root))
        startup = json.loads((root / "control/background-startup.json").read_text())
        assert startup["status"] == "failed"
        assert "frozen author plan changed" in startup["error"]
        assert json.loads((root / "batch.json").read_text())["status"] == "failed"
        assert not (root / "entries").exists()
    finally:
        if control.read(root) and not control_exited(root):
            control.stop(root, timeout=20, force=True)
