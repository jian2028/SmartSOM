"""Native public search stop drains real training and sampling processes."""

import multiprocessing
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
from test_learning_study_integration import tiny_recipe

from smartsom import api
from smartsom.config.codec import canonical_json
from smartsom.config.experiment import ExperimentConfig, apply_overrides
from smartsom.experiments import control, learning_study
from smartsom.experiments.catalog import read_json
from smartsom.experiments.production_training import verify_checkpoint

_REAL_WORKER = learning_study._worker


def _gated_worker(directory, attempt, *args):
    """Pause only the progress callback after a genuine saved first update."""
    original = api.train_prepared

    def train(prepared, *, on_progress=None, **kwargs):
        def progress(event):
            result = on_progress(event) if on_progress else None
            if event.get("stage") == "learning_metrics" and event["ppo_updates"] == 2:
                scope = control.CURRENT.get()
                scope.capture()
                table = control.processes()
                children = [
                    row for row in table.values() if row["parent"] == os.getpid()
                ]
                assert children, "the native sampling child must be running"
                control.write_json(
                    Path(attempt) / "gate-ready.json",
                    {"worker": table[os.getpid()], "children": children},
                )
                deadline = time.monotonic() + 90
                while not control.requested(attempt):
                    if time.monotonic() >= deadline:
                        raise TimeoutError("test did not request search stop")
                    time.sleep(0.01)
            return result

        return original(prepared, on_progress=progress, **kwargs)

    api.train_prepared = train
    try:
        _REAL_WORKER(directory, attempt, *args)
    finally:
        api.train_prepared = original


def _search_driver(config_json):
    learning_study._worker = _gated_worker
    result = api.search(ExperimentConfig.model_validate_json(config_json))
    assert result.status == "interrupted"


def _remember_processes(owned):
    table = control.processes()
    parents = {
        identity["pid"] for identity in owned.values() if control.alive(identity, table)
    }
    changed = True
    while changed:
        changed = False
        for pid, row in table.items():
            if (
                pid not in parents
                and row["parent"] in parents
                and control._parent_precedes_child(table[row["parent"]], row)
            ):
                owned[pid, row["created"]] = row
                parents.add(pid)
                changed = True


def _cleanup_processes(owned):
    # Emergency cleanup targets only observed descendants and exact kernel births.
    signals = (None,) if os.name == "nt" else (signal.SIGTERM, signal.SIGKILL)
    for signum in signals:
        _remember_processes(owned)
        for identity in reversed(list(owned.values())):
            if control.alive(identity, control.processes()):
                try:
                    if os.name == "nt":
                        from smartsom.experiments.windows_processes import terminate

                        terminate(identity)
                    else:
                        os.kill(identity["pid"], signum)
                except ProcessLookupError:
                    pass
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            table = control.processes()
            if not any(control.alive(identity, table) for identity in owned.values()):
                return
            time.sleep(0.05)
    assert not any(
        control.alive(identity, control.processes()) for identity in owned.values()
    ), "a verified test-owned process survived emergency cleanup"


def _wait(process, predicate, root, owned, timeout=90):
    deadline = time.monotonic() + timeout
    while True:
        _remember_processes(owned)
        if predicate():
            return
        if process.exitcode is not None or time.monotonic() > deadline:
            logs = {
                str(path): path.read_text(encoding="utf-8", errors="replace")[-8000:]
                for path in root.rglob("worker.log")
            }
            raise AssertionError(f"search did not reach the saved boundary: {logs}")
        time.sleep(0.05)


def test_public_optuna_stop_drains_owned_children_and_resumes(tmp_path):
    from smartsom.experiments.search import require_optuna

    optuna = (
        require_optuna()
        if os.environ.get("SMARTSOM_REQUIRE_SEARCH") == "1"
        else pytest.importorskip("optuna")
    )
    config = apply_overrides(
        tiny_recipe(tmp_path),
        [
            ("training.total_steps", 64),
            ("training.max_decisions", 32),
            ("runtime.sampling_processes", 1),
            ("runtime.max_concurrent", 1),
            ("checkpointing.every_updates", 1),
            ("validation.enabled", True),
            ("validation.every_updates", 4),
            ("validation.replications", 1),
            ("search.method", "optuna"),
            ("search.trials", 3),
            ("search.objective", "makespan"),
            ("search.failure_policy", "all_complete"),
            (
                "search.space",
                {
                    "algorithm.learning_rate": {
                        "type": "float",
                        "low": 0.0001,
                        "high": 0.0004,
                    }
                },
            ),
        ],
    )
    context = multiprocessing.get_context("spawn")
    driver = context.Process(target=_search_driver, args=(canonical_json(config),))
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(180)"])
    owned = {}
    root = None
    try:
        driver.start()
        identity = control.processes()[driver.pid]
        owned[identity["pid"], identity["created"]] = identity
        output = Path(config.output.root)
        _wait(
            driver,
            lambda: bool(list(output.glob("*/trials/*/attempt-*/gate-ready.json"))),
            output,
            owned,
        )
        gate = next(output.glob("*/trials/*/attempt-*/gate-ready.json"))
        root = gate.parents[3]
        identities = read_json(gate)
        workers = [identities["worker"], *identities["children"]]
        assert workers[0]["parent"] == driver.pid
        _wait(
            driver,
            lambda: all(
                any(
                    (item["pid"], item["created"])
                    == (identity["pid"], identity["created"])
                    for item in control.read(root)["members"]
                )
                for identity in workers
            ),
            output,
            owned,
        )
        # The gate is in update two; update one has already been sealed by the real learner.
        first = next(
            root.glob("trials/*/attempt-*/replication-*/*/checkpoints/update-000001")
        )
        assert verify_checkpoint(first)["steps"] == 16
        plan_before = (root / "config/plan.json").read_bytes()
        stopped = control.stop(root, timeout=30)
        assert stopped["status"] == "stopped" and stopped["remaining"] == []
        driver.join(timeout=10)
        assert driver.exitcode == 0
        assert unrelated.poll() is None
        table = control.processes()
        assert all(not control.alive(identity, table) for identity in owned.values())
        summary = read_json(root / "summary.json")
        assert summary["status"] == "interrupted" and summary["pending"] == 3
        assert (
            len(summary["entries"]) == len(read_json(root / "run.json")["trials"]) == 1
        )
        entry = summary["entries"][0]
        assert entry["status"] == "interrupted"
        row = entry["runs"][0]
        assert 0 < row["environment_steps"] < config.training.total_steps
        child = root / "trials" / entry["id"] / row["run_dir"]
        last = read_json(child / "checkpoints/last.json")["checkpoint"]
        assert (
            verify_checkpoint(child / "checkpoints" / last)["steps"]
            == row["environment_steps"]
        )
        database = optuna.load_study(
            study_name="smartsom-search", storage=f"sqlite:///{root / 'optuna.sqlite3'}"
        )
        assert len(database.trials) == 1 and database.trials[0].state.name == "RUNNING"
        parameters = database.trials[0].params
        resumed = api.search(resume=root)
        assert resumed.pending == 0 and len(resumed.entries) == 3
        assert all(
            entry["status"] in {"completed", "ineligible"} for entry in resumed.entries
        )
        assert all(
            entry["runs"][0]["environment_steps"] == 64 for entry in resumed.entries
        )
        assert len(database.trials) == 3 and database.trials[0].params == parameters
        assert all(trial.state.is_finished() for trial in database.trials)
        assert (root / "config/plan.json").read_bytes() == plan_before
        assert api.search(resume=root).entries == resumed.entries
        assert unrelated.poll() is None
    finally:
        try:
            if driver.is_alive() and root is not None:
                control.stop(root, timeout=1, force=True)
        finally:
            try:
                _cleanup_processes(owned)
                if driver.pid is not None:
                    driver.join(timeout=10)
            finally:
                if unrelated.poll() is None:
                    unrelated.kill()
                unrelated.wait(timeout=10)
