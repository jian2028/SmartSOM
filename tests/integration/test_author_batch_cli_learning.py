"""Public directory-batch training, committed stop, and fresh-process recovery.

One tiny CPU SB3 learner uses the real Ray queue, with fixed execution and no
performance calibration. This checks engineering continuation, not policy quality.
"""

import copy
import json
import os
import pickle
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import pytest
import yaml
from test_author_learning import ROOT, _inputs, _require_cpu

pytestmark = pytest.mark.learning
TOTAL_TICKS = 512
TICKS_PER_UPDATE = 8
CLI = [sys.executable, "-m", "smartsom.experiments.cli"]
DISPLAY = ["--progress", "off", "--no-verbose", "--log-format", "json"]

EXIT_REGISTRY = """
import json, runpy, sys
from pathlib import Path
import psutil
registry, mode, target, *arguments = sys.argv[1:]
try:
    if mode == '-m':
        sys.argv = [target, *arguments]
        runpy.run_module(target, run_name='__main__', alter_sys=True)
    elif mode == '-c':
        sys.argv = ['-c', *arguments]
        exec(compile(target, '<string>', 'exec'), {'__name__': '__main__'})
    else:
        raise ValueError('unsupported public Python entrypoint')
finally:
    identities = []
    for child in psutil.Process().children(recursive=True):
        try:
            identities.append([child.pid, child.create_time()])
        except psutil.NoSuchProcess:
            pass
    temporary = Path(registry + '.tmp')
    temporary.write_text(json.dumps(identities))
    temporary.replace(registry)
"""


class _OwnedProcesses(dict):
    def __init__(self, identities, registry):
        super().__init__(identities)
        self.registry = registry


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _batch_inputs(tmp_path):
    source = _inputs(tmp_path / "inputs", "sb3")
    document = yaml.safe_load(source.read_text())
    # Several inexpensive updates leave room for the separate public stop CLI to
    # start. The public stop is asynchronous: reaching the full budget before it
    # takes effect fails this test rather than retrying or rewinding a checkpoint.
    document["training"].update(
        total_ticks=TOTAL_TICKS, ticks_per_update=TICKS_PER_UPDATE
    )
    document["validation"]["every_updates"] = 8
    document["execution"] = {
        "executor": "tune",
        "tuning": "auto",
        "scheduling": "fixed",
        "calibration_level": "off",
        "max_concurrent": 1,
        "preflight": "quick",
    }
    directory = source.parent / "experiments"
    directory.mkdir()
    for name in ("factory", "workload", "algorithm"):
        document[name] = "../" + document[name]
    (directory / "tiny.yaml").write_text(yaml.safe_dump(document, sort_keys=False))
    return directory, Path(document["output"]["root"])


def _remember_children(owned):
    import psutil

    registry = getattr(owned, "registry", None)
    if registry is not None and registry.exists():
        for pid, born in json.loads(registry.read_text()):
            try:
                process = psutil.Process(pid)
                if process.create_time() == born:
                    owned[(pid, born)] = process
            except psutil.NoSuchProcess:
                pass
    for process in list(owned.values()):
        try:
            if process.is_running():
                for child in process.children(recursive=True):
                    owned[(child.pid, child.create_time())] = child
        except psutil.NoSuchProcess:
            pass


def _alive(owned):
    import psutil

    result = []
    for (pid, born), process in owned.items():
        try:
            if (
                process.is_running()
                and process.create_time() == born
                and process.status() != psutil.STATUS_ZOMBIE
            ):
                result.append(process)
        except psutil.NoSuchProcess:
            pass
    return result


@contextmanager
def _command(directory, environment, *arguments, register_exit_children=False):
    import psutil

    directory.mkdir()
    stdout, stderr = directory / "stdout.log", directory / "stderr.log"
    registry = directory / "exit-children.json" if register_exit_children else None
    launch = [*CLI, *map(str, arguments)]
    if registry is not None:
        # Same Python process executes the actual CLI and records its descendants
        # before SystemExit detaches them, closing the parent's polling race.
        launch = [CLI[0], "-c", EXIT_REGISTRY, str(registry), *launch[1:]]
    with stdout.open("w") as out, stderr.open("w") as err:
        process = subprocess.Popen(
            launch,
            cwd=ROOT,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=out,
            stderr=err,
        )
        identities = {}
        try:
            owner = psutil.Process(process.pid)
            born = owner.create_time()
            if process.poll() is None:
                identities[(owner.pid, born)] = owner
        except psutil.NoSuchProcess:
            # An immediate exit may precede even the initial parent observation.
            # The exit registry still supplies the exact descendant identities.
            pass
        owned = _OwnedProcesses(identities, registry)
        try:
            yield process, owned, stdout, stderr
        finally:
            # Only exact descendants observed from this test's Popen are eligible
            # for emergency cleanup. Never address an ambient/shared Ray cluster.
            _remember_children(owned)
            remaining = _alive(owned)
            for child in reversed(remaining):
                try:
                    child.terminate()
                except psutil.NoSuchProcess:
                    pass
            _, remaining = psutil.wait_procs(remaining, timeout=5)
            for child in remaining:
                if child in _alive(owned):
                    try:
                        child.kill()
                    except psutil.NoSuchProcess:
                        pass
            psutil.wait_procs(remaining, timeout=5)
            process.wait(timeout=10)


def _wait(command, predicate, *, timeout=180, allow_exit=False):
    process, owned, stdout, stderr = command
    deadline = time.monotonic() + timeout
    while True:
        _remember_children(owned)
        result = predicate()
        if result:
            return result
        if (process.poll() is not None and not allow_exit) or (
            time.monotonic() >= deadline
        ):
            pytest.fail(
                f"public CLI did not reach the expected boundary (exit={process.poll()}):"
                f"\n{stdout.read_text()}\n{stderr.read_text()}"
            )
        time.sleep(0.02)


def _finish(command, *, code=0):
    process, owned, stdout, stderr = command
    _wait(command, lambda: process.poll() is not None, allow_exit=True)
    assert process.returncode == code, stdout.read_text() + stderr.read_text()
    _wait(command, lambda: not _alive(owned), timeout=10, allow_exit=True)
    return _json(stdout)


def _allocated(output):
    roots = [path.parent for path in output.glob("*/batch.json")]
    assert len(roots) <= 1
    return roots[0] if roots else None


def _entry(root):
    state = _json(root / "batch.json")
    tune = state.get("tune_directory")
    if not tune:
        return None
    tune_root = root / tune
    path = tune_root / "batch.json"
    if not path.is_file():
        return None
    ledger = _json(path)
    row = ledger.get("entries", {}).get("tiny__entry-0001", {})
    return (tune_root, ledger, row) if row.get("checkpoint") else None


def _state(checkpoint):
    # The test unpickles only its own checksum-verified native commit.
    from smartsom.experiments.tuning_session import verify_commit

    verify_commit(checkpoint)
    with (Path(checkpoint) / "continuation.pkl").open("rb") as stream:
        return pickle.load(stream)


def _assert_exact_state(actual, expected):
    import torch
    from test_tuning_native_continuation import assert_state_equal

    def check_tensor_types(actual, expected):
        # torch.equal alone permits equal values with different tensor dtypes.
        if isinstance(expected, torch.Tensor):
            assert isinstance(actual, torch.Tensor)
            assert actual.dtype == expected.dtype and actual.shape == expected.shape
        elif isinstance(expected, dict):
            for key in expected:
                check_tensor_types(actual[key], expected[key])
        elif isinstance(expected, (list, tuple)):
            for a, b in zip(actual, expected, strict=True):
                check_tensor_types(a, b)

    check_tensor_types(actual, expected)
    assert_state_equal(actual, expected, exact=True)


def test_public_batch_run_stops_then_resumes_native_training_and_evaluation(tmp_path):
    _require_cpu()
    from smartsom.config.experiment_v3 import PreparedComposition
    from smartsom.experiments.tuning_ray import _set_threads
    from smartsom.experiments.tuning_session import AdaptiveSession, verify_identity

    directory, output = _batch_inputs(tmp_path)
    environment = {
        **os.environ,
        "PYTHONPATH": str(ROOT / "src"),
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    with _command(
        tmp_path / "batch-cli",
        environment,
        "batch-run",
        directory,
        "--no-background",
        *DISPLAY,
    ) as command:
        root = _wait(command, lambda: _allocated(output))
        observed = _wait(command, lambda: _entry(root))
        assert 0 < observed[2]["physical_ticks"] < TOTAL_TICKS
        with _command(
            tmp_path / "stop-cli", environment, "stop", root, "--timeout", "60"
        ) as stop_command:
            stopped = _finish(stop_command)
        assert stopped["status"] == "stopped" and not stopped["remaining"]
        assert _finish(command, code=1)["status"] == "stopped"

    tune_root, before_ledger, before = _entry(root)
    checkpoint = Path(before["checkpoint"])
    record = _json(checkpoint / "record.json")
    frozen = PreparedComposition(**before["selected_prepared"])
    marker = verify_identity(frozen, record, checkpoint)
    assert marker["phase"] == "training"
    assert 0 < marker["physical_ticks"] < TOTAL_TICKS
    assert marker["updates"] * TICKS_PER_UPDATE == marker["physical_ticks"]
    partial = _state(checkpoint)
    assert partial["ticks"] == marker["physical_ticks"]
    assert sum(partial["optimizations"].values()) > 0
    assert partial["learners"]["central"]["optimizer"]["state"]
    immutable = {
        path: path.read_bytes()
        for path in (
            checkpoint / "commit.json",
            root / "smoke.json",
            tune_root / "calibration.json",
        )
    }

    # Continue the exact partial state as an uninterrupted native reference.
    # This also checks immediate complete-state restoration before any new update;
    # independent fresh actors need not share their ambient process RNG seeds.
    reference_record = copy.deepcopy(record)
    reference_record["tuning"]["continuation"] = str(checkpoint)
    limits = _set_threads(1)
    reference = None
    try:
        reference = AdaptiveSession(
            frozen, tmp_path / "reference", reference_record, threads=1
        )
        _assert_exact_state(reference.session.state_dict(), partial)
        remaining = (TOTAL_TICKS - partial["ticks"]) // TICKS_PER_UPDATE
        for _ in range(remaining):
            if reference.step()["done"]:
                break
        else:
            pytest.fail("native reference did not finish within its remaining updates")
        expected = _state(reference.last_commit)
    finally:
        if reference is not None:
            reference.close()
        limits.restore_original_limits()

    with _command(
        tmp_path / "resume-cli",
        environment,
        "resume",
        root,
        "--no-background",
        *DISPLAY,
    ) as command:
        resumed = _finish(command)
    assert resumed["status"] == "completed"
    assert _json(root / "batch.json")["files"]["tiny"]["status"] == "completed"
    _, after_ledger, after = _entry(root)
    assert after["status"] == "completed"
    assert len(after_ledger["segments"]) == len(before_ledger["segments"]) + 1
    assert len(after["attempts"]) == len(before["attempts"]) + 1
    assert after["physical_ticks"] == TOTAL_TICKS > before["physical_ticks"]
    assert after["updates"] == TOTAL_TICKS // TICKS_PER_UPDATE > before["updates"]
    final_checkpoint = Path(after["checkpoint"])
    final_record = _json(final_checkpoint / "record.json")
    final_marker = verify_identity(frozen, final_record, final_checkpoint)
    assert final_marker["phase"] == "experiment_complete"
    assert final_marker["threads"] == 1
    final = _state(final_checkpoint)
    # Model weights, optimizer moments/steps, policy and ambient RNG, simulator,
    # sampler, validation state and raw episode history all match the reference.
    _assert_exact_state(final, expected)
    assert sum(final["optimizations"].values()) > sum(partial["optimizations"].values())
    assert any(final_record["changed_weights"].values())
    assert final["sampling_layout"] == {"num_envs": 1, "sampling_processes": 0}
    attempt = Path(after["attempts"][-1]["run_dir"])
    evaluation = _json(attempt / "evaluation/tuning-final.json")
    cases = json.loads(frozen.evaluation_json)
    assert [
        (row["case_id"], row["replication"], row["seed"]) for row in evaluation
    ] == [(case["case"], case["replication"], case["seed"]) for case in cases]
    assert len(evaluation) == 1 and not evaluation[0]["engineering_failure"]
    summary = _json(attempt / "evaluation/summary.json")
    assert summary["requested"] == 1 and summary["exceptions"] == 0
    assert summary["completed"] + summary["truncated"] == 1
    assert {path: path.read_bytes() for path in immutable} == immutable
