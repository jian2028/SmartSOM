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
import textwrap
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

OWNERSHIP_HOOK = r"""
import fcntl, inspect, json, os, subprocess, threading, time
from pathlib import Path
import psutil

journal = Path(os.environ['SMARTSOM_TEST_OWNERSHIP_JOURNAL'])
closed = journal.with_suffix('.closed')
overflow = journal.with_suffix('.overflow')
descriptor_mutex = threading.Lock()
active_descriptors = set()


def record(pid):
    try:
        born = psutil.Process(pid).create_time()
    except psutil.NoSuchProcess:
        return True
    row = (json.dumps([pid, born]) + '\n').encode()
    with descriptor_mutex:
        descriptor = os.open(journal, os.O_WRONLY | os.O_APPEND)
        active_descriptors.add(descriptor)
    try:
        deadline = time.monotonic() + 1
        while True:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    return False
                time.sleep(.01)
        if os.fstat(descriptor).st_size + len(row) > 1048576:
            overflow.touch()
            return False
        while row:
            row = row[os.write(descriptor, row):]
    finally:
        with descriptor_mutex:
            os.close(descriptor)
            active_descriptors.discard(descriptor)
    return True


if not record(os.getpid()) or closed.exists():
    os._exit(125)


def before_fork():
    # Protect only descriptor bookkeeping, never the duration of writer I/O.
    # A writer can therefore be released by the parent after fork returns.
    try:
        if not descriptor_mutex.acquire(timeout=1):
            os._exit(125)
    except BaseException:
        os._exit(125)


def after_parent():
    try:
        descriptor_mutex.release()
    except BaseException:
        os._exit(125)


def after_fork():
    global descriptor_mutex, active_descriptors
    try:
        # flock follows the inherited open-file description. Drop every child
        # copy before opening a new one, so vanished writer threads cannot keep
        # their parent's lock held forever in this child.
        for descriptor in active_descriptors:
            os.close(descriptor)
        active_descriptors = set()
        descriptor_mutex = threading.Lock()
        if not record(os.getpid()) or closed.exists():
            os._exit(125)
    except BaseException:
        # Python ignores exceptions from at-fork callbacks; never let a failed
        # registration fall through into unregistered user code.
        os._exit(125)


os.register_at_fork(before=before_fork, after_in_parent=after_parent,
                    after_in_child=after_fork)
original = subprocess.Popen
signature = inspect.signature(original.__init__)
hook_directory = str(journal.parent / 'ownership-hook')


class TrackedPopen(original):
    def __init__(self, *args, **kwargs):
        if closed.exists():
            raise RuntimeError('owned command cleanup has started')
        bound = signature.bind(self, *args, **kwargs)
        environment = bound.arguments.get('env')
        if environment is not None:
            environment = dict(environment)
            paths = environment.get('PYTHONPATH', '').split(os.pathsep)
            if hook_directory not in paths:
                paths.insert(0, hook_directory)
            environment['PYTHONPATH'] = os.pathsep.join(paths)
            environment['SMARTSOM_TEST_OWNERSHIP_JOURNAL'] = str(journal)
            bound.arguments['env'] = environment
        super().__init__(*bound.args[1:], **bound.kwargs)
        try:
            if not record(self.pid):
                raise RuntimeError('owned command registration failed')
        except BaseException:
            # Creation has happened already: retain this exact Popen ownership
            # while failing registration, including native executables.
            self.terminate()
            try:
                self.wait(timeout=1)
            except subprocess.TimeoutExpired:
                self.kill()
                self.wait(timeout=1)
            raise


subprocess.Popen = TrackedPopen
"""


class _OwnedProcesses(dict):
    def __init__(self, identities, registry):
        super().__init__(identities)
        self.registry = registry
        self.registry_offset = 0
        self.registry_pending = False
        self.registry_corrupt = False


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
        # Writers serialize appends; consume only complete lines without waiting
        # on their lock. A stopped writer must not block bounded cleanup of the
        # identities already committed before it acquired that lock.
        with registry.open("rb") as stream:
            stream.seek(owned.registry_offset)
            data = stream.read(1048577)
        boundary = data.rfind(b"\n") + 1
        owned.registry_pending = boundary != len(data)
        owned.registry_offset += boundary
        records = []
        for line in data[:boundary].splitlines():
            try:
                pid, born = json.loads(line)
                if (
                    type(pid) is not int
                    or pid <= 0
                    or not isinstance(born, (int, float))
                    or born <= 0
                ):
                    raise ValueError("invalid process identity")
                records.append((pid, born))
            except (ValueError, TypeError):
                owned.registry_corrupt = True
        for pid, born in records:
            if (pid, born) in owned:
                continue
            try:
                process = psutil.Process(pid)
                if process.create_time() == born:
                    owned[(pid, born)] = process
            except psutil.NoSuchProcess:
                pass
    # children(recursive=True) builds a whole-host ppid map on macOS. Repeating
    # it for every Ray member makes this external observer consume almost a CPU,
    # starving the actual broker on small runners. Read one fresh host table and
    # walk only descendants of exact, still-matching owned identities.
    children = {}
    for process in psutil.process_iter(["ppid"]):
        try:
            children.setdefault(process.info["ppid"], []).append(process.pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    pending = []
    for (pid, born), process in owned.items():
        try:
            if process.is_running() and psutil.Process(pid).create_time() == born:
                pending.append((pid, born))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    visited = {pid for pid, _ in pending}
    while pending:
        parent, expected_parent_birth = pending.pop()
        for pid in children.get(parent, ()):
            if pid in visited:
                continue
            visited.add(pid)
            try:
                child, ancestor = psutil.Process(pid), psutil.Process(parent)
                born = child.create_time()
                if (
                    child.ppid() == parent
                    and ancestor.create_time() == expected_parent_birth
                ):
                    owned[(pid, born)] = child
                    pending.append((pid, born))
            except (psutil.NoSuchProcess, psutil.AccessDenied):
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
def _command(
    directory, environment, *arguments, register_exit_children=False, startup_hook=""
):
    import psutil

    directory.mkdir()
    stdout, stderr = directory / "stdout.log", directory / "stderr.log"
    registry = directory / "owned-processes.jsonl" if register_exit_children else None
    launch = [*CLI, *map(str, arguments)]
    observer_started, observer_cpu = time.monotonic(), time.process_time()
    if registry is not None:
        # A private inherited startup hook registers each Python generation before
        # user code. Creator registration also covers direct native subprocesses.
        # Keep the real CLI invocation intact instead of replacing its main module.
        registry.touch(mode=0o600)
        hook = directory / "ownership-hook"
        hook.mkdir(mode=0o700)
        startup = (
            "import os\ntry:\n"
            + textwrap.indent(OWNERSHIP_HOOK, "    ")
            + "\n"
            + textwrap.indent(startup_hook, "    ")
            + "\nexcept BaseException:\n"
            + "    import traceback; traceback.print_exc(); os._exit(125)\n"
        )
        # Python ordinarily ignores sitecustomize errors. Fail closed instead of
        # allowing an unregistered generation to run, and validate before launch.
        compile(startup, "sitecustomize.py", "exec")
        (hook / "sitecustomize.py").write_text(startup)
        environment = dict(environment)
        environment["SMARTSOM_TEST_OWNERSHIP_JOURNAL"] = str(registry)
        environment["PYTHONPATH"] = os.pathsep.join(
            [str(hook), environment.get("PYTHONPATH", "")]
        )
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
            # The inherited journal still supplies exact descendant identities.
            pass
        owned = _OwnedProcesses(identities, registry)
        try:
            yield process, owned, stdout, stderr
        finally:
            if registry is not None:
                registry.with_suffix(".closed").touch()
            # Close registration's launch gate, then drain identities repeatedly:
            # children born during shutdown must not escape a one-shot snapshot.
            deadline = time.monotonic() + 10
            while True:
                _remember_children(owned)
                remaining = _alive(owned)
                if not remaining and not owned.registry_pending:
                    break
                for child in reversed(remaining):
                    try:
                        if child in _alive(owned):
                            if time.monotonic() < deadline:
                                child.terminate()
                            else:
                                child.kill()
                    except psutil.NoSuchProcess:
                        pass
                psutil.wait_procs(remaining, timeout=0.1)
                if not remaining:
                    time.sleep(0.01)
                if time.monotonic() >= deadline + 2:
                    break
            process.wait(timeout=10)
            _remember_children(owned)
            remaining = _alive(owned)
            (directory / "observer-summary.json").write_text(
                json.dumps(
                    {
                        "pid": os.getpid(),
                        "birth": psutil.Process().create_time(),
                        "wall_seconds": time.monotonic() - observer_started,
                        "parent_cpu_seconds": time.process_time() - observer_cpu,
                        "identities_observed": len(owned),
                        "remaining": [[p.pid, p.create_time()] for p in remaining],
                        "journal_pending": owned.registry_pending,
                        "journal_corrupt": owned.registry_corrupt,
                    }
                )
            )
            assert not remaining, "exact owned processes survived bounded cleanup"
            assert not owned.registry_pending, (
                "owned identity journal remained incomplete"
            )
            assert not owned.registry_corrupt, (
                "owned identity journal contained invalid rows"
            )
            if registry is not None:
                assert not registry.with_suffix(".overflow").exists(), (
                    "owned process identity journal exceeded 1 MiB"
                )


def _wait(command, predicate, *, timeout=180, allow_exit=False):
    process, owned, stdout, stderr = command
    deadline = time.monotonic() + timeout
    next_scan = 0.0
    while True:
        # Journal registration is independent of observation. Retain 50 Hz
        # ancestry observation for commands without it; journalled commands need
        # only 4 Hz whole-host scans and an immediate scan at every boundary.
        scanned = False
        if getattr(owned, "registry", None) is None or time.monotonic() >= next_scan:
            _remember_children(owned)
            next_scan = time.monotonic() + 0.25
            scanned = True
        result = predicate()
        if result:
            if not scanned:
                _remember_children(owned)
                result = predicate()
            if result and not getattr(owned, "registry_pending", False):
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


@pytest.mark.parametrize(
    "phase,exit_kind",
    [
        ("preflight", "stop"),
        ("preflight", "interrupt"),
        ("preflight", "exception"),
        ("training", "stop"),
    ],
)
def test_public_live_queue_stop_waits_for_adopted_driver_and_preserves_commit(
    tmp_path, phase, exit_kind
):
    _require_cpu()
    from smartsom.experiments import control
    from smartsom.experiments.tuning_session import verify_commit

    directory, output = _batch_inputs(tmp_path)
    barrier = tmp_path / "smoke-entered"
    environment = {
        **os.environ,
        "PYTHONPATH": str(ROOT / "src"),
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "SMARTSOM_TEST_SMOKE_BARRIER": str(barrier),
    }
    hook = (
        r"""
import time
from pathlib import Path
from smartsom.experiments import preflight
from smartsom.experiments.control import StopRequested
original_smoke = preflight.smoke

def slow_smoke(prepared, **kwargs):
    Path(os.environ['SMARTSOM_TEST_SMOKE_BARRIER']).touch()
    while not kwargs['cancelled']():
        time.sleep(.02)
    raise StopRequested('controlled slow smoke reached cooperative stop')

preflight.smoke = slow_smoke
"""
        if phase == "preflight"
        else ""
    )
    with _command(
        tmp_path / "source-cli",
        environment,
        "batch-run",
        directory,
        "--no-background",
        *DISPLAY,
        register_exit_children=True,
        startup_hook=hook,
    ) as source_command:
        root = _wait(source_command, lambda: _allocated(output))
        if phase == "preflight":
            _wait(source_command, barrier.is_file)
            assert _json(root / "batch.json")["files"]["tiny"]["status"] == "queued"
            checkpoint = None
        else:
            _, _, entry = _wait(source_command, lambda: _entry(root))
            assert 0 < entry["physical_ticks"] < TOTAL_TICKS
            checkpoint = Path(entry["checkpoint"])
            verify_commit(checkpoint)
        original_plan = (root / "plan.json").read_bytes()
        original_owner = control.read(root)
        commit_bytes = (checkpoint / "commit.json").read_bytes() if checkpoint else None
        queue_hook = (
            r"""
from pathlib import Path
from smartsom.experiments import live_queue
original_apply_requests = live_queue.apply_requests

def injected_failure(root, state):
    if (Path(root) / 'inject-error').exists():
        raise RuntimeError('injected queue polling failure')
    return original_apply_requests(root, state)

live_queue.apply_requests = injected_failure
"""
            if exit_kind == "exception"
            else ""
        )
        with _command(
            tmp_path / "queue-cli",
            environment,
            "batch-run",
            directory,
            "--live-queue",
            "--adopt-run",
            root,
            "--max-concurrent",
            "1",
            "--no-background",
            *DISPLAY,
            register_exit_children=True,
            startup_hook=queue_hook,
        ) as queue_command:

            def allocated_queue():
                paths = list(root.parent.glob("*-live-queue-*/batch.json"))
                return paths[0].parent if paths else None

            queue = _wait(queue_command, allocated_queue)
            _wait(
                queue_command,
                lambda: _json(queue / "batch.json")["status"] == "running",
            )
            assert control.read(root)["id"] == original_owner["id"]
            assert control.read(root)["owner"] == original_owner["owner"]
            if exit_kind == "stop":
                with _command(
                    tmp_path / "queue-stop-cli",
                    environment,
                    "stop",
                    queue,
                    "--timeout",
                    "60",
                ) as stop_command:
                    stopped = _finish(stop_command)
                assert stopped["status"] == "stopped" and not stopped["remaining"]
                assert _finish(queue_command)["status"] == "stopped"
            else:
                if exit_kind == "interrupt":
                    import signal

                    queue_command[0].send_signal(signal.SIGINT)
                else:
                    (queue / "inject-error").touch()
                _wait(
                    queue_command,
                    lambda: queue_command[0].poll() is not None,
                    allow_exit=True,
                )
                assert queue_command[0].returncode == (
                    130 if exit_kind == "interrupt" else 1
                )
                _wait(
                    queue_command, lambda: not _alive(queue_command[1]), allow_exit=True
                )
            assert _finish(source_command, code=1)["status"] == "stopped"
            assert not _alive(source_command[1]) and not _alive(queue_command[1])
            assert _json(root / "control/stop.json")["id"] == original_owner["id"]
            assert _json(queue / "batch.json")["status"] == (
                "failed" if exit_kind == "exception" else "stopped"
            )
            assert (root / "plan.json").read_bytes() == original_plan
            if checkpoint:
                verify_commit(checkpoint)
                assert (checkpoint / "commit.json").read_bytes() == commit_bytes
            if checkpoint:
                # The drained queue releases ownership; ordinary source resume
                # still restores its own committed optimizer and workflow state.
                with _command(
                    tmp_path / "original-resume-cli",
                    environment,
                    "resume",
                    root,
                    "--no-background",
                    *DISPLAY,
                    register_exit_children=True,
                ) as resume_command:
                    assert _finish(resume_command)["status"] == "completed"
                assert _entry(root)[2]["physical_ticks"] == TOTAL_TICKS


@pytest.mark.parametrize("preflight_level", ["quick", "full"])
def test_real_original_and_appended_native_job_share_admission_from_preflight(
    tmp_path, monkeypatch, preflight_level
):
    _require_cpu()
    from smartsom.experiments import background, control, live_queue, queue_admission

    directory, output = _batch_inputs(tmp_path)
    source_document = yaml.safe_load((directory / "tiny.yaml").read_text())
    source_document["execution"]["preflight"] = preflight_level
    (directory / "tiny.yaml").write_text(
        yaml.safe_dump(source_document, sort_keys=False)
    )
    barrier, release = tmp_path / "before-tune", tmp_path / "release-tune"
    environment = {
        **os.environ,
        "PYTHONPATH": str(ROOT / "src"),
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "SMARTSOM_TEST_QUEUE_BARRIER": str(barrier),
        "SMARTSOM_TEST_QUEUE_RELEASE": str(release),
    }
    monkeypatch.setenv("PYTHONPATH", str(ROOT / "src"))
    metrics_trace = tmp_path / "queue-metrics.jsonl"
    metrics_hook = tmp_path / "queue-metrics-hook"
    metrics_hook.mkdir()
    # Observe the actual two-root sampler and its public snapshot without
    # replacing measured counters, resource probes or scheduling decisions.
    (metrics_hook / "sitecustomize.py").write_text(
        r"""
import json
import os
from pathlib import Path
from smartsom.telemetry import execution_metrics, runtime
original_fields = execution_metrics.resource_fields
original_publish = runtime.RuntimeDisplay.publish
last_queue_fields = None

def fields(snapshot, **kwargs):
    global last_queue_fields
    result = original_fields(snapshot, **kwargs)
    roots = kwargs.get('roots') or []
    if len(roots) == 2:
        last_queue_fields = ([str(root) for root in roots], result)
    return result

def publish(view, **kwargs):
    result = original_publish(view, **kwargs)
    if view.metrics_producer == 'live-queue':
        assert view.execution_telemetry.resource_monitor is None, 'duplicate resource sampler'
        assert view.execution_telemetry.throughput.last is None, 'duplicate throughput sampler'
        if last_queue_fields and view.tuning:
            roots, expected = last_queue_fields
            actual = view.tuning['resources']
            for name in ('owned_cpu_cores', 'owned_cpu_status', 'cpus_used'):
                assert actual.get(name) == expected.get(name), 'queue metric overwritten: ' + name
            with Path(os.environ['SMARTSOM_QUEUE_METRICS_TRACE']).open('a') as stream:
                stream.write(json.dumps({'roots': roots, 'owned_cpu_cores': actual.get('owned_cpu_cores')}) + '\n')
    return result

execution_metrics.resource_fields = fields
runtime.RuntimeDisplay.publish = publish
"""
    )
    monkeypatch.setenv("SMARTSOM_QUEUE_METRICS_TRACE", str(metrics_trace))
    monkeypatch.setenv("PYTHONPATH", str(metrics_hook) + os.pathsep + str(ROOT / "src"))
    hook = r"""
import time
from pathlib import Path
from smartsom.experiments import author_batch
from smartsom.experiments.control import boundary
original_run_tune = author_batch._run_tune

def held_tune(root, *args, **kwargs):
    Path(os.environ['SMARTSOM_TEST_QUEUE_BARRIER']).touch()
    while not Path(os.environ['SMARTSOM_TEST_QUEUE_RELEASE']).exists():
        boundary(root, fresh=True)
        time.sleep(.02)
    return original_run_tune(root, *args, **kwargs)

author_batch._run_tune = held_tune
"""
    with _command(
        tmp_path / "source-cli",
        environment,
        "batch-run",
        directory,
        "--no-background",
        *DISPLAY,
        register_exit_children=True,
        startup_hook=hook,
    ) as source_command:
        root = _wait(source_command, lambda: _allocated(output))
        _wait(source_command, barrier.is_file)
        original = (root / "plan.json").read_bytes()
        queue = live_queue.adopt(root, limit=1)
        added = tmp_path / "later-inputs"
        added.mkdir()
        document = yaml.safe_load((directory / "tiny.yaml").read_text())
        for key in ("factory", "workload", "algorithm"):
            document[key] = str((directory / document[key]).resolve())
        (added / "later.yaml").write_text(yaml.safe_dump(document, sort_keys=False))
        live_queue.append(queue, added)
        started = background.launch(
            queue, display_options={"progress": "off", "verbose": False}
        )
        assert started["startup_status"] == "ready"
        samples, telemetry_samples = [], []
        try:
            _wait(
                source_command,
                lambda: "native:later" in queue_admission.snapshot(root)["slots"],
            )
            assert _json(root / "batch.json")["files"]["tiny"]["status"] == "running"
            # The original file label is coarse, but it has no allocation yet.
            assert (
                live_queue.read(queue / "batch.json")["jobs"]["tiny"]["status"]
                == "queued"
            )
            release.touch()

            def drained():
                slots = queue_admission.snapshot(root)["slots"]
                samples.append(dict(slots))
                progress = queue / "logs/progress.json"
                if progress.is_file():
                    telemetry_samples.append(_json(progress))
                assert len(slots) <= 1, slots
                return all(
                    job["status"] == "completed"
                    for job in live_queue.read(queue / "batch.json")["jobs"].values()
                )

            _wait(source_command, drained, timeout=180, allow_exit=True)
            assert _finish(source_command)["status"] == "completed"
            assert _entry(root)[2]["physical_ticks"] == TOTAL_TICKS
            assert (
                live_queue.read(queue / "jobs/later/batch.json")["status"]
                == "completed"
            )
            assert any("native:later" in s for s in samples)
            assert telemetry_samples
            metric_rows = [
                json.loads(line) for line in metrics_trace.read_text().splitlines()
            ]
            assert metric_rows and all(
                row["roots"] == [str(queue), str(root)] for row in metric_rows
            )
            assert any(row["owned_cpu_cores"] is not None for row in metric_rows)
            assert (
                control.read(queue)["owner"]["pid"]
                != control.read(root)["owner"]["pid"]
            )
            assert any(
                task.get("execution", {}).get("training_ticks", 0) > 0
                for sample in telemetry_samples
                for task in sample.get("tasks", [])
                if task["id"].startswith("tiny::")
            ), "adopted Tune observations must survive directory and queue projection"
            assert any(
                task.get("execution", {}).get("training_ticks", 0) > 0
                for sample in telemetry_samples
                for task in sample.get("tasks", [])
                if task["id"].startswith("later::")
            ), "native entry observations must survive its parent driver"
            assert not any(
                task.get("execution", {}).get("reason")
                for sample in telemetry_samples
                for task in sample.get("tasks", [])
                if task.get("execution", {}).get("training_ticks", 0) > 0
            )
            for prefix in ("tiny::", "later::"):
                plans = [
                    task["execution"]["plan"]
                    for sample in telemetry_samples
                    for task in sample.get("tasks", [])
                    if task["id"].startswith(prefix)
                    and task.get("execution", {}).get("plan")
                ]
                assert plans and all(plan.get("eta_group") for plan in plans)
                assert all(
                    plan.get("validation_horizons") and plan.get("evaluation_horizons")
                    for plan in plans
                )
            assert any(
                sample.get("tuning", {}).get("resources", {}).get("cpus_used")
                is not None
                for sample in telemetry_samples
            )
            assert any(
                sample.get("tuning", {}).get("resources", {}).get("owned_cpu_cores")
                is not None
                for sample in telemetry_samples
            )
            assert any(
                sample.get("tuning", {}).get("training_ticks_per_second") is not None
                for sample in telemetry_samples
            )
            assert any(
                sample.get("tuning", {}).get("total_ticks_per_second") is not None
                for sample in telemetry_samples
            )
            assert all(
                sample.get("execution", {}).get("batch", {}).get("open_queue")
                for sample in telemetry_samples
                if sample.get("execution")
            )
            assert any("source:tiny__entry-0001" in s for s in samples)
            if preflight_level == "full":
                assert any(
                    slot.get("status") == "preflight"
                    for sample in samples
                    for key, slot in sample.items()
                    if key.startswith("source:")
                )
                tune_root = root / _json(root / "batch.json")["tune_directory"]
                full = _json(tune_root / "preflight.json")
                assert full["level"] == "full" and full["smoke_done"] == 1
            original_owner = (root / "control/owner.json").read_bytes()
            original_claim = (root / "control/admission.json").read_bytes()
            original_state = (root / "batch.json").read_bytes()
            for background_mode in ("--no-background", "--background"):
                with _command(
                    tmp_path / ("refused-resume" + background_mode),
                    environment,
                    "resume",
                    root,
                    background_mode,
                    *DISPLAY,
                    register_exit_children=True,
                ) as refused:
                    _wait(
                        refused, lambda: refused[0].poll() is not None, allow_exit=True
                    )
                    assert refused[0].returncode == 2
                    _wait(refused, lambda: not _alive(refused[1]), allow_exit=True)
                assert (root / "control/owner.json").read_bytes() == original_owner
                assert (root / "control/admission.json").read_bytes() == original_claim
                assert (root / "batch.json").read_bytes() == original_state
            # Rejected resumes leave this still-open queue fully controllable.
            live_queue.close(queue)
            _wait(
                source_command,
                lambda: live_queue.read(queue / "batch.json")["status"] == "completed",
                allow_exit=True,
            )
            assert not control.stop(queue, timeout=60)["remaining"]
            assert "queue" not in queue_admission.snapshot(root)
            assert (root / "plan.json").read_bytes() == original
        finally:
            if control.read(queue):
                assert not control.stop(queue, timeout=60)["remaining"]


def test_real_native_resume_final_seeds_completion_and_distinguishes_initial_control(
    tmp_path,
):
    _require_cpu()
    directory, output = _batch_inputs(tmp_path)
    path = directory / "tiny.yaml"
    document = yaml.safe_load(path.read_text())
    document["execution"].update(
        executor="native", tuning="off", preflight="quick", background=False
    )
    document["training"].update(total_ticks=16, ticks_per_update=8, record_initial=True)
    document["validation"]["every_updates"] = 1
    document["evaluation"]["initial_control"] = True
    path.write_text(yaml.safe_dump(document, sort_keys=False))
    trace, stopped = tmp_path / "phase-events.jsonl", tmp_path / "stopped-before-final"
    environment = {
        **os.environ,
        "PYTHONPATH": str(ROOT / "src"),
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "SMARTSOM_PHASE_TRACE": str(trace),
        "SMARTSOM_PHASE_STOP": str(stopped),
    }
    hook = r"""
import json
from pathlib import Path
from smartsom.experiments import composable
from smartsom.experiments.control import StopRequested
from smartsom.telemetry.runtime import CURRENT
from smartsom.telemetry.execution_metrics import source_execution
real_evaluate = composable.evaluate
def observed_evaluate(*args, **kwargs):
    marker = Path(os.environ['SMARTSOM_PHASE_STOP'])
    if not marker.exists():
        marker.touch()
        raise StopRequested('engineering stop before final evaluation')
    view = CURRENT.get()
    before = source_execution(view.snapshot())
    result = real_evaluate(*args, **kwargs)
    after = source_execution(view.snapshot())
    with Path(os.environ['SMARTSOM_PHASE_TRACE']).open('a') as stream:
        stream.write(json.dumps({'purpose': kwargs.get('purpose'), 'before': before, 'after': after}) + '\n')
    return result
composable.evaluate = observed_evaluate
"""
    with _command(
        tmp_path / "native-stop",
        environment,
        "run",
        path,
        "--no-background",
        *DISPLAY,
        register_exit_children=True,
        startup_hook=hook,
    ) as command:
        result = _finish(command, code=1)
    root = Path(result["run_directory"])
    ledger_path = root / "entries/entry-0001/stages.json"
    ledger = _json(ledger_path)
    assert ledger["stages"]["training"]["status"] == "completed"
    training = Path(ledger["stages"]["training"]["run_dir"])
    immutable = {
        file: file.read_bytes()
        for file in [
            training / "run.json",
            *training.glob("checkpoints/**/continuation.pkl"),
        ]
    }
    with _command(
        tmp_path / "native-resume",
        environment,
        "resume",
        root,
        "--no-background",
        *DISPLAY,
        register_exit_children=True,
        startup_hook=hook,
    ) as command:
        assert _finish(command)["status"] == "completed"
    rows = [json.loads(line) for line in trace.read_text().splitlines()]
    final, initial = rows
    before = final["before"]
    assert before["physical_ticks"] == 16
    assert before["training_ticks"] == before["execution_ticks"] == 0
    assert (
        before["estimate"]["phase_work"]["training"]
        == before["estimate"]["phase_work"]["validation"]
        == 0
    )
    assert before["estimate"]["phase_work"]["evaluation"] > 0
    assert initial["purpose"] == "initial_control"
    final_work = final["after"]["phases"]["evaluation"]["work"]
    assert final_work > 0
    assert initial["after"]["phases"]["evaluation"]["work"] == final_work
    assert initial["after"]["phases"]["initial"]["work"] > 0
    assert (
        initial["after"]["execution_ticks"]
        == final_work + initial["after"]["phases"]["initial"]["work"]
    )
    snapshot = _json(root / "entries/entry-0001/logs/progress.json")["execution"][
        "experiment"
    ]
    assert snapshot["phases"]["finalization"]["mean_seconds"] > 0
    assert snapshot["estimate"]["seconds"] == 0
    assert {file: file.read_bytes() for file in immutable} == immutable


def test_real_rule_directory_publishes_current_cpu_and_execution_throughput(tmp_path):
    _require_cpu()
    directory, output = _batch_inputs(tmp_path)
    algorithm = {
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
    }
    (directory.parent / "algorithm.yaml").write_text(
        yaml.safe_dump(algorithm, sort_keys=False)
    )
    path = directory / "tiny.yaml"
    document = yaml.safe_load(path.read_text())
    document.update(task="evaluate")
    document.pop("training")
    document["validation"] = {"enabled": False}
    document["execution"].update(
        executor="native", tuning="off", preflight="quick", background=False
    )
    path.write_text(yaml.safe_dump(document, sort_keys=False))
    barrier = tmp_path / "case-observed"
    environment = {
        **os.environ,
        "PYTHONPATH": str(ROOT / "src"),
        "SMARTSOM_RULE_METRIC_BARRIER": str(barrier),
    }
    hook = r"""
import time
from pathlib import Path
from smartsom.telemetry import runtime
original_emit = runtime.emit
def held_case(task, event, **kwargs):
    result = original_emit(task, event, **kwargs)
    marker = Path(os.environ['SMARTSOM_RULE_METRIC_BARRIER'])
    view = runtime.CURRENT.get()
    if view and view.kind == 'evaluation' and event.get('evaluation_tick') == 0 and not marker.exists():
        marker.touch()
        time.sleep(6.1)
    return result
runtime.emit = held_case
"""
    samples = []
    with _command(
        tmp_path / "rules-cli",
        environment,
        "batch-run",
        directory,
        "--calibration-level",
        "off",
        "--no-background",
        *DISPLAY,
        register_exit_children=True,
        startup_hook=hook,
    ) as command:
        root = _wait(command, lambda: _allocated(output))
        _wait(command, barrier.is_file)

        def completed():
            progress = root / "logs/progress.json"
            if progress.is_file():
                samples.append(_json(progress))
            return _json(root / "batch.json")["status"] == "completed"

        _wait(command, completed, timeout=60, allow_exit=True)
        assert _finish(command)["status"] == "completed"
    assert any(
        sample.get("tuning", {}).get("resources", {}).get("cpus_used") is not None
        for sample in samples
    )
    assert any(
        sample.get("tuning", {}).get("resources", {}).get("owned_cpu_cores") is not None
        for sample in samples
    )
    assert any(
        (sample.get("tuning", {}).get("total_ticks_per_second") or 0) > 0
        for sample in samples
    )
    assert all(
        sample["tuning"]["native_stage"] and not sample["tuning"].get("online")
        for sample in samples
        if sample.get("tuning")
    )


@pytest.mark.parametrize("explicit_stages", [False, True])
def test_real_two_stage_live_adoption_drains_last_trial_before_next_smoke(
    tmp_path, monkeypatch, explicit_stages
):
    _require_cpu()
    from smartsom.experiments import background, control, live_queue, queue_admission

    directory, output = _batch_inputs(tmp_path)
    first = yaml.safe_load((directory / "tiny.yaml").read_text())
    second = copy.deepcopy(first)
    second["seed"] = 102
    if explicit_stages:
        first["batch"] = dict(stage=0)
        second["batch"] = dict(stage=1)
    (directory / "tiny.yaml").write_text(yaml.safe_dump(first, sort_keys=False))
    (directory / "zz-next.yaml").write_text(yaml.safe_dump(second, sort_keys=False))
    barrier, release = tmp_path / "before-tune", tmp_path / "release-tune"
    environment = {
        **os.environ,
        "PYTHONPATH": str(ROOT / "src"),
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "SMARTSOM_TEST_QUEUE_BARRIER": str(barrier),
        "SMARTSOM_TEST_QUEUE_RELEASE": str(release),
    }
    monkeypatch.setenv("PYTHONPATH", str(ROOT / "src"))
    hook = r"""
import time
from pathlib import Path
from smartsom.experiments import author_batch
from smartsom.experiments.control import boundary
original_run_tune = author_batch._run_tune

def held_tune(root, *args, **kwargs):
    Path(os.environ['SMARTSOM_TEST_QUEUE_BARRIER']).touch()
    while not Path(os.environ['SMARTSOM_TEST_QUEUE_RELEASE']).exists():
        boundary(root, fresh=True)
        time.sleep(.02)
    return original_run_tune(root, *args, **kwargs)

author_batch._run_tune = held_tune
"""
    with _command(
        tmp_path / "source-cli",
        environment,
        "batch-run",
        directory,
        "--no-background",
        *DISPLAY,
        register_exit_children=True,
        startup_hook=hook,
    ) as command:
        root = _wait(command, lambda: _allocated(output))
        _wait(command, barrier.is_file)
        assert len(_json(root / "batch.json")["tune_directories"]) == 2
        frozen = (root / "plan.json").read_bytes()
        queue = live_queue.adopt(root, limit=1)
        background.launch(queue, display_options={"progress": "off", "verbose": False})
        samples = []
        try:
            release.touch()

            def drained():
                slots = queue_admission.snapshot(root)["slots"]
                samples.append(dict(slots))
                assert len(slots) <= 1, slots
                return all(
                    row["status"] == "completed"
                    for row in live_queue.read(queue / "batch.json")["jobs"].values()
                )

            _wait(command, drained, timeout=180, allow_exit=True)
            assert _finish(command)["status"] == "completed"
            state = _json(root / "batch.json")
            for stage, file_id in [("0", "tiny"), ("1", "zz-next")]:
                tune_root = root / state["tune_directories"][stage]
                entry = _json(tune_root / "batch.json")["entries"][
                    file_id + "__entry-0001"
                ]
                assert (
                    entry["status"] == "completed"
                    and entry["physical_ticks"] == TOTAL_TICKS
                )
                from smartsom.experiments.tuning_session import verify_commit

                verify_commit(entry["checkpoint"])
            assert any("source:tiny__entry-0001" in s for s in samples)
            assert any("source:zz-next__entry-0001" in s for s in samples)
            assert not queue_admission.snapshot(root)["slots"]
            live_queue.close(queue)
            _wait(
                command,
                lambda: live_queue.read(queue / "batch.json")["status"] == "completed",
                allow_exit=True,
            )
            assert not control.stop(queue, timeout=60)["remaining"]
            assert (root / "plan.json").read_bytes() == frozen
        finally:
            assert not control.stop(queue, timeout=60)["remaining"]
