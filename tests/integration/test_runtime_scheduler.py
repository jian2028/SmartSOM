"""Real spawned worker lifecycle acceptance; CI owner: control.

The counter workers exercise scheduler/process/checkpoint mechanics, not PPO or DQN.
Their pause boundary is an atomic persisted counter update.
"""

import json
import os
import time
from pathlib import Path

import pytest

from smartsom.experiments import control, queue_admission
from smartsom.experiments import runtime_control as runtime
from smartsom.experiments.runtime_workers import NativeScheduler
from smartsom.experiments.tuning_resources import ResourceSnapshot


@pytest.fixture
def author_input(tmp_path):
    import yaml

    source = Path(__file__).resolve().parents[2] / "configs/base-diagnostic"
    directory = tmp_path / "author"
    directory.mkdir()
    documents = {
        key: yaml.safe_load((source / name).read_text())
        for key, name in (
            ("factory", "small-factory.yaml"),
            ("workload", "small-workload.yaml"),
            ("algorithm", "rule.yaml"),
            ("experiment", "small.yaml"),
        )
    }
    documents["workload"].update(orders=4, initial_orders=2)
    documents["experiment"].update(
        factory="factory.yaml",
        workload="workload.yaml",
        algorithm="algorithm.yaml",
        tracking={"mode": "disabled"},
        run={
            "task": "smoke",
            "smoke_ticks": 20,
            "methods": ["rule"],
            "output": str(tmp_path / "runs" / "rule"),
        },
    )
    for key, value in documents.items():
        (directory / (key + ".yaml")).write_text(yaml.safe_dump(value))
    return directory / "experiment.yaml", documents


def _counter_worker(directory, parent_root):
    root = Path(directory)
    scope = control.Scope()
    scope.inherit_driver(parent_root)
    scope.bind(root)
    token = control.CURRENT.set(scope)
    checkpoint = root / "counter.json"
    previous = (
        json.loads(checkpoint.read_text())
        if checkpoint.exists()
        else {"counter": 0, "generations": []}
    )
    previous["generations"].append({"pid": os.getpid(), "initial": previous["counter"]})
    status = "failed"
    try:
        while True:
            previous["counter"] += 1
            previous["pid"] = os.getpid()
            control.write_json(checkpoint, previous)
            control.boundary(root, fresh=True)
            if (root / "complete").exists():
                status = "completed"
                break
            time.sleep(0.025)
    except control.PauseRequested:
        if (root / "delay-pause").exists():
            control.write_json(
                root / "pause-boundary.json", {"counter": previous["counter"]}
            )
            deadline = time.monotonic() + 10
            while not (root / "release-pause").exists():
                if control.requested(root, fresh=True):
                    status = "stopped"
                    return
                if time.monotonic() > deadline:
                    raise OSError("test checkpoint release deadline expired")
                time.sleep(0.02)
        if (root / "fail-save").exists():
            raise OSError("injected checkpoint write failure") from None
        status = "paused"
    except control.StopRequested:
        status = "stopped"
    finally:
        control.write_json(root / "batch.json", {"status": status})
        scope.finish(status)
        control.CURRENT.reset(token)


def _until(scheduler, predicate, timeout=30):
    deadline = time.monotonic() + timeout
    state = {}
    while time.monotonic() < deadline:
        state = scheduler.tick()
        if predicate(state):
            return state
        time.sleep(0.04)
    pytest.fail(f"scheduler deadline: {state}")


def _read(root, task):
    return json.loads((root / task / "counter.json").read_text())


@pytest.fixture
def scheduler_factory(tmp_path):
    created = []

    def make(count, limit, *, observer=None, keep_open=False):
        root = tmp_path / f"batch-{len(created)}"
        root.mkdir()
        runtime.initialize(root, limit, keep_open=keep_open)
        scope = control.Scope()
        scope.bind(root)
        queue_admission.initialize(root, supported=True)
        jobs = {}
        for i in range(count):
            task = f"t{i}"
            child = root / task
            child.mkdir()
            jobs[task] = {"root": str(child), "status": "queued"}
        scheduler = NativeScheduler(
            root, jobs, worker=_counter_worker, observer=observer
        )
        created.append((scheduler, scope))
        return scheduler

    yield make
    for scheduler, scope in created:
        runtime.submit(scheduler.root, "stop")
        _until(scheduler, lambda state: not scheduler.processes)
        scope.finish("stopped")


def test_live_four_six_two_keeps_existing_processes_and_progress(scheduler_factory):
    class IdleHost:
        def snapshot(self):
            return ResourceSnapshot(
                cpus=8, memory_total=16 * 1024**3, memory_available=14 * 1024**3
            )

    # Exercise operator concurrency with deterministic capacity; resource pressure
    # and refusal have separate tests using the same unchanged admission broker.
    scheduler = scheduler_factory(8, 4, observer=IdleHost())
    # These sleeping counters have an explicitly lightweight declared shape.
    # Six processes do not represent six full-CPU learners on this host.
    for job in scheduler.jobs.values():
        job["cpus"] = 0.25
    root = scheduler.root
    _until(
        scheduler,
        lambda state: (
            state["active"] == 4
            and all((root / f"t{i}" / "counter.json").exists() for i in range(4))
        ),
    )
    original = {f"t{i}": _read(root, f"t{i}") for i in range(4)}
    runtime.submit(root, "limit", value=6)
    _until(
        scheduler,
        lambda state: (
            state["active"] == 6
            and all((root / f"t{i}" / "counter.json").exists() for i in range(6))
        ),
    )
    for task, prior in original.items():
        current = _read(root, task)
        assert current["pid"] == prior["pid"]
        assert current["counter"] >= prior["counter"]
        assert len(current["generations"]) == 1
    runtime.submit(root, "limit", value=2)
    state = scheduler.tick()
    assert state["active"] == 6
    assert state["draining"] and state["status"] == "draining"
    for i in range(4):
        (root / f"t{i}" / "complete").touch()
    _until(scheduler, lambda state: state["active"] == 2)
    assert scheduler.jobs["t6"]["status"] == "queued"
    assert scheduler.jobs["t7"]["status"] == "queued"
    for i in (4, 5):
        assert len(_read(root, f"t{i}")["generations"]) == 1
    for i in range(8):
        (root / f"t{i}" / "complete").touch()
    _until(scheduler, lambda state: not scheduler.unfinished)
    assert runtime.finish_if_drained(root, unfinished=scheduler.unfinished)


def test_task_pause_refills_resume_queues_global_resume_keeps_manual_hold(
    scheduler_factory,
):
    scheduler = scheduler_factory(3, 1)
    root = scheduler.root
    _until(scheduler, lambda state: (root / "t0/counter.json").exists())
    original = _read(root, "t0")
    runtime.submit(root, "pause", task_id="t0")
    _until(
        scheduler,
        lambda state: (
            scheduler.jobs["t0"]["status"] == "paused"
            and (root / "t1/counter.json").exists()
        ),
    )
    paused = _read(root, "t0")
    assert paused["counter"] >= original["counter"]
    runtime.submit(root, "pause")
    _until(scheduler, lambda state: state["active"] == 0)
    assert scheduler.unfinished
    assert not runtime.finish_if_drained(root, unfinished=scheduler.unfinished)
    assert not (root / "t2/counter.json").exists()
    runtime.submit(root, "resume")
    _until(
        scheduler,
        lambda state: (
            (root / "t1/counter.json").exists()
            and len(_read(root, "t1")["generations"]) == 2
        ),
    )
    assert scheduler.jobs["t0"]["status"] == "paused"
    active_pid = _read(root, "t1")["pid"]
    runtime.submit(root, "resume", task_id="t0")
    scheduler.tick()
    assert scheduler.jobs["t0"]["status"] == "queued"
    assert _read(root, "t1")["pid"] == active_pid
    (root / "t1/complete").touch()
    _until(scheduler, lambda state: len(_read(root, "t0")["generations"]) == 2)
    resumed = _read(root, "t0")
    assert resumed["generations"][-1]["initial"] == paused["counter"]
    assert resumed["counter"] > paused["counter"]


def test_stopped_task_does_not_resume_with_batch_and_drain_can_exit(scheduler_factory):
    scheduler = scheduler_factory(2, 1)
    root = scheduler.root
    _until(scheduler, lambda state: (root / "t0/counter.json").exists())
    runtime.submit(root, "stop", task_id="t0")
    _until(
        scheduler,
        lambda state: (
            scheduler.jobs["t0"]["status"] == "stopped"
            and (root / "t1/counter.json").exists()
        ),
    )
    stopped = _read(root, "t0")
    runtime.submit(root, "resume")
    scheduler.tick()
    assert scheduler.jobs["t0"]["status"] == "stopped"
    assert _read(root, "t0") == stopped
    (root / "t1/complete").touch()
    _until(scheduler, lambda state: not scheduler.unfinished)
    assert runtime.finish_if_drained(root, unfinished=scheduler.unfinished)
    assert runtime.snapshot(root)["tasks"]["t0"]["status"] == "stopped"
    assert runtime.snapshot(root)["status"] == "partial_stopped"


class _NoMemory:
    def snapshot(self):
        return ResourceSnapshot(
            cpus=8, memory_total=8 * 1024**3, memory_available=128 * 1024**2
        )


def test_resource_wait_reports_truth_and_keep_open_is_explicit(scheduler_factory):
    scheduler = scheduler_factory(1, 4, observer=_NoMemory())
    scheduler.tick()
    state = scheduler.tick()
    assert state["active"] == 0
    assert "headroom" in state["blocking_reason"]
    assert state["tasks"]["t0"]["status"] == "queued"
    idle = scheduler_factory(0, 1, keep_open=True)
    idle.tick()
    assert not runtime.finish_if_drained(idle.root, unfinished=idle.unfinished)
    runtime.submit(idle.root, "close")
    assert runtime.finish_if_drained(idle.root, unfinished=idle.unfinished)


def test_only_individually_paused_work_stays_controllable(scheduler_factory):
    scheduler = scheduler_factory(1, 1)
    root = scheduler.root
    _until(scheduler, lambda state: (root / "t0/counter.json").exists())
    runtime.submit(root, "pause", task_id="t0")
    state = _until(scheduler, lambda state: scheduler.jobs["t0"]["status"] == "paused")
    assert state["active"] == 0
    assert state["status"] == "paused"
    assert scheduler.unfinished
    assert not runtime.finish_if_drained(root, unfinished=scheduler.unfinished)
    paused = _read(root, "t0")
    runtime.submit(root, "resume", task_id="t0")
    _until(scheduler, lambda state: len(_read(root, "t0")["generations"]) == 2)
    assert _read(root, "t0")["generations"][-1]["initial"] == paused["counter"]


def test_remove_saves_running_task_and_preserves_tombstone_on_restart(
    scheduler_factory,
):
    scheduler = scheduler_factory(2, 1)
    root = scheduler.root
    _until(scheduler, lambda state: (root / "t0/counter.json").exists())
    before = _read(root, "t0")
    target = runtime.removal_target(root, "t0")
    runtime.submit(
        root,
        "remove",
        task_id="t0",
        confirmation=target["confirmation"],
        expected_revision=target["revision"],
    )
    _until(
        scheduler,
        lambda state: (
            scheduler.jobs["t0"]["status"] == "removed"
            and (root / "t1/counter.json").exists()
        ),
    )
    assert _read(root, "t0")["counter"] >= before["counter"]
    assert (root / "t0/batch.json").is_file()
    rejected = runtime.submit(root, "resume", task_id="t0")
    state = runtime.apply(root)
    assert state["commands"][rejected["id"]]["status"] == "rejected"
    (root / "t1/complete").touch()
    _until(scheduler, lambda state: not scheduler.unfinished)
    assert runtime.finish_if_drained(root, unfinished=scheduler.unfinished)
    assert runtime.snapshot(root)["status"] == "partial_removed"
    runtime.restart(root)
    assert "t0" in runtime.snapshot(root)["removed_tasks"]
    assert scheduler.tick()["tasks"]["t0"]["status"] == "removed"


def test_failed_checkpoint_never_reports_safe_pause(scheduler_factory):
    scheduler = scheduler_factory(1, 1)
    root = scheduler.root
    _until(scheduler, lambda state: (root / "t0/counter.json").exists())
    (root / "t0/fail-save").touch()
    runtime.submit(root, "pause", task_id="t0")
    state = _until(scheduler, lambda state: scheduler.jobs["t0"]["status"] == "failed")
    assert state["active"] == 0
    assert state["tasks"]["t0"]["status"] != "paused"
    assert state["tasks"]["t0"]["exit_code"] != 0
    assert _read(root, "t0")["counter"] > 0


def test_public_batch_default_control_append_frozen_rules_and_keep_open(
    author_input, tmp_path
):
    """Run the public CLI and actual rule experiments, without a replacement worker."""
    import copy
    import subprocess
    import sys

    import yaml

    path, documents = author_input
    source = Path(__file__).resolve().parents[2]
    inputs = path.parent / "initial"
    additions = path.parent / "additional"
    inputs.mkdir()
    additions.mkdir()
    for folder, name, seed in (
        (inputs, "original", 9001),
        (additions, "added", 9002),
        (additions, "removed", 9003),
    ):
        document = copy.deepcopy(documents["experiment"])
        document.update(
            factory="../factory.yaml",
            workload="../workload.yaml",
            algorithm="../algorithm.yaml",
        )
        document["run"]["seed"] = seed
        (folder / f"{name}.yaml").write_text(yaml.safe_dump(document))
    cli = [sys.executable, "-m", "smartsom.experiments.cli"]
    env = dict(os.environ, PYTHONPATH=str(source / "src"), PYTHONDONTWRITEBYTECODE="1")
    log = tmp_path / "public-runtime.log"
    parent = None

    def poll(predicate, timeout=45):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            value = predicate()
            if value:
                return value
            if process.poll() is not None:
                pytest.fail(
                    f"public driver exited {process.returncode}: {log.read_text()}"
                )
            time.sleep(0.1)
        pytest.fail(f"public driver timeout: {log.read_text()}")

    def command(*arguments):
        result = subprocess.run(
            cli + list(map(str, arguments)),
            cwd=source,
            env=env,
            text=True,
            capture_output=True,
            timeout=15,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        return result

    with log.open("w") as output:
        process = subprocess.Popen(
            cli
            + [
                "batch-run",
                str(inputs),
                "--keep-open",
                "--foreground",
                "--progress",
                "off",
                "--no-verbose",
                "--log-format",
                "json",
            ],
            cwd=source,
            env=env,
            stdout=output,
            stderr=subprocess.STDOUT,
        )
        try:
            parent = poll(
                lambda: next(
                    Path(documents["experiment"]["run"]["output"]).parent.glob(
                        "*/control/runtime.json"
                    ),
                    None,
                )
            ).parents[1]
            poll(
                lambda: (
                    runtime.snapshot(parent)
                    .get("tasks", {})
                    .get("original", {})
                    .get("status")
                    == "completed"
                )
            )
            assert process.poll() is None
            frozen = (parent / "plan.json").read_bytes()
            command("batch-limit", parent, 2)
            poll(lambda: runtime.snapshot(parent)["limit"] == 2)
            command("batch-pause", parent)
            poll(lambda: runtime.snapshot(parent)["paused"])
            command("batch-add", parent, additions, "--command-id", "append-once")
            poll(
                lambda: (
                    runtime.snapshot(parent)
                    .get("tasks", {})
                    .get("added", {})
                    .get("status")
                    == "paused"
                )
            )
            added_root = Path(runtime.snapshot(parent)["additions"]["added"]["root"])
            assert added_root.is_dir()
            # Delete original author files after append has frozen all inputs.
            (additions / "added.yaml").unlink()
            command("batch-add", parent, additions, "--command-id", "append-once")
            command("batch-pause", parent, "--task", "added")
            poll(lambda: "added" in runtime.snapshot(parent)["paused_tasks"])
            target = runtime.removal_target(parent, "removed")
            runtime.submit(
                parent,
                "remove",
                task_id="removed",
                confirmation=target["confirmation"],
                expected_revision=target["revision"],
            )
            poll(
                lambda: (
                    runtime.snapshot(parent)
                    .get("tasks", {})
                    .get("removed", {})
                    .get("status")
                    == "removed"
                )
            )
            removed_root = Path(
                runtime.snapshot(parent)["additions"]["removed"]["root"]
            )
            assert (removed_root / "plan.json").exists()
            command("batch-resume", parent)
            poll(lambda: not runtime.snapshot(parent)["paused"])
            state = runtime.snapshot(parent)
            assert state["tasks"]["added"]["status"] == "paused"
            assert state["status"] == "paused"
            assert process.poll() is None
            command("batch-resume", parent, "--task", "added")
            poll(
                lambda: (
                    runtime.snapshot(parent)
                    .get("tasks", {})
                    .get("added", {})
                    .get("status")
                    == "completed"
                )
            )
            assert (
                json.loads((added_root / "batch.json").read_text())["status"]
                == "completed"
            )
            assert (parent / "plan.json").read_bytes() == frozen
            assert process.poll() is None  # keep-open survives both empty queues.
            command("batch-close", parent)
            assert process.wait(timeout=30) == 0, log.read_text()
            assert runtime.snapshot(parent)["status"] == "partial_removed"
        finally:
            if process.poll() is None and parent:
                runtime.submit(parent, "stop")
                process.wait(timeout=30)


def test_public_batch_default_drains_without_live_queue(author_input):
    import copy
    import subprocess
    import sys

    import yaml

    path, documents = author_input
    folder = path.parent / "default-batch"
    folder.mkdir()
    document = copy.deepcopy(documents["experiment"])
    document.update(
        factory="../factory.yaml",
        workload="../workload.yaml",
        algorithm="../algorithm.yaml",
    )
    (folder / "ordinary.yaml").write_text(yaml.safe_dump(document))
    source = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "smartsom.experiments.cli",
            "batch-run",
            str(folder),
            "--foreground",
            "--progress",
            "off",
            "--no-verbose",
            "--log-format",
            "json",
        ],
        cwd=source,
        env=dict(os.environ, PYTHONPATH=str(source / "src")),
        text=True,
        capture_output=True,
        timeout=45,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    record = next(Path(document["run"]["output"]).parent.glob("*/control/runtime.json"))
    state = json.loads(record.read_text())
    assert state["status"] == "completed"
    assert not state["keep_open"]
    assert state["tasks"]["ordinary"]["status"] == "completed"


def test_native_adaptive_budget_ceiling_preserves_operator_target(tmp_path):
    from smartsom.experiments.evidence import write_json

    root = tmp_path / "adaptive"
    root.mkdir()
    runtime.initialize(root, 6, mode="adaptive")
    scope = control.Scope()
    scope.bind(root)
    queue_admission.initialize(root, supported=True)
    child = root / "one"
    child.mkdir()
    write_json(child / "batch.json", {"status": "queued"})
    scheduler = NativeScheduler(
        root,
        {"one": {"root": str(child), "status": "queued", "peak_memory": 512 * 1024**2}},
        worker=_counter_worker,
        observer=_NoMemory(),
    )
    try:
        runtime.submit(root, "budget", value={"cpus": 2, "memory": 1024**3})
        state = scheduler.tick()
        assert state["limit"] == 6 and state["native_effective_limit"] == 0
        assert state["native_adaptation"] == "resource_headroom"
        assert scheduler.peak_memory["one"] == 512 * 1024**2
        runtime.submit(root, "mode", value="manual")
        state = scheduler.tick()
        assert state["limit"] == state["native_effective_limit"] == 6
        assert not scheduler.processes
        assert "headroom" in state["blocking_reason"]
    finally:
        scope.finish("stopped")


def test_native_adaptive_completed_and_impossible_jobs_do_not_block_small_work(
    scheduler_factory,
):
    scheduler = scheduler_factory(3, 6)
    scheduler.jobs["t0"].update(status="completed", cpus=999, peak_memory=10**15)
    scheduler.jobs["t1"]["cpus"] = 999
    runtime.submit(scheduler.root, "mode", value="adaptive")
    runtime.submit(scheduler.root, "budget", value={"cpus": 2, "memory": 4 * 1024**3})
    state = _until(scheduler, lambda state: "t2" in scheduler.processes)
    assert state["limit"] == 6
    assert state["native_effective_limit"] > 0
    assert scheduler.jobs["t1"]["status"] == "queued"
    assert "cannot fit" in scheduler.jobs["t1"]["blocking_reason"]
    assert "t0" not in scheduler.processes and "t1" not in scheduler.processes


def test_native_external_pressure_marks_large_waiter_infeasible(scheduler_factory):
    class Busy:
        def snapshot(self):
            return ResourceSnapshot(
                cpus=8,
                memory_total=32 * 1024**3,
                memory_available=24 * 1024**3,
                cpu_load=80,
            )

    scheduler = scheduler_factory(1, 2, observer=Busy())
    scheduler.jobs["t0"]["cpus"] = 4
    scheduler.tick()
    shared = queue_admission.snapshot(scheduler.root)
    assert shared["waiters"]["native:t0"]["feasible"] is False
    assert queue_admission.reserve(
        scheduler.root,
        "source:small",
        owner_id=shared["owner_id"],
        cpus=1,
        memory=320 * 1024**2,
    )
    queue_admission.retire(scheduler.root, "source:small", owner_id=shared["owner_id"])


def test_native_held_status_uses_explicit_child_owner(tmp_path):
    runtime.initialize(tmp_path, 2)
    scheduler = NativeScheduler(
        tmp_path,
        {
            "foo": {
                "root": str(tmp_path / "foo"),
                "status": "running",
                "own_slots": True,
            }
        },
    )
    state = runtime.publish(
        tmp_path, active=0, tasks={"foo__unrelated": {"status": "paused"}}
    )
    assert scheduler._persist(state)["status"] != "paused"
    state = runtime.publish(
        tmp_path,
        active=0,
        tasks={
            "inner": {"status": "paused", "file_id": "foo"},
            "foo__unrelated": {"status": "running"},
        },
    )
    assert scheduler._persist(state)["status"] == "paused"


def test_ready_large_worker_gets_capacity_before_continuing_small_arrivals(
    scheduler_factory, monkeypatch
):
    """Real workers drain; a controlled clock avoids a 30-second aging sleep."""
    from types import SimpleNamespace

    class IdleHost:
        def snapshot(self):
            return ResourceSnapshot(
                cpus=8, memory_total=16 * 1024**3, memory_available=14 * 1024**3
            )

    age = [0.0]
    monkeypatch.setattr(
        queue_admission,
        "time",
        SimpleNamespace(monotonic=lambda: age[0], sleep=time.sleep),
    )
    scheduler = scheduler_factory(2, 12, observer=IdleHost())
    root = scheduler.root
    runtime.submit(root, "budget", value={"cpus": 4, "memory": 12 * 1024**3})
    _until(
        scheduler,
        lambda _: all((root / k / "counter.json").exists() for k in ("t0", "t1")),
    )
    # The prerequisite completes before its dependent work becomes runnable.
    (root / "t0/complete").touch()
    _until(scheduler, lambda _: scheduler.jobs["t0"]["status"] == "completed")
    large = root / "large"
    large.mkdir()
    scheduler.jobs["large"] = {"root": str(large), "status": "queued", "cpus": 4}
    scheduler.tick()
    waiting = queue_admission.snapshot(root)["waiters"]["native:large"]
    assert waiting["feasible"]
    assert not (large / "counter.json").exists()
    age[0] = 31.0
    # Put every new small job before the large one to defeat accidental dict-order
    # fairness. Each could fit beside t1 without the aged shared reservation.
    for index in range(6):
        key = f"new-{index}"
        child = root / key
        child.mkdir()
        jobs = dict(scheduler.jobs)
        scheduler.jobs.clear()
        scheduler.jobs[key] = {"root": str(child), "status": "queued", "cpus": 1}
        scheduler.jobs.update(jobs)
        scheduler.tick()
        assert set(scheduler.processes) == {"t1"}
        age[0] += 1
    before = _read(root, "t1")
    (root / "t1/complete").touch()
    _until(scheduler, lambda _: (large / "counter.json").exists())
    assert scheduler.jobs["t1"]["status"] == "completed"
    assert _read(root, "t1")["counter"] >= before["counter"]
    assert set(scheduler.processes) == {"large"}
    assert queue_admission.snapshot(root)["slots"]["native:large"]["cpus"] == 4
    assert all(not (root / f"new-{i}/counter.json").exists() for i in range(6))


def test_mixed_source_and_native_large_waiter_ages_across_shared_owned_load(
    scheduler_factory, monkeypatch
):
    """Real workers with controlled host readings; no hardware throughput claim."""
    from types import SimpleNamespace

    import psutil

    class Host:
        busy = True

        def snapshot(self):
            return ResourceSnapshot(
                cpus=8,
                memory_total=16 * 1024**3,
                memory_available=12 * 1024**3,
                cpu_load=37.5 if self.busy else 0,
            )

    clock = [0.0]
    monkeypatch.setattr(
        queue_admission,
        "time",
        SimpleNamespace(monotonic=lambda: clock[0], sleep=time.sleep),
    )
    host = Host()
    scheduler = scheduler_factory(0, 12, observer=host)
    root = scheduler.root
    runtime.submit(root, "budget", value={"cpus": 7, "memory": 12 * 1024**3})
    runtime.apply(root)
    owner = queue_admission.snapshot(root)["owner_id"]
    source = root / "source"
    source.mkdir()
    assert queue_admission.reserve(
        root, "source:held", owner_id=owner, cpus=3, memory=320 * 1024**2
    )
    process = scheduler.context.Process(
        target=_counter_worker, args=(str(source), str(root))
    )
    process.start()
    try:
        _until(scheduler, lambda _: (source / "counter.json").exists())
        identity = psutil.Process(process.pid)

        def observe_source():
            queue_admission.update_resources(
                root,
                "source:held",
                owner_id=owner,
                memory=320 * 1024**2,
                observed_memory=identity.memory_info().rss,
                observed_cpus=3,
                observed_processes=[(identity.pid, identity.create_time())],
            )

        observe_source()
        large = root / "large"
        large.mkdir()
        scheduler.jobs["large"] = {"root": str(large), "status": "queued", "cpus": 5}
        scheduler.tick()
        assert queue_admission.snapshot(root)["waiters"]["native:large"]["feasible"]
        assert not (large / "counter.json").exists()
        clock[0] = 31
        for index in range(4):
            observe_source()
            key = f"small-{index}"
            child = root / key
            child.mkdir()
            old = dict(scheduler.jobs)
            scheduler.jobs.clear()
            scheduler.jobs[key] = {"root": str(child), "status": "queued", "cpus": 1}
            scheduler.jobs.update(old)
            scheduler.tick()
            assert not scheduler.processes
            assert not queue_admission.reserve(
                root,
                f"source:new-{index}",
                owner_id=owner,
                cpus=1,
                memory=320 * 1024**2,
                host=host.snapshot(),
            )
            assert (
                queue_admission.snapshot(root)["waiters"]["native:large"]["since"] == 0
            )
            clock[0] += 1
        before = json.loads((source / "counter.json").read_text())
        (source / "complete").touch()
        process.join(timeout=10)
        assert process.exitcode == 0
        assert (
            json.loads((source / "counter.json").read_text())["counter"]
            >= before["counter"]
        )
        queue_admission.retire(root, "source:held", owner_id=owner)
        host.busy = False
        scheduler.tick()
        assert set(scheduler.processes) == {"large"}
        assert all(not (root / f"small-{i}/counter.json").exists() for i in range(4))
        _until(scheduler, lambda _: (large / "counter.json").exists())
    finally:
        (source / "complete").touch()
        process.join(timeout=10)
        queue_admission.retire(root, "source:held", owner_id=owner)


def test_native_resource_dto_clears_current_after_real_worker_exit(scheduler_factory):
    scheduler = scheduler_factory(1, 1)
    root = scheduler.root
    initial = scheduler.tick()["tasks"]["t0"]["resources"]
    assert initial["cpu_cores"] is None and initial["memory"] is None
    _until(
        scheduler,
        lambda state: state["tasks"]["t0"]["resources"]["status"] == "current",
    )
    sampled = runtime.snapshot(root)["tasks"]["t0"]["resources"]
    assert sampled["memory"] > 0
    assert sampled["memory_peak"] >= sampled["memory"]
    assert sampled["observed_at"] is not None
    (root / "t0/complete").touch()
    state = _until(
        scheduler, lambda state: state["tasks"]["t0"]["status"] == "completed"
    )
    final = state["tasks"]["t0"]["resources"]
    assert final["status"] == "inactive"
    assert final["cpu_cores"] is None and final["memory"] is None
    assert final["memory_peak"] >= sampled["memory_peak"]


def test_selected_resume_waits_for_global_continue_and_real_resource_guard(
    scheduler_factory,
):
    scheduler = scheduler_factory(1, 1)
    root = scheduler.root
    _until(scheduler, lambda _: (root / "t0/counter.json").exists())
    runtime.submit(root, "pause", task_id="t0")
    _until(scheduler, lambda state: state["tasks"]["t0"]["status"] == "paused")
    checkpoint = _read(root, "t0")
    host_observer = scheduler.observer
    scheduler.observer = _NoMemory()
    runtime.submit(root, "pause")
    runtime.submit(root, "resume", task_id="t0")
    state = scheduler.tick()
    assert state["paused"]
    assert "t0" not in state["paused_tasks"]
    assert state["active"] == 0
    assert state["tasks"]["t0"]["status"] == "paused"
    runtime.submit(root, "resume")
    state = scheduler.tick()
    assert state["tasks"]["t0"]["status"] == "queued"
    assert state["active"] == 0
    assert "headroom" in state["blocking_reason"]
    assert _read(root, "t0") == checkpoint
    scheduler.observer = host_observer
    _until(scheduler, lambda _: len(_read(root, "t0")["generations"]) == 2)
    assert _read(root, "t0")["generations"][-1]["initial"] == checkpoint["counter"]


@pytest.mark.parametrize("save_fails", [False, True])
def test_delayed_checkpoint_progress_is_observed_and_global_targets_stay_fixed(
    scheduler_factory, save_fails
):
    from types import SimpleNamespace

    # Controlled host readings isolate checkpoint waiting from unrelated host load.
    observer = SimpleNamespace(
        snapshot=lambda: ResourceSnapshot(8, 16 * 1024**3, 14 * 1024**3)
    )
    scheduler = scheduler_factory(2, 2, observer=observer)
    root = scheduler.root
    _until(
        scheduler,
        lambda _: all((root / f"t{i}/counter.json").exists() for i in range(2)),
    )
    (root / "t0/delay-pause").touch()
    if save_fails:
        (root / "t0/fail-save").touch()
    receipt = runtime.submit(root, "pause")
    _until(
        scheduler,
        lambda state: (
            (root / "t0/pause-boundary.json").exists()
            and state["tasks"]["t1"]["status"] == "paused"
        ),
    )
    initial = runtime.snapshot(root)["commands"][receipt["id"]]
    assert initial["status"] == "applied"
    assert initial["progress"]["state"] == "waiting"
    assert initial["progress"]["targets"] == ["t0", "t1"]
    assert initial["progress"]["completed_count"] == 1
    assert initial["progress"]["tasks"]["t0"]["phase"] == "safe_pausing"
    assert initial["progress"]["remaining_seconds"] is None
    repeated = runtime.submit(root, "pause")
    assert repeated["id"] == receipt["id"]
    assert runtime.snapshot(root)["revision"] == receipt["revision"]
    late = root / "late"
    late.mkdir()
    with runtime.transaction(root) as state:
        state["additions"]["late"] = {"root": str(late), "status": "queued"}
    state = _until(
        scheduler,
        lambda state: (
            state["commands"][receipt["id"]]["progress"]["elapsed_seconds"]
            > initial["progress"]["elapsed_seconds"] + 0.1
        ),
    )
    assert state["commands"][receipt["id"]]["progress"]["total_count"] == 2
    assert not (late / "counter.json").exists()
    (root / "t0/release-pause").touch()
    state = _until(
        scheduler,
        lambda state: (
            state["commands"][receipt["id"]]["progress"]["state"] != "waiting"
        ),
    )
    progress = state["commands"][receipt["id"]]["progress"]
    assert progress["state"] == ("failed" if save_fails else "completed")
    assert progress["completed_count"] == (1 if save_fails else 2)
    assert progress["failed_count"] == int(save_fails)
    assert progress["finished_at"] >= progress["started_at"]
    assert state["tasks"]["t0"]["exit_code"] == (1 if save_fails else 0)
    assert state["active"] == 0


def test_manual_priority_launches_next_tasks_without_restarting_active_worker(
    scheduler_factory,
):
    from types import SimpleNamespace

    observer = SimpleNamespace(
        snapshot=lambda: ResourceSnapshot(8, 16 * 1024**3, 14 * 1024**3)
    )
    scheduler = scheduler_factory(4, 1, observer=observer)
    root = scheduler.root
    _until(scheduler, lambda state: (root / "t0/counter.json").exists())
    pid = _read(root, "t0")["pid"]
    runtime.submit(root, "priority", task_id="t3", value="add")
    runtime.submit(root, "priority", task_id="t2", value="add")
    scheduler.tick()
    assert _read(root, "t0")["pid"] == pid
    assert not (root / "t3/counter.json").exists()
    (root / "t0/complete").touch()
    _until(scheduler, lambda state: (root / "t3/counter.json").exists())
    assert not (root / "t1/counter.json").exists()
    assert not (root / "t2/counter.json").exists()
    (root / "t3/complete").touch()
    _until(scheduler, lambda state: (root / "t2/counter.json").exists())
    assert not (root / "t1/counter.json").exists()
