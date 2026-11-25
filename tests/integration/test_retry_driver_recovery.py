"""Isolated real driver crash/recovery with a bounded synthetic worker."""

import json
import multiprocessing
import os
from pathlib import Path

import pytest

from smartsom.experiments import author_batch, control
from smartsom.experiments import runtime_control as runtime
from smartsom.experiments.runtime_workers import NativeScheduler
from smartsom.experiments.tuning_resources import ResourceSnapshot
from tests.integration.test_runtime_scheduler import (
    _controlled_retry_counter_worker,
    _until,
)
from tests.unit.test_incremental_batch import batch


def _producer_crash(root_text, boundary):
    os.setsid()  # Restrict cleanup to this test driver's own process session.
    from types import SimpleNamespace

    from smartsom.experiments import queue_admission

    root = Path(root_text)
    runtime._RETRY_ENABLED = True
    scope = control.Scope()
    token = control.CURRENT.set(scope)
    scheduler = None
    try:
        scope.bind(root)
        _, plan, state = author_batch.load(root)
        author_batch._runtime_initialize(root, plan, state, keep_open=True)
        queue_admission.initialize(root, supported=True)
        selected = state["selection"]["tasks"][0]
        row = next(item for item in plan["files"] if item["id"] == selected)
        child = root / row["child"]
        (child / "trial.lock").touch()
        observer = SimpleNamespace(
            snapshot=lambda: ResourceSnapshot(8, 16 * 1024**3, 14 * 1024**3)
        )
        scheduler = NativeScheduler(
            root,
            {selected: {"root": str(child), "status": "queued"}},
            worker=_controlled_retry_counter_worker,
            observer=observer,
        )
        _until(scheduler, lambda _: scheduler.jobs[selected]["status"] == "failed")
        assert not scheduler.processes
        receipt = runtime.submit(
            root, "retry", task_id=selected, command_id="crash-retry"
        )
        assert receipt["status"] == "received"
        if boundary in {"applied_queued", "spawned_unregistered"}:
            scheduler._consume_retries(runtime.snapshot(root))
            observed = runtime.snapshot(root)
            assert observed["commands"]["crash-retry"]["status"] == "applied"
            assert observed["tasks"][selected]["status"] == "queued"
        runtime.submit(root, "close", command_id="close-for-test")
        state["status"] = "failed"
        state["files"][selected]["status"] = "failed"
        control.write_json(root / "batch.json", state)
        control.write_json(
            root / "crash-fixture.json",
            {"selected": selected, "owner_id": scope.id, "boundary": boundary},
        )
        if boundary == "spawned_unregistered":
            import time

            real_context = multiprocessing.get_context("spawn")

            class CrashAfterSpawn:
                def Process(self, **kwargs):
                    process = real_context.Process(**kwargs)

                    class Proxy:
                        def start(self):
                            process.start()
                            deadline = time.monotonic() + 10
                            while time.monotonic() < deadline:
                                if (child / "fixture-held-ready.json").exists():
                                    os._exit(24)
                                time.sleep(0.02)
                            raise RuntimeError(
                                "test worker did not register before deadline"
                            )

                    return Proxy()

            scheduler.context = CrashAfterSpawn()
            scheduler.worker = _held_retry_worker
            scheduler.tick()
            raise RuntimeError("spawn crash injection was not reached")
    except BaseException as exc:
        control.write_json(root / "producer-error.json", {"error": repr(exc)})
        if scheduler is not None:
            scheduler.close()
        scope.finish("failed")
        control.CURRENT.reset(token)
        raise
    # All workers are joined. Simulate a driver crash without finalizing its scope.
    os._exit(23)


def _held_retry_worker(directory, parent_root, retry_once=False):
    """Bounded orphan fixture; parent releases it after checking the recovery gate."""
    import time

    root = Path(directory)
    scope = control.Scope()
    scope.inherit_driver(parent_root)
    scope.bind(root)
    try:
        owner = control.read(root)["owner"]
        control.write_json(
            root / "fixture-held-ready.json",
            {"identity": owner, "retry_once": retry_once},
        )
        deadline = time.monotonic() + 90
        while (
            not (root / "fixture-held-release").exists() and time.monotonic() < deadline
        ):
            time.sleep(0.02)
    finally:
        scope.finish("stopped")


def _recover_driver(root_text):
    os.setsid()  # Restrict cleanup to this test driver's own process session.
    from smartsom.experiments import author_batch

    root = Path(root_text)
    runtime._RETRY_ENABLED = True
    author_batch._quiet_worker = _controlled_retry_counter_worker
    scope = control.Scope()
    token = control.CURRENT.set(scope)
    status = "failed"
    try:
        result = author_batch.execute_saved(root)
        control.write_json(
            root / "recovered-fixture.json", {"result": result, "owner_id": scope.id}
        )
        status = result.get("status", "failed")
    except BaseException as exc:
        control.write_json(root / "recover-error.json", {"error": repr(exc)})
        raise
    finally:
        scope.finish(status)
        control.CURRENT.reset(token)


def _join_process(process, timeout=60):
    process.join(timeout)
    if process.is_alive():
        import signal

        # A live multiprocessing child plus matching session ID confines the
        # signal to the session that this test explicitly created.
        try:
            if os.getsid(process.pid) == process.pid:
                os.killpg(process.pid, signal.SIGTERM)
            else:
                process.terminate()
        except ProcessLookupError:
            pass  # Child exited after is_alive(); it still must be joined.
        process.join(10)
        pytest.fail(
            "isolated test driver exceeded its deadline; inspect its dedicated test session"
        )


@pytest.mark.skipif(
    os.name != "posix", reason="isolated session crash test requires POSIX"
)
@pytest.mark.parametrize("boundary", ["received", "applied_queued"])
def test_new_driver_recovers_retry_after_real_driver_crash(tmp_path, boundary):
    """Real frozen batch/driver rebind and synthetic finite worker, no training."""
    import hashlib

    from smartsom.experiments.launch import compile_launch

    folder = batch(tmp_path)
    compiled = compile_launch(
        folder,
        ids=["medium-base"],
        first_train_seeds=1,
        operations={"output_root": str(tmp_path / "results")},
    )
    root = Path(author_batch.allocate(compiled))
    plan_before = hashlib.sha256((root / "plan.json").read_bytes()).hexdigest()
    context = multiprocessing.get_context("spawn")
    producer = context.Process(target=_producer_crash, args=(str(root), boundary))
    producer.start()
    _join_process(producer)
    assert producer.exitcode == 23, (
        (root / "producer-error.json").read_text()
        if (root / "producer-error.json").exists()
        else producer.exitcode
    )
    first = json.loads((root / "crash-fixture.json").read_text())
    runner = context.Process(target=_recover_driver, args=(str(root),))
    runner.start()
    _join_process(runner)
    assert runner.exitcode == 0, (
        (root / "recover-error.json").read_text()
        if (root / "recover-error.json").exists()
        else runner.exitcode
    )
    recovered = json.loads((root / "recovered-fixture.json").read_text())
    assert recovered["owner_id"] != first["owner_id"]
    assert hashlib.sha256((root / "plan.json").read_bytes()).hexdigest() == plan_before
    saved = json.loads((root / "batch.json").read_text())
    selected = first["selected"]
    assert saved["selection"]["tasks"] == [selected]
    assert saved["files"][selected]["status"] == "completed"
    assert all(
        value["status"] == "deferred"
        for key, value in saved["files"].items()
        if key != selected
    )
    queue = runtime.snapshot(root)
    assert queue["commands"]["crash-retry"]["status"] == "applied"
    assert queue["commands"]["close-for-test"]["status"] == "applied"
    from smartsom.experiments import queue_admission

    assert not queue_admission.snapshot(root).get("slots")
    assert "previous driver exit is not verified" in control.read(root).get(
        "deletion_blockers", []
    )
    assert len(queue["tasks"][selected]["retry_history"]) == 1
    plan = json.loads((root / "plan.json").read_text())
    row = next(item for item in plan["files"] if item["id"] == selected)
    result = json.loads((root / row["child"] / "resumed.json").read_text())
    assert result == {"prior_ticks": 10, "retry_once": True}


@pytest.mark.skipif(
    os.name != "posix", reason="isolated session crash test requires POSIX"
)
def test_new_driver_blocks_when_retry_worker_spawn_outcome_is_unknown(tmp_path):
    """A live owned child during the start/registration gap must never be duplicated."""
    import time

    from smartsom.experiments import queue_admission
    from smartsom.experiments.launch import compile_launch

    folder = batch(tmp_path)
    compiled = compile_launch(
        folder,
        ids=["medium-base"],
        first_train_seeds=1,
        operations={"output_root": str(tmp_path / "results")},
    )
    root = Path(author_batch.allocate(compiled))
    context = multiprocessing.get_context("spawn")
    producer = context.Process(
        target=_producer_crash, args=(str(root), "spawned_unregistered")
    )
    producer.start()
    _join_process(producer)
    assert producer.exitcode == 24, (
        (root / "producer-error.json").read_text()
        if (root / "producer-error.json").exists()
        else producer.exitcode
    )
    first = json.loads((root / "crash-fixture.json").read_text())
    selected = first["selected"]
    plan = json.loads((root / "plan.json").read_text())
    row = next(item for item in plan["files"] if item["id"] == selected)
    child = root / row["child"]
    held = json.loads((child / "fixture-held-ready.json").read_text())
    assert held["retry_once"] is True
    owner_before = (root / "control/owner.json").read_bytes()
    try:
        runner = context.Process(target=_recover_driver, args=(str(root),))
        runner.start()
        _join_process(runner)
        assert runner.exitcode != 0
        error = json.loads((root / "recover-error.json").read_text())["error"]
        assert "previous retry launch requires worker ownership verification" in error
        assert (root / "control/owner.json").read_bytes() == owner_before
        state = runtime.snapshot(root)
        assert state["tasks"][selected]["status"] == "starting"
        assert state["tasks"][selected]["retry_pending"] is True
        assert "native:" + selected in queue_admission.snapshot(root).get("slots", {})
        assert control.alive(held["identity"], control.processes())
    finally:
        (child / "fixture-held-release").touch()
        deadline = time.monotonic() + 25
        while time.monotonic() < deadline and control.alive(
            held["identity"], control.processes()
        ):
            time.sleep(0.05)
        assert not control.alive(held["identity"], control.processes())


def _fail_then_continue_counter(directory, parent_root, retry_once=False):
    _controlled_retry_counter_worker(
        directory,
        parent_root,
        retry_once=(Path(directory) / "continuation/current.json").exists(),
    )


@pytest.mark.parametrize("selection_count", [1, 2])
def test_ended_driver_retries_only_selected_shared_task(
    tmp_path, monkeypatch, selection_count
):
    """Real native workers and continuation hashes; no learner or long training."""
    from smartsom.experiments import batch_retry
    from smartsom.experiments.cli import main
    from smartsom.experiments.launch import compile_launch

    folder = batch(tmp_path)
    compiled = compile_launch(
        folder,
        ids=["medium-base", "medium-sl", "small-base"],
        first_train_seeds=1,
        operations={
            "output_root": str(tmp_path / "run"),
            "tracking": {"mode": "disabled"},
        },
    )
    root = author_batch.allocate(compiled)
    monkeypatch.setattr(author_batch, "_quiet_worker", _fail_then_continue_counter)

    def execute(**kwargs):
        scope = control.Scope()
        token = control.CURRENT.set(scope)
        try:
            scope.bind(root)
            result = author_batch.execute_saved(root, **kwargs)
            scope.finish(result["status"])
            return result
        finally:
            control.CURRENT.reset(token)

    result = execute()
    assert result["status"] == "failed"
    selected = ["medium-base--seed-4101", "medium-sl--seed-4101"][:selection_count]
    peer_variant = "medium-sl" if selection_count == 1 else "small-base"
    peer = peer_variant + "--seed-4101"
    peer_root = root / "trials" / peer_variant / "seed-4101"
    before = {
        str(p.relative_to(peer_root)): p.read_bytes()
        for p in peer_root.rglob("*")
        if p.is_file()
    }
    preview = batch_retry.inspect_ended_targets(root, selected)
    assert preview[0]["ticks"] == 10
    args = ["retry", str(root), "--foreground", "--memory-headroom-ratio", ".15"]
    for key in selected:
        args.extend(["--task-id", key])
    assert main(args) == 0
    result = json.loads((root / "batch.json").read_text())
    assert all(result["files"][key]["status"] == "completed" for key in selected)
    assert runtime.snapshot(root)["memory_policy"]["memory_headroom_ratio"] == 0.15
    assert result["files"][peer]["status"] == "failed"
    assert {
        str(p.relative_to(peer_root)): p.read_bytes()
        for p in peer_root.rglob("*")
        if p.is_file()
    } == before
    assert (
        json.loads((root / "trials/medium-base/seed-4101/resumed.json").read_text())[
            "prior_ticks"
        ]
        == 10
    )
    assert not (peer_root / "resumed.json").exists()
