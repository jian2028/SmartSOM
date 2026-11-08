"""Bounded real-process remove and cleanup of an isolated fixture batch."""

import hashlib
import io
import json
import multiprocessing
import os
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

import pytest

from smartsom.experiments import author_batch, control, queue_admission
from smartsom.experiments import runtime_control as runtime
from smartsom.experiments.runtime_workers import NativeScheduler
from smartsom.experiments.tuning_resources import ResourceSnapshot
from tests.integration.test_retry_driver_recovery import _join_process
from tests.integration.test_runtime_scheduler import _counter_worker, _until
from tests.unit.test_incremental_batch import batch, compile_batch


def _cleanup_fixture_driver(root_text):
    os.setsid()
    from smartsom.experiments.cli import main

    root = Path(root_text)
    scope = control.Scope()
    token = control.CURRENT.set(scope)
    scheduler = None
    status = "failed"
    try:
        scope.bind(root)
        _, plan, state = author_batch.load(root)
        author_batch._runtime_initialize(root, plan, state, limit=2, keep_open=True)
        queue_admission.initialize(root, supported=True)
        ids = state["selection"]["tasks"]
        rows = [next(row for row in plan["files"] if row["id"] == key) for key in ids]
        children = [root / row["child"] for row in rows]
        preserved = (root / "plan.json", children[0] / "inputs/experiment.yaml")
        preserved_hashes = {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in preserved
        }
        for index, child in enumerate(children):
            (child / "last.pt").write_bytes(f"fixture-{index}".encode())
        jobs = {
            row["id"]: {"root": str(child), "status": "queued"}
            for row, child in zip(rows, children, strict=True)
        }
        observer = SimpleNamespace(
            snapshot=lambda: ResourceSnapshot(8, 16 * 1024**3, 14 * 1024**3)
        )
        scheduler = NativeScheduler(
            root, jobs, worker=_counter_worker, observer=observer
        )
        _until(
            scheduler,
            lambda _: all((child / "counter.json").exists() for child in children),
        )
        first, peer = ids
        peer_before = json.loads((children[1] / "counter.json").read_text())

        def invoke(arguments):
            output = io.StringIO()
            with redirect_stdout(output):
                assert main(arguments) == 0
            return json.loads(output.getvalue())

        target = invoke(["remove", str(root), "--task-id", first, "--show-target"])
        invoke(
            [
                "remove",
                str(root),
                "--task-id",
                first,
                "--batch-id",
                target["batch_id"],
                "--expected-revision",
                str(target["revision"]),
                "--confirm-target",
                target["confirmation"],
                "--command-id",
                "remove-fixture",
            ]
        )
        _until(scheduler, lambda _: scheduler.jobs[first]["status"] == "removed")
        assert first not in scheduler.processes
        assert "native:" + first not in queue_admission.snapshot(root).get("slots", {})
        preview = invoke(["clean", str(root), "--task-id", first, "--show-target"])
        assert any(item["name"] == "last.pt" for item in preview["items"])
        receipt = invoke(
            [
                "clean",
                str(root),
                "--task-id",
                first,
                "--confirm-target",
                preview["confirmation"],
            ]
        )
        assert receipt["status"] == "complete"
        assert not (children[0] / "last.pt").exists()
        assert (children[1] / "last.pt").read_bytes() == b"fixture-1"
        assert (
            scheduler.jobs[peer]["status"] == "running"
            and scheduler.processes[peer].is_alive()
        )
        peer_after_cleanup = json.loads((children[1] / "counter.json").read_text())
        assert peer_after_cleanup["pid"] == peer_before["pid"]
        _until(
            scheduler,
            lambda _: (
                json.loads((children[1] / "counter.json").read_text())["counter"]
                > peer_after_cleanup["counter"]
            ),
        )
        assert (
            json.loads((children[1] / "counter.json").read_text())["pid"]
            == peer_before["pid"]
        )
        assert {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in preserved
        } == preserved_hashes
        assert first in runtime.snapshot(root)["removed_tasks"]
        (children[1] / "complete").touch()
        _until(scheduler, lambda _: not scheduler.processes)
        assert scheduler.jobs[peer]["status"] == "completed"
        control.write_json(
            root / "cleanup-fixture-result.json",
            {"status": "passed", "target": first, "peer": peer},
        )
        status = "completed"
    except BaseException as exc:
        control.write_json(root / "cleanup-fixture-error.json", {"error": repr(exc)})
        raise
    finally:
        if scheduler is not None:
            scheduler.close()
        scope.finish(status)
        control.CURRENT.reset(token)


@pytest.mark.skipif(os.name != "posix", reason="isolated session test requires POSIX")
def test_official_remove_and_cleanup_keeps_real_sibling_running(tmp_path):
    """Only this fixture's owned session and temporary batch may be changed."""
    folder = batch(tmp_path)
    compiled = compile_batch(folder, first_train_seeds=1)
    root = Path(author_batch.allocate(compiled))
    process = multiprocessing.get_context("spawn").Process(
        target=_cleanup_fixture_driver, args=(str(root),)
    )
    process.start()
    _join_process(process)
    assert process.exitcode == 0, (
        (root / "cleanup-fixture-error.json").read_text()
        if (root / "cleanup-fixture-error.json").exists()
        else process.exitcode
    )
    assert (
        json.loads((root / "cleanup-fixture-result.json").read_text())["status"]
        == "passed"
    )
