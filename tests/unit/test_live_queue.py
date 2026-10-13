"""Shared admission and cooperative queue lifecycle, without native learning."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from smartsom.experiments import control, live_queue, queue_admission
from smartsom.experiments.evidence import write_json


@pytest.fixture
def source(tmp_path, monkeypatch):
    root = tmp_path / "source"
    root.mkdir()
    (root / "control").mkdir()
    write_json(
        root / "control/owner.json",
        dict(
            schema="smartsom.run-control/v1",
            id="original",
            root=str(root),
            driver_root=str(root),
            status="running",
            owner=dict(pid=101, created="1", state="S"),
            members=[],
        ),
    )
    table = {101: dict(pid=101, created="1", state="S")}
    monkeypatch.setattr(control, "processes", lambda: table)
    monkeypatch.setattr(live_queue, "implementation_identity", lambda: "code")
    monkeypatch.setattr(live_queue, "source_identity", lambda: {})
    write_json(
        root / "plan.json",
        dict(
            schema="smartsom.author-batch-plan/v1",
            source={},
            files=[
                dict(
                    id="prior",
                    entry_ids=["entry"],
                    task="train-evaluate",
                    gate=None,
                )
            ],
        ),
    )
    write_json(
        root / "batch.json",
        dict(
            status="prepared",
            files={"prior": dict(status="queued")},
        ),
    )
    queue_admission.initialize(root, supported=True)
    return root, table


def test_atomic_slots_include_pending_and_retiring_and_limit_drains(source):
    root, _ = source
    queue_admission.claim(root, "queue", limit=2, owner_id="original")

    def enter(i):
        return queue_admission.reserve(root, f"native:{i}", owner_id="original")

    with ThreadPoolExecutor(max_workers=12) as pool:
        acquired = list(pool.map(enter, range(24)))
    assert sum(acquired) == 2
    before = queue_admission.snapshot(root)["slots"]
    queue_admission.configure(root, "queue", limit=1)
    assert not queue_admission.reserve(root, "source:a", owner_id="original")
    assert queue_admission.snapshot(root)["slots"] == before
    for key in before:
        queue_admission.retire(root, key, owner_id="original")
    assert queue_admission.reserve(
        root, "source:a", owner_id="original", status="stopping"
    )
    assert not enter(30)
    queue_admission.retire(root, "source:a", owner_id="original")
    assert enter(30)
    queue_admission.configure(root, "queue", limit=3, stopping=True)
    assert not enter(31)


def test_active_legacy_and_measured_calibration_fail_closed(source):
    root, _ = source
    (root / "control/admission.json").unlink()
    with pytest.raises(ValueError, match="cooperating"):
        live_queue.adopt(root)
    queue_admission.initialize(root, supported=False)
    with pytest.raises(ValueError, match="off/online"):
        live_queue.adopt(root)


def test_duplicate_adoption_and_restarted_owner_are_rejected(source):
    root, _ = source
    queue = live_queue.adopt(root)
    with pytest.raises(ValueError, match="already belongs"):
        live_queue.adopt(root)
    owner = control.read(root)
    owner["id"] = "another-generation"
    write_json(root / "control/owner.json", owner)
    _, plan, _ = live_queue.load(queue)
    with pytest.raises(ValueError, match="ownership changed"):
        live_queue._stop_source(root, plan)


@pytest.mark.parametrize("phase", ["preflight", "queued", "running", "completed"])
@pytest.mark.parametrize("exit_kind", ["stop", "interrupt", "exception"])
def test_queue_waits_for_original_owner_in_every_phase(
    source, monkeypatch, phase, exit_kind
):
    root, table = source
    original = (root / "plan.json").read_bytes()
    if phase != "queued":
        if phase != "completed":
            queue_admission.reserve(
                root, "source:prior__entry", owner_id="original", status=phase
            )
        else:
            write_json(
                root / "batch.json",
                dict(status="completed", files={"prior": dict(status="completed")}),
            )
    queue = live_queue.adopt(root, limit=1)
    calls = []

    def stop(path, *, timeout, expected_owner_id):
        assert Path(path) == root and timeout == 0 and expected_owner_id == "original"
        calls.append(path)
        write_json(root / "control/stop.json", dict(id="original"))
        # File completion is intentionally insufficient for two observations.
        assert live_queue.read(queue / "batch.json")["status"] not in live_queue.FINAL
        if len(calls) == 3:
            table.clear()
        return dict(remaining=list(table))

    monkeypatch.setattr(control, "stop", stop)
    monkeypatch.setattr(live_queue, "requested", lambda _: exit_kind == "stop")
    monkeypatch.setattr(live_queue, "bind", lambda _: None)
    monkeypatch.setattr(live_queue.time, "sleep", lambda _: None)
    from smartsom.experiments.tuning_resources import ResourceMonitor

    monkeypatch.setattr(
        ResourceMonitor,
        "snapshot",
        lambda _: SimpleNamespace(
            memory_available=64 * 1024**3, cpus=4, cpu_load=10, unavailable=()
        ),
    )
    display = SimpleNamespace(configure_tuning=lambda _: None, publish=lambda **_: None)
    token = live_queue.CURRENT.set(display)
    if exit_kind != "stop":

        def fail(*_):
            raise (
                KeyboardInterrupt()
                if exit_kind == "interrupt"
                else RuntimeError("queue poll failed")
            )

        monkeypatch.setattr(live_queue, "apply_requests", fail)
    try:
        if exit_kind == "stop":
            assert live_queue.execute_saved.__wrapped__(queue)["status"] == "stopped"
        else:
            with pytest.raises(
                KeyboardInterrupt if exit_kind == "interrupt" else RuntimeError
            ):
                live_queue.execute_saved.__wrapped__(queue)
    finally:
        live_queue.CURRENT.reset(token)
    final = live_queue.read(queue / "batch.json")
    assert final["status"] == ("failed" if exit_kind == "exception" else "stopped")
    assert len(calls) == 3 and not table
    assert control.requested(root, fresh=True)
    assert (root / "plan.json").read_bytes() == original


def test_collect_uses_shared_allocations_and_releases_completed_files(source):
    root, _ = source
    jobs = {}
    for name in ("a", "b", "c", "queued"):
        jobs[name] = dict(
            id=name,
            kind="adopted",
            root=str(root),
            entry_ids=[name + "__entry"],
            status="running",
        )
    write_json(
        root / "batch.json",
        dict(status="running", files={name: dict(status="running") for name in jobs}),
    )
    (root / "logs").mkdir()
    write_json(
        root / "logs/progress.json",
        dict(
            tasks=[
                dict(
                    id=name + "__entry",
                    status="completed" if name in {"a", "b"} else "queued",
                    values={},
                )
                for name in jobs
            ]
        ),
    )
    queue_admission.reserve(
        root, "source:c__entry", owner_id="original", status="running"
    )
    state = dict(jobs=jobs, admission_root=str(root), limit=2)
    live_queue.collect(state)
    assert [jobs[name]["status"] for name in jobs] == [
        "completed",
        "completed",
        "running",
        "queued",
    ]
    assert state["reserved"] == state["active"] == 1
    jobs["new"] = dict(id="new", kind="native", status="queued")
    assert live_queue.admission_jobs(state, memory_available=64 * 1024**3) == [
        jobs["new"]
    ]


def test_stop_race_does_not_cancel_a_replacement_owner(source, monkeypatch):
    root, _ = source
    queue = live_queue.adopt(root)
    _, plan, _ = live_queue.load(queue)
    owner = control.read(root)
    owner["id"] = "replacement"
    write_json(root / "control/owner.json", owner)
    # Simulate replacement after the first ownership observation but before stop.
    monkeypatch.setattr(live_queue, "_source_live", lambda *_: True)
    with pytest.raises(ValueError, match="identity changed before stopping"):
        live_queue._stop_source(root, plan)
    assert not (root / "control/stop.json").exists()
