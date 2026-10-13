"""Shared admission and cooperative queue lifecycle, without native learning."""

import json
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


def test_failed_spawn_releases_only_unstarted_queue_claim(source, monkeypatch):
    from smartsom.experiments import background

    root, _ = source
    assert queue_admission.reserve(
        root, "source:prior__entry", owner_id="original", status="running"
    )
    queue = live_queue.adopt(root, limit=1)
    original_plan = (root / "plan.json").read_bytes()
    original_state = (root / "batch.json").read_bytes()
    before = queue_admission.snapshot(root)["slots"]

    def fail(*args, **kwargs):
        raise OSError("controlled Mac spawn failure")

    monkeypatch.setattr(background.subprocess, "Popen", fail)
    with pytest.raises(OSError, match="spawn failure"):
        background.launch(queue)
    state = queue_admission.snapshot(root)
    assert "queue" not in state and state["slots"] == before
    assert (root / "plan.json").read_bytes() == original_plan
    assert (root / "batch.json").read_bytes() == original_state
    retry = live_queue.adopt(root, limit=1)
    assert queue_admission.snapshot(root)["queue"] == str(retry)
    assert queue_admission.snapshot(root)["slots"] == before


def test_uncertain_created_process_retains_queue_claim(source, monkeypatch):
    import os

    from smartsom.experiments import background

    root, _ = source
    queue = live_queue.adopt(root)
    before = queue_admission.snapshot(root)
    monkeypatch.setattr(
        background.subprocess,
        "Popen",
        lambda *args, **kwargs: SimpleNamespace(pid=2728, poll=lambda: None),
    )
    monkeypatch.setattr(
        control, "_birth", lambda pid: "launcher" if pid == os.getpid() else None
    )
    with pytest.raises(ValueError, match="process birth"):
        background.launch(queue)
    assert queue_admission.snapshot(root) == before


def test_rejected_source_rebind_preserves_owner_claim_and_ledgers(source):
    from smartsom.experiments import background

    root, table = source
    live_queue.adopt(root)
    owner = control.read(root)
    owner["status"] = "completed"
    write_json(root / "control/owner.json", owner)
    table.clear()
    before = {
        name: (root / name).read_bytes()
        for name in (
            "control/owner.json",
            "control/admission.json",
            "batch.json",
            "run.json",
        )
        if (root / name).exists()
    }
    scope = control.Scope()
    with pytest.raises(ValueError, match="cannot be restarted"):
        scope.bind(root)
    with pytest.raises(ValueError, match="cannot be restarted"):
        background.launch(root, resume=True)
    assert not scope.roots and scope.thread is None
    assert {name: (root / name).read_bytes() for name in before} == before


def test_full_tune_preflight_waits_for_shared_slot_before_smoke(source, monkeypatch):
    import threading
    from concurrent.futures import ThreadPoolExecutor

    from smartsom.experiments import composable, preflight

    root, _ = source
    live_queue.adopt(root, limit=1)
    queue_admission.reserve(root, "native:later", owner_id="original", status="running")
    child = root / "tune"
    child.mkdir()
    (child / "control").mkdir()
    write_json(
        child / "control/owner.json",
        dict(id="tune", driver_root=str(root), status="running"),
    )
    monkeypatch.setattr(composable, "prepared_from_run", lambda _: object())
    monkeypatch.setattr(composable, "verify_prepared_rules", lambda _: None)
    blocked, entered = threading.Event(), threading.Event()
    reserve = queue_admission.reserve

    def observe(*args, **kwargs):
        result = reserve(*args, **kwargs)
        if not result:
            blocked.set()
        return result

    monkeypatch.setattr(queue_admission, "reserve", observe)

    def smoke(*args, **kwargs):
        slots = queue_admission.snapshot(root)["slots"]
        assert slots == {"source:prior__entry": {"status": "preflight"}}
        entered.set()
        return dict(status="passed", ticks=1)

    monkeypatch.setattr(preflight, "smoke", smoke)
    plan = dict(
        schema="smartsom.tune-batch/v1",
        preflight="full",
        provenance=dict(kind="author-batch"),
        entries=[dict(experiment_id="prior__entry", prepared={})],
    )
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(preflight.run, child, plan)
        try:
            assert blocked.wait(2) and not entered.is_set()
        finally:
            queue_admission.retire(root, "native:later", owner_id="original")
        result = future.result(5)
    assert result["status"] == "passed" and result["smoke_done"] == 1
    assert entered.is_set() and not queue_admission.snapshot(root)["slots"]


def test_active_v1_driver_is_rejected_and_completed_adoption_claims_source(source):
    root, table = source
    state = queue_admission.snapshot(root)
    state["schema"] = "smartsom.queue-admission/v1"
    write_json(root / "control/admission.json", state)
    with pytest.raises(ValueError, match="cooperating"):
        live_queue.adopt(root)
    owner = control.read(root)
    owner["status"] = "completed"
    write_json(root / "control/owner.json", owner)
    write_json(
        root / "batch.json",
        dict(status="completed", files={"prior": dict(status="completed")}),
    )
    table.clear()
    queue = live_queue.adopt(root)
    _, plan, _ = live_queue.load(queue)
    assert plan["admission_root"] == str(root)
    assert queue_admission.snapshot(root)["queue"] == str(queue)
    assert queue_admission.snapshot(root)["schema"] == queue_admission.SCHEMA
    original = (root / "control/owner.json").read_bytes()
    with pytest.raises(ValueError, match="cannot be restarted"):
        control.Scope().bind(root)
    assert (root / "control/owner.json").read_bytes() == original


@pytest.mark.parametrize("failure", ["directory", "lock", "request"])
def test_pre_spawn_preparation_and_failed_diagnostics_still_rollback_claim(
    source, monkeypatch, failure
):
    from contextlib import contextmanager
    from pathlib import Path

    from smartsom.experiments import background

    root, _ = source
    queue_admission.reserve(
        root, "source:existing", owner_id="original", status="running"
    )
    queue = live_queue.adopt(root, limit=1)
    before = queue_admission.snapshot(root)["slots"]
    mkdir, writer = Path.mkdir, control.write_json

    def fail_write(path, data):
        if Path(path).name in {"background-request.json", "background-startup.json"}:
            raise OSError("controlled diagnostic write failure")
        writer(path, data)

    @contextmanager
    def fail_lock(_):
        raise OSError("controlled launch lock failure")
        yield

    def fail_mkdir(path, *args, **kwargs):
        if path == queue / "logs":
            raise OSError("controlled log directory failure")
        return mkdir(path, *args, **kwargs)

    monkeypatch.setattr(control, "write_json", fail_write)
    if failure == "directory":
        monkeypatch.setattr(Path, "mkdir", fail_mkdir)
    elif failure == "lock":
        monkeypatch.setattr(background, "_launch_lock", fail_lock)
    monkeypatch.setattr(
        background.subprocess, "Popen", lambda *a, **k: pytest.fail("no spawn")
    )
    with pytest.raises(OSError, match="controlled") as caught:
        background.launch(queue)
    assert any(
        "failure could not be recorded" in note for note in caught.value.__notes__
    )
    state = queue_admission.snapshot(root)
    assert "queue" not in state and state["slots"] == before
    retry = live_queue.adopt(root)
    assert queue_admission.snapshot(root)["queue"] == str(retry)


def test_failed_preparation_preserves_other_live_launcher(source, monkeypatch):
    from smartsom.experiments import background

    root, _ = source
    queue = live_queue.adopt(root)
    (queue / "control").mkdir(exist_ok=True)
    write_json(
        queue / "control/background-launch.lock",
        dict(pid=777, created="other", nonce="other"),
    )
    before = queue_admission.snapshot(root)
    ledgers = {name: (queue / name).read_bytes() for name in ("batch.json", "run.json")}
    monkeypatch.setattr(
        control, "_birth", lambda pid: "other" if pid == 777 else "self"
    )
    with pytest.raises(ValueError, match="another background launch"):
        background.launch(queue)
    assert queue_admission.snapshot(root) == before
    assert {name: (queue / name).read_bytes() for name in ledgers} == ledgers


def test_interrupted_spawn_without_returned_handle_retains_unknown_claim(
    source, monkeypatch
):
    from smartsom.experiments import background

    root, _ = source
    queue = live_queue.adopt(root)
    before = queue_admission.snapshot(root)

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt("constructor interrupted after possible fork")

    monkeypatch.setattr(background.subprocess, "Popen", interrupted)
    with pytest.raises(KeyboardInterrupt):
        background.launch(queue)
    assert queue_admission.snapshot(root) == before


def test_native_directory_entries_share_smoke_actor_and_appended_job_gate(
    source, monkeypatch
):
    from smartsom.experiments import author_driver

    root, _ = source
    child = root / "experiments/rule"
    child.mkdir(parents=True)
    source_plan = json.loads((root / "plan.json").read_text())
    source_plan["files"][0]["child"] = "experiments/rule"
    write_json(root / "plan.json", source_plan)
    token = control.CURRENT.set(SimpleNamespace(driver_root=root))
    try:
        admission = author_driver._native_source_admission(child)
        queue_admission.claim(root, "queue", limit=1, owner_id="original")
        assert queue_admission.reserve(root, "native:appended", owner_id="original")
        assert not author_driver._native_source_reserve(admission, "entry")
        queue_admission.retire(root, "native:appended", owner_id="original")
        assert author_driver._native_source_reserve(admission, "entry")
        key = source_plan["files"][0]["id"] + "__entry"
        assert (
            queue_admission.snapshot(root)["slots"]["source:" + key]["status"]
            == "pending"
        )
        author_driver._native_source_started(admission, "entry")
        assert (
            queue_admission.snapshot(root)["slots"]["source:" + key]["status"]
            == "running"
        )
        assert not queue_admission.reserve(root, "native:appended", owner_id="original")
        author_driver._native_source_retire(admission, "entry")
        assert queue_admission.reserve(root, "native:appended", owner_id="original")
    finally:
        control.CURRENT.reset(token)
