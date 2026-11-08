"""Retry journal acceptance and coordinator validation with isolated fixtures."""

import hashlib
from pathlib import Path

import pytest

from smartsom.experiments import batch_retry
from smartsom.experiments import runtime_control as runtime
from smartsom.experiments.runtime_workers import NativeScheduler


@pytest.fixture
def failed_queue(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "_RETRY_ENABLED", True)
    root = tmp_path / "queue"
    child = root / "trials" / "seed-4103"
    (child / "continuation").mkdir(parents=True)
    (child / "trial.lock").touch()
    (child / "continuation/current.json").write_text('{"generation":"test"}')
    runtime.initialize(root, 2)
    job = {
        "root": str(child),
        "status": "failed",
        "pid": 42,
        "exit_code": 1,
        "own_slots": False,
    }
    with runtime.transaction(root) as state:
        state.update(status="running", active=1)
        state["tasks"]["trial--seed-4103"] = dict(job)
        state["tasks"]["other"] = {
            "root": str(root / "other"),
            "status": "running",
            "pid": 43,
        }
    scheduler = NativeScheduler.__new__(NativeScheduler)
    scheduler.root = root
    scheduler.jobs = {"trial--seed-4103": dict(job)}
    scheduler.processes = {}
    return root, child, scheduler


def fake_info(root, child):
    return {
        "owner_root": str(root.resolve()),
        "child": str(child),
        "generation": "test",
        "pointer_sha256": hashlib.sha256(
            (child / "continuation/current.json").read_bytes()
        ).hexdigest(),
    }


def test_retry_journal_is_idempotent_and_single_pending(failed_queue):
    root, _, _ = failed_queue
    one = runtime.submit(root, "retry", task_id="trial--seed-4103", command_id="fixed")
    assert (
        runtime.submit(root, "retry", task_id="trial--seed-4103", command_id="fixed")
        == one
    )
    with pytest.raises(ValueError, match="already pending"):
        runtime.submit(root, "retry", task_id="trial--seed-4103", command_id="other")
    assert runtime.apply(root)["commands"]["fixed"]["status"] == "received"
    assert not runtime.finish_if_drained(root)


def test_scheduler_consumes_only_its_failed_job(failed_queue, monkeypatch):
    root, child, scheduler = failed_queue
    monkeypatch.setattr(
        batch_retry,
        "inspect_target",
        lambda root_, key, state, **kwargs: fake_info(root, child),
    )
    runtime.submit(root, "retry", task_id="trial--seed-4103", command_id="fixed")
    scheduler._consume_retries(runtime.snapshot(root))
    state = runtime.snapshot(root)
    assert state["commands"]["fixed"]["status"] == "applied"
    assert state["tasks"]["trial--seed-4103"]["status"] == "queued"
    assert scheduler.jobs["trial--seed-4103"]["status"] == "queued"
    assert state["tasks"]["other"]["pid"] == 43
    assert len(scheduler.jobs["trial--seed-4103"]["retry_history"]) == 1
    with pytest.raises(ValueError, match="exact failed or paused task"):
        runtime.submit(root, "retry", task_id="trial--seed-4103", command_id="other")


def test_validation_failure_keeps_failure_state(failed_queue, monkeypatch):
    root, _, scheduler = failed_queue

    def reject(*args, **kwargs):
        raise ValueError("checkpoint invalid")

    monkeypatch.setattr(batch_retry, "inspect_target", reject)
    runtime.submit(root, "retry", task_id="trial--seed-4103", command_id="fixed")
    scheduler._consume_retries(runtime.snapshot(root))
    state = runtime.snapshot(root)
    assert state["commands"]["fixed"]["status"] == "rejected"
    assert state["commands"]["fixed"]["reason"] == "checkpoint invalid"
    assert state["tasks"]["trial--seed-4103"]["status"] == "failed"
    assert scheduler.jobs["trial--seed-4103"]["status"] == "failed"


def test_retry_respects_cleanup_holds_and_stage_fence(failed_queue):
    root, _, _ = failed_queue
    for field, value in (
        ("cleanup_active", {"task_id": "other"}),
        ("retry_phase_closed", True),
        ("stopped_tasks", ["trial--seed-4103"]),
    ):
        with runtime.transaction(root) as state:
            state[field] = value
        with pytest.raises(ValueError):
            runtime.submit(root, "retry", task_id="trial--seed-4103")
        with runtime.transaction(root) as state:
            state[field] = [] if field == "stopped_tasks" else False


def test_disabled_draft_does_not_consume_retry(failed_queue, monkeypatch):
    root, _, scheduler = failed_queue
    runtime.submit(root, "retry", task_id="trial--seed-4103", command_id="fixed")
    monkeypatch.setattr(runtime, "_RETRY_ENABLED", False)
    state = runtime.apply(root)
    scheduler._consume_retries(state)
    assert state["commands"]["fixed"]["status"] == "rejected"
    assert scheduler.jobs["trial--seed-4103"]["status"] == "failed"


def test_retry_selected_paused_task_keeps_other_and_global_pause(
    failed_queue, monkeypatch
):
    root, child, scheduler = failed_queue
    key = "trial--seed-4103"
    scheduler.jobs[key]["status"] = "paused"
    with runtime.transaction(root) as state:
        state["tasks"][key]["status"] = "paused"
        state["paused_tasks"] = [key, "other"]
        state["paused"] = True
        state["active"] = 0
    monkeypatch.setattr(
        batch_retry,
        "inspect_target",
        lambda root_, key_, state, **kwargs: fake_info(root, child),
    )
    runtime.submit(root, "retry", task_id=key, command_id="paused-retry")
    scheduler._consume_retries(runtime.snapshot(root))
    state = runtime.snapshot(root)
    assert state["commands"]["paused-retry"]["status"] == "applied"
    assert state["paused_tasks"] == ["other"]
    assert state["paused"] is True
    assert scheduler.jobs[key]["status"] == "queued"


def test_manual_retry_stops_after_one_new_failure(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace

    from smartsom.experiments import launch, reservation
    from smartsom.telemetry import runtime as telemetry

    root = tmp_path / "run"
    root.mkdir()
    previous = [
        {"time": 1, "type": "OSError", "message": "old failure", "stopped": False}
    ]
    (root / "attempts.json").write_text(json.dumps(previous))
    experiment = SimpleNamespace(validate_execution=lambda: None)
    args = SimpleNamespace(_prepared_inputs=([experiment], "identity"))
    settings = SimpleNamespace(task="train", tune="off", max_failures=10)
    monkeypatch.setattr(reservation, "load", lambda _: None)
    monkeypatch.setattr(reservation, "arguments", lambda _: (args, settings))
    monkeypatch.setattr(telemetry, "bind", lambda _: None)
    monkeypatch.setattr(launch, "check_resource_budget", lambda *a, **k: None)
    calls = []

    def fail_once(_):
        calls.append(1)
        raise OSError("new failure")

    monkeypatch.setattr(reservation.runtime, "train", fail_once)
    with pytest.raises(OSError, match="new failure"):
        reservation.execute_saved(root, retry_once=True)
    attempts = json.loads((root / "attempts.json").read_text())
    assert len(calls) == 1 and len(attempts) == 2
    assert attempts[0] == previous[0] and attempts[1]["message"] == "new failure"
    assert json.loads((root / "progress.json").read_text())["status"] == "failed"


def test_applied_retry_survives_before_in_memory_update(failed_queue, monkeypatch):
    root, child, scheduler = failed_queue
    key = "trial--seed-4103"
    monkeypatch.setattr(
        batch_retry,
        "inspect_target",
        lambda root_, key_, state, **kwargs: fake_info(root, child),
    )

    class InterruptedDict(dict):
        def update(self, value):
            if value.get("status") == "queued":
                raise KeyboardInterrupt("driver interrupted after journal")
            return super().update(value)

    scheduler.jobs[key] = InterruptedDict(scheduler.jobs[key])
    runtime.submit(root, "retry", task_id=key, command_id="fixed")
    with pytest.raises(KeyboardInterrupt):
        scheduler._consume_retries(runtime.snapshot(root))
    observed = runtime.snapshot(root)
    assert observed["commands"]["fixed"]["status"] == "applied"
    attempt = observed["tasks"][key]["retry_attempt"]
    fresh = NativeScheduler.__new__(NativeScheduler)
    fresh.root = root
    fresh.jobs = {key: {"root": str(child), "status": "failed", "pid": 42}}
    fresh.processes = {}
    fresh._restore_queued_retries(observed)
    assert fresh.jobs[key]["status"] == "queued"
    assert fresh.jobs[key]["retry_attempt"] == attempt
    plan = {"incremental": True, "files": [{"id": key}]}
    state = {
        "status": "failed",
        "selection": {"tasks": [key]},
        "files": {key: {"status": "failed"}},
    }
    assert batch_retry.restore_selected_state(root, plan, state, observed) == [key]
    assert state["files"][key]["status"] == "queued"


def test_started_retry_never_respawns_without_ownership_resolution(
    failed_queue, monkeypatch
):
    root, child, scheduler = failed_queue
    key = "trial--seed-4103"
    monkeypatch.setattr(
        batch_retry,
        "inspect_target",
        lambda root_, key_, state, **kwargs: fake_info(root, child),
    )
    runtime.submit(root, "retry", task_id=key, command_id="fixed")
    scheduler._consume_retries(runtime.snapshot(root))
    with runtime.transaction(root) as state:
        state["tasks"][key]["status"] = "starting"
    observed = runtime.snapshot(root)
    fresh = NativeScheduler.__new__(NativeScheduler)
    fresh.root = root
    fresh.jobs = {key: {"root": str(child), "status": "failed", "pid": 42}}
    fresh.processes = {}
    fresh._restore_queued_retries(observed)
    assert fresh.jobs[key]["status"] == "failed"
    assert "may have started" in fresh.jobs[key]["retry_recovery_required"]
    with pytest.raises(ValueError, match="uncertain ownership"):
        runtime.submit(root, "retry", task_id=key)
    plan = {"incremental": True, "files": [{"id": key}]}
    old = {
        "status": "failed",
        "selection": {"tasks": [key]},
        "files": {key: {"status": "failed"}},
    }
    with pytest.raises(ValueError, match="may have started"):
        batch_retry.restore_selected_state(root, plan, old, observed)


def test_newer_pause_supersedes_retry(failed_queue, monkeypatch):
    root, child, scheduler = failed_queue
    key = "trial--seed-4103"
    monkeypatch.setattr(
        batch_retry,
        "inspect_target",
        lambda root_, key_, state, **kwargs: fake_info(root, child),
    )
    runtime.submit(root, "retry", task_id=key, command_id="retry-first")
    runtime.submit(root, "pause", task_id=key, command_id="pause-later")
    state = runtime.apply(root)
    scheduler._consume_retries(state)
    state = runtime.snapshot(root)
    assert state["commands"]["retry-first"]["status"] == "rejected"
    assert "newer pause" in state["commands"]["retry-first"]["reason"]
    assert key in state["paused_tasks"]
    assert scheduler.jobs[key]["status"] == "failed"


def test_restore_added_task_does_not_expand_original_selection(
    failed_queue, monkeypatch
):
    root, child, scheduler = failed_queue
    key = "trial--seed-4103"
    monkeypatch.setattr(
        batch_retry,
        "inspect_target",
        lambda root_, key_, state, **kwargs: fake_info(root, child),
    )
    runtime.submit(root, "retry", task_id=key, command_id="fixed")
    scheduler._consume_retries(runtime.snapshot(root))
    observed = runtime.snapshot(root)
    added = dict(observed["tasks"].pop(key))
    added.update(owner_root=str(root), owner_task=key)
    observed["additions"][key] = added
    plan = {
        "incremental": True,
        "files": [{"id": key, "child": str(child.relative_to(root))}],
    }
    old = {
        "status": "failed",
        "selection": {"tasks": []},
        "files": {key: {"status": "queued"}},
    }
    assert batch_retry.restore_selected_state(root, plan, old, observed) == [key]
    assert old["selection"]["tasks"] == [] and old["status"] == "prepared"


def test_restore_foreign_pending_attempt_fails_closed(failed_queue):
    root, _, _ = failed_queue
    observed = runtime.snapshot(root)
    observed["additions"]["foreign"] = {"retry_pending": True, "status": "starting"}
    plan = {"files": []}
    old = {"status": "failed", "selection": {"tasks": []}, "files": {}}
    with pytest.raises(ValueError, match="external-owner retry"):
        batch_retry.restore_selected_state(root, plan, old, observed)


def test_paused_retry_retains_once_only_policy_after_driver_restart(
    failed_queue, monkeypatch
):
    root, child, scheduler = failed_queue
    key = "trial--seed-4103"
    monkeypatch.setattr(
        batch_retry,
        "inspect_target",
        lambda root_, key_, state, **kwargs: fake_info(root, child),
    )
    runtime.submit(root, "retry", task_id=key, command_id="fixed")
    scheduler._consume_retries(runtime.snapshot(root))
    with runtime.transaction(root) as state:
        state["tasks"][key].update(status="paused", retry_pending=False)
        state["paused_tasks"] = [key]
    observed = runtime.snapshot(root)
    fresh = NativeScheduler.__new__(NativeScheduler)
    fresh.root = root
    fresh.jobs = {key: {"root": str(child), "status": "paused"}}
    fresh.processes = {}
    fresh._restore_queued_retries(observed)
    assert fresh.jobs[key]["status"] == "paused"
    assert fresh.jobs[key]["retry_once_active"] is True
    assert fresh.jobs[key]["retry_attempt"] == observed["tasks"][key]["retry_attempt"]


def test_submit_preview_uses_fixed_targets_and_reports_partial_success(
    failed_queue, monkeypatch
):
    root, child, _ = failed_queue
    key = "trial--seed-4103"
    with runtime.transaction(root) as state:
        state["tasks"]["bad"] = {"status": "failed", "root": str(child), "pid": 44}

    def inspect(root_, key_, state, **kwargs):
        if key_ == "bad":
            raise ValueError("invalid checkpoint")
        return fake_info(root, child)

    monkeypatch.setattr(batch_retry, "inspect_target", inspect)
    preview = batch_retry.preview_failed(root, include_paused=True)
    with runtime.transaction(root) as state:
        state["tasks"]["later"] = {"status": "failed", "root": str(child), "pid": 45}
    result = batch_retry.submit_preview(root, preview)
    assert [item["task_id"] for item in result["outcomes"]] == ["bad", key]
    assert result["outcomes"][0]["status"] == "rejected"
    assert result["outcomes"][1]["status"] == "received"
    assert all(item["task_id"] != "later" for item in result["outcomes"])
    state = runtime.snapshot(root)
    assert state["tasks"]["later"]["status"] == "failed"
    assert (
        len(
            [
                item
                for item in state["commands"].values()
                if item["payload"]["kind"] == "retry"
            ]
        )
        == 1
    )


def test_submit_preview_rejects_changed_failure_attempt(failed_queue, monkeypatch):
    root, child, _ = failed_queue
    key = "trial--seed-4103"
    monkeypatch.setattr(
        batch_retry,
        "inspect_target",
        lambda root_, key_, state, **kwargs: fake_info(root, child),
    )
    preview = batch_retry.preview_failed(root, [key], include_paused=True)
    with runtime.transaction(root) as state:
        state["tasks"][key]["pid"] = 999
    result = batch_retry.submit_preview(root, preview)
    assert result["outcomes"][0]["status"] == "rejected"
    assert "changed since preview" in result["outcomes"][0]["reason"]
    assert not runtime.snapshot(root)["commands"]


def test_driver_recovery_ignores_ordinary_transient_addition_views():
    observed = {
        "tasks": {"ordinary": {"status": "paused"}},
        "additions": {"ordinary": {"status": "queued"}},
    }
    batch_retry.check_driver_recovery(observed, enabled=True)
    batch_retry.check_driver_recovery(observed, enabled=False)


def test_driver_recovery_blocks_uncertain_retry_before_resume():
    for marker in (
        {"retry_recovery_required": "unknown", "status": "failed"},
        {"retry_pending": True, "status": "starting"},
        {"retry_pending": True, "status": "running"},
    ):
        with pytest.raises(ValueError, match="requires worker ownership verification"):
            batch_retry.check_driver_recovery({"tasks": {"a": marker}}, enabled=True)
    with pytest.raises(ValueError, match="retry recovery is not enabled"):
        batch_retry.check_driver_recovery(
            {"tasks": {"a": {"status": "paused", "retry_once_active": True}}},
            enabled=False,
        )


def test_requeued_paused_retry_addition_preserves_once_policy(
    failed_queue, monkeypatch
):
    root, child, scheduler = failed_queue
    key = "trial--seed-4103"
    monkeypatch.setattr(
        batch_retry,
        "inspect_target",
        lambda root_, key_, state, **kwargs: fake_info(root, child),
    )
    runtime.submit(root, "retry", task_id=key, command_id="fixed")
    scheduler._consume_retries(runtime.snapshot(root))
    state = runtime.snapshot(root)
    addition = dict(state["tasks"][key])
    addition.update(status="queued", retry_pending=False, retry_once_active=True)
    state["tasks"][key]["status"] = "paused"
    state["additions"][key] = addition
    batch_retry.sync_retry_addition(state, key, addition)
    assert batch_retry.task_view(state, key)["status"] == "queued"
    fresh = NativeScheduler.__new__(NativeScheduler)
    fresh.root = root
    fresh.jobs = {key: {"root": str(child), "status": "queued"}}
    fresh.processes = {}
    fresh._restore_queued_retries(state)
    assert fresh.jobs[key]["status"] == "queued"
    assert fresh.jobs[key]["retry_once_active"] is True


@pytest.mark.parametrize("interrupt", [False, True, "after_start"])
def test_uncertain_retry_spawn_keeps_slot_and_blocks_new_admission(
    failed_queue, monkeypatch, interrupt
):
    from types import SimpleNamespace

    from smartsom.experiments import queue_admission

    root, child, scheduler = failed_queue
    key = "trial--seed-4103"
    job = scheduler.jobs[key]
    job.update(
        status="queued", retry_pending=True, retry_once_active=True, resources={}
    )
    with runtime.transaction(root) as state:
        state["tasks"][key].update(job)
        state["active"] = 0

    class Host:
        def snapshot(self):
            return SimpleNamespace(
                cpus=8,
                cpu_load=0,
                memory_total=16 * 1024**3,
                memory_available=14 * 1024**3,
            )

    class BrokenProcess:
        def start(self):
            if interrupt == "after_start":
                return
            if interrupt:
                raise KeyboardInterrupt("injected uncertain spawn")
            raise OSError("injected uncertain spawn")

    class Context:
        def Process(self, **kwargs):
            return BrokenProcess()

    slots = {}
    retired = []
    monkeypatch.setattr(
        queue_admission, "snapshot", lambda _: {"owner_id": "owner", "slots": slots}
    )
    monkeypatch.setattr(queue_admission, "note_wait", lambda *a, **kw: None)

    def reserve(root_, slot, **kw):
        slots[slot] = {"status": "running"}
        return True

    monkeypatch.setattr(queue_admission, "reserve", reserve)
    monkeypatch.setattr(queue_admission, "retire", lambda *a, **kw: retired.append(a))
    if interrupt == "after_start":
        from smartsom.experiments import author_batch

        class InterruptRegistration(dict):
            def __setitem__(self, key_, value):
                raise KeyboardInterrupt("injected after spawn before registration")

        scheduler.processes = InterruptRegistration()
        monkeypatch.setattr(author_batch, "_request_owned_stop", lambda _: None)
    scheduler.observer = Host()
    scheduler.context = Context()
    scheduler.worker = lambda *a: None
    scheduler.samples = 2
    scheduler._retry_reconciled = True
    scheduler.peak_memory = {}
    scheduler.stop_sent = set()
    scheduler.pause_sent = set()
    if interrupt:
        with pytest.raises(KeyboardInterrupt):
            scheduler.tick()
        if interrupt == "after_start":
            scheduler.close()
    else:
        scheduler.tick()
    assert not retired
    assert "native:" + key in slots
    assert scheduler.jobs[key]["retry_recovery_required"]
    assert scheduler.unfinished
    scheduler.tick()
    assert "ownership is uncertain" in runtime.snapshot(root)["blocking_reason"]


def test_one_bulk_retry_selects_only_paused_and_failed(failed_queue, monkeypatch):
    root, child, _ = failed_queue
    failed, paused = "trial--seed-4103", "trial--seed-4102"
    with runtime.transaction(root) as state:
        state["tasks"][paused] = {
            "root": str(child),
            "status": "paused",
            "pid": 44,
            "exit_code": 0,
        }
        state["paused_tasks"] = [paused]
        state["tasks"]["deferred"] = {"status": "deferred"}
    monkeypatch.setattr(
        batch_retry,
        "inspect_target",
        lambda root_, key_, state, **kwargs: fake_info(root, child),
    )
    preview = batch_retry.preview_failed(root, include_paused=True)
    assert {x["task_id"] for x in preview["targets"]} == {failed, paused}
    result = batch_retry.submit_preview(root, preview)
    assert {x["task_id"] for x in result["outcomes"]} == {failed, paused}
    assert all(x["status"] == "received" for x in result["outcomes"])
    state = runtime.snapshot(root)
    assert state["tasks"]["deferred"]["status"] == "deferred"
    assert state["tasks"]["other"]["status"] == "running"
    assert state["paused_tasks"] == [paused]  # changed only by coordinator


def test_stale_scheduler_persist_does_not_erase_committed_retry(
    failed_queue, monkeypatch
):
    from smartsom.experiments import queue_admission

    root, child, scheduler = failed_queue
    key = "trial--seed-4103"
    monkeypatch.setattr(
        batch_retry,
        "inspect_target",
        lambda root_, key_, state, **kwargs: fake_info(root, child),
    )
    monkeypatch.setattr(queue_admission, "snapshot", lambda _: {})
    stale = dict(scheduler.jobs[key])
    runtime.submit(root, "retry", task_id=key, command_id="fixed")
    scheduler._consume_retries(runtime.snapshot(root))
    committed = runtime.snapshot(root)["tasks"][key]
    scheduler.jobs[key] = stale
    scheduler._persist(runtime.snapshot(root))
    actual = runtime.snapshot(root)["tasks"][key]
    assert actual["retry_attempt"] == committed["retry_attempt"]
    assert actual["retry_pending"] is True and actual["status"] == "queued"
    assert actual["retry_history"] == committed["retry_history"]


def test_interrupted_apply_then_scheduler_close_preserves_retry(
    failed_queue, monkeypatch
):
    from types import SimpleNamespace

    from smartsom.experiments import author_batch, queue_admission

    root, child, scheduler = failed_queue
    key = "trial--seed-4103"
    monkeypatch.setattr(
        batch_retry,
        "inspect_target",
        lambda root_, key_, state, **kwargs: fake_info(root, child),
    )
    monkeypatch.setattr(queue_admission, "snapshot", lambda _: {})
    monkeypatch.setattr(author_batch, "_request_owned_stop", lambda _: None)

    class Host:
        def snapshot(self):
            return SimpleNamespace(
                cpus=8,
                cpu_load=0,
                memory_total=16 * 1024**3,
                memory_available=14 * 1024**3,
            )

    class InterruptedDict(dict):
        def update(self, value):
            if value.get("status") == "queued":
                raise KeyboardInterrupt("injected after journal commit")
            return super().update(value)

    scheduler.jobs[key] = InterruptedDict(scheduler.jobs[key])
    scheduler.observer = Host()
    scheduler.samples = 0
    scheduler._retry_reconciled = True
    scheduler.peak_memory = {}
    scheduler.stop_sent = set()
    scheduler.pause_sent = set()
    runtime.submit(root, "retry", task_id=key, command_id="fixed")
    try:
        scheduler._consume_retries(runtime.snapshot(root))
        pytest.fail("injected interruption did not occur")
    except KeyboardInterrupt:
        scheduler.close()
    state = runtime.snapshot(root)
    assert state["commands"]["fixed"]["status"] == "applied"
    assert state["tasks"][key]["retry_pending"] is True
    attempt = state["tasks"][key]["retry_attempt"]
    runtime.restart(root)
    fresh = NativeScheduler.__new__(NativeScheduler)
    fresh.root, fresh.jobs, fresh.processes = (
        root,
        {key: {"root": str(child), "status": "failed", "pid": 42}},
        {},
    )
    fresh._restore_queued_retries(runtime.snapshot(root))
    assert fresh.jobs[key]["status"] == "queued"
    assert fresh.jobs[key]["retry_attempt"] == attempt


@pytest.mark.parametrize("entry", ["api", "foreground", "background", "child"])
def test_uncertain_retry_rejected_before_owner_or_ready_writes(
    failed_queue, monkeypatch, entry
):
    import json

    from smartsom import api
    from smartsom.experiments import background, foreground

    root, _, _ = failed_queue
    (root / "plan.json").write_text(
        json.dumps({"schema": "smartsom.author-batch-plan/v1"})
    )
    (root / "batch.json").write_text("{}")
    owner = root / "control/owner.json"
    owner.write_text('{"id":"old-owner"}')
    with runtime.transaction(root) as state:
        state["tasks"]["trial--seed-4103"].update(retry_recovery_required="uncertain")
    before = {
        str(path): path.read_bytes()
        for path in (
            root / "plan.json",
            root / "batch.json",
            owner,
            root / "control/runtime.json",
        )
    }
    with pytest.raises(ValueError, match="requires worker ownership verification"):
        if entry == "api":
            api.resume(root)
        elif entry == "foreground":
            foreground.run(root, resume=True)
        elif entry == "background":
            background.launch(root, resume=True)
        else:
            background.child(root, nonce="unused", resume=True)
    assert all(Path(path).read_bytes() == data for path, data in before.items())
    assert not (root / "control/background-startup.json").exists()


def test_cli_retry_selects_failed_and_preserves_paused_snapshot(
    failed_queue, monkeypatch, capsys
):
    import json

    from smartsom.experiments import catalog, cli

    root, child, _ = failed_queue
    monkeypatch.setattr(
        "smartsom.experiments.control.read",
        lambda root: {"status": "running", "owner": {}},
    )
    monkeypatch.setattr("smartsom.experiments.control.alive", lambda *args: True)

    key = "trial--seed-4103"
    with runtime.transaction(root) as state:
        state["tasks"]["paused-seed"] = {
            "root": str(child),
            "status": "paused",
            "pid": 44,
        }
    monkeypatch.setattr(catalog, "resolve_run", lambda _: root)
    monkeypatch.setattr(catalog, "require_current", lambda _: None)
    monkeypatch.setattr(
        batch_retry,
        "inspect_target",
        lambda root_, key_, state, **kwargs: fake_info(root, child),
    )
    assert cli.main(["retry", str(root), "--all"]) == 0
    captured = capsys.readouterr()
    preview = json.loads(captured.err)["preview"]
    result = json.loads(captured.out)
    assert {item["task_id"] for item in preview["targets"]} == {key}
    assert {item["status"] for item in result["outcomes"]} == {"received"}
    assert runtime.snapshot(root)["tasks"]["other"]["status"] == "running"


def test_cli_retry_is_gated_when_not_ready(failed_queue, monkeypatch, capsys):
    from smartsom.experiments import catalog, cli

    root, _, _ = failed_queue
    monkeypatch.setattr(catalog, "resolve_run", lambda _: root)
    monkeypatch.setattr(catalog, "require_current", lambda _: None)
    monkeypatch.setattr(runtime, "_RETRY_ENABLED", False)
    assert cli.main(["retry", str(root), "--task-id", "trial--seed-4103"]) != 0
    assert "not yet enabled" in capsys.readouterr().err
    assert not runtime.snapshot(root)["commands"]


def test_received_retry_survives_driver_recovery_for_selected_failure(
    failed_queue, monkeypatch
):
    from smartsom.experiments.incremental_batch import selected_rows

    root, child, scheduler = failed_queue
    key = "trial--seed-4103"
    monkeypatch.setattr(
        batch_retry,
        "inspect_target",
        lambda root_, key_, state, **kwargs: fake_info(root, child),
    )
    runtime.submit(root, "retry", task_id=key, command_id="fixed")
    observed = runtime.snapshot(root)
    plan = {
        "incremental": True,
        "files": [{"id": key, "child": str(child.relative_to(root)), "stage": 0}],
    }
    old = {
        "status": "failed",
        "selection": {"tasks": [key]},
        "files": {key: {"status": "failed"}},
    }
    assert batch_retry.restore_selected_state(root, plan, old, observed) == []
    assert old["status"] == "prepared"
    rows = selected_rows(plan, old, batch_retry.pending_retry_targets(observed))
    assert [row["id"] for row in rows] == [key]
    assert (
        old["files"][key]["status"] == "failed"
    )  # coordinator validates before queuing
    scheduler._consume_retries(observed)
    assert runtime.snapshot(root)["commands"]["fixed"]["status"] == "applied"


def test_received_retry_for_addition_keeps_original_selection(
    failed_queue, monkeypatch
):
    from smartsom.experiments.incremental_batch import selected_rows

    root, child, scheduler = failed_queue
    key = "trial--seed-4103"
    monkeypatch.setattr(
        batch_retry,
        "inspect_target",
        lambda root_, key_, state, **kwargs: fake_info(root, child),
    )
    with runtime.transaction(root) as state:
        addition = state["tasks"].pop(key)
        addition.update(owner_root=str(root), owner_task=key)
        state["additions"][key] = addition
    runtime.submit(root, "retry", task_id=key, command_id="fixed")
    observed = runtime.snapshot(root)
    plan = {
        "incremental": True,
        "files": [{"id": key, "child": str(child.relative_to(root)), "stage": 0}],
    }
    old = {
        "status": "failed",
        "selection": {"tasks": []},
        "files": {key: {"status": "failed"}},
    }
    batch_retry.restore_selected_state(root, plan, old, observed)
    assert old["status"] == "prepared" and old["selection"]["tasks"] == []
    assert selected_rows(plan, old, batch_retry.pending_retry_targets(observed)) == []
    scheduler._consume_retries(observed)
    assert runtime.snapshot(root)["commands"]["fixed"]["status"] == "applied"


def test_same_bulk_preview_replay_returns_original_receipts(failed_queue, monkeypatch):
    root, child, _ = failed_queue
    key = "trial--seed-4103"
    monkeypatch.setattr(
        batch_retry,
        "inspect_target",
        lambda root_, key_, state, **kwargs: fake_info(root, child),
    )
    preview = batch_retry.preview_failed(root, [key], include_paused=True)
    first = batch_retry.submit_preview(root, preview)
    again = batch_retry.submit_preview(root, preview)
    assert first == again
    assert len(runtime.snapshot(root)["commands"]) == 1


def test_retry_pause_resume_retains_single_attempt_policy(failed_queue, monkeypatch):
    import json
    from types import SimpleNamespace

    from smartsom.experiments import queue_admission

    root, child, scheduler = failed_queue
    key = "trial--seed-4103"
    monkeypatch.setattr(
        batch_retry,
        "inspect_target",
        lambda root_, key_, state, **kwargs: fake_info(root, child),
    )
    monkeypatch.setattr(queue_admission, "snapshot", lambda _: {})

    class Host:
        def snapshot(self):
            return SimpleNamespace(
                cpus=8,
                cpu_load=0,
                memory_total=16 * 1024**3,
                memory_available=14 * 1024**3,
            )

    launches = []

    class FakeProcess:
        def __init__(self, args):
            self.args = args
            self.pid = 100 + len(launches)
            self.exitcode = 0 if not launches else 1

        def start(self):
            launches.append(self.args)

        def is_alive(self):
            return False

        def join(self):
            pass

    class Context:
        def Process(self, *, target, args):
            return FakeProcess(args)

    scheduler.observer = Host()
    scheduler.context = Context()
    scheduler.worker = lambda *a: None
    scheduler.samples = 2
    scheduler._retry_reconciled = True
    scheduler.peak_memory = {}
    scheduler.stop_sent = set()
    scheduler.pause_sent = set()
    runtime.submit(root, "retry", task_id=key, command_id="fixed")
    scheduler.tick()
    assert launches[0][-1] is True
    (child / "batch.json").write_text(json.dumps({"status": "paused"}))
    runtime.submit(root, "pause", task_id=key)
    scheduler.tick()
    assert scheduler.jobs[key]["status"] == "paused"
    assert scheduler.jobs[key]["retry_once_active"] is True
    runtime.submit(root, "resume", task_id=key)
    scheduler.tick()
    assert len(launches) == 2 and launches[1][-1] is True
    (child / "batch.json").write_text(json.dumps({"status": "failed"}))
    scheduler.tick()
    assert scheduler.jobs[key]["status"] == "failed"
    assert scheduler.jobs[key]["retry_once_active"] is False
    assert runtime.snapshot(root)["tasks"][key]["retry_once_active"] is False


def test_received_paused_retry_survives_stage_job_reconstruction(failed_queue):
    root, _, _ = failed_queue
    key = "trial--seed-4103"
    with runtime.transaction(root) as state:
        state["tasks"][key]["status"] = "paused"
        state["paused_tasks"] = [key]
    runtime.submit(root, "retry", task_id=key, command_id="fixed")
    observed = runtime.snapshot(root)
    jobs = {key: {"status": "paused"}, "ordinary": {"status": "paused"}}
    batch_retry.prepare_stage_jobs(jobs, observed)
    assert jobs[key]["status"] == "paused"
    assert jobs["ordinary"]["status"] == "queued"


def test_received_paused_retry_passes_run_children_recovery_path(
    failed_queue, monkeypatch
):
    from smartsom.experiments import author_batch, runtime_workers

    root, child, _ = failed_queue
    key = "trial--seed-4103"
    with runtime.transaction(root) as state:
        state["tasks"][key]["status"] = "paused"
        state["paused_tasks"] = [key]
    runtime.submit(root, "retry", task_id=key, command_id="fixed")
    monkeypatch.setattr(
        batch_retry,
        "inspect_target",
        lambda root_, key_, state, **kwargs: fake_info(root, child),
    )
    monkeypatch.setattr(author_batch, "_save", lambda *a: None)
    monkeypatch.setattr(author_batch, "_publish", lambda *a: None)
    monkeypatch.setattr(author_batch, "requested", lambda _: False)

    class FakeScheduler:
        def __init__(self, root_, jobs):
            self.root, self.jobs, self.processes = root_, jobs, {}

        def tick(self):
            assert self.jobs[key]["status"] == "paused"
            NativeScheduler._consume_retries(self, runtime.snapshot(root))
            assert runtime.snapshot(root)["commands"]["fixed"]["status"] == "applied"
            self.jobs[key]["status"] = "completed"
            return {"status": "running", "stopping": False}

        @property
        def unfinished(self):
            return False

        def close(self):
            raise AssertionError("unexpected close")

    monkeypatch.setattr(runtime_workers, "NativeScheduler", FakeScheduler)
    row = {
        "id": key,
        "child": str(child.relative_to(root)),
        "entry_ids": [key],
        "stage": 0,
        "task": "train",
    }
    plan = {"files": [row]}
    state = {"files": {key: {"status": "paused"}}, "status": "running"}
    assert author_batch._run_children(root, plan, state, [row], parallel_files=2)
    assert key not in runtime.snapshot(root)["paused_tasks"]


def test_conflicting_retry_views_block_without_overwriting_journal(
    failed_queue, monkeypatch
):
    from smartsom.experiments import queue_admission

    root, _, scheduler = failed_queue
    key = "trial--seed-4103"
    with runtime.transaction(root) as state:
        state["tasks"][key].update(
            status="failed", retry_pending=True, retry_attempt="attempt"
        )
        state["additions"][key] = dict(state["tasks"][key], status="starting")
    before = runtime.snapshot(root)
    scheduler._restore_queued_retries(before)
    assert scheduler.jobs[key]["retry_recovery_required"]
    monkeypatch.setattr(queue_admission, "snapshot", lambda _: {})
    scheduler._persist(before)
    after = runtime.snapshot(root)
    assert after["tasks"][key] == before["tasks"][key]
    assert after["additions"][key] == before["additions"][key]
    assert scheduler.unfinished


def test_rich_retry_selected_and_all_use_fixed_preview(failed_queue, monkeypatch):
    from types import SimpleNamespace

    from smartsom.telemetry.monitor import _retry_from_rich

    root, child, _ = failed_queue
    key = "trial--seed-4103"
    with runtime.transaction(root) as state:
        state["tasks"]["paused-seed"] = {
            "root": str(child),
            "status": "paused",
            "pid": 44,
        }
    monkeypatch.setattr(
        batch_retry,
        "inspect_target",
        lambda root_, key_, state, **kwargs: fake_info(root, child),
    )
    display = SimpleNamespace()
    notice = _retry_from_rich(root, display, key)
    assert key in notice and "paused-seed" not in notice
    assert len(runtime.snapshot(root)["commands"]) == 1
    notice = _retry_from_rich(root, display)
    assert "paused-seed" in notice and key in notice
    assert "already pending" in notice
    assert runtime.snapshot(root)["tasks"]["other"]["status"] == "running"
    assert len(display.runtime_local_commands) == 3


def test_rich_retry_gate_keeps_all_tasks_unchanged(failed_queue, monkeypatch):
    from types import SimpleNamespace

    from smartsom.telemetry.monitor import _retry_from_rich

    root, _, _ = failed_queue
    monkeypatch.setattr(runtime, "_RETRY_ENABLED", False)
    with pytest.raises(ValueError, match="尚未启用"):
        _retry_from_rich(root, SimpleNamespace())
    assert not runtime.snapshot(root)["commands"]


@pytest.mark.parametrize(
    ("status", "enabled", "expected"),
    [
        ("failed", True, "retry"),
        ("failed", False, "retry"),
        ("paused", True, "resume"),
        ("paused", False, "resume"),
        ("stopped", True, "resume"),
        ("running", True, "resume"),
    ],
)
def test_rich_selected_resume_dispatch(status, enabled, expected):
    from smartsom.telemetry.monitor import _selected_resume_action

    assert _selected_resume_action(status, enabled) == expected


def test_rich_retry_rejections_remain_in_feedback(failed_queue, monkeypatch):
    from types import SimpleNamespace

    from smartsom.telemetry.monitor import _retry_from_rich
    from smartsom.telemetry.tuning_dashboard import runtime_feedback_detail

    root, _, _ = failed_queue
    monkeypatch.setattr(
        batch_retry,
        "inspect_target",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("no checkpoint")),
    )
    view = SimpleNamespace(tuning={})
    notice = _retry_from_rich(root, view, "trial--seed-4103")
    assert "no checkpoint" in notice
    detail = runtime_feedback_detail(view)
    assert any("no checkpoint" in line for line in detail)
    assert any("冻结目标 1" in line for line in detail)
    assert not runtime.snapshot(root)["commands"]


def test_bulk_retry_resumes_paused_tasks_without_training_checkpoint(failed_queue):
    root, _, _ = failed_queue
    with runtime.transaction(root) as state:
        state["tasks"].update(
            {
                "never-started": {"status": "paused", "pid": None},
                "evaluation": {"status": "paused", "pid": 55, "task": "evaluate"},
                "multiworker": {"status": "paused", "own_slots": True},
                "deferred": {"status": "deferred"},
            }
        )
        state["paused_tasks"] = ["never-started", "evaluation", "multiworker"]
    preview = batch_retry.preview_failed(
        root, ["never-started", "evaluation", "multiworker"], include_paused=True
    )
    assert all(item["action"] == "resume" for item in preview["targets"])
    result = batch_retry.submit_preview(root, preview)
    assert all(item["status"] == "received" for item in result["outcomes"])
    with runtime.transaction(root) as state:
        runtime._apply(state)
    state = runtime.snapshot(root)
    assert not state["paused_tasks"]
    assert state["tasks"]["deferred"]["status"] == "deferred"
    assert state["tasks"]["other"]["status"] == "running"


def test_bulk_retry_paused_target_changes_rejected_at_submit_and_apply(failed_queue):
    root, _, _ = failed_queue
    with runtime.transaction(root) as state:
        state["tasks"]["paused"] = {"status": "paused", "pid": 44}
        state["paused_tasks"] = ["paused"]
    preview = batch_retry.preview_failed(root, ["paused"], include_paused=True)
    with runtime.transaction(root) as state:
        state["tasks"]["paused"]["status"] = "failed"
    assert (
        batch_retry.submit_preview(root, preview)["outcomes"][0]["status"] == "rejected"
    )
    with runtime.transaction(root) as state:
        state["tasks"]["paused"]["status"] = "paused"
    preview = batch_retry.preview_failed(root, ["paused"], include_paused=True)
    result = batch_retry.submit_preview(root, preview)
    command = result["outcomes"][0]["receipt"]["id"]
    with runtime.transaction(root) as state:
        state["tasks"]["paused"]["status"] = "failed"
        runtime._apply(state)
    state = runtime.snapshot(root)
    assert state["commands"][command]["status"] == "rejected"
    assert "paused" in state["paused_tasks"]


def test_bulk_resume_new_pause_cycle_creates_new_command(failed_queue):
    root, _, _ = failed_queue
    with runtime.transaction(root) as state:
        state["tasks"]["held"] = {"status": "paused", "pid": None}
        state["paused_tasks"] = ["held"]
    first = batch_retry.preview_failed(root, ["held"], include_paused=True)
    first_receipt = batch_retry.submit_preview(root, first)["outcomes"][0]["receipt"]
    with runtime.transaction(root) as state:
        runtime._apply(state)
    runtime.submit(root, "pause", task_id="held")
    with runtime.transaction(root) as state:
        runtime._apply(state)
        assert "held" in state["paused_tasks"]
    second = batch_retry.preview_failed(root, ["held"], include_paused=True)
    assert first["preview_sha256"] != second["preview_sha256"]
    second_receipt = batch_retry.submit_preview(root, second)["outcomes"][0]["receipt"]
    assert first_receipt["id"] != second_receipt["id"]
    with runtime.transaction(root) as state:
        runtime._apply(state)
        assert "held" not in state["paused_tasks"]
    assert (
        batch_retry.submit_preview(root, second)["outcomes"][0]["receipt"]["id"]
        == second_receipt["id"]
    )


def test_conflicting_starting_retry_tick_preserves_both_journal_views(
    failed_queue, monkeypatch
):
    from types import SimpleNamespace

    from smartsom.experiments import queue_admission

    root, _, scheduler = failed_queue
    key = "trial--seed-4103"
    with runtime.transaction(root) as state:
        state["tasks"][key].update(
            status="failed", retry_pending=True, retry_attempt="attempt"
        )
        state["additions"][key] = dict(state["tasks"][key], status="starting")
    before = runtime.snapshot(root)
    scheduler._restore_queued_retries(before)
    scheduler._retry_reconciled = True
    scheduler.jobs[key].update(status="starting", retry_pending=True)
    scheduler.observer = SimpleNamespace(
        snapshot=lambda: SimpleNamespace(
            cpus=8, cpu_load=0, memory_total=16 * 1024**3, memory_available=14 * 1024**3
        )
    )
    scheduler.samples = 2
    scheduler.peak_memory = {}
    scheduler.stop_sent = set()
    scheduler.pause_sent = set()
    monkeypatch.setattr(queue_admission, "snapshot", lambda _: {})
    scheduler.tick()
    after = runtime.snapshot(root)
    assert after["tasks"][key] == before["tasks"][key]
    assert after["additions"][key] == before["additions"][key]
    assert scheduler.jobs[key]["retry_recovery_required"]


def test_protected_retry_conflict_freezes_entire_foreign_owner(
    failed_queue, monkeypatch
):
    from smartsom.experiments import batch_append

    root, _, _ = failed_queue
    with runtime.transaction(root) as state:
        state["additions"] = {
            "conflict": {"owner_root": "/foreign/a", "status": "failed"},
            "safe-same-owner": {"owner_root": "/foreign/a", "status": "completed"},
            "safe-other-owner": {"owner_root": "/foreign/b", "status": "completed"},
        }
    seen = []
    monkeypatch.setattr(
        batch_append, "_synchronize_locked", lambda root_, jobs: seen.append(dict(jobs))
    )
    state = runtime.snapshot(root)
    safe = {
        key: state["additions"][key] for key in ("safe-same-owner", "safe-other-owner")
    }
    batch_append.synchronize(root, safe, protected={"conflict"})
    assert set(seen[0]) == {"safe-other-owner"}
    assert (
        runtime.snapshot(root)["additions"]["conflict"]
        == state["additions"]["conflict"]
    )


def test_retry_feature_is_enabled_by_default():
    assert runtime._RETRY_ENABLED is True


def test_restore_selected_ignores_transient_ordinary_addition_views(failed_queue):
    root, child, _ = failed_queue
    key = "ordinary"
    plan = {
        "incremental": True,
        "files": [{"id": key, "child": str(child.relative_to(root)), "stage": 0}],
    }
    saved = {
        "status": "running",
        "selection": {"tasks": []},
        "files": {key: {"status": "deferred"}},
    }
    observed = {
        "tasks": {key: {"status": "deferred"}},
        "additions": {key: {"status": "queued"}},
        "commands": {},
    }
    assert batch_retry.restore_selected_state(root, plan, saved, observed) == []
    assert saved["files"][key]["status"] == "deferred"


def test_restore_selected_allows_nonincremental_null_selection(failed_queue):
    root, child, _ = failed_queue
    key = "plain-task"
    plan = {
        "incremental": False,
        "files": [{"id": key, "child": str(child.relative_to(root)), "stage": 0}],
    }
    saved = {
        "status": "running",
        "selection": None,
        "files": {key: {"status": "queued"}},
    }
    observed = {"tasks": {key: {"status": "queued"}}, "additions": {}, "commands": {}}
    assert batch_retry.restore_selected_state(root, plan, saved, observed) == []


def test_restore_nonincremental_null_selection_with_applied_retry(failed_queue):
    root, child, _ = failed_queue
    key = "trial--seed-4103"
    plan = {
        "incremental": False,
        "files": [{"id": key, "child": str(child.relative_to(root)), "stage": 0}],
    }
    saved = {
        "status": "failed",
        "selection": None,
        "files": {key: {"status": "failed"}},
    }
    job = {
        "root": str(child),
        "status": "queued",
        "retry_pending": True,
        "retry_attempt": "attempt",
        "retry_history": [{"command_id": "command"}],
    }
    observed = {
        "tasks": {key: job},
        "additions": {},
        "commands": {"command": {"status": "applied", "retry_attempt": "attempt"}},
    }
    assert batch_retry.restore_selected_state(root, plan, saved, observed) == [key]
    assert saved["files"][key]["status"] == "queued"
    assert saved["status"] == "prepared"
