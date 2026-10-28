"""Public runtime control contracts (CI owner: control)."""

import multiprocessing
from concurrent.futures import ProcessPoolExecutor

import pytest

from smartsom.experiments import runtime_control as runtime


def _submit_delta(arguments):
    root, identity = arguments
    receipt = runtime.submit(root, "delta", value=1, command_id=identity)
    # Independent windows can also acknowledge queued work concurrently.
    runtime.apply(root)
    return receipt


def test_concurrent_controllers_preserve_every_delta_and_idempotency(tmp_path):
    runtime.initialize(tmp_path, 2)
    with ProcessPoolExecutor(
        max_workers=4, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        receipts = list(
            pool.map(_submit_delta, [(str(tmp_path), f"c{i}") for i in range(16)])
        )
    state = runtime.apply(tmp_path)
    assert state["limit"] == 18
    assert sorted(row["revision"] for row in receipts) == list(range(1, 17))
    assert state["applied_revision"] == 16
    assert all(row["status"] == "applied" for row in state["commands"].values())
    repeated = runtime.submit(tmp_path, "delta", value=1, command_id="c0")
    assert repeated["revision"] == receipts[0]["revision"]
    assert runtime.apply(tmp_path)["limit"] == 18
    with pytest.raises(ValueError, match="different payload"):
        runtime.submit(tmp_path, "delta", value=-1, command_id="c0")
    with pytest.raises(ValueError, match="revision changed"):
        runtime.submit(tmp_path, "limit", value=3, expected_revision=0)
    assert runtime.snapshot(tmp_path)["revision"] == 16


def test_receipt_is_distinct_from_applied_intent_and_running_observation(tmp_path):
    runtime.initialize(tmp_path, 4)
    runtime.publish(tmp_path, active=4)
    receipt = runtime.submit(tmp_path, "limit", value=2)
    assert receipt["status"] == "received"
    assert runtime.snapshot(tmp_path)["limit"] == 4
    applied = runtime.apply(tmp_path)
    assert applied["commands"][receipt["id"]]["status"] == "applied"
    assert applied["active"] == 4
    state = runtime.publish(tmp_path, active=4)
    assert state["status"] == "draining"
    assert state["draining"] is True
    state = runtime.publish(tmp_path, active=1, blocking_reason="insufficient memory")
    assert state["limit"] == 2
    assert state["active"] == 1
    assert state["blocking_reason"] == "insufficient memory"
    assert runtime.publish(tmp_path, active=2)["blocking_reason"] is None


def test_restart_preserves_last_control_settings_and_individual_holds(tmp_path):
    runtime.initialize(tmp_path, 4)
    runtime.publish(
        tmp_path, active=0, tasks={"a": {"status": "queued"}, "b": {"status": "queued"}}
    )
    runtime.submit(tmp_path, "limit", value=6)
    runtime.submit(tmp_path, "mode", value="adaptive")
    runtime.submit(tmp_path, "pause", task_id="a")
    runtime.submit(tmp_path, "stop", task_id="b")
    runtime.submit(tmp_path, "pause")
    state = runtime.apply(tmp_path)
    assert state["paused_tasks"] == ["a"]
    assert state["stopped_tasks"] == ["b"]
    assert state["paused"]
    runtime.submit(tmp_path, "stop")
    runtime.apply(tmp_path)
    assert runtime.finish_if_drained(tmp_path)
    assert runtime.snapshot(tmp_path)["status"] == "stopped"
    runtime.restart(tmp_path)
    state = runtime.initialize(tmp_path, 1, mode="manual")
    assert (state["limit"], state["mode"]) == (6, "adaptive")
    assert state["paused_tasks"] == ["a"]
    assert state["stopped_tasks"] == ["b"]
    assert not state["stopping"]
    assert not state["paused"]
    runtime.submit(tmp_path, "resume")
    assert runtime.apply(tmp_path)["stopped_tasks"] == ["b"]
    runtime.submit(tmp_path, "resume", task_id="b")
    state = runtime.apply(tmp_path)
    assert state["stopped_tasks"] == []
    assert any(
        row["payload"]["kind"] == "stop" and row["payload"]["task_id"] == "b"
        for row in state["commands"].values()
    )


def test_invalid_task_and_out_of_range_delta_are_rejected_without_mutation(tmp_path):
    runtime.initialize(tmp_path, 1)
    first = runtime.submit(tmp_path, "delta", value=-1)
    second = runtime.submit(tmp_path, "pause", task_id="missing")
    state = runtime.apply(tmp_path)
    assert state["limit"] == 1
    assert state["paused_tasks"] == []
    assert state["commands"][first["id"]]["status"] == "rejected"
    assert state["commands"][second["id"]]["status"] == "rejected"


def test_drain_keep_open_and_pause_are_different(tmp_path):
    runtime.initialize(tmp_path, 1, keep_open=True)
    assert not runtime.finish_if_drained(tmp_path)
    runtime.submit(tmp_path, "pause")
    runtime.apply(tmp_path)
    runtime.submit(tmp_path, "close")
    assert not runtime.finish_if_drained(tmp_path)
    runtime.submit(tmp_path, "resume")
    assert not runtime.finish_if_drained(tmp_path, unfinished=True)
    assert runtime.finish_if_drained(tmp_path)
    assert runtime.snapshot(tmp_path)["status"] == "completed"
    receipt = runtime.submit(tmp_path, "limit", value=2)
    assert receipt["status"] == "rejected"


def test_overlapping_retries_apply_same_command_once(tmp_path):
    runtime.initialize(tmp_path, 2)
    with ProcessPoolExecutor(
        max_workers=4, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        receipts = list(pool.map(_submit_delta, [(str(tmp_path), "same-command")] * 8))
    assert {row["revision"] for row in receipts} == {1}
    state = runtime.apply(tmp_path)
    assert state["limit"] == 3
    assert len(state["commands"]) == 1


def test_removal_challenge_is_readonly_and_bound_to_exact_task_and_revision(tmp_path):
    runtime.initialize(tmp_path, 2)
    runtime.publish(
        tmp_path,
        active=1,
        tasks={"a": {"status": "running"}, "b": {"status": "queued"}},
    )
    before = runtime.snapshot(tmp_path)
    target = runtime.removal_target(tmp_path, "a")
    assert runtime.snapshot(tmp_path) == before
    for identity, confirmation in (
        ("a", None),
        ("a", "wrong"),
        ("b", target["confirmation"]),
    ):
        with pytest.raises(ValueError, match="confirmation"):
            runtime.submit(
                tmp_path,
                "remove",
                task_id=identity,
                confirmation=confirmation,
                expected_revision=target["revision"],
            )
        assert runtime.snapshot(tmp_path) == before
    runtime.submit(tmp_path, "delta", value=1)
    with pytest.raises(ValueError, match="revision changed"):
        runtime.submit(
            tmp_path,
            "remove",
            task_id="a",
            confirmation=target["confirmation"],
            expected_revision=target["revision"],
        )


def test_removal_rejects_completed_or_already_removed_target_and_retry_is_idempotent(
    tmp_path,
):
    runtime.initialize(tmp_path, 2)
    runtime.publish(
        tmp_path,
        active=1,
        tasks={"a": {"status": "running"}, "b": {"status": "queued"}},
    )
    completed = runtime.removal_target(tmp_path, "a")
    runtime.publish(tmp_path, active=0, tasks={"a": {"status": "completed"}})
    with pytest.raises(ValueError, match="no longer active"):
        runtime.submit(
            tmp_path,
            "remove",
            task_id="a",
            confirmation=completed["confirmation"],
            expected_revision=completed["revision"],
        )
    target = runtime.removal_target(tmp_path, "b")
    arguments = dict(
        task_id="b",
        confirmation=target["confirmation"],
        expected_revision=target["revision"],
        command_id="remove-b",
    )
    receipt = runtime.submit(tmp_path, "remove", **arguments)
    runtime.apply(tmp_path)
    retried = runtime.submit(tmp_path, "remove", **arguments)
    assert retried["revision"] == receipt["revision"]
    assert retried["status"] == "applied"
    with pytest.raises(ValueError, match="already removed"):
        runtime.removal_target(tmp_path, "b")
    assert runtime.snapshot(tmp_path)["removed_tasks"] == ["b"]


def _reserve_mixed(arguments):
    from smartsom.experiments import queue_admission

    root, key = arguments
    return queue_admission.reserve(root, key, owner_id="shared", cpus=1, memory=400)


def test_shared_budget_atomically_caps_mixed_scheduler_reservations(tmp_path):
    from smartsom.experiments import queue_admission
    from smartsom.experiments.evidence import write_json

    runtime.initialize(tmp_path, 20)
    write_json(tmp_path / "control/owner.json", {"id": "shared"})
    queue_admission.initialize(tmp_path, supported=True)
    runtime.submit(tmp_path, "budget", value={"cpus": 3, "memory": 1000})
    runtime.apply(tmp_path)
    with ProcessPoolExecutor(
        max_workers=4, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        results = list(
            pool.map(
                _reserve_mixed,
                [
                    (str(tmp_path), ("source:" if i % 2 else "native:") + str(i))
                    for i in range(12)
                ],
            )
        )
    assert sum(results) == 2
    slots = queue_admission.snapshot(tmp_path)["slots"]
    assert sum(slot["cpus"] for slot in slots.values()) == 2
    assert sum(slot["memory"] for slot in slots.values()) == 800
    key = next(iter(slots))
    queue_admission.update_resources(tmp_path, key, owner_id="shared", memory=1100)
    assert not queue_admission.reserve(
        tmp_path, "native:extra", owner_id="shared", cpus=1, memory=1
    )
    assert queue_admission.snapshot(tmp_path)["slots"][key]["memory"] == 1100


def test_shared_waiter_fairness_ignores_impossible_and_held_requests(
    tmp_path, monkeypatch
):
    from smartsom.experiments import queue_admission
    from smartsom.experiments.evidence import write_json

    runtime.initialize(tmp_path, 20)
    runtime.publish(tmp_path, active=0, tasks={"old": {"status": "queued"}})
    write_json(tmp_path / "control/owner.json", {"id": "shared"})
    queue_admission.initialize(tmp_path, supported=True)
    runtime.submit(tmp_path, "budget", value={"cpus": 4, "memory": 2000})
    runtime.apply(tmp_path)
    clock = [0.0]
    monkeypatch.setattr(queue_admission.time, "monotonic", lambda: clock[0])
    queue_admission.note_wait(
        tmp_path, "source:old", owner_id="shared", cpus=4, memory=1500
    )
    clock[0] = 31
    assert not queue_admission.reserve(
        tmp_path, "native:young", owner_id="shared", cpus=1, memory=100
    )
    queue_admission.note_wait(
        tmp_path, "source:old", owner_id="shared", cpus=4, memory=1500, feasible=False
    )
    assert queue_admission.reserve(
        tmp_path, "native:young", owner_id="shared", cpus=1, memory=100
    )
    queue_admission.retire(tmp_path, "native:young", owner_id="shared")
    queue_admission.note_wait(
        tmp_path, "source:old", owner_id="shared", cpus=4, memory=1500
    )
    clock[0] = 62
    runtime.submit(tmp_path, "pause", task_id="old")
    runtime.apply(tmp_path)
    assert queue_admission.reserve(
        tmp_path, "native:young", owner_id="shared", cpus=1, memory=100
    )


def test_task_identifiers_do_not_imply_parentage_from_name_prefix(tmp_path):
    runtime.initialize(tmp_path, 2)
    runtime.publish(
        tmp_path,
        active=0,
        tasks={"foo": {"status": "queued"}, "foo__bar": {"status": "queued"}},
    )
    runtime.submit(tmp_path, "pause", task_id="foo")
    state = runtime.apply(tmp_path)
    assert runtime.matches("foo", state["paused_tasks"])
    assert not runtime.matches("foo__bar", state["paused_tasks"])
    assert state["tasks"]["foo__bar"]["status"] == "queued"


def test_shared_waiter_requires_fresh_verified_usage_and_rechecks_current_host(
    tmp_path, monkeypatch
):
    import os
    from types import SimpleNamespace

    import psutil

    from smartsom.experiments import queue_admission
    from smartsom.experiments.evidence import write_json
    from smartsom.experiments.tuning_resources import ResourceSnapshot

    runtime.initialize(tmp_path, 8)
    write_json(tmp_path / "control/owner.json", {"id": "shared"})
    queue_admission.initialize(tmp_path, supported=True)
    clock = [0.0]
    monkeypatch.setattr(
        queue_admission,
        "time",
        SimpleNamespace(monotonic=lambda: clock[0], sleep=lambda _: None),
    )
    host = ResourceSnapshot(8, 16 * 1024**3, 12 * 1024**3, cpu_load=37.5)
    assert queue_admission.reserve(
        tmp_path, "native:busy", owner_id="shared", cpus=3, memory=320 * 1024**2
    )
    process = psutil.Process(os.getpid())

    def observe(birth):
        queue_admission.update_resources(
            tmp_path,
            "native:busy",
            owner_id="shared",
            memory=320 * 1024**2,
            observed_memory=process.memory_info().rss,
            observed_cpus=3,
            observed_processes=[(process.pid, birth)],
        )

    def wait():
        queue_admission.note_wait(
            tmp_path,
            "source:large",
            owner_id="shared",
            cpus=5,
            memory=320 * 1024**2,
            host=host,
            cpu_reserve=1.6,
        )
        return queue_admission.snapshot(tmp_path)["waiters"]["source:large"]

    # Future reservations and a stale/reused PID are never actual reclaimable CPU.
    assert not wait()["feasible"]
    observe(process.create_time() - 1)
    assert not wait()["feasible"]
    observe(process.create_time())
    assert wait()["feasible"]
    clock[0] = 6
    assert not wait()["feasible"]
    observe(process.create_time())
    assert wait()["feasible"]
    clock[0] = 37
    observe(process.create_time())
    assert wait()["since"] == 6
    assert not queue_admission.reserve(
        tmp_path, "native:small", owner_id="shared", cpus=1, memory=1, host=host
    )
    queue_admission.retire(tmp_path, "native:busy", owner_id="shared")
    changed = ResourceSnapshot(8, 16 * 1024**3, 12 * 1024**3, cpu_load=25)
    # 5 CPUs no longer fit after the Tune 1.6-CPU reserve, though a small job fits.
    assert queue_admission.reserve(
        tmp_path, "native:small", owner_id="shared", cpus=1, memory=1, host=changed
    )
    assert not queue_admission.snapshot(tmp_path)["waiters"]["source:large"]["feasible"]


def test_performance_waiter_cannot_age_above_shared_memory_envelope(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    from smartsom.experiments import queue_admission
    from smartsom.experiments.evidence import write_json
    from smartsom.experiments.tuning_resources import GIB, ResourceSnapshot

    runtime.initialize(tmp_path, 4)
    write_json(tmp_path / "control/owner.json", {"id": "shared"})
    queue_admission.initialize(tmp_path, supported=True)
    clock = [0.0]
    monkeypatch.setattr(
        queue_admission,
        "time",
        SimpleNamespace(monotonic=lambda: clock[0], sleep=lambda _: None),
    )
    host = ResourceSnapshot(8, 16 * GIB, 12 * GIB, external_cpu_load=0)
    # 9 GiB fits the Tune performance 10% reserve, but not shared 20% reserve.
    queue_admission.note_wait(
        tmp_path,
        "source:large",
        owner_id="shared",
        cpus=1,
        memory=9 * GIB,
        host=host,
        cpu_reserve=1,
        memory_reserve=int(16 * GIB * 0.1),
    )
    assert not queue_admission.snapshot(tmp_path)["waiters"]["source:large"]["feasible"]
    clock[0] = 31
    assert queue_admission.reserve(
        tmp_path, "native:small", owner_id="shared", cpus=1, memory=GIB, host=host
    )


def test_placeholder_settlement_retains_newer_persistent_growth(tmp_path):
    from smartsom.experiments import queue_admission
    from smartsom.experiments.evidence import write_json

    runtime.initialize(tmp_path, 2)
    write_json(tmp_path / "control/owner.json", {"id": "shared"})
    queue_admission.initialize(tmp_path, supported=True)
    assert queue_admission.reserve(
        tmp_path,
        "source:a",
        owner_id="shared",
        cpus=1,
        memory=1000,
        temporary_memory=True,
        memory_floor=200,
    )
    queue_admission.update_resources(
        tmp_path, "source:a", owner_id="shared", memory=800
    )
    assert (
        queue_admission.settle_memory_placeholder(
            tmp_path, "source:a", owner_id="shared", memory=300
        )
        == 800
    )
    assert (
        queue_admission.settle_memory_placeholder(
            tmp_path, "source:a", owner_id="shared", memory=100
        )
        is None
    )
    assert queue_admission.snapshot(tmp_path)["slots"]["source:a"]["memory"] == 800


def test_transition_progress_waits_for_observation_and_snapshot_never_writes(tmp_path):
    runtime.initialize(tmp_path, 2)
    runtime.publish(tmp_path, active=1, tasks={"a": {"status": "running"}})
    receipt = runtime.submit(tmp_path, "pause", task_id="a")
    assert receipt["status"] == "received"
    assert runtime.submit(tmp_path, "pause", task_id="a")["id"] == receipt["id"]
    state = runtime.apply(tmp_path)
    assert state["commands"][receipt["id"]]["progress"]["state"] == "waiting"
    path = tmp_path / "control/runtime.json"
    before = (path.read_bytes(), path.stat().st_mtime_ns)
    runtime.snapshot(tmp_path)
    assert (path.read_bytes(), path.stat().st_mtime_ns) == before
    resume = runtime.submit(tmp_path, "resume", task_id="a")
    state = runtime.apply(tmp_path)
    assert state["commands"][receipt["id"]]["progress"]["state"] == "superseded"
    assert state["commands"][resume["id"]]["progress"]["state"] == "completed"
    frozen = state["commands"][receipt["id"]]
    runtime.publish(tmp_path, active=0, tasks={"a": {"status": "paused"}})
    assert runtime.snapshot(tmp_path)["commands"][receipt["id"]] == frozen


def test_missing_or_failed_transition_is_never_reported_safely_complete(tmp_path):
    runtime.initialize(tmp_path, 1)
    runtime.publish(tmp_path, active=1, tasks={"a": {"status": "running"}})
    receipt = runtime.submit(tmp_path, "pause")
    runtime.apply(tmp_path)
    with runtime.transaction(tmp_path) as state:
        state["tasks"].clear()
    state = runtime.publish(tmp_path, active=0)
    progress = state["commands"][receipt["id"]]["progress"]
    assert progress["state"] == "waiting"
    assert progress["tasks"]["a"]["phase"] == "unavailable"
    state = runtime.publish(tmp_path, active=0, status="stopped")
    assert state["commands"][receipt["id"]]["progress"]["state"] == "failed"


def test_partial_pause_failure_keeps_other_targets_live_and_avoids_double_counts(
    tmp_path,
):
    runtime.initialize(tmp_path, 3)
    runtime.publish(
        tmp_path,
        active=2,
        tasks={
            "matrix": {"status": "running", "own_slots": True},
            "a": {"status": "running", "file_id": "matrix"},
            "b": {"status": "running", "file_id": "matrix"},
        },
    )
    receipt = runtime.submit(tmp_path, "pause")
    runtime.apply(tmp_path)
    state = runtime.publish(
        tmp_path,
        active=1,
        tasks={
            "a": {
                "status": "failed",
                "error": "checkpoint write failed",
                "file_id": "matrix",
            },
            "b": {"status": "pausing", "file_id": "matrix"},
        },
    )
    progress = state["commands"][receipt["id"]]["progress"]
    assert progress["targets"] == ["a", "b"]
    assert progress["state"] == "waiting"
    assert progress["failed_count"] == 1
    assert progress["tasks"]["a"]["reason"] == "checkpoint write failed"
    assert runtime.submit(tmp_path, "pause")["id"] == receipt["id"]
    state = runtime.publish(
        tmp_path, active=0, tasks={"b": {"status": "completed", "file_id": "matrix"}}
    )
    progress = state["commands"][receipt["id"]]["progress"]
    assert progress["state"] == "failed"
    assert progress["paused_count"] == 0
    assert progress["naturally_completed_count"] == 1
    assert progress["settled_count"] == 2


@pytest.mark.parametrize(
    "kind,pending,finished",
    [("resume", "queued", "running"), ("remove", "removing", "removed")],
)
def test_resume_and_remove_progress_follow_actual_transition(
    tmp_path, kind, pending, finished
):
    runtime.initialize(tmp_path, 1)
    runtime.publish(tmp_path, active=0, tasks={"a": {"status": "paused"}})
    extra = {}
    if kind == "remove":
        target = runtime.removal_target(tmp_path, "a")
        extra = dict(
            confirmation=target["confirmation"], expected_revision=target["revision"]
        )
    receipt = runtime.submit(tmp_path, kind, task_id="a", **extra)
    runtime.apply(tmp_path)
    state = runtime.publish(
        tmp_path,
        active=0,
        tasks={
            "a": {"status": pending, "blocking_reason": "resource admission pending"}
        },
    )
    progress = state["commands"][receipt["id"]]["progress"]
    assert progress["state"] == "waiting"
    assert progress["tasks"]["a"]["reason"] == "resource admission pending"
    state = runtime.publish(
        tmp_path, active=int(finished == "running"), tasks={"a": {"status": finished}}
    )
    assert state["commands"][receipt["id"]]["progress"]["state"] == "completed"


def test_global_target_set_freezes_at_apply_not_received_or_later_publish(tmp_path):
    runtime.initialize(tmp_path, 2)
    runtime.publish(tmp_path, active=1, tasks={"a": {"status": "running"}})
    receipt = runtime.submit(tmp_path, "pause")
    runtime.publish(tmp_path, active=2, tasks={"b": {"status": "running"}})
    state = runtime.apply(tmp_path)
    assert state["commands"][receipt["id"]]["progress"]["targets"] == ["a", "b"]
    state = runtime.publish(tmp_path, active=2, tasks={"later": {"status": "queued"}})
    assert state["commands"][receipt["id"]]["progress"]["targets"] == ["a", "b"]


def test_pending_pause_dedup_uses_revision_not_json_command_id_order(tmp_path):
    runtime.initialize(tmp_path, 1)
    runtime.publish(tmp_path, active=1, tasks={"worker": {"status": "running"}})
    first = runtime.submit(tmp_path, "pause", command_id="z")
    resumed = runtime.submit(tmp_path, "resume", command_id="a")
    last = runtime.submit(tmp_path, "pause", command_id="b")
    assert [first["revision"], resumed["revision"], last["revision"]] == [1, 2, 3]
    assert last["id"] == "b"
    assert runtime.submit(tmp_path, "pause", command_id="c")["id"] == "b"
    state = runtime.apply(tmp_path)
    assert state["paused"] is True
    assert state["commands"]["z"]["progress"]["state"] == "superseded"
    assert state["commands"]["b"]["progress"]["state"] == "waiting"


def test_priority_queue_is_ordered_persistent_and_rejects_started_targets(tmp_path):
    runtime.initialize(tmp_path, 2)
    runtime.publish(
        tmp_path,
        active=1,
        tasks={
            "a": {"status": "running"},
            "b": {"status": "queued"},
            "c": {"status": "queued"},
            "c__child": {"status": "queued"},
        },
    )
    first = runtime.submit(
        tmp_path, "priority", task_id="c", value="add", command_id="first"
    )
    assert (
        runtime.submit(
            tmp_path, "priority", task_id="c", value="add", command_id="first"
        )
        == first
    )
    runtime.submit(tmp_path, "priority", task_id="b", value="add")
    state = runtime.apply(tmp_path)
    assert state["priority_tasks"] == ["c", "b"]
    assert runtime.priority_order(state, ["b", "c__child", "c"]) == [
        "c",
        "b",
        "c__child",
    ]
    runtime.submit(tmp_path, "priority", task_id="b", value="earlier")
    assert runtime.apply(tmp_path)["priority_tasks"] == ["b", "c"]
    runtime.restart(tmp_path)
    assert runtime.snapshot(tmp_path)["priority_tasks"] == ["b", "c"]
    stale = runtime.submit(tmp_path, "priority", task_id="c", value="first")
    runtime.publish(tmp_path, active=2, tasks={"c": {"status": "running"}})
    state = runtime.apply(tmp_path)
    assert state["commands"][stale["id"]]["status"] == "rejected"
    assert state["priority_tasks"] == ["b"]
    assert state["tasks"]["a"]["status"] == "running"
    runtime.submit(tmp_path, "priority", task_id="b", value="remove")
    assert runtime.apply(tmp_path)["priority_tasks"] == []


def test_priority_overrides_waiter_age_but_not_resource_or_hold_guards(
    tmp_path, monkeypatch
):
    from smartsom.experiments import queue_admission
    from smartsom.experiments.evidence import write_json

    runtime.initialize(tmp_path, 2)
    runtime.publish(
        tmp_path, active=0, tasks={key: {"status": "queued"} for key in ("old", "next")}
    )
    write_json(tmp_path / "control/owner.json", {"id": "owner"})
    queue_admission.initialize(tmp_path, supported=True)
    runtime.submit(tmp_path, "budget", value={"cpus": 2, "memory": 2000})
    runtime.apply(tmp_path)
    clock = [0.0]
    monkeypatch.setattr(queue_admission.time, "monotonic", lambda: clock[0])
    queue_admission.note_wait(
        tmp_path, "source:old", owner_id="owner", cpus=1, memory=500
    )
    clock[0] = 40
    queue_admission.note_wait(
        tmp_path, "native:next", owner_id="owner", cpus=1, memory=500
    )
    runtime.submit(tmp_path, "priority", task_id="next", value="add")
    runtime.apply(tmp_path)
    assert not queue_admission.reserve(
        tmp_path, "source:old", owner_id="owner", cpus=1, memory=500
    )
    assert not queue_admission.reserve(
        tmp_path, "native:next", owner_id="owner", cpus=3, memory=500
    )
    assert queue_admission.reserve(
        tmp_path, "native:next", owner_id="owner", cpus=1, memory=500
    )
    queue_admission.retire(tmp_path, "native:next", owner_id="owner")
    queue_admission.note_wait(
        tmp_path, "native:next", owner_id="owner", cpus=1, memory=500
    )
    runtime.submit(tmp_path, "pause", task_id="next")
    runtime.apply(tmp_path)
    assert queue_admission.reserve(
        tmp_path, "source:old", owner_id="owner", cpus=1, memory=500
    )
