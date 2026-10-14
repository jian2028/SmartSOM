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
    runtime.submit(tmp_path, "resume")
    state = runtime.apply(tmp_path)
    assert state["paused_tasks"] == ["a"]
    assert state["stopped_tasks"] == ["b"]
    assert not state["paused"]
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


def test_matrix_waiter_uses_external_pressure_and_owned_memory(tmp_path):
    from smartsom.experiments import queue_admission
    from smartsom.experiments.author_driver import _native_source_reserve
    from smartsom.experiments.evidence import write_json
    from smartsom.experiments.tuning_resources import ResourceSnapshot

    runtime.initialize(tmp_path, 4)
    write_json(tmp_path / "control/owner.json", {"id": "shared"})
    queue_admission.initialize(tmp_path, supported=True)
    observed = ResourceSnapshot(
        cpus=8,
        memory_total=32 * 1024**3,
        memory_available=24 * 1024**3,
        cpu_load=80,
        external_cpu_load=80,
    )
    assert not _native_source_reserve(
        (tmp_path, "shared", "matrix"), "large", cpus=4, observed=observed
    )
    assert (
        queue_admission.snapshot(tmp_path)["waiters"]["source:matrix__large"][
            "feasible"
        ]
        is False
    )
    assert _native_source_reserve(
        (tmp_path, "shared", "matrix"), "small", cpus=1, observed=observed
    )


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


def test_matrix_entry_resources_are_samples_and_clear_on_unknown_or_exit(tmp_path):
    from smartsom.experiments.author_driver import _entry_resource_observation

    row = {"status": "running", "resource_estimate": {"cpus": 8, "memory": 10**10}}
    _entry_resource_observation(row)
    assert row["resources"]["status"] == "unavailable"
    assert row["resources"]["memory"] is None
    assert row["resources"]["cpu_cores"] is None
    _entry_resource_observation(row, memory=1024, cpu=None)
    assert row["resources"]["cpu_cores"] is None
    _entry_resource_observation(row, memory=512, cpu=0.25)
    sampled = dict(row["resources"])
    assert sampled["memory_peak"] == 1024
    assert sampled["cpu_cores"] == 0.25
    _entry_resource_observation(row)
    assert row["resources"]["status"] == "unavailable"
    assert row["resources"]["memory"] is None
    assert row["resources"]["memory_peak"] == 1024
    _entry_resource_observation(row, inactive=True)
    assert row["resources"]["status"] == "inactive"
    assert row["resources"]["cpu_cores"] is None
    assert row["resources"]["memory_peak"] == 1024
    assert row["resources"]["observed_at"] == sampled["observed_at"]
    runtime.initialize(tmp_path, 2)
    runtime.publish(
        tmp_path, active=0, tasks={"matrix__entry": dict(row, file_id="matrix")}
    )
    assert (
        runtime.snapshot(tmp_path)["tasks"]["matrix__entry"]["resources"]
        == row["resources"]
    )


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
