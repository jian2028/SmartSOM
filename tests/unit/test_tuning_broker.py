"""Admission/lifecycle regressions without Ray startup or training processes."""

import json
import pickle
import sys
from types import SimpleNamespace

import pytest

from smartsom.experiments.tuning_broker import AdaptiveBroker
from smartsom.experiments.tuning_resources import (
    GIB,
    GPUInfo,
    ProcessUsage,
    ResourceSnapshot,
)


def resources(cpu=1, memory=GIB, gpu=0):
    return {"CPU": cpu, "GPU": gpu, "memory": memory}


def test_admission_diagnostics_explain_capacity_without_mutating_reservations():
    from smartsom.experiments.tuning_resources import ResourceLease, ResourceRequest

    broker = AdaptiveBroker([{"experiment_id": "a"}], {"a": [profile()]})
    broker._snapshot = ResourceSnapshot(
        3,
        7 * GIB,
        3 * GIB,
        external_cpu_load=37.9,
        processes=(
            ProcessUsage(10, "large RSS", 0.01, GIB),
            ProcessUsage(11, "busy small", 1.1, 1024),
        ),
    )
    manager = broker._manager()
    manager.leases["a"] = ResourceLease("a", ResourceRequest(1, GIB), "pending", (11,))
    before = dict(manager.leases)
    capacity = manager.capacity(broker._snapshot)
    broker.clock = lambda: pytest.fail("diagnostic read must not sample")
    data = broker.admission_diagnostics()
    assert data["capacity"]["cpus"] == pytest.approx(0.863)
    assert data["cpu_reserve"] == 1 and data["ram_reserve"] == 2 * GIB
    assert data["reserved_cpus"] == 1
    assert data["reserved_peak_ram"] == int(1.25 * GIB)
    assert data["top_cpu_processes"][0]["pid"] == 11
    assert data["lease_count"] == 1 and data["leases"][0]["state"] == "pending"
    assert manager.leases == before and manager.capacity(broker._snapshot) == capacity


def profile(cpu=1, memory=GIB, gpu=0, concurrency=2, **costs):
    return {
        **resources(cpu, memory, gpu),
        "concurrency": concurrency,
        "throughput": 10 * cpu,
        "update_seconds": 10 / cpu,
        "restart_seconds": 1,
        "gpu_memory": 0,
        **costs,
    }


def trial(identity="a", actor="actor-a", epoch=0, cpu=1, overhead=0, error=None):
    return SimpleNamespace(
        config={
            "experiment_id": identity,
            "execution_contract": {"allocation_epoch": epoch, "cpu_overhead": overhead},
        },
        temporary_state=SimpleNamespace(
            ray_actor=None
            if actor is None
            else SimpleNamespace(_actor_id=SimpleNamespace(hex=lambda: actor))
        ),
        placement_group_factory=SimpleNamespace(required_resources=resources(cpu)),
        get_pickled_error=lambda: error,
    )


@pytest.fixture
def observed(tmp_path, monkeypatch):
    facts = SimpleNamespace(
        now=100.0,
        actors={},
        processes={},
        descendants=(),
        exclusions=[],
        snapshot=ResourceSnapshot(16, 64 * GIB, 64 * GIB, external_cpu_load=0),
    )
    monkeypatch.setattr(
        AdaptiveBroker, "_actor_state", lambda self, actor: facts.actors.get(actor, {})
    )
    monkeypatch.setattr(
        AdaptiveBroker,
        "_process_state",
        lambda self, pid, created: facts.processes.get((pid, created), "live"),
    )
    monkeypatch.setattr(
        AdaptiveBroker, "_descendants", lambda self, pid, created: facts.descendants
    )

    def create(
        *,
        names=("a",),
        options=None,
        sampling=0,
        overhead=0,
        execution="adaptive",
        groups=None,
        **broker_options,
    ):
        entries = [
            {
                "experiment_id": name,
                "run_dir": str(tmp_path / name),
                "remaining_updates": 100,
                "execution_contract": {"allocation_epoch": 0, "cpu_overhead": overhead},
                "prepared": {
                    "config_json": json.dumps(
                        {"runtime": {"sampling_processes": sampling}}
                    )
                },
            }
            for name in names
        ]
        if groups is not None:
            for entry in entries:
                entry["calibration_group"] = groups[entry["experiment_id"]]
        broker = AdaptiveBroker(
            entries,
            options
            or {name: [profile(overhead + 1), profile(overhead + 4)] for name in names},
            execution=execution,
            clock=lambda: facts.now,
            **broker_options,
        )

        def snapshot(**kwargs):
            facts.exclusions.append(kwargs["exclude_pids"])
            return facts.snapshot

        broker._manager().monitor = SimpleNamespace(snapshot=snapshot)
        return broker

    def ack(
        broker,
        name="a",
        pid=111,
        created=1.0,
        epoch=0,
        threads=1,
        phase="training",
        children=(),
        complete=True,
        **changes,
    ):
        data = {
            "experiment_id": name,
            "pid": pid,
            "pid_create_time": created,
            "allocation_epoch": epoch,
            "threads": threads,
            "phase": phase,
            "child_processes": [{"pid": p, "create_time": c} for p, c in children],
            "children_complete": complete,
            **changes,
        }
        path = tmp_path / name / "tuning-runtime.json"
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(data))
        return data

    return facts, create, ack


def running(observed, *, cpu=1, sampling=0, overhead=0):
    facts, create, ack = observed
    broker = create(sampling=sampling, overhead=overhead)
    assert broker.try_acquire("a", resources(cpu))
    facts.actors["actor-a"] = {"State": "ALIVE", "Pid": 111}
    ack(broker, threads=cpu - overhead)
    item = trial(cpu=cpu, overhead=overhead)
    broker.note_actor("a", item)
    assert broker._manager().leases["a"].state == "running"
    return broker, item


def test_continuous_queue_obeys_stage_file_and_negative_mixed_limits(observed):
    _, create, _ = observed
    names = ("first__a", "first__b", "later__c")
    groups = {"first__a": "g1", "first__b": "g1", "later__c": "g2"}
    broker = create(names=names, groups=groups)
    for identity, file_id in (
        ("first__a", "first"),
        ("first__b", "first"),
        ("later__c", "later"),
    ):
        broker.entries[identity]["file_id"] = file_id
    broker.global_limit = 2
    broker.file_limits = {"first": 1, "later": 2}
    assert broker.try_acquire("first__a", resources())
    assert not broker.try_acquire("first__b", resources())
    assert broker.try_acquire("later__c", resources())
    broker.release("later__c")
    broker.incompatible_group_pairs = {frozenset(("g1", "g2"))}
    assert not broker.try_acquire("later__c", resources())
    broker.release("first__a")
    assert broker.try_acquire("later__c", resources())


def test_uncalibrated_admission_ramps_after_observed_formal_update(observed):
    _, create, _ = observed
    names = ("first__a", "later__b")
    broker = create(names=names, groups={name: "g" for name in names})
    broker.global_limit = 2
    broker.uncalibrated_groups = {"g"}
    broker._ramp_limit = 1
    assert broker.try_acquire("first__a", resources())
    assert not broker.try_acquire("later__b", resources())
    broker._memory_peaks["first__a"] = GIB
    broker.observe_formal_update("first__a")
    assert broker._ramp_limit == 2
    assert broker.try_acquire("later__b", resources())


def test_online_broker_waits_for_real_throughput_and_pauses_only_at_boundary(
    observed, tmp_path
):
    facts, create, ack = observed
    names = ("a", "b", "c")
    context = {"g": {"hardware": "host", "shape": "task", "hint": None}}
    broker = create(
        names=names,
        groups={name: "g" for name in names},
        options={name: [profile(concurrency=3)] for name in names},
        uncalibrated_groups=("g",),
        online_context=context,
        online_root=tmp_path,
        online_output=tmp_path / "runs",
    )
    assert broker.try_acquire("a", resources())
    assert not broker.try_acquire("b", resources())

    facts.actors["actor-a"] = {"State": "ALIVE", "Pid": 111}
    ack(broker)
    broker.note_actor("a", trial())
    facts.snapshot = ResourceSnapshot(
        16,
        64 * GIB,
        64 * GIB,
        external_cpu_load=0,
        processes=(ProcessUsage(111, "worker", 1, GIB, True),),
    )
    for updates, ticks, elapsed in ((1, 100, 0), (3, 400, 30), (5, 700, 60)):
        facts.now = 100 + elapsed
        broker.observe_formal_update(
            "a", {"phase": "training", "updates": updates, "physical_ticks": ticks}
        )
    assert broker._online["g"].limit == 2
    group = broker.summary()["online"]["g"]
    assert group["active"] == 1 and group["limit"] == 2
    assert group["latest_throughput"] == 10 and group["measured_concurrency"] == 1
    assert broker.try_acquire("b", resources())
    # Admission reserves capacity before the actor actually starts.
    assert broker.summary()["online"]["g"]["active"] == 1
    assert not broker.try_acquire("c", resources())
    from smartsom.experiments.performance_profiles import select_online

    cached = select_online(tmp_path / "runs", hardware="host", shape="task")
    assert cached["concurrency"] == 1 and cached["throughput"] == 10
    facts.actors["actor-b"] = {"State": "ALIVE", "Pid": 222}
    ack(broker, name="b", pid=222)
    broker.note_actor("b", trial(identity="b", actor="actor-b"))
    assert broker.summary()["online"]["g"]["active"] == 2
    broker._online["g"].limit = 1
    assert broker.should_pause("b") and not broker.should_pause("a")
    broker.defer_resize("b", resources(), resources())
    assert not broker.should_pause("a")
    assert not broker.try_acquire("b", resources())
    # Actor death and complete child exit, rather than a scheduler PAUSE label,
    # release its reservation; the lower online limit then prevents readmission.
    facts.actors["actor-b"]["State"] = "DEAD"
    facts.processes[(222, 1.0)] = "dead"
    broker.refresh(force=True)
    assert not broker.try_acquire("b", resources())


def test_successful_admission_clears_priming_wait_without_leaking_other_entry_reason(
    observed,
):
    facts, create, ack = observed
    broker = create(names=("a", "b"))
    from dataclasses import replace

    facts.snapshot = replace(
        facts.snapshot, unavailable=("owned CPU load first sample unavailable",)
    )
    assert not broker.try_acquire("a", resources())
    assert "primed" in broker.summary()["entries"][0]["resource_change_reason"]
    facts.snapshot = replace(facts.snapshot, unavailable=())
    facts.now += 6
    assert broker.try_acquire("a", resources())
    facts.actors["actor-a"] = {"State": "ALIVE", "Pid": 111}
    ack(broker)
    broker.note_actor("a", trial())
    broker._reason("b", "waiting for independent memory budget")
    entries = {e["experiment_id"]: e for e in broker.summary()["entries"]}
    assert entries["a"]["status"] == "running"
    assert entries["a"]["resource_change_reason"] is None
    assert (
        entries["b"]["resource_change_reason"]
        == "waiting for independent memory budget"
    )


def test_compatible_seed_groups_grow_only_after_observed_rss_and_stable_windows(
    observed,
):
    from test_tuning_batch import learner_seed_entries

    from smartsom.experiments.tuning_batch import _groups

    entries = learner_seed_entries()
    groups, mapping = _groups(entries, online=True)
    assert len(groups) == 1
    group = next(iter(groups))
    names = tuple(row["experiment_id"] for row in entries)
    facts, create, ack = observed
    broker = create(
        names=names,
        groups=mapping,
        options={name: [profile(concurrency=3)] for name in names},
        uncalibrated_groups=(group,),
        online_context={group: {"hint": None}},
    )
    assert broker.try_acquire(names[0], resources())
    assert not broker.try_acquire(names[1], resources())
    facts.actors["actor-a"] = {"State": "ALIVE", "Pid": 111}
    ack(broker, name=names[0])
    broker.note_actor(names[0], trial(identity=names[0]))
    facts.snapshot = ResourceSnapshot(
        16,
        64 * GIB,
        64 * GIB,
        external_cpu_load=0,
        processes=(ProcessUsage(111, "worker", 1, GIB, True),),
    )
    for updates, ticks, elapsed in ((1, 100, 0), (3, 400, 30), (5, 700, 60)):
        facts.now = 100 + elapsed
        broker.observe_formal_update(
            names[0], {"phase": "training", "updates": updates, "physical_ticks": ticks}
        )
    assert broker._online[group].limit == 2
    assert broker.try_acquire(names[1], resources())
    assert not broker.try_acquire(names[2], resources())


def test_online_resume_does_not_relabel_unobserved_group_history(observed, tmp_path):
    from smartsom.experiments.performance_profiles import ONLINE_SCHEMA, select_online

    _, create, _ = observed
    path = tmp_path / "online-performance.json"
    previous = {"accepted": True, "concurrency": 4, "throughput": 80}
    old_profile = {
        "hardware": "old-host",
        "shape": "old-task",
        "concurrency": 4,
        "throughput": 80,
        "at": 1,
        "report": str(path.resolve()),
        "evidence": previous,
    }
    path.write_text(
        json.dumps(
            {
                "schema": ONLINE_SCHEMA,
                "groups": {"old": {"measurements": [previous]}},
                "profiles": [old_profile],
            }
        )
    )
    context = {
        "old": {"hardware": "new-host", "shape": "old-task", "hint": None},
        "fresh": {"hardware": "new-host", "shape": "fresh-task", "hint": None},
    }
    broker = create(
        names=("a", "b"),
        groups={"a": "old", "b": "fresh"},
        online_context=context,
        online_root=tmp_path,
        online_output=tmp_path / "runs",
    )
    # A different group's fresh decision must not republish untouched history.
    for updates, ticks, elapsed in ((1, 100, 0), (3, 400, 30), (5, 700, 60)):
        broker._online["fresh"].observe(
            ("b",), "b", ticks=ticks, updates=updates, now=elapsed
        )
    broker._save_online()
    report = json.loads(path.read_text())
    assert report["groups"]["old"]["measurements"] == [previous]
    assert report["groups"]["old"]["best_concurrency"] is None
    assert old_profile in report["profiles"]
    assert len(report["profiles"]) == 2
    assert (
        select_online(tmp_path / "runs", hardware="new-host", shape="old-task") is None
    )
    assert (
        select_online(tmp_path / "runs", hardware="new-host", shape="fresh-task")[
            "throughput"
        ]
        == 10
    )


def test_online_cannot_grow_without_observed_memory_or_during_pressure(observed):
    facts, create, ack = observed
    names = ("a", "b")
    broker = create(
        names=names,
        groups={name: "g" for name in names},
        online_context={"g": {"hint": None}},
        uncalibrated_groups=("g",),
    )
    assert broker.try_acquire("a", resources())
    facts.actors["actor-a"] = {"State": "ALIVE", "Pid": 111}
    ack(broker)
    broker.note_actor("a", trial())
    for update in range(1, 8):
        facts.now += 30
        broker.observe_formal_update(
            "a",
            {"phase": "training", "updates": update, "physical_ticks": update * 100},
        )
    assert broker._online["g"].limit == 1
    assert not broker.try_acquire("b", resources())
    facts.snapshot = ResourceSnapshot(
        16,
        64 * GIB,
        GIB,
        external_cpu_load=0,
        processes=(ProcessUsage(111, "worker", 1, GIB, True),),
    )
    for update in range(8, 14):
        facts.now += 30
        broker.observe_formal_update(
            "a",
            {"phase": "training", "updates": update, "physical_ticks": update * 100},
        )
    assert broker._memory_peaks["a"] == GIB
    assert broker._online["g"].limit == 1
    assert broker._online["g"].history == []
    assert broker.should_pause("a")


def test_online_ignores_final_evaluation_before_its_result_arrives(observed):
    facts, create, ack = observed
    broker = create(
        groups={"a": "g"},
        options={"a": [profile()]},
        online_context={"g": {"hint": None}},
    )
    assert broker.try_acquire("a", resources())
    facts.actors["actor-a"] = {"State": "ALIVE", "Pid": 111}
    ack(broker, phase="evaluation")
    broker.note_actor("a", trial())
    facts.snapshot = ResourceSnapshot(
        16,
        64 * GIB,
        64 * GIB,
        external_cpu_load=0,
        processes=(ProcessUsage(111, "worker", 1, GIB, True),),
    )
    # Another actor's callback can observe this worker in evaluation before the
    # worker returns its final experiment_complete result. Do not time that
    # membership as a comparable training-throughput window.
    for update in (1, 3, 5):
        facts.now += 30
        broker.observe_formal_update(
            "a",
            {"phase": "training", "updates": update, "physical_ticks": update * 100},
        )
    assert broker._online["g"].history == []
    assert broker._online["g"].limit == 1


def test_pending_staged_running_and_pre_actor_rollback(observed):
    facts, create, ack = observed
    broker = create()
    assert broker.try_acquire("a", resources())
    assert broker.try_acquire("a", resources())
    assert broker.summary()["entries"][0]["actual_cpus"] is None
    broker.release("a")
    assert not broker._manager().leases
    assert broker.try_acquire("a", resources())
    facts.actors["actor-a"] = {"State": "ALIVE", "Pid": 111}
    broker.note_actor("a", trial())
    assert broker._manager().leases["a"].state == "staged"
    broker.release("a")
    assert "a" in broker._manager().leases
    ack(broker)
    broker.refresh(force=True)
    row = broker.summary()["entries"][0]
    assert (row["status"], row["actual_cpus"], row["threads"]) == ("running", 1, 1)


@pytest.mark.parametrize(
    "changes",
    [
        {"experiment_id": "another"},
        {"allocation_epoch": 2},
        {"threads": 4},
        {"pid": 222},
        {"pid_create_time": None},
        {"phase": "starting"},
        {"phase": "closed"},
    ],
)
def test_ack_requires_matching_identity_epoch_pid_and_threads(observed, changes):
    facts, create, ack = observed
    broker = create()
    assert broker.try_acquire("a", resources())
    facts.actors["actor-a"] = {"State": "ALIVE", "Pid": 111}
    ack(broker, **changes)
    broker.note_actor("a", trial())
    assert broker.summary()["entries"][0]["actual_cpus"] is None


def test_pid_reuse_does_not_ack_applied_threads(observed):
    facts, create, ack = observed
    broker = create()
    assert broker.try_acquire("a", resources())
    facts.actors["actor-a"] = {"State": "ALIVE", "Pid": 111}
    facts.processes[111, 1.0] = "dead"
    ack(broker)
    broker.note_actor("a", trial())
    assert broker.summary()["entries"][0]["actual_cpus"] is None


def test_resize_waits_for_actor_and_child_then_acknowledges_new_epoch(observed):
    facts, _, ack = observed
    broker, item = running(observed)
    ack(broker, children=((333, 3.0),))
    broker.refresh(force=True)
    broker.defer_resize("a", resources(), resources(4))
    assert not broker.try_acquire("a", resources(4))
    facts.actors["actor-a"]["State"] = "DEAD"
    broker.refresh(force=True)
    assert "a" in broker._manager().leases
    facts.processes[333, 3.0] = "dead"
    broker.refresh(force=True)
    assert "a" not in broker._manager().leases
    assert broker.try_acquire("a", resources(4))
    # RCS's second admission must permit staging while acknowledgement is pending.
    assert broker.try_acquire("a", resources(4))
    replacement = trial(actor="actor-new", epoch=1, cpu=4)
    broker.notify_allocation("a", replacement)
    facts.actors["actor-new"] = {"State": "ALIVE", "Pid": 222}
    broker.note_actor("a", replacement)
    row = broker.summary()["entries"][0]
    assert row["pending_resize"] and row["actual_cpus"] is None
    ack(broker, pid=222, created=2.0, epoch=1, threads=4)
    broker.refresh(force=True)
    row = broker.summary()["entries"][0]
    assert not row["pending_resize"]
    assert (row["actual_cpus"], row["allocation_epoch"]) == (4, 1)


def test_failed_constructor_id_recovery_frees_next_preset_after_dead(observed):
    RayActorError = pytest.importorskip("ray.exceptions").RayActorError

    facts, create, _ = observed
    options = {name: [profile(concurrency=1)] for name in ("a", "b")}
    broker = create(names=("a", "b"), options=options)
    assert broker.try_acquire("a", resources())
    error = RayActorError(actor_id="failed-start", actor_init_failed=True)
    facts.actors["failed-start"] = {"State": "ALIVE", "Pid": 111}
    broker.actor_failed("a", trial(actor=None, error=error))
    assert not broker.try_acquire("b", resources())
    facts.actors["failed-start"]["State"] = "DEAD"
    broker.refresh(force=True)
    assert broker.try_acquire("b", resources())
    assert not broker.try_acquire("a", resources())


def test_unavailable_actor_is_not_confirmed_dead(observed):
    ActorUnavailableError = pytest.importorskip("ray.exceptions").ActorUnavailableError

    facts, create, _ = observed
    broker = create()
    assert broker.try_acquire("a", resources())
    error = ActorUnavailableError("temporarily unavailable", b"\x00" * 16)
    facts.actors[error.actor_id] = {"State": "ALIVE", "Pid": 111}
    broker.actor_failed("a", trial(actor=None, error=error))
    broker.refresh(force=True)
    assert "a" in broker._manager().leases


def test_unknown_failure_preserves_reservation_and_reports_unresolved(observed):
    _, create, _ = observed
    broker = create()
    assert broker.try_acquire("a", resources())
    broker.actor_failed("a", trial(actor=None, error=ValueError("setup failed")))
    broker.refresh(force=True)
    row = broker.summary()["entries"][0]
    assert row["status"] == "unresolved" and row["actual_cpus"] is None
    assert "a" in broker._manager().leases
    assert broker.unresolved_failures() == ("a",)
    broker.release("a")
    assert "a" in broker._manager().leases


def test_missing_optional_ray_keeps_failure_identity_unknown(monkeypatch):
    import builtins

    original = builtins.__import__

    def without_ray(name, *args, **kwargs):
        if name == "ray.exceptions":
            raise ImportError("optional Ray is absent")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_ray)
    assert (
        AdaptiveBroker._failure_actor(trial(error=ValueError("setup failed"))) is None
    )


def test_sampling_setup_without_complete_ownership_stays_reserved(observed):
    RayActorError = pytest.importorskip("ray.exceptions").RayActorError

    facts, create, ack = observed
    broker = create(sampling=1)
    assert broker.try_acquire("a", resources())
    facts.actors["failed"] = {"State": "DEAD", "Pid": 111}
    ack(broker, complete=False, phase="closed")
    broker.actor_failed("a", trial(actor=None, error=RayActorError(actor_id="failed")))
    broker.refresh(force=True)
    assert "a" in broker._manager().leases
    ack(broker, complete=True, phase="closed")
    broker.refresh(force=True)
    assert "a" not in broker._manager().leases


def test_new_busy_ack_revokes_previous_complete_child_registry(observed):
    RayActorError = pytest.importorskip("ray.exceptions").RayActorError
    facts, create, ack = observed
    broker = create(sampling=1)
    assert broker.try_acquire("a", resources())
    facts.actors["actor-a"] = {"State": "ALIVE", "Pid": 111}
    ack(broker, complete=True)
    broker.note_actor("a", trial())
    assert broker._ownership_complete["a"]
    ack(broker, complete=False, phase="training")
    broker.refresh(force=True)
    assert not broker._ownership_complete["a"]
    facts.actors["actor-a"]["State"] = "DEAD"
    broker.actor_failed("a", trial(error=RayActorError(actor_id="actor-a")))
    assert broker.unresolved_failures() == ("a",)
    assert "a" in broker._manager().leases


def test_children_registry_survives_reparenting_and_ray_idle_parent(observed):
    facts, _, ack = observed
    broker, _ = running(observed)
    ack(broker, children=((333, 3.0),))
    broker.refresh(force=True)
    # Closed ack's empty current child tree cannot erase an observed child.
    ack(broker, phase="closed", children=())
    facts.actors["actor-a"]["State"] = "DEAD"
    facts.processes[111, 1.0] = "live"
    facts.processes[333, 3.0] = "unknown"
    broker.refresh(force=True)
    assert "a" in broker._manager().leases
    facts.processes[333, 3.0] = "dead"
    broker.refresh(force=True)
    assert "a" not in broker._manager().leases


def test_pressure_victim_excludes_pending_and_waits_for_stopping(observed):
    facts, create, ack = observed
    broker = create(names=("a", "z"))
    assert broker.try_acquire("a", resources(4))
    assert broker.try_acquire("z", resources())
    facts.actors["actor-a"] = {"State": "ALIVE", "Pid": 111}
    ack(broker, threads=4)
    broker.note_actor("a", trial(cpu=4))
    facts.snapshot = ResourceSnapshot(4, 64 * GIB, 64 * GIB, external_cpu_load=0)
    broker.refresh(force=True)
    assert broker.should_pause("a") and not broker.should_pause("z")
    broker.defer_resize("a", resources(4), resources())
    assert not broker.should_pause("z")


def test_initial_cooldown_expires_and_growth_must_amortize(observed):
    facts, _, _ = observed
    broker, item = running(observed)
    assert broker.allocation(None, item, {"updates": 0}, None) is None
    facts.now += 61
    assert broker.allocation(None, item, {"updates": 0}, None) == resources(4)
    assert broker.allocation(None, item, {"updates": 100}, None) is None
    broker.profiles["a"][1]["restart_seconds"] = 1000
    assert broker.allocation(None, item, {"updates": 99}, None) is None


def test_cpu_ray_bundle_may_omit_zero_gpu(observed):
    broker, item = running(observed)
    item.placement_group_factory.required_resources.pop("GPU")
    assert broker.allocation(None, item, {"updates": 0}, None) is None


def test_memory_growth_budget_and_memory_pressure_shrink(observed):
    facts, create, ack = observed
    options = {"a": [profile(memory=GIB), profile(4, memory=40 * GIB)]}
    broker = create(options=options)
    assert broker.try_acquire("a", resources())
    facts.actors["actor-a"] = {"State": "ALIVE", "Pid": 111}
    ack(broker)
    item = trial()
    broker.note_actor("a", item)
    facts.now += 61
    facts.snapshot = ResourceSnapshot(16, 64 * GIB, 20 * GIB, external_cpu_load=0)
    broker.refresh(force=True)
    assert broker.allocation(None, item, {"updates": 0}, None) is None


def test_fixed_still_gates_new_admission_without_resizing(observed):
    facts, create, _ = observed
    broker = create(execution="fixed")
    facts.snapshot = ResourceSnapshot(16, 64 * GIB, 64 * GIB, external_cpu_load=100)
    assert not broker.try_acquire("a", resources())
    assert broker.allocation(None, trial(actor=None), {"updates": 0}, None) is None
    assert not broker.should_pause("a")


def test_first_cpu_sample_wait_has_no_deadline(observed):
    facts, create, _ = observed
    broker = create()
    facts.snapshot = ResourceSnapshot(
        16,
        64 * GIB,
        64 * GIB,
        external_cpu_load=None,
        unavailable=("CPU load first sample unavailable",),
    )
    assert not broker.try_acquire("a", resources())
    facts.now += 100000
    broker.refresh(force=True)
    assert not broker.try_acquire("a", resources())
    assert "a" not in broker._manager().leases


@pytest.mark.parametrize(
    "bad", [resources(1.5), resources(gpu=0.5), resources(memory=True), resources(8)]
)
def test_fractional_or_unmeasured_profiles_are_rejected(observed, bad):
    _, create, _ = observed
    broker = create()
    with pytest.raises(ValueError):
        broker.try_acquire("a", bad)


def test_sampling_overhead_and_epoch_remain_frozen(observed):
    _, create, _ = observed
    broker = create(sampling=2, overhead=2)
    assert broker.try_acquire("a", resources(3))
    with pytest.raises(ValueError, match="overhead"):
        broker.notify_allocation("a", trial(overhead=3))
    broker.notify_allocation("a", trial(epoch=2, overhead=2))
    with pytest.raises(ValueError, match="regressed"):
        broker.notify_allocation("a", trial(epoch=1, overhead=2))


def test_gpu_budget_uses_matching_profile_and_worst_possible_assignment(observed):
    facts, create, _ = observed
    options = {
        "a": [profile(gpu=1, gpu_memory=GIB), profile(4, gpu=1, gpu_memory=8 * GIB)]
    }
    broker = create(options=options)
    facts.snapshot = ResourceSnapshot(
        16,
        64 * GIB,
        64 * GIB,
        gpus=(
            GPUInfo("0", "GPU", 16 * GIB, 16 * GIB),
            GPUInfo("1", "GPU", 16 * GIB, 4 * GIB),
        ),
        external_cpu_load=0,
    )
    assert not broker.try_acquire("a", resources(4, gpu=1))
    assert broker.try_acquire("a", resources(gpu=1))
    assert broker._manager().leases["a"].request.gpu_memory == GIB


def test_ambiguous_gpu_observation_does_not_admit(observed):
    facts, create, _ = observed
    broker = create(options={"a": [profile(gpu=1, gpu_memory=GIB)]})
    facts.snapshot = ResourceSnapshot(
        16,
        64 * GIB,
        64 * GIB,
        gpus=(GPUInfo("0", "MIG", reason="unavailable"),),
        external_cpu_load=0,
    )
    assert not broker.try_acquire("a", resources(gpu=1))


def test_unknown_gpu_peak_is_not_zero_memory_admission(observed):
    facts, create, _ = observed
    broker = create(options={"a": [profile(gpu=1)]})
    facts.snapshot = ResourceSnapshot(
        16,
        64 * GIB,
        64 * GIB,
        gpus=(GPUInfo("0", "GPU", 16 * GIB, 16 * GIB),),
        external_cpu_load=0,
    )
    assert not broker.try_acquire("a", resources(gpu=1))
    assert "unavailable" in broker.summary()["entries"][0]["resource_change_reason"]


def test_online_gpu_bootstrap_reserves_observed_worst_device_budget(observed):
    import math

    facts, create, _ = observed
    broker = create(
        names=("a", "b"),
        options={name: [profile(gpu=1)] for name in ("a", "b")},
        groups={"a": "g", "b": "g"},
        online_context={"g": {"hint": None}},
        uncalibrated_groups=("g",),
    )
    facts.snapshot = ResourceSnapshot(
        16,
        64 * GIB,
        64 * GIB,
        gpus=(
            GPUInfo("0", "GPU", 16 * GIB, 16 * GIB),
            GPUInfo("1", "GPU", 16 * GIB, 4 * GIB),
        ),
        external_cpu_load=0,
    )
    assert broker.try_acquire("a", resources(gpu=1))
    lease = broker._manager().leases["a"]
    assert lease.request.gpu_memory > 0
    assert (
        math.ceil(lease.request.gpu_memory * broker._manager().peak_factor) <= 4 * GIB
    )
    assert not broker.try_acquire("b", resources(gpu=1))


@pytest.mark.parametrize("available", [None, 0])
def test_online_gpu_bootstrap_requires_live_memory_capacity(observed, available):
    facts, create, _ = observed
    broker = create(
        options={"a": [profile(gpu=1)]},
        groups={"a": "g"},
        online_context={"g": {"hint": None}},
    )
    facts.snapshot = ResourceSnapshot(
        16,
        64 * GIB,
        64 * GIB,
        gpus=(GPUInfo("0", "GPU", 16 * GIB, available),),
        external_cpu_load=0,
    )
    assert not broker.try_acquire("a", resources(gpu=1))


def test_same_resource_pause_keeps_epoch_but_waits_for_new_actor_ack(observed):
    facts, _, ack = observed
    broker, item = running(observed)
    broker.defer_resize("a", resources(), resources())
    facts.actors["actor-a"]["State"] = "DEAD"
    broker.refresh(force=True)
    assert broker.try_acquire("a", resources())
    facts.actors["next"] = {"State": "ALIVE", "Pid": 222}
    broker.note_actor("a", trial(actor="next", epoch=0))
    assert broker.summary()["entries"][0]["pending_resize"]
    ack(broker, pid=222, created=2.0, epoch=0)
    broker.refresh(force=True)
    row = broker.summary()["entries"][0]
    assert row["allocation_epoch"] == 0 and not row["pending_resize"]


def test_pickle_preserves_reservations_but_reconstructs_monitor_lazily(observed):
    _, create, _ = observed
    broker = create()
    assert broker.try_acquire("a", resources())
    broker._pending["a"] = resources(4)
    restored = pickle.loads(pickle.dumps(broker))
    assert restored._broker is None and restored._snapshot is None
    assert (
        restored._manager().leases["a"].request == broker._manager().leases["a"].request
    )
    assert restored._pending["a"] == resources(4)
    assert restored._manager().monitor is not broker._manager().monitor


def test_optional_import_does_not_load_ray_or_psutil(tmp_path):
    import subprocess

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import smartsom.experiments.tuning_broker; assert 'ray' not in sys.modules and 'psutil' not in sys.modules",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize(
    "created,running,status,expected",
    [
        (1.0, True, "zombie", "dead"),
        (2.0, True, "sleeping", "dead"),
        (1.0, False, "sleeping", "dead"),
        (1.0, True, "sleeping", "live"),
    ],
)
def test_process_identity_gate_accepts_zombie_and_pid_reuse(
    monkeypatch, created, running, status, expected
):
    class ProcessError(Exception):
        pass

    class NoSuchProcess(ProcessError):
        pass

    process = SimpleNamespace(
        create_time=lambda: created, is_running=lambda: running, status=lambda: status
    )
    monkeypatch.setitem(
        sys.modules,
        "psutil",
        SimpleNamespace(
            Process=lambda pid: process,
            NoSuchProcess=NoSuchProcess,
            Error=ProcessError,
            STATUS_ZOMBIE="zombie",
        ),
    )
    broker = AdaptiveBroker([], {})
    assert broker._process_state(111, 1.0) == expected


@pytest.mark.parametrize("missing,expected", [(True, "dead"), (False, "unknown")])
def test_missing_process_and_inaccessible_process_are_distinct(
    monkeypatch, missing, expected
):
    class ProcessError(Exception):
        pass

    class NoSuchProcess(ProcessError):
        pass

    def lookup(pid):
        raise NoSuchProcess() if missing else ProcessError()

    monkeypatch.setitem(
        sys.modules,
        "psutil",
        SimpleNamespace(
            Process=lookup,
            NoSuchProcess=NoSuchProcess,
            Error=ProcessError,
            STATUS_ZOMBIE="zombie",
        ),
    )
    assert AdaptiveBroker([], {})._process_state(111, 1.0) == expected


def test_ram_addback_excludes_driver_ray_services_and_unowned_workers(observed):
    facts, create, ack = observed
    broker = create(names=("a", "b"))
    assert broker.try_acquire("a", resources())
    facts.actors["actor-a"] = {"State": "ALIVE", "Pid": 111}
    ack(broker)
    broker.note_actor("a", trial())
    facts.snapshot = ResourceSnapshot(
        16,
        16 * GIB,
        4 * GIB,
        external_cpu_load=0,
        processes=(
            ProcessUsage(111, "worker", 1, GIB, True),
            ProcessUsage(100, "driver", 1, 3 * GIB, True),
            ProcessUsage(101, "raylet", 1, 3 * GIB, True),
            ProcessUsage(102, "unknown-worker", 1, 3 * GIB, True),
        ),
    )
    broker.refresh(force=True)
    assert {p.pid for p in broker._snapshot.processes if p.excluded} == {111}
    assert not broker.try_acquire("b", resources())
    assert all(p["protected"] for p in broker.summary()["resources"]["processes"])
    # A Ray idle parent remains real occupied RAM after its logical actor dies.
    facts.actors["actor-a"]["State"] = "DEAD"
    broker.refresh(force=True)
    assert not any(p.excluded for p in broker._snapshot.processes)


def test_downward_cpu_pressure_is_immediate_recovery_uses_ema(observed):
    facts, create, _ = observed
    broker = create()
    broker.refresh(force=True)
    facts.snapshot = ResourceSnapshot(16, 64 * GIB, 64 * GIB, external_cpu_load=80)
    facts.now += 5
    broker.refresh(force=True)
    assert broker._snapshot.external_cpu_load == 80
    facts.snapshot = ResourceSnapshot(16, 64 * GIB, 64 * GIB, external_cpu_load=0)
    facts.now += 5
    broker.refresh(force=True)
    assert 0 < broker._snapshot.external_cpu_load < 80


def test_reused_child_pid_is_not_excluded_from_external_cpu_load(observed):
    facts, _, ack = observed
    broker, _ = running(observed)
    ack(broker, children=((333, 3.0),))
    broker.refresh(force=True)
    assert 333 in facts.exclusions[-1]
    facts.processes[333, 3.0] = "dead"
    broker.refresh(force=True)
    assert 333 not in facts.exclusions[-1]
    assert (333, 3.0) in broker._children["a"]


@pytest.mark.parametrize("execution", ["adaptive", "fixed"])
def test_observed_replay_growth_blocks_new_admission(observed, execution):
    facts, create, ack = observed
    broker = create(names=("a", "b"), execution=execution)
    assert broker.try_acquire("a", resources())
    facts.actors["actor-a"] = {"State": "ALIVE", "Pid": 111}
    ack(broker, children=((333, 3.0),))
    broker.note_actor("a", trial())
    facts.snapshot = ResourceSnapshot(
        16,
        16 * GIB,
        6 * GIB,
        external_cpu_load=0,
        processes=(
            ProcessUsage(111, "learner", 1, 2 * GIB, True),
            ProcessUsage(333, "sampler", 1, 6 * GIB, True),
        ),
    )
    broker.refresh(force=True)
    assert broker._resources["a"] == resources()  # Calibrated identity is frozen.
    assert broker._manager().leases["a"].request.memory == 8 * GIB
    assert not broker.try_acquire("b", resources())
    assert "RAM" in broker.summary()["entries"][1]["resource_change_reason"]


def test_observed_replay_growth_creates_pressure_and_survives_restart(observed):
    facts, _, ack = observed
    broker, _ = running(observed)
    facts.snapshot = ResourceSnapshot(
        16,
        16 * GIB,
        3 * GIB,
        external_cpu_load=0,
        processes=(ProcessUsage(111, "learner", 1, 8 * GIB, True),),
    )
    broker.refresh(force=True)
    assert broker.should_pause("a")
    broker.defer_resize("a", resources(), resources())
    facts.actors["actor-a"]["State"] = "DEAD"
    broker.refresh(force=True)
    assert not broker._manager().leases
    assert not broker.try_acquire("a", resources())
    # A real exit frees RAM; the next actor still needs its observed replay peak.
    facts.snapshot = ResourceSnapshot(16, 16 * GIB, 16 * GIB, external_cpu_load=0)
    broker.refresh(force=True)
    assert broker.try_acquire("a", resources())
    assert broker._manager().leases["a"].request.memory == 8 * GIB
    facts.actors["replacement"] = {"State": "ALIVE", "Pid": 222}
    ack(broker, pid=222, created=2.0)
    broker.note_actor("a", trial(actor="replacement"))
    restored = pickle.loads(pickle.dumps(broker))
    assert restored._memory_peaks["a"] == 8 * GIB
    assert restored._manager().leases["a"].request.memory == 8 * GIB


def test_reused_child_rss_is_external_not_owned_memory(observed):
    facts, _, ack = observed
    broker, _ = running(observed)
    ack(broker, children=((333, 3.0),))
    facts.processes[333, 3.0] = "dead"
    facts.snapshot = ResourceSnapshot(
        16,
        64 * GIB,
        32 * GIB,
        external_cpu_load=0,
        processes=(
            ProcessUsage(111, "learner", 1, GIB, True),
            ProcessUsage(333, "other-program", 1, 20 * GIB, True),
        ),
    )
    broker.refresh(force=True)
    assert broker._manager().leases["a"].request.memory == GIB
    assert {p.pid for p in broker._snapshot.processes if p.excluded} == {111}


def test_independent_calibration_groups_share_global_budget(observed):
    facts, create, _ = observed
    names = ("a", "b", "c")
    broker = create(
        names=names,
        options={name: [profile(cpu=4, concurrency=1)] for name in names},
        groups={name: f"group-{name}" for name in names},
    )
    assert broker.try_acquire("a", resources(4))
    assert broker.try_acquire("b", resources(4))
    facts.snapshot = ResourceSnapshot(8, 64 * GIB, 64 * GIB, external_cpu_load=0)
    broker.refresh(force=True)
    assert not broker.try_acquire("c", resources(4))
    assert "CPU" in broker.summary()["entries"][2]["resource_change_reason"]


def test_shared_calibration_group_keeps_measured_concurrency_cap(observed):
    _, create, _ = observed
    names = ("a", "b")
    broker = create(
        names=names,
        options={name: [profile(concurrency=1)] for name in names},
        groups={name: "same-model" for name in names},
    )
    assert broker.try_acquire("a", resources())
    assert not broker.try_acquire("b", resources())
    assert "concurrency" in broker.summary()["entries"][1]["resource_change_reason"]


def test_independent_cuda_groups_use_distinct_whole_gpu_reservations(observed):
    facts, create, _ = observed
    names = ("a", "b", "c")
    broker = create(
        names=names,
        options={
            name: [profile(gpu=1, gpu_memory=GIB, concurrency=1)] for name in names
        },
        groups={name: f"group-{name}" for name in names},
    )
    facts.snapshot = ResourceSnapshot(
        16,
        64 * GIB,
        64 * GIB,
        external_cpu_load=0,
        gpus=(
            GPUInfo("0", "GPU", 16 * GIB, 16 * GIB),
            GPUInfo("1", "GPU", 16 * GIB, 16 * GIB),
        ),
    )
    assert broker.try_acquire("a", resources(gpu=1))
    assert broker.try_acquire("b", resources(gpu=1))
    assert not broker.try_acquire("c", resources(gpu=1))
    assert {lease.request.gpu for lease in broker._manager().leases.values()} == {
        "0",
        "1",
    }


def test_dead_failed_setup_reports_incomplete_sampling_ownership(observed):
    RayActorError = pytest.importorskip("ray.exceptions").RayActorError

    facts, create, ack = observed
    broker = create(sampling=1, overhead=1)
    assert broker.try_acquire("a", resources(2))
    facts.actors["failed-start"] = {"State": "ALIVE", "Pid": 111}
    broker.actor_failed(
        "a", trial(actor=None, overhead=1, error=RayActorError(actor_id="failed-start"))
    )
    assert broker.unresolved_failures() == ()
    facts.actors["failed-start"]["State"] = "DEAD"
    broker.refresh(force=True)
    assert broker.unresolved_failures() == ("a",)
    assert "a" in broker._manager().leases
    # A later complete registry resolves uncertainty, while its live child must
    # still exit before the resource lease can be released.
    ack(broker, phase="closed", children=((333, 3.0),), complete=True)
    broker.refresh(force=True)
    assert broker.unresolved_failures() == ()
    assert "a" in broker._manager().leases
    facts.processes[333, 3.0] = "dead"
    broker.refresh(force=True)
    assert not broker._manager().leases


def test_missing_ack_on_staged_actor_is_not_an_unresolved_failure(observed):
    facts, create, _ = observed
    broker = create(sampling=1, overhead=1)
    assert broker.try_acquire("a", resources(2))
    facts.actors["actor-a"] = {"State": "ALIVE", "Pid": 111}
    broker.note_actor("a", trial(cpu=2, overhead=1))
    assert broker.unresolved_failures() == ()
    facts.actors["actor-a"]["State"] = "DEAD"
    broker.refresh(force=True)
    assert broker.unresolved_failures() == ()
    assert "a" in broker._manager().leases


def test_failed_setup_without_actor_death_evidence_is_not_reported_dead(observed):
    RayActorError = pytest.importorskip("ray.exceptions").RayActorError

    _, create, _ = observed
    broker = create(sampling=1, overhead=1)
    assert broker.try_acquire("a", resources(2))
    broker.actor_failed(
        "a",
        trial(actor=None, overhead=1, error=RayActorError(actor_id="not-observable")),
    )
    assert broker.unresolved_failures() == ()
    assert "a" in broker._manager().leases


def test_actor_query_uses_ray_258_global_state_actor_table(monkeypatch):
    seen = []
    result = {"ActorID": "actor-id", "State": "ALIVE", "Pid": 111}

    def actor_table(actor):
        seen.append(actor)
        return result

    # GlobalState exposes actor_table, not the module-level actors convenience
    # function. Omitting actors catches the actual pinned API mismatch.
    monkeypatch.setitem(
        sys.modules,
        "ray",
        SimpleNamespace(
            _private=SimpleNamespace(
                state=SimpleNamespace(state=SimpleNamespace(actor_table=actor_table))
            ),
            exceptions=SimpleNamespace(RayError=RuntimeError),
        ),
    )
    assert AdaptiveBroker([], {})._actor_state("actor-id") == result
    assert seen == ["actor-id"]


def test_explicit_fixed_slots_admit_three_unknown_groups(observed, monkeypatch):
    monkeypatch.setenv("SMARTSOM_FIXED_ADMISSION_SLOTS", "3")
    _, create, _ = observed
    names = ("a", "b", "c")
    broker = create(
        names=names,
        groups={name: name for name in names},
        execution="fixed",
        global_limit=3,
        uncalibrated_groups=names,
    )
    assert broker._ramp_limit == 3
    for name in names:
        assert broker.try_acquire(name, resources())
    assert len(broker._manager().leases) == 3


def test_explicit_fixed_slots_retain_ram_capacity_check(observed, monkeypatch):
    monkeypatch.setenv("SMARTSOM_FIXED_ADMISSION_SLOTS", "3")
    _, create, _ = observed
    broker = create(
        names=("a",),
        execution="fixed",
        global_limit=3,
        uncalibrated_groups=("g",),
        groups={"a": "g"},
        options={"a": [profile(memory=128 * GIB)]},
    )
    assert not broker.try_acquire("a", resources(memory=128 * GIB))


def test_explicit_fixed_slots_reject_adaptive_mode(observed, monkeypatch):
    monkeypatch.setenv("SMARTSOM_FIXED_ADMISSION_SLOTS", "3")
    _, create, _ = observed
    with pytest.raises(ValueError, match="fixed admission slots"):
        create(global_limit=3)


def test_shared_queue_pending_retiring_and_native_owners_use_one_cap(
    observed, tmp_path
):
    from smartsom.experiments import queue_admission
    from smartsom.experiments.evidence import write_json

    facts, create, ack = observed
    root = tmp_path / "directory"
    root.mkdir()
    (root / "control").mkdir()
    write_json(root / "control/owner.json", dict(id="original"))
    queue_admission.initialize(root, supported=True)
    queue_admission.claim(root, "queue", limit=2, owner_id="original")
    # New jobs can fill the cap while the original driver is still in preflight.
    for name in ("one", "two"):
        assert queue_admission.reserve(
            root, "native:" + name, owner_id="original", status="running"
        )
    broker = create(names=("a", "b", "c"), admission_root=root)
    assert not broker.try_acquire("a", resources())
    queue_admission.retire(root, "native:one", owner_id="original")
    assert broker.try_acquire("a", resources())
    assert queue_admission.snapshot(root)["slots"]["source:a"]["status"] == "pending"
    assert not broker.try_acquire("b", resources())
    facts.actors["actor-a"] = dict(State="ALIVE", Pid=111)
    ack(broker, children=((333, 3.0),))
    broker.note_actor("a", trial())
    broker.refresh(force=True)
    assert queue_admission.snapshot(root)["slots"]["source:a"]["status"] == "running"
    # Lowering the cap does not pause or kill the existing learner.
    queue_admission.configure(root, "queue", limit=1)
    before = dict(broker._manager().leases)
    assert not broker.try_acquire("b", resources())
    assert broker._manager().leases == before
    broker.defer_resize("a", resources(), resources(4))
    facts.actors["actor-a"]["State"] = "DEAD"
    facts.processes[333, 3.0] = "unknown"
    broker.refresh(force=True)
    assert queue_admission.snapshot(root)["slots"]["source:a"]["status"] == "stopping"
    queue_admission.retire(root, "native:two", owner_id="original")
    assert not broker.try_acquire("b", resources())
    facts.processes[333, 3.0] = "dead"
    broker.refresh(force=True)
    assert "source:a" not in queue_admission.snapshot(root)["slots"]
    assert broker.try_acquire("b", resources())
    broker.release("b")
    assert not queue_admission.snapshot(root)["slots"]


def test_cohort_drain_waits_for_children_and_never_releases_unknown_ownership(
    observed, tmp_path, monkeypatch
):
    from smartsom.experiments import queue_admission, tuning_broker
    from smartsom.experiments.evidence import write_json

    facts, create, ack = observed
    root = tmp_path / "directory"
    root.mkdir()
    (root / "control").mkdir()
    write_json(root / "control/owner.json", dict(id="original"))
    queue_admission.initialize(root, supported=True)
    queue_admission.claim(root, "queue", limit=1, owner_id="original")
    broker = create(admission_root=root)
    assert broker.try_acquire("a", resources())
    facts.actors["actor-a"] = dict(State="ALIVE", Pid=111)
    ack(broker, children=((333, 3.0),))
    broker.note_actor("a", trial())
    facts.actors["actor-a"]["State"] = "DEAD"
    facts.processes[333, 3.0] = "unknown"

    def wait(_):
        assert "source:a" in queue_admission.snapshot(root)["slots"]
        facts.now += 1

    monkeypatch.setattr(tuning_broker.time, "sleep", wait)
    with pytest.raises(RuntimeError, match="exit remains unverified"):
        broker.drain_shared(timeout=2)
    assert "source:a" in queue_admission.snapshot(root)["slots"]
    facts.processes[333, 3.0] = "dead"
    broker.drain_shared()
    assert not broker._manager().leases and not queue_admission.snapshot(root)["slots"]
    # The next stage uses a fresh broker against the same root and cap.
    following = create(names=("next",), admission_root=root)
    assert following.try_acquire("next", resources())
    following.drain_shared()  # a reservation that never staged is safe to roll back
    assert not queue_admission.snapshot(root)["slots"]


def test_unknown_owned_cpu_blocks_new_work_and_cpu_only_preemption_but_not_ram(
    observed,
):
    facts, create, _ = observed
    broker = create(names=("a", "b", "c"))
    assert broker.try_acquire("a", resources())
    assert broker.try_acquire("b", resources())
    facts.snapshot = ResourceSnapshot(
        4,
        64 * GIB,
        64 * GIB,
        external_cpu_load=30,
        unavailable=("owned CPU load first sample unavailable",),
    )
    broker.refresh(force=True)
    assert broker._manager().capacity(broker._snapshot).cpus < 2
    # The conservative CPU capacity is an upper-bound external estimate while
    # an owned PID is unprimed. It cannot certify an external-pressure preemption.
    assert not broker._pressure()[0]
    assert not broker.should_pause("a") and not broker.should_pause("b")
    assert not broker.try_acquire("c", resources())
    assert "primed owned CPU" in broker._reasons["c"]
    facts.snapshot = ResourceSnapshot(
        4,
        64 * GIB,
        GIB,
        external_cpu_load=30,
        unavailable=("owned CPU load first sample unavailable",),
    )
    broker.refresh(force=True)
    assert broker._pressure()[0]  # measured memory pressure remains immediate
    facts.snapshot = ResourceSnapshot(4, 64 * GIB, 64 * GIB, external_cpu_load=30)
    broker.refresh(force=True)
    assert broker._pressure()[0]  # primed genuine CPU pressure remains immediate


def test_dispatched_before_start_callback_retains_shared_cap_until_verified_dead(
    observed, tmp_path, monkeypatch
):
    from smartsom.experiments import queue_admission, tuning_broker
    from smartsom.experiments.evidence import write_json

    facts, create, _ = observed
    root = tmp_path / "directory"
    (root / "control").mkdir(parents=True)
    write_json(root / "control/owner.json", dict(id="original"))
    queue_admission.initialize(root, supported=True)
    queue_admission.claim(root, "queue", limit=1, owner_id="original")
    broker = create(admission_root=root)
    assert broker.try_acquire("a", resources())
    broker.handoff("a")
    broker.note_dispatched_actor("a", trial().temporary_state.ray_actor)
    # No on_trial_start or native setup acknowledgement arrives.
    facts.actors["actor-a"] = dict(State="PENDING_CREATION", Pid=0)

    def wait(_):
        assert not queue_admission.reserve(root, "native:next", owner_id="original")
        facts.now += 1

    monkeypatch.setattr(tuning_broker.time, "sleep", wait)
    with pytest.raises(RuntimeError, match="exit remains unverified"):
        broker.drain_shared(timeout=2)
    assert "source:a" in queue_admission.snapshot(root)["slots"]
    facts.actors["actor-a"]["State"] = "DEAD"
    broker.drain_shared()
    assert queue_admission.reserve(root, "native:next", owner_id="original")


def test_handed_off_unknown_trial_is_not_search_rollback(observed):
    _, create, _ = observed
    broker = create()
    assert broker.try_acquire("a", resources())
    broker.handoff("a")
    broker.release("a")
    assert "a" in broker._manager().leases
    broker.cancel_undispatched("a")  # audited closed cohort, not absent callback
    assert not broker._manager().leases


def test_runtime_manual_target_overrides_advisory_limits_without_restart(
    observed, tmp_path
):
    from smartsom.experiments.runtime_control import initialize, submit

    facts, create, _ = observed
    root = tmp_path / "runtime"
    initialize(root, 4)
    broker = create(
        names=tuple("abcdef"),
        execution="fixed",
        global_limit=1,
        file_limits={"a": 1},
        control_root=root,
    )
    for identity in "abcd":
        assert broker.try_acquire(identity, resources())
    original = dict(broker._manager().leases)
    submit(root, "limit", value=6)
    assert broker.try_acquire("e", resources())
    assert broker.try_acquire("f", resources())
    submit(root, "limit", value=2)
    assert not broker.should_pause("a")
    assert all(broker._manager().leases[k] == v for k, v in original.items())
    assert len(broker._manager().leases) == 6
    assert broker.summary()["runtime_control"]["limit"] == 2


def test_runtime_pause_polls_without_waiting_for_resource_sample(observed, tmp_path):
    from smartsom.experiments.runtime_control import initialize, snapshot, submit

    _, create, _ = observed
    root = tmp_path / "runtime"
    initialize(root, 2)
    broker = create(names=("a", "b"), execution="fixed", control_root=root)
    assert broker.try_acquire("a", resources())
    submit(root, "pause", task_id="a")
    assert broker.should_pause("a")
    assert broker.try_acquire("b", resources())
    submit(root, "pause")
    assert broker.should_pause("b")
    submit(root, "resume")
    assert broker.should_pause("a")
    assert not broker.should_pause("b")
    assert snapshot(root)["paused_tasks"] == ["a"]


def test_runtime_manual_target_keeps_resource_admission(observed, tmp_path):
    from smartsom.experiments.runtime_control import initialize

    facts, create, _ = observed
    root = tmp_path / "runtime"
    initialize(root, 6)
    facts.snapshot = ResourceSnapshot(2, 64 * GIB, 64 * GIB, external_cpu_load=0)
    broker = create(names=("a", "b"), execution="fixed", control_root=root)
    assert broker.try_acquire("a", resources())
    assert not broker.try_acquire("b", resources())
    assert "CPU" in broker._reasons["b"]


def test_runtime_task_stop_is_distinct_and_batch_resume_preserves_it(
    observed, tmp_path
):
    from smartsom.experiments.runtime_control import initialize, submit

    _, create, _ = observed
    root = tmp_path / "runtime"
    initialize(root, 2)
    broker = create(names=("a", "b"), control_root=root)
    broker.refresh()
    submit(root, "stop", task_id="a")
    assert broker.should_stop("a")
    submit(root, "resume")
    assert broker.should_stop("a")
    assert not broker.try_acquire("a", resources())
    assert broker.try_acquire("b", resources())
    submit(root, "resume", task_id="a")
    assert not broker.should_stop("a")


def test_runtime_adaptive_limit_is_ceiling_not_manual_target(observed, tmp_path):
    from smartsom.experiments.runtime_control import initialize

    _, create, _ = observed
    root = tmp_path / "runtime"
    initialize(root, 6, mode="adaptive")
    broker = create(names=("a", "b", "c"), global_limit=1, control_root=root)
    assert broker.try_acquire("a", resources())
    assert not broker.try_acquire("b", resources())


def test_runtime_aged_large_request_reserves_capacity_without_starving_itself(
    observed, tmp_path
):
    from smartsom.experiments.runtime_control import initialize

    facts, create, _ = observed
    root = tmp_path / "runtime"
    initialize(root, 6)
    facts.snapshot = ResourceSnapshot(4, 64 * GIB, 64 * GIB, external_cpu_load=0)
    broker = create(
        names=("active", "large", "small"),
        control_root=root,
        options={
            "active": [profile()],
            "large": [profile(cpu=3)],
            "small": [profile()],
        },
    )
    assert broker.try_acquire("active", resources())
    assert not broker.try_acquire("large", resources(cpu=3))
    facts.now += 31
    assert not broker.try_acquire("small", resources())
    assert "older task large" in broker._reasons["small"]
    broker.release("active")
    assert broker.try_acquire("large", resources(cpu=3))


def test_runtime_removal_retains_reservation_until_observed_release(observed, tmp_path):
    from smartsom.experiments import runtime_control as runtime

    _, create, _ = observed
    root = tmp_path / "runtime"
    runtime.initialize(root, 2)
    broker = create(names=("a", "b"), control_root=root)
    assert broker.try_acquire("a", resources())
    broker.refresh()
    target = runtime.removal_target(root, "a")
    runtime.submit(
        root,
        "remove",
        task_id="a",
        confirmation=target["confirmation"],
        expected_revision=target["revision"],
    )
    assert broker.should_stop("a")
    assert runtime.snapshot(root)["tasks"]["a"]["status"] == "removing"
    assert "a" in broker._manager().leases
    broker.release("a")
    broker.refresh()
    assert runtime.snapshot(root)["tasks"]["a"]["status"] == "removed"
    assert not broker.try_acquire("a", resources())
    assert broker.try_acquire("b", resources())


def test_runtime_budget_command_preserves_observed_node_capacity(observed, tmp_path):
    from smartsom.experiments import runtime_control as runtime

    _, create, _ = observed
    root = tmp_path / "runtime"
    runtime.initialize(root, 6)
    broker = create(names=("a", "b"), control_root=root)
    runtime.submit(root, "budget", value={"cpus": 1.5, "memory": 4 * GIB})
    assert broker.try_acquire("a", resources())
    assert not broker.try_acquire("b", resources())
    assert broker._snapshot.cpus == 16
    assert broker._manager().capacity(broker._snapshot).cpus == 1.5


def test_tune_and_native_reservations_share_node_memory_budget(observed, tmp_path):
    from smartsom.experiments import queue_admission, runtime_control
    from smartsom.experiments.evidence import write_json

    _, create, _ = observed
    root = tmp_path / "shared"
    runtime_control.initialize(root, 8)
    write_json(root / "control/owner.json", {"id": "original"})
    queue_admission.initialize(root, supported=True)
    runtime_control.submit(root, "budget", value={"cpus": 8, "memory": 2 * GIB})
    runtime_control.apply(root)
    assert queue_admission.reserve(
        root, "native:one", owner_id="original", cpus=1, memory=GIB
    )
    broker = create(names=("a",), admission_root=root)
    assert not broker.try_acquire("a", resources())
    queue_admission.retire(root, "native:one", owner_id="original")
    assert broker.try_acquire("a", resources())
    slot = queue_admission.snapshot(root)["slots"]["source:a"]
    assert slot["cpus"] == 1 and slot["memory"] >= int(1.25 * GIB)
    assert not queue_admission.reserve(
        root, "native:two", owner_id="original", cpus=1, memory=GIB
    )


def test_tune_waiter_ages_against_shared_native_load_even_when_slots_full(
    observed, tmp_path
):
    import os

    import psutil

    from smartsom.experiments import queue_admission, runtime_control
    from smartsom.experiments.evidence import write_json
    from smartsom.experiments.tuning_resources import ResourceLease, ResourceRequest

    facts, create, _ = observed
    root = tmp_path / "shared-fairness"
    runtime_control.initialize(root, 1)
    write_json(root / "control/owner.json", {"id": "original"})
    queue_admission.initialize(root, supported=True)
    assert queue_admission.reserve(
        root, "native:busy", owner_id="original", cpus=3, memory=GIB
    )
    process = psutil.Process(os.getpid())
    queue_admission.update_resources(
        root,
        "native:busy",
        owner_id="original",
        memory=GIB,
        observed_memory=process.memory_info().rss,
        observed_cpus=3,
        observed_processes=[(process.pid, process.create_time())],
    )
    facts.snapshot = ResourceSnapshot(
        8, 16 * GIB, 12 * GIB, cpu_load=37.5, external_cpu_load=37.5
    )
    broker = create(
        names=("a", "b"),
        admission_root=root,
        options={"a": [profile(cpu=1)], "b": [profile(cpu=5)]},
    )
    broker._resources["a"] = resources()
    broker._manager().leases["a"] = ResourceLease(
        "a", ResourceRequest(1, GIB), "running", ()
    )
    assert not broker.try_acquire("b", resources(cpu=5))
    waiter = queue_admission.snapshot(root)["waiters"]["source:b"]
    assert waiter["feasible"]
    assert waiter["cpu_reserve"] == 1.6
    assert "source:b" not in queue_admission.snapshot(root)["slots"]
    assert "b" not in broker._manager().leases


def test_task_resource_dto_uses_observations_not_reservations_and_expires(
    observed, tmp_path
):
    from smartsom.experiments import runtime_control
    from smartsom.experiments.tuning_resources import ResourceLease, ResourceRequest

    facts, create, _ = observed
    root = tmp_path / "resource-display"
    runtime_control.initialize(root, 2)
    broker = create(control_root=root)
    broker._resources["a"] = resources(cpu=4, memory=2 * GIB)
    broker._manager().leases["a"] = ResourceLease(
        "a", ResourceRequest(4, 2 * GIB), "running", (111,)
    )
    broker._owners["a"] = (111, 1.0)
    broker._actors["a"] = "actor-a"
    facts.actors["actor-a"] = {"State": "ALIVE"}
    broker._refresh_control()

    def publish():
        broker._publish_control()
        return runtime_control.snapshot(root)["tasks"]["a"]["resources"]

    unknown = publish()
    assert unknown["status"] == "unavailable"
    assert unknown["cpu_cores"] is None and unknown["memory"] is None
    assert unknown["memory_peak"] is None
    broker._budget_snapshot(
        ResourceSnapshot(
            16,
            64 * GIB,
            32 * GIB,
            processes=(ProcessUsage(111, "learner", None, 128 * 1024**2),),
        )
    )
    first = publish()
    assert first["status"] == "current"
    assert first["cpu_cores"] is None  # CPU requires a primed sample.
    assert first["memory"] == 128 * 1024**2
    assert first["observed_at"] is not None
    broker._budget_snapshot(
        ResourceSnapshot(
            16,
            64 * GIB,
            32 * GIB,
            processes=(ProcessUsage(111, "learner", 0.4, 100 * 1024**2),),
        )
    )
    current = publish()
    assert current["cpu_cores"] == 0.4 and current["memory"] == 100 * 1024**2
    assert current["memory_peak"] == 128 * 1024**2
    facts.now += 6
    stale = publish()
    assert stale["status"] == "stale"
    assert stale["cpu_cores"] is None and stale["memory"] is None
    assert stale["memory_peak"] == current["memory_peak"]
    del broker._manager().leases["a"]
    inactive = publish()
    assert inactive["status"] == "inactive"
    assert inactive["cpu_cores"] is None and inactive["memory"] is None
    assert inactive["memory_peak"] == current["memory_peak"]
    legacy = broker.__getstate__()
    legacy.pop("_resource_observations")
    restored = AdaptiveBroker.__new__(AdaptiveBroker)
    restored.__setstate__(legacy)
    assert restored._resource_observations == {}
    assert restored._memory_peaks["a"] == current["memory_peak"]


@pytest.mark.parametrize("mode", ["performance", "balanced"])
def test_unknown_first_worker_placeholder_fits_shared_envelope(
    observed, tmp_path, mode
):
    from smartsom.experiments import queue_admission, runtime_control
    from smartsom.experiments.evidence import write_json

    facts, create, _ = observed
    root = tmp_path / "unknown-startup"
    runtime_control.initialize(root, 1)
    write_json(root / "control/owner.json", {"id": "original"})
    queue_admission.initialize(root, supported=True)
    facts.snapshot = ResourceSnapshot(8, 16 * GIB, 12 * GIB, external_cpu_load=0)
    broker = create(
        admission_root=root,
        mode=mode,
        groups={"a": "unknown"},
        uncalibrated_groups={"unknown"},
    )
    capacity = queue_admission.envelope_capacity(
        root, owner_id="original", host=facts.snapshot
    )
    assert broker.try_acquire("a", resources())
    slot = queue_admission.snapshot(root)["slots"]["source:a"]
    assert GIB < slot["memory"] <= capacity["memory"]
    assert broker._manager().leases["a"].request.memory >= GIB


@pytest.mark.parametrize("bound", ["authored", "observed", "model"])
def test_unknown_placeholder_never_clamps_real_memory_requirement(
    observed, tmp_path, monkeypatch, bound
):
    from smartsom.experiments import queue_admission, runtime_control
    from smartsom.experiments.evidence import write_json

    facts, create, _ = observed
    root = tmp_path / "real-memory-bound"
    runtime_control.initialize(root, 1)
    write_json(root / "control/owner.json", {"id": "original"})
    queue_admission.initialize(root, supported=True)
    facts.snapshot = ResourceSnapshot(8, 16 * GIB, 12 * GIB, external_cpu_load=0)
    authored = 10 * GIB if bound == "authored" else GIB
    broker = create(
        admission_root=root,
        mode="performance",
        groups={"a": "unknown"},
        uncalibrated_groups={"unknown"},
        options={"a": [profile(memory=authored)]},
    )
    if bound == "observed":
        broker._memory_peaks["a"] = 10 * GIB
    if bound == "model":
        monkeypatch.setattr(
            broker, "_stage_request", lambda *args: {"memory": 10 * GIB, "cpus": 1}
        )
    assert not broker.try_acquire("a", resources(memory=authored))
    assert broker._request("a", resources(memory=authored)).memory >= 10 * GIB
    assert not queue_admission.snapshot(root)["slots"]
    assert not queue_admission.snapshot(root)["waiters"]["source:a"]["feasible"]


def test_fixed_three_shared_placeholders_release_only_after_commit_and_allow_growth(
    observed, tmp_path, monkeypatch
):
    from smartsom.experiments import queue_admission, runtime_control
    from smartsom.experiments.evidence import write_json

    monkeypatch.setenv("SMARTSOM_FIXED_ADMISSION_SLOTS", "3")
    facts, create, _ = observed
    root = tmp_path / "fixed-shared"
    runtime_control.initialize(root, 3)
    write_json(root / "control/owner.json", {"id": "original"})
    queue_admission.initialize(root, supported=True)
    facts.snapshot = ResourceSnapshot(8, 16 * GIB, 12 * GIB, external_cpu_load=0)
    names = ("a", "b", "c", "d")
    broker = create(
        names=names,
        admission_root=root,
        mode="performance",
        execution="fixed",
        global_limit=3,
        groups={name: "unknown" for name in names},
        uncalibrated_groups={"unknown"},
    )
    for name in names[:3]:
        assert broker.try_acquire(name, resources())
    before = queue_admission.snapshot(root)["slots"]["source:a"]
    assert before["temporary_memory"] > before["memory_floor"]
    runtime_control.submit(root, "limit", value=4)
    assert not broker.try_acquire("d", resources())
    broker._memory_peaks["a"] = GIB
    broker.observe_formal_update("a")
    for _ in range(2):
        broker.refresh(force=True)
    after = queue_admission.snapshot(root)["slots"]["source:a"]
    assert "temporary_memory" not in after
    assert after["memory"] == after["memory_floor"] == int(1.25 * GIB)
    assert after["memory"] < before["memory"]
    assert broker.try_acquire("d", resources())
    assert len(queue_admission.snapshot(root)["slots"]) == 4
    broker.observe_formal_update("a")
    assert (
        queue_admission.snapshot(root)["slots"]["source:a"]["memory"] == after["memory"]
    )
