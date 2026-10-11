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
