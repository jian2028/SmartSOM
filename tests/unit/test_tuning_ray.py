"""Ray admission, resize lifecycle and native-session contract checks."""

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("ray")
from ray.tune import Trainable
from ray.tune.execution.placement_groups import PlacementGroupFactory
from ray.tune.experiment import Experiment, Trial
from ray.tune.schedulers.trial_scheduler import TrialScheduler

from smartsom.experiments import tuning_ray as adapter


class Broker:
    def __init__(self):
        self.denied = set()
        self.leases = {}
        self.deferred = []
        self.pausing = set()

    def try_acquire(self, identity, resources):
        if identity in self.denied:
            return False
        self.leases[identity] = dict(resources)
        return True

    def defer_resize(self, identity, current, requested):
        self.deferred.append((identity, dict(current), dict(requested)))
        self.denied.add(identity)

    def release(self, identity):
        self.leases.pop(identity, None)

    def refresh(self):
        pass

    def should_pause(self, identity):
        return identity in self.pausing


def resources(cpu, gpu=0):
    return {"CPU": cpu, "GPU": gpu}


def trial(identity, cpu=1, gpu=0):
    item = Trial(
        "stub",
        stub=True,
        trial_id=identity,
        config={"experiment_id": identity},
        placement_group_factory=PlacementGroupFactory([resources(cpu, gpu)]),
    )
    item.create_placement_group_factory()
    return item


def controller(trials):
    return SimpleNamespace(get_trials=lambda: trials, _reuse_actors=False)


@pytest.fixture
def numerical_environment(monkeypatch):
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        monkeypatch.setenv(key, os.environ.get(key, "1"))


def test_version_contract_is_explicit(monkeypatch):
    monkeypatch.setattr(adapter.ray, "__version__", "2.59.0")
    with pytest.raises(RuntimeError, match="Revalidate"):
        adapter.PresetSearchAlgorithm([], {}, Broker())


def test_bounded_retention_is_per_trial_and_preserves_legacy_history(tmp_path):
    class Dummy(Trainable):
        def step(self):
            return {"done": True}

    entries = [
        {
            "experiment_id": "bounded",
            "prepared": {
                "config_json": json.dumps(
                    {"checkpointing": {"retention": {"mode": "latest_full_and_best"}}}
                )
            },
        },
        {"experiment_id": "legacy"},
    ]
    search = adapter.PresetSearchAlgorithm(
        entries, {entry["experiment_id"]: resources(1) for entry in entries}, Broker()
    )
    search.add_configurations(
        Experiment("mixed", Dummy, num_samples=2, storage_path=str(tmp_path))
    )
    first, second = search.next_trial(), search.next_trial()
    assert first.run_metadata.checkpoint_manager.checkpoint_config.num_to_keep == 1
    assert first.run_metadata.checkpoint_manager.checkpoint_config.checkpoint_at_end
    assert second.run_metadata.checkpoint_manager.checkpoint_config.num_to_keep is None


@pytest.mark.parametrize(
    "value",
    [{"CPU": 0}, {"CPU": 1.5}, {"CPU": 1, "GPU": 0.5}, {"CPU": 1, "memory": -1}],
)
def test_invalid_resource_requests_fail_before_admission(value):
    with pytest.raises(ValueError):
        adapter.resource_dict(value)


def test_search_waits_without_creating_pending_and_skips_blocked_head(tmp_path):
    class Dummy(Trainable):
        def step(self):
            return {"done": True}

    broker = Broker()
    broker.denied.update({"a", "b"})
    entries = [{"experiment_id": identity} for identity in ("a", "b")]
    search = adapter.PresetSearchAlgorithm(
        entries, {"a": resources(3), "b": resources(1)}, broker
    )
    search.add_configurations(
        Experiment("test", Dummy, num_samples=2, storage_path=str(tmp_path))
    )
    assert search.next_trial() is None
    assert not search.is_finished()
    assert broker.leases == {}
    broker.denied.remove("b")
    created = search.next_trial()
    assert created.config == entries[1]
    assert created.status == Trial.PENDING
    assert not search.is_finished()
    broker.denied.clear()
    other = search.next_trial()
    assert other.config == entries[0]
    assert other.trial_id != created.trial_id
    assert search.is_finished()
    assert search.next_trial() is None
    search.save_to_dir(tmp_path)
    assert (
        len(json.loads((tmp_path / "preset-search-state.json").read_text())["issued"])
        == 2
    )
    with pytest.raises(RuntimeError, match="new Tune segment"):
        search.restore_from_dir(tmp_path)


def test_initial_resources_change_before_actor_staging_and_base_stays_uniform():
    broker = Broker()
    selected = {"a": resources(3), "b": {**resources(2), "memory": 1024}}
    scheduler = adapter.SafeResourceChangingScheduler(selected=selected, broker=broker)
    trials = [trial("a"), trial("b")]
    for item in trials:
        scheduler.on_trial_add(controller(trials), item)
    assert adapter.resource_dict(scheduler.base_trial_resources) == resources(1)
    assert adapter.resource_dict(trials[0].placement_group_factory) == resources(3)
    assert adapter.resource_dict(trials[1].placement_group_factory) == selected["b"]
    assert trials[0].config["execution_contract"]["allocation_epoch"] == 1
    with pytest.raises(RuntimeError, match="varying base resources"):
        scheduler.on_trial_add(controller(trials), trial("c", cpu=2))


def test_every_queued_resize_is_applied_and_old_leases_block_resume():
    broker = Broker()

    def allocation(*args):
        return resources(2)

    scheduler = adapter.SafeResourceChangingScheduler(
        selected={"a": resources(3), "b": resources(3)},
        broker=broker,
        allocation_function=allocation,
    )
    trials = [trial("a"), trial("b")]
    state = controller(trials)
    for item in trials:
        scheduler.on_trial_add(state, item)
        item.set_status(Trial.RUNNING)
        assert scheduler.on_trial_result(state, item, {}) == TrialScheduler.PAUSE
        item.set_status(Trial.PAUSED)
    assert len(broker.deferred) == 2
    assert scheduler.choose_trial_to_run(state) is None
    assert all(
        adapter.resource_dict(t.placement_group_factory)["CPU"] == 3 for t in trials
    )
    broker.denied.clear()  # observer has confirmed both old actors exited
    assert scheduler.choose_trial_to_run(state) in trials
    assert scheduler._trials_to_reallocate == {}
    assert all(
        adapter.resource_dict(t.placement_group_factory)["CPU"] == 2 for t in trials
    )
    assert all(t.config["execution_contract"]["allocation_epoch"] == 2 for t in trials)


def test_minimum_allocation_can_pause_and_wait_without_stopping():
    broker = Broker()
    broker.pausing.add("a")
    scheduler = adapter.SafeResourceChangingScheduler(
        selected={"a": resources(1)}, broker=broker
    )
    item = trial("a")
    state = controller([item])
    scheduler.on_trial_add(state, item)
    item.set_status(Trial.RUNNING)
    assert scheduler.on_trial_result(state, item, {}) == TrialScheduler.PAUSE
    item.set_status(Trial.PAUSED)
    assert scheduler.choose_trial_to_run(state) is None
    assert item.status == Trial.PAUSED
    broker.denied.clear()
    broker.pausing.clear()
    assert scheduler.choose_trial_to_run(state) is item


def test_frozen_device_cannot_change_during_resize():
    broker = Broker()
    scheduler = adapter.SafeResourceChangingScheduler(
        selected={"a": resources(1)},
        broker=broker,
        allocation_function=lambda *args: resources(1, 1),
    )
    item = trial("a")
    scheduler.on_trial_add(controller([item]), item)
    item.set_status(Trial.RUNNING)
    with pytest.raises(ValueError, match="device cohorts"):
        scheduler.on_trial_result(controller([item]), item, {})


def test_threads_are_set_before_state_restore_and_old_threads_are_not_loaded(
    tmp_path, monkeypatch
):
    events = []

    class Limits:
        def restore_original_limits(self):
            events.append("release_threads")

    def set_threads(threads):
        events.append(("threads", threads))
        return Limits()

    class Session:
        def __init__(self, prepared, root, record, **kwargs):
            events.append(("session", kwargs, record["tuning"]))

        def step(self):
            return {"updates": 19, "ticks": 300, "done": True, "commit_id": "c19"}

        def load_checkpoint(self, path):
            events.append(("load", path))

        def save_checkpoint(self, path):
            events.append(("save", path))

        def close(self):
            events.append("close")

    monkeypatch.setattr(adapter, "_set_threads", set_threads)
    monkeypatch.setattr(adapter, "_session_factory", Session)
    config = {
        "experiment_id": "stable",
        "prepared": {"config_json": json.dumps({"runtime": {"device": "cpu"}})},
        "run_dir": str(tmp_path),
        "record": {"schema": "native"},
        "execution_contract": {"allocation_epoch": 2, "cpu_overhead": 1},
        "continuation": str(tmp_path / "previous"),
    }
    trainable = object.__new__(adapter.SmartSOMTrainable)
    trainable.config = config
    trainable._trial_info = SimpleNamespace(
        trial_resources=PlacementGroupFactory([resources(4)]), trial_id="new-trial"
    )
    trainable.setup(config)
    assert events[0] == ("threads", 3)
    assert events[1][1] == {"allocation_epoch": 2, "threads": 3}
    assert events[2] == ("threads", 3)
    assert events[3] == ("load", tmp_path / "previous")
    result = trainable.step()
    assert result["updates"] == 19 and result["experiment_id"] == "stable"
    assert "training_iteration" not in result
    target = str(tmp_path / "ray-checkpoint")
    assert trainable.save_checkpoint(target) == target
    trainable.load_checkpoint(target)
    assert events[-1] == ("load", Path(target))
    trainable.cleanup()
    assert events[-3:] == ["close", "release_threads", "release_threads"]


def test_worker_setup_failure_records_cause_before_ray_reports_trial_error(
    tmp_path, monkeypatch
):
    class Limits:
        def restore_original_limits(self):
            pass

    def fail_session(*args, **kwargs):
        raise ValueError("worker implementation differs from frozen batch")

    monkeypatch.setattr(adapter, "_set_threads", lambda _threads: Limits())
    monkeypatch.setattr(adapter, "_session_factory", fail_session)
    config = {
        "experiment_id": "stable",
        "prepared": {"config_json": json.dumps({"runtime": {"device": "cpu"}})},
        "run_dir": str(tmp_path),
        "record": {"schema": "native"},
        "execution_contract": {"cpu_overhead": 0},
    }
    trainable = object.__new__(adapter.SmartSOMTrainable)
    trainable.config = config
    trainable._trial_info = SimpleNamespace(
        trial_resources=PlacementGroupFactory([resources(1)]), trial_id="trial"
    )
    with pytest.raises(ValueError, match="worker implementation differs"):
        trainable.setup(config)
    saved = json.loads((tmp_path / "logs/worker-error.json").read_text())
    assert saved == {
        "exception": "ValueError",
        "message": "worker implementation differs from frozen batch",
        "phase": "setup",
    }


def test_early_worker_setup_validation_failure_also_records_cause(tmp_path):
    config = {
        "run_dir": str(tmp_path),
        "execution_contract": {"cpu_overhead": -1},
    }
    trainable = object.__new__(adapter.SmartSOMTrainable)
    trainable.config = config
    trainable._trial_info = SimpleNamespace(
        trial_resources=PlacementGroupFactory([resources(1)]), trial_id="trial"
    )
    with pytest.raises(ValueError, match="cpu_overhead"):
        trainable.setup(config)
    saved = json.loads((tmp_path / "logs/worker-error.json").read_text())
    assert saved["phase"] == "setup"
    assert "cpu_overhead" in saved["message"]


def test_initialized_interop_threads_cannot_silently_ignore_a_conflict(
    monkeypatch, numerical_environment
):
    def fail(value):
        raise RuntimeError("already initialized")

    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            set_num_threads=lambda value: None,
            get_num_interop_threads=lambda: 4,
            set_num_interop_threads=fail,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "threadpoolctl",
        SimpleNamespace(
            threadpool_limits=lambda **kw: None, threadpool_info=lambda: []
        ),
    )
    with pytest.raises(RuntimeError, match="inter-op"):
        adapter._set_threads(2)


def test_thread_pool_limits_are_verified_and_a_mismatch_is_explicit(
    monkeypatch, numerical_environment
):
    events = []
    pools = [{"user_api": "blas", "num_threads": 2, "prefix": "openblas"}]

    class Limits:
        def restore_original_limits(self):
            events.append("restore")

    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            set_num_threads=lambda value: events.append(("torch", value)),
            get_num_threads=lambda: 2,
            get_num_interop_threads=lambda: 1,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "threadpoolctl",
        SimpleNamespace(
            threadpool_limits=lambda **kw: Limits(), threadpool_info=lambda: pools
        ),
    )
    limits = adapter._set_threads(2)
    assert events == [("torch", 2)]
    assert isinstance(limits, Limits)
    pools[0]["num_threads"] = 4
    with pytest.raises(RuntimeError, match="openblas"):
        adapter._set_threads(2)
    assert events[-1] == "restore"


def test_build_requires_driver_runtime_without_starting_or_stopping_ray(monkeypatch):
    monkeypatch.setattr(adapter.ray, "is_initialized", lambda: False)
    monkeypatch.setattr(
        adapter.ray, "init", lambda **kwargs: pytest.fail("adapter initialized Ray")
    )
    monkeypatch.setattr(
        adapter.ray, "shutdown", lambda: pytest.fail("adapter stopped Ray")
    )
    with pytest.raises(RuntimeError, match="batch driver"):
        adapter.build_tuner(
            [{"experiment_id": "a"}],
            selected={"a": resources(1)},
            broker=Broker(),
            storage_path="/tmp/not-created",
            name="segment",
            device="cpu",
        )


def test_real_session_factory_checks_live_source_before_learner(monkeypatch, tmp_path):
    from smartsom.experiments import composable, evidence, tuning_session

    live = {
        "python": "3.12",
        "platform": "test",
        "packages": {},
        "git": {"commit": None},
    }
    monkeypatch.setattr(evidence, "source_identity", lambda: live)
    monkeypatch.setattr(composable, "implementation_identity", lambda: "changed")
    monkeypatch.setattr(
        tuning_session,
        "AdaptiveSession",
        lambda *args, **kwargs: pytest.fail(
            "constructed learner before checking source"
        ),
    )
    with pytest.raises(ValueError, match="worker implementation"):
        adapter._session_factory(
            {},
            tmp_path,
            {"source": live, "implementation_sha256": "frozen"},
            allocation_epoch=1,
            threads=1,
        )
