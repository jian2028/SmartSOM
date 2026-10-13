"""A failed Ray trial must leave a durable reason and end its Tune segment."""

import json
import sys

import pytest

ray = pytest.importorskip("ray")
from ray import tune  # noqa: E402
from ray.tune.error import TuneError  # noqa: E402

from smartsom.experiments.composable import implementation_identity  # noqa: E402
from smartsom.experiments.evidence import source_identity  # noqa: E402
from smartsom.experiments.tuning_callbacks import EvidenceCallback  # noqa: E402
from smartsom.experiments.tuning_ray import _record_worker_error  # noqa: E402

pytestmark = pytest.mark.learning


@pytest.mark.parametrize("cancel", [False, True])
def test_real_tune_first_admission_wait_can_progress_or_stop(
    tmp_path, monkeypatch, cancel
):
    from ray.tune.execution import insufficient_resources_manager

    from smartsom.experiments.control import StopRequested
    from smartsom.experiments.tuning_ray import build_tuner

    monkeypatch.setenv("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")
    ray._private.ray_constants.RAY_ENABLE_UV_RUN_RUNTIME_ENV = False
    # Exercise the same delayed Ray warning branch immediately, without
    # spending a minute or saturating the host to deny the first admission.
    monkeypatch.setattr(
        insufficient_resources_manager,
        "_get_insufficient_resources_warning_threshold",
        lambda: 0,
    )

    class Broker:
        ready = False
        denied = 0

        def handoff(self, identity):
            pass

        def note_dispatched_actor(self, identity, actor):
            pass

        def cancel_undispatched(self, identity):
            pass

        def try_acquire(self, identity, resources):
            if not self.ready:
                self.denied += 1
            return self.ready

    broker = Broker()

    class Admission(tune.Callback):
        def on_step_begin(self, iteration, trials, **info):
            if iteration >= 4 and not broker.ready:
                assert broker.denied >= 3
                assert not trials
                if cancel:
                    raise StopRequested("cancel before first admission")
                broker.ready = True

    class Once(tune.Trainable):
        def step(self):
            return {"done": True}

        def save_checkpoint(self, checkpoint_dir):
            return {}

    ray.init(
        address="local",
        num_cpus=1,
        include_dashboard=False,
        log_to_driver=False,
        runtime_env={"py_executable": sys.executable},
    )
    try:
        tuner = build_tuner(
            [{"experiment_id": "first"}],
            selected={"first": {"CPU": 1, "GPU": 0}},
            broker=broker,
            storage_path=tmp_path,
            name="admission",
            device="cpu",
            callbacks=[Admission()],
            trainable=Once,
        )
        if cancel:
            with pytest.raises(StopRequested, match="before first admission"):
                tuner.fit()
        else:
            grid = tuner.fit()
            assert len(grid) == 1 and not grid.errors
    finally:
        ray.shutdown()
    assert not ray.is_initialized()


def test_unresolved_failed_trial_exits_real_tune_and_saves_reason(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")
    ray._private.ray_constants.RAY_ENABLE_UV_RUN_RUNTIME_ENV = False
    (tmp_path / "batch.json").write_text(
        json.dumps({"entries": {"entry": {"status": "queued"}}})
    )
    (tmp_path / "plan.json").write_text(
        json.dumps(
            {
                "source": source_identity(),
                "implementation_sha256": implementation_identity(),
            }
        )
    )

    class Broker:
        failed = False
        drained = False

        def drain_shared(self):
            self.drained = True

        def refresh(self):
            pass

        def note_actor(self, identity, trial):
            assert identity == "entry"

        def actor_failed(self, identity, trial):
            assert identity == "entry"
            self.failed = True

        def unresolved_failures(self):
            return ("entry",) if self.failed else ()

    def broken(config):
        try:
            raise ValueError("intentional worker setup failure")
        except ValueError as exc:
            _record_worker_error(config["run_dir"], exc, "setup")
            raise

    ray.init(
        address="local",
        num_cpus=1,
        include_dashboard=False,
        log_to_driver=False,
        runtime_env={"py_executable": sys.executable},
    )
    try:
        with pytest.raises(TuneError, match="reservation retained, batch stopped"):
            tune.Tuner(
                broken,
                param_space={
                    "experiment_id": "entry",
                    "run_dir": str(tmp_path / "attempt"),
                },
                run_config=tune.RunConfig(
                    name="failure-exit",
                    storage_path=str(tmp_path / "ray"),
                    verbose=0,
                    failure_config=tune.FailureConfig(max_failures=0),
                    callbacks=[EvidenceCallback(tmp_path, [], Broker())],
                ),
            ).fit()
    finally:
        ray.shutdown()

    row = json.loads((tmp_path / "batch.json").read_text())["entries"]["entry"]
    assert row["status"] == "failed"
    assert "intentional worker setup failure" in row["failure"]["message"]
    assert not ray.is_initialized()


def test_real_dispatch_cancel_before_ready_drains_shared_cap_without_start_callback(
    tmp_path, monkeypatch
):
    """Cancel after Actor.remote but before even scheduling its ready future."""
    import time
    from types import SimpleNamespace

    from smartsom.experiments import queue_admission
    from smartsom.experiments.control import StopRequested
    from smartsom.experiments.evidence import write_json
    from smartsom.experiments.tuning_broker import AdaptiveBroker
    from smartsom.experiments.tuning_ray import build_tuner
    from smartsom.experiments.tuning_resources import GIB, ResourceSnapshot

    monkeypatch.setenv("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")
    ray._private.ray_constants.RAY_ENABLE_UV_RUN_RUNTIME_ENV = False
    root = tmp_path / "source"
    (root / "control").mkdir(parents=True)
    write_json(root / "control/owner.json", dict(id="original"))
    queue_admission.initialize(root, supported=True)
    queue_admission.claim(root, "queue", limit=1, owner_id="original")

    class CanceledBroker(AdaptiveBroker):
        dispatched = None

        def note_dispatched_actor(self, identity, actor):
            super().note_dispatched_actor(identity, actor)
            self.dispatched = actor._actor_id.hex()
            assert not queue_admission.reserve(root, "native:next", owner_id="original")
            raise StopRequested("cancel after dispatch before ready")

    broker = CanceledBroker(
        [{"experiment_id": "entry", "run_dir": str(tmp_path / "entry")}],
        {
            "entry": [
                {
                    "CPU": 1,
                    "GPU": 0,
                    "memory": GIB,
                    "concurrency": 1,
                    "throughput": 1,
                    "update_seconds": 1,
                    "restart_seconds": 1,
                    "gpu_memory": 0,
                }
            ]
        },
        admission_root=root,
    )
    broker._manager().monitor = SimpleNamespace(
        snapshot=lambda **kwargs: ResourceSnapshot(
            4, 64 * GIB, 64 * GIB, external_cpu_load=0
        )
    )

    class SlowSetup(tune.Trainable):
        def setup(self, config):
            time.sleep(1)

        def step(self):
            return {"done": True}

    class Drain(tune.Callback):
        def on_trial_start(self, **kwargs):
            pytest.fail("start callback must never be scheduled")

        def on_experiment_end(self, trials, **kwargs):
            broker.drain_shared(timeout=20)

    ray.init(
        address="local",
        num_cpus=1,
        include_dashboard=False,
        log_to_driver=False,
        runtime_env={"py_executable": sys.executable},
    )
    try:
        tuner = build_tuner(
            [{"experiment_id": "entry"}],
            selected={"entry": {"CPU": 1, "GPU": 0, "memory": GIB}},
            broker=broker,
            storage_path=tmp_path,
            name="dispatch-cancel",
            device="cpu",
            callbacks=[Drain()],
            trainable=SlowSetup,
        )
        with pytest.raises(StopRequested, match="after dispatch before ready"):
            tuner.fit()
        assert broker.dispatched is not None
        assert broker._actor_state(broker.dispatched)["State"] == "DEAD"
        assert not broker._manager().leases
        assert queue_admission.reserve(root, "native:next", owner_id="original")
    finally:
        ray.shutdown()
