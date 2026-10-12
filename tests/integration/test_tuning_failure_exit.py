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
