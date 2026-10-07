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
