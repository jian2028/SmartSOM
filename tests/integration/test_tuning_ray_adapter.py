"""A real local Tune segment, using a tiny checkpointable engineering stand-in."""

import json
import tempfile
import time
from pathlib import Path

import pytest

pytest.importorskip("ray")
import ray
from ray.tune import Trainable

from smartsom.experiments.tuning_ray import build_tuner

pytestmark = pytest.mark.learning


class Broker:
    """Admission stand-in; Ray still enforces the actual logical CPU requests."""

    def __init__(self):
        self.leases = {}
        self.resizes = []

    def try_acquire(self, identity, resources):
        self.leases[identity] = resources
        return True

    def defer_resize(self, identity, current, requested):
        self.resizes.append((identity, current, requested))

    def release(self, identity):
        self.leases.pop(identity, None)

    def refresh(self):
        pass


def test_two_trials_resize_and_restore_inside_existing_runtime(tmp_path):
    class NativeCounter(Trainable):
        def setup(self, config):
            self.updates = 40  # native progress is independent of Tune iteration
            self.restore_count = 0
            self.cpu = self.trial_resources.required_resources["CPU"]

        def step(self):
            time.sleep(0.03)
            self.updates += 1
            return {
                "updates": self.updates,
                "restores": self.restore_count,
                "allocated_cpu": self.cpu,
                "done": self.updates == 43,
            }

        def save_checkpoint(self, checkpoint_dir):
            (Path(checkpoint_dir) / "state.json").write_text(
                json.dumps({"updates": self.updates, "restores": self.restore_count})
            )
            return checkpoint_dir

        def load_checkpoint(self, checkpoint_dir):
            state = json.loads((Path(checkpoint_dir) / "state.json").read_text())
            self.updates = state["updates"]
            self.restore_count = state["restores"] + 1

    def allocation(controller, trial, result, scheduler):
        return {"CPU": 1, "GPU": 0} if result["updates"] == 41 else None

    broker = Broker()
    # Short socket paths are required by macOS. The test owns this runtime;
    # build_tuner neither initializes nor shuts down the driver's runtime.
    with tempfile.TemporaryDirectory(
        prefix="smartsom-ray-", dir=Path("/tmp").resolve()
    ) as root:
        ray.init(
            num_cpus=4,
            include_dashboard=False,
            log_to_driver=False,
            object_store_memory=80 * 1024**2,
            _temp_dir=root,
        )
        try:
            tuner = build_tuner(
                [{"experiment_id": "a"}, {"experiment_id": "b"}],
                selected={
                    "a": {"CPU": 2, "GPU": 0},
                    "b": {"CPU": 2, "GPU": 0},
                },
                broker=broker,
                storage_path=tmp_path,
                name="real-counter-segment",
                device="cpu",
                trainable=NativeCounter,
                allocation_function=allocation,
            )
            results = tuner.fit()
            assert ray.is_initialized()
            assert len(results) == 2 and not results.errors
            assert {entry[0] for entry in broker.resizes} == {"a", "b"}
            for result in results:
                assert result.metrics["updates"] == 43
                assert result.metrics["training_iteration"] == 3
                assert result.metrics["allocated_cpu"] == 1
                assert result.metrics["restores"] == 1
                assert result.metrics_dataframe["allocated_cpu"].tolist() == [2, 1, 1]
        finally:
            ray.shutdown()
