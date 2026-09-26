"""Real Ray pause/resume with a controlled resize, not a calibration benchmark.

The admission stand-in deliberately grants a fixed 2-to-1 CPU allocation. Ray
still creates/stops/restores genuine SmartSOMTrainable actors. The tiny native
PPO/DQN cases check complete continuation and final evidence, not policy quality
or the real resource broker's machine-load and process-release guarantees.
"""

import copy
import json
import pickle
import tempfile
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from smartsom import api
from smartsom.config.codec import canonical_json
from smartsom.config.experiment import prepare
from smartsom.experiments.composable import allocate
from smartsom.experiments.tuning_session import (
    AdaptiveSession,
    scientific_identity,
    verify_commit,
)

torch = pytest.importorskip("torch")
ray = pytest.importorskip("ray")
Callback = pytest.importorskip("ray.tune").Callback
from smartsom.experiments.tuning_ray import (  # noqa: E402
    SmartSOMTrainable,
    build_tuner,
)

pytestmark = pytest.mark.learning
ROOT = Path(__file__).resolve().parents[2]


class ControlledAdmission:
    """Grant requests deterministically; this is not the production broker."""

    def __init__(self):
        self.leases = {}
        self.resizes = []

    def try_acquire(self, identity, resources):
        self.leases[identity] = dict(resources)
        return True

    def defer_resize(self, identity, current, requested):
        self.resizes.append((identity, dict(current), dict(requested)))

    def release(self, identity):
        self.leases.pop(identity, None)

    def refresh(self):
        pass


def _tiny_prepared(algorithm, output):
    config = api.load_config(ROOT / f"configs/test/runs/train_all_{algorithm}.yaml")
    config.training.total_ticks = 8
    config.training.ticks_per_update = 4
    config.training.record_initial = False
    config.validation.enabled = True
    config.validation.every_updates = 1
    config.validation.replications = 1
    config.evaluation.replications = 1
    config.evaluation.checkpoint = "last"
    config.output.root = str(output)
    config.scenario_overrides["tick_limit"] = 8
    config.runtime.numerical_threads = 1
    config.runtime.sampling_processes = 0
    frozen = prepare(config)
    parameters = json.loads(frozen.parameters_json)
    if algorithm == "dqn":
        parameters.update(batch_size=2, warmup_ticks=0, target_update_ticks=4)
    else:
        parameters.update(batch_size=2, n_epochs=1)
    # Empty validation makes best/patience transitions predictable while both
    # training updates still execute the real manufacturing/learner backends.
    cases = json.loads(frozen.validation_json)
    for case in cases:
        case["scenario"]["demands"] = []
    return replace(
        frozen,
        parameters_json=canonical_json(parameters),
        validation_json=canonical_json(cases),
    )


def _state(checkpoint):
    return pickle.loads((Path(checkpoint) / "continuation.pkl").read_bytes())


def _assert_native_state_equal(actual, expected):
    import numpy as np

    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
    elif isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_native_state_equal(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected, strict=True):
            _assert_native_state_equal(a, b)
    elif hasattr(expected, "snapshot"):
        assert actual.snapshot() == expected.snapshot()
    else:
        assert actual == expected


@pytest.mark.parametrize("algorithm", ("ppo", "dqn"))
def test_real_native_trainable_resizes_and_preserves_complete_state(
    algorithm, tmp_path
):
    def seed_process_rng(seed):
        import random

        import numpy as np

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    # Native per-policy/optimizer RNGs use frozen named seeds. Ambient process
    # RNGs are also saved, but start differently in independent fresh actors;
    # normalize the first process solely for this complete-state comparison.
    seed_process_rng(321)
    frozen = _tiny_prepared(algorithm, tmp_path / "runs")
    baseline_root, baseline_record, baseline_frozen = allocate(frozen, "training")
    baseline_record["tuning"] = {"experiment_id": "continuous"}
    baseline = AdaptiveSession(
        baseline_frozen, baseline_root, baseline_record, threads=2
    )
    try:
        while not baseline.step()["done"]:
            pass
        expected = _state(baseline.last_commit)
        assert sum(expected["optimizations"].values()) > 0
    finally:
        baseline.close()

    run_root, record, frozen = allocate(frozen, "training")
    original = canonical_json(asdict(frozen))
    identity = "ray-native-" + algorithm
    controls = {
        "names": ["initial", "rule", "random"],
        "pair_key": "ray-native-controls-" + algorithm,
        "directory": str(tmp_path / "controls"),
    }

    class Observer(Callback):
        def __init__(self):
            self.results = []

        def on_trial_result(self, iteration, trials, trial, result, **info):
            marker = verify_commit(result["checkpoint"])
            self.results.append((copy.deepcopy(result), marker))

    class ObservedNative(SmartSOMTrainable):
        """Only instrument the genuine loader; no training/state substitution."""

        def setup(self, config):
            cpu = self.trial_resources.required_resources["CPU"]
            # Poison the restarted actor's ambient RNGs: the genuine native
            # loader must replace them with checkpoint state, not fresh seeds.
            seed_process_rng(321 if cpu == 2 else 654)
            super().setup(config)

        def load_checkpoint(self, checkpoint_dir):
            before = self.session.session.updates
            marker = verify_commit(checkpoint_dir)
            super().load_checkpoint(checkpoint_dir)
            proof = {
                "before_updates": before,
                "updates": self.session.session.updates,
                "physical_ticks": self.session.session.ticks,
                "threads": torch.get_num_threads(),
                "allocation_epoch": self.session.epoch,
                "commit_id": marker["commit_id"],
            }
            # Ray 2.58 does not invoke Callback.on_trial_restore, so observe its
            # actual restore RPC rather than infer restoration from a callback.
            with (self.session.root / "ray-native-restores.jsonl").open("a") as f:
                f.write(json.dumps(proof) + "\n")

    def allocation(controller, trial, result, scheduler):
        return {"CPU": 1, "GPU": 0} if result["updates"] == 1 else None

    observer = Observer()
    broker = ControlledAdmission()
    entry = {
        "experiment_id": identity,
        "prepared": asdict(frozen),
        "run_dir": str(run_root),
        "record": record,
        "control_spec": controls,
        "execution_contract": {
            "schema": "smartsom.tune-execution/v1",
            "cpu_overhead": 0,
            "allocation_epoch": 0,
        },
    }
    # This test owns its local runtime. Short macOS socket paths are deliberate.
    with tempfile.TemporaryDirectory(
        prefix="som-native-ray-", dir=Path("/tmp").resolve()
    ) as d:
        ray.init(
            num_cpus=4,
            include_dashboard=False,
            log_to_driver=False,
            object_store_memory=80 * 1024**2,
            _temp_dir=d,
        )
        try:
            tuner = build_tuner(
                [entry],
                selected={identity: {"CPU": 2, "GPU": 0}},
                broker=broker,
                storage_path=tmp_path / "ray-storage",
                name="native-" + algorithm,
                device="cpu",
                allocation_function=allocation,
                callbacks=[observer],
                trainable=ObservedNative,
            )
            results = tuner.fit()
            assert ray.is_initialized()
            assert len(results) == 1 and not results.errors
            result = results[0]
            assert result.metrics["done"]
            assert result.metrics["physical_ticks"] == 8
            assert (
                result.metrics["updates"] == result.metrics["training_iteration"] == 2
            )
            assert result.metrics["phase"] == "experiment_complete"
            restores = [
                json.loads(line)
                for line in (run_root / "ray-native-restores.jsonl")
                .read_text()
                .splitlines()
            ]
            assert restores == [
                {
                    "before_updates": 0,
                    "updates": 1,
                    "physical_ticks": 4,
                    "threads": 1,
                    "allocation_epoch": 2,
                    "commit_id": observer.results[0][1]["commit_id"],
                }
            ]
            assert broker.resizes == [
                (identity, {"CPU": 2.0, "GPU": 0.0}, {"CPU": 1.0, "GPU": 0.0})
            ]
            reports = [r for r, _ in observer.results]
            assert [r["actual_threads"] for r in reports] == [2, 1]
            assert [r["allocation_epoch"] for r in reports] == [1, 2]
            assert [r["updates"] for r in reports] == [1, 2]
            assert [r["physical_ticks"] for r in reports] == [4, 8]
            assert reports[0]["pid"] != reports[1]["pid"]
            assert {r["experiment_id"] for r in reports} == {identity}
            first = _state(reports[0]["checkpoint"])
            assert first["best_update"] == 1 and first["no_improvement"] == 0
            assert (run_root / "logs/validation-000001.json").is_file()
            assert (run_root / "logs/validation-000002.json").is_file()
            final_marker = observer.results[-1][1]
            assert final_marker["scientific_sha256"] == scientific_identity(frozen)
            assert final_marker["threads"] == 1
            assert _state(result.metrics["checkpoint"])["best_update"] == 1
            assert _state(result.metrics["checkpoint"])["no_improvement"] == 1
            _assert_native_state_equal(_state(result.metrics["checkpoint"]), expected)
            # Both the native final commit and Ray's own copied checkpoint have
            # a validated complete payload, including initial/best support.
            with result.checkpoint.as_directory() as checkpoint:
                copied = verify_commit(checkpoint)
                assert copied["commit_id"] == final_marker["commit_id"]
                for name in ("update-000000", "update-000001"):
                    assert (
                        Path(checkpoint) / "native/support/checkpoints" / name
                    ).is_dir()
            saved_original = json.loads(
                (run_root / "config/original-prepared.json").read_text()
            )
            assert canonical_json(saved_original) == original
            final_record = json.loads(
                (Path(result.metrics["checkpoint"]) / "record.json").read_text()
            )
            assert final_record["tuning"]["experiment_status"] == "completed"
            assert set(final_record["tuning"]["controls"]) == set(controls["names"])
            assert (run_root / "evaluation/evidence/case-0000/trace.jsonl").is_file()
            for name in (
                "tuning-final",
                "controls/initial",
                "controls/rule",
                "controls/random",
            ):
                rows = json.loads(
                    (run_root / "evaluation" / (name + ".json")).read_text()
                )
                assert len(rows) == 1 and not rows[0]["engineering_failure"]
                assert rows[0]["seed"] == json.loads(frozen.evaluation_json)[0]["seed"]
        finally:
            ray.shutdown()
