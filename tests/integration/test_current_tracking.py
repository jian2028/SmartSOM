"""Real SDK offline segments and TensorBoard use local artifacts only."""

import json
import random

import pytest


def test_real_wandb_offline_segments_preserve_rng_and_experiment_identity(
    tmp_path, monkeypatch
):
    pytest.importorskip("wandb")
    torch = pytest.importorskip("torch")
    np = pytest.importorskip("numpy")
    from smartsom.config.base_diagnostic import TrackingOptions
    from smartsom.telemetry.experiment_tracking import ExperimentTracking

    monkeypatch.setenv("WANDB_MODE", "offline")
    original = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    ids = []
    for step in (80, 160):
        tracker = ExperimentTracking(
            tmp_path, TrackingOptions(mode="offline"), job_type="train"
        )
        tracker.start(
            {
                "source": {
                    "commit": "engineering-acceptance",
                    "implementation_sha256": "test",
                },
                "inputs": {},
            },
            {},
        )
        tracker.log({"phase": "train", "physical_ticks": step, "reward": -2.5})
        tracker.close()
        ids.append(tracker.status["experiment_id"])
    status = json.loads((tmp_path / "tracking/status.json").read_text())
    assert ids[0] == ids[1]
    assert status["sequence"] == 2 and status["sync_status"] == "offline"
    assert len({s["run_id"] for s in status["segments"]}) == 2
    assert len(list((tmp_path / "tracking").rglob("*.wandb"))) == 2
    assert random.getstate() == original[0]
    assert np.array_equal(np.random.get_state()[1], original[1][1])
    assert torch.equal(torch.get_rng_state(), original[2])


def test_tensorboard_and_local_journal_share_metric_values(tmp_path):
    pytest.importorskip("tensorboard")
    import random

    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    from smartsom.config.base_diagnostic import TrackingOptions
    from smartsom.telemetry.experiment_tracking import ExperimentTracking

    before = random.getstate()
    tracking = ExperimentTracking(
        tmp_path, TrackingOptions(mode="disabled", tensorboard=True), job_type="train"
    )
    tracking.start({"source": {}, "inputs": {}}, {})
    tracking.log({"phase": "train", "physical_ticks": 80, "reward": -12.5})
    tracking.close()
    events = EventAccumulator(str(tmp_path / "tensorboard")).Reload()
    assert events.Scalars("train/reward")[0].value == -12.5
    assert events.Scalars("train/reward")[0].step == 80
    assert (
        json.loads((tmp_path / "tracking/events.jsonl").read_text())["reward"] == -12.5
    )
    assert random.getstate() == before


def test_evaluation_series_separate_method_and_selection(tmp_path):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    from smartsom.config.base_diagnostic import TrackingOptions
    from smartsom.telemetry.experiment_tracking import ExperimentTracking

    tracking = ExperimentTracking(
        tmp_path,
        TrackingOptions(mode="disabled", tensorboard=True),
        job_type="evaluate",
    )
    tracking.start({"source": {}, "inputs": {}}, {})
    for selection, reward in (("sample", -5), ("argmax", -8)):
        tracking.log(
            {
                "phase": "validation",
                "method": "configured",
                "selection": selection,
                "training_progress": {"episodes": 4},
                "reward": reward,
            }
        )
    tracking.close()
    events = EventAccumulator(str(tmp_path / "tensorboard")).Reload()
    assert events.Scalars("validation/configured/sample/reward")[0].value == -5
    assert events.Scalars("validation/configured/argmax/reward")[0].value == -8
    assert events.Scalars("validation/configured/argmax/reward")[0].step == 4
