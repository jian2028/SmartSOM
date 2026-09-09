"""Presentation and tracking must preserve training RNGs and failure causes."""

import importlib.util
import json
import os
import random

import pytest

from smartsom.api import load_preset
from smartsom.telemetry.training import TrainingDisplay, isolated_random_state


def require(package):
    if importlib.util.find_spec(package) is None:
        if os.environ.get("SMARTSOM_REQUIRE_TRACKING") == "1":
            pytest.fail(f"required tracking dependency {package} is missing")
        pytest.skip(f"optional {package} is absent")


def configure(root):
    (root / "logs").mkdir(parents=True)
    config = load_preset("sb3_micro")
    config.logging.tensorboard = False
    config.logging.progress = "off"
    config.logging.verbose = 0
    return config


def test_noninteractive_json_has_no_terminal_escape_and_keeps_numeric_metrics(
    tmp_path, capsys
):
    config = configure(tmp_path)
    config.logging.format = "json"
    config.logging.verbose = 1
    with TrainingDisplay(tmp_path, config) as display:
        display(
            {
                "stage": "completed",
                "sampled_steps": 256,
                "ppo_updates": 1,
                "metrics": {"entropy": 1.25, "unavailable": None},
            }
        )
    output = capsys.readouterr().err
    assert "\x1b" not in output
    assert json.loads(output)["metrics"]["entropy"] == 1.25
    row = json.loads((tmp_path / "logs/events.jsonl").read_text())
    assert row["ppo_updates"] == 1
    assert row["metrics"]["unavailable"] is None


def test_progress_callback_cannot_consume_global_rng(tmp_path):
    config = configure(tmp_path)
    random.seed(918)
    before = random.getstate()
    with TrainingDisplay(
        tmp_path, config, on_progress=lambda event: random.random()
    ) as display:
        display({"stage": "sampling", "sampled_steps": 1})
    assert random.getstate() == before


def test_numpy_torch_rng_isolation_on_callback_failure():
    require("torch")
    import numpy as np
    import torch

    random.seed(12)
    np.random.seed(12)
    torch.manual_seed(12)
    expected = (random.random(), np.random.random(), torch.rand(2))
    random.seed(12)
    np.random.seed(12)
    torch.manual_seed(12)
    with pytest.raises(RuntimeError, match="callback failed"):
        with isolated_random_state():
            random.random()
            np.random.random()
            torch.rand(2)
            raise RuntimeError("callback failed")
    assert random.random() == expected[0]
    assert np.random.random() == expected[1]
    assert torch.equal(torch.rand(2), expected[2])


def test_cleanup_preserves_original_failure_and_closes_other_sinks(tmp_path):
    config = configure(tmp_path)
    cause = ValueError("original physical failure")

    class BrokenWriter:
        def close(self):
            raise OSError("disk failed")

    with pytest.raises(ValueError, match="original physical failure") as captured:
        with TrainingDisplay(tmp_path, config) as display:
            display.writer = BrokenWriter()
            raise cause
    assert captured.value is cause
    assert display.log.closed
    assert "disk failed" in cause.__notes__[0]


def test_tensorboard_writes_same_named_scalar_stream(tmp_path):
    require("tensorboard")
    require("torch")
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    config = configure(tmp_path)
    config.logging.tensorboard = True
    with TrainingDisplay(tmp_path, config) as display:
        display(
            {
                "stage": "learning_metrics",
                "sampled_steps": 256,
                "ppo_updates": 1,
                "metrics": {"entropy": 0.75},
            }
        )
    accumulator = EventAccumulator(str(tmp_path / "logs/tensorboard")).Reload()
    point = accumulator.Scalars("learner/entropy")[0]
    assert (point.step, point.value) == (256, 0.75)


def test_wandb_offline_integration_does_not_publish(tmp_path, monkeypatch):
    require("wandb")
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_SILENT", "true")
    monkeypatch.setenv("WANDB_DISABLE_GIT", "true")
    config = configure(tmp_path)
    config.logging.wandb = True
    config.logging.wandb_project = "smartsom-local-integration"
    config.logging.wandb_mode = "offline"
    with TrainingDisplay(tmp_path, config) as display:
        display(
            {
                "stage": "completed",
                "sampled_steps": 256,
                "ppo_updates": 1,
                "metrics": {"entropy": 0.75},
            }
        )
        assert display.wandb_run.settings.mode == "offline"
    files = list((tmp_path / "logs/wandb").glob("offline-run-*/run-*.wandb"))
    assert len(files) == 1 and files[0].stat().st_size > 0
