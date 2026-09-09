"""Presentation and tracking must preserve training RNGs and failure causes."""

import importlib.util
import io
import json
import os
import random
import re

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


def capture_text(display):
    from rich.console import Console

    output = io.StringIO()
    display.console = Console(file=output, width=120, force_terminal=False)
    return output


def test_throttled_metrics_remain_visible_in_periodic_and_final_tables(
    tmp_path, monkeypatch
):
    config = configure(tmp_path)
    config.logging.verbose = 1
    config.logging.every_seconds = 60
    clock = [100.0]
    monkeypatch.setattr("smartsom.telemetry.training.time.monotonic", lambda: clock[0])
    with TrainingDisplay(tmp_path, config) as display:
        output = capture_text(display)
        display({"stage": "sampling", "sampled_steps": 32, "completed_episodes": 0})
        clock[0] = 101.0
        # Names and signs observed in a real SB3 engineering example.
        display(
            {
                "stage": "learning_metrics",
                "sampled_steps": 32,
                "ppo_updates": 1,
                "learner_updates": 2,
                "metrics": {
                    "train/loss": 999.314697265625,
                    "train/entropy_loss": -1.7448177933692932,
                    "train/approx_kl": 0.00027121230959892273,
                },
            }
        )
        assert "entropy_loss" not in output.getvalue()
        output.seek(0)
        output.truncate()
        clock[0] = 161.0
        display({"stage": "sampling", "sampled_steps": 96})
        assert "entropy_loss" in output.getvalue()
        output.seek(0)
        output.truncate()
        clock[0] = 162.0
        display(
            {
                "stage": "completed",
                "sampled_steps": 128,
                "learner_updates": 8,
                "completed_episodes": 3,
                "failed_episodes": 0,
            }
        )
    text = output.getvalue()
    assert "Environment steps" in text
    assert "Latest received learner metrics" in text
    for value in ("train", "entropy_loss", "-1.74482", "999.315", "approx_kl"):
        assert value in text
    # The last observed PPO count is 1; do not infer 4 from sampled steps.
    assert re.search(r"PPO updates\s*│\s*1\s*│\s*Learner updates\s*│\s*8", text)
    rows = [
        json.loads(line)
        for line in (tmp_path / "logs/events.jsonl").read_text().splitlines()
    ]
    assert "metrics" not in rows[-1] and "ppo_updates" not in rows[-1]
    assert rows[1]["metrics"]["train/entropy_loss"] == -1.7448177933692932


@pytest.mark.parametrize(
    "scope,metrics,visible",
    [
        (
            "train",
            {
                "loss": 999.314697265625,
                "entropy_loss": -1.7448177933692932,
                "approx_kl": 0.00027121230959892273,
            },
            ("loss", "entropy_loss", "-1.74482", "approx_kl"),
        ),
        (
            "default_policy",
            {
                "total_loss": 10.293593406677246,
                "entropy": 1.6871527433395386,
                "mean_kl_loss": 0.0004071553994435817,
            },
            ("total_loss", "10.2936", "entropy", "1.68715", "mean_kl_loss"),
        ),
    ],
)
def test_backend_metric_names_are_not_renamed(tmp_path, scope, metrics, visible):
    config = configure(tmp_path)
    with TrainingDisplay(tmp_path, config) as display:
        output = capture_text(display)
        display(
            {
                "stage": "completed",
                "sampled_steps": 128,
                "metrics": {f"{scope}/{key}": value for key, value in metrics.items()},
            }
        )
    text = output.getvalue()
    assert scope in text
    assert all(value in text for value in visible)
    if scope == "default_policy":
        assert "entropy_loss" not in text


def test_resource_rounds_actions_and_role_metrics_are_separate(tmp_path):
    config = configure(tmp_path)
    with TrainingDisplay(tmp_path, config) as display:
        output = capture_text(display)
        display(
            {
                "stage": "learning_metrics",
                "sampled_steps": 128,
                "ppo_updates": 4,
                "learner_updates": 4,
                "metrics": {
                    "machine_policy/total_loss": 0.029030868783593178,
                    "machine_policy/entropy": 0.08148603141307831,
                    "machine_policy/mean_kl_loss": 0.00007068678678479046,
                    "agv_policy/total_loss": 0.01460867840796709,
                    "agv_policy/entropy": 0.30514463782310486,
                    "agv_policy/mean_kl_loss": 0.00010004257637774572,
                    "__all_modules__/total_loss": 99999,
                },
            }
        )
        display(
            {
                "stage": "completed",
                "sampled_steps": 128,
                "agent_steps": 1536,
                "physical_actions": 140,
                "completed_episodes": 2,
                "failed_episodes": 2,
            }
        )
    text = output.getvalue()
    assert "Joint rounds" in text and "Environment steps" not in text
    assert re.search(r"Agent steps\s*│\s*1536\s*│\s*Physical actions\s*│\s*140", text)
    # Each role's values occupy its own row block, never an aggregate or another role.
    agv = text.split("agv_policy", 1)[1].split("machine_policy", 1)[0]
    machine = text.split("machine_policy", 1)[1]
    assert "0.0146087" in agv and "0.305145" in agv
    assert "0.0290309" in machine and "0.081486" in machine
    assert "0.0290309" not in agv and "0.0146087" not in machine
    assert "99999" not in text and "__all_modules__" not in text


def test_missing_metrics_and_updates_are_na_including_missing_role(tmp_path):
    config = configure(tmp_path)
    with TrainingDisplay(tmp_path, config) as display:
        output = capture_text(display)
        display(
            {
                "stage": "failed",
                "sampled_steps": 128,
                "agent_steps": 0,
                "physical_actions": 0,
                "metrics": {
                    "machine_policy/total_loss": 0,
                    "machine_policy/entropy": None,
                },
            }
        )
    text = output.getvalue()
    assert "agv_policy" in text and "machine_policy" in text
    assert re.search(r"agv_policy\s*│\s*N/A\s*│\s*N/A\s*│\s*N/A", text)
    assert re.search(r"PPO updates\s*│\s*N/A\s*│\s*Learner updates\s*│\s*N/A", text)
    assert "entropy" in text and "total_loss" in text


def test_verbose_zero_noninteractive_text_only_renders_final_without_escapes(
    tmp_path, capsys
):
    config = configure(tmp_path)
    with TrainingDisplay(tmp_path, config) as display:
        display({"stage": "sampling", "sampled_steps": 32})
        display(
            {
                "stage": "learning_metrics",
                "sampled_steps": 32,
                "metrics": {"train/entropy_loss": -1.25},
            }
        )
        assert capsys.readouterr().err == ""
        display({"stage": "interrupted", "sampled_steps": 32})
    text = capsys.readouterr().err
    assert "\x1b" not in text
    assert "interrupted" in text and "entropy_loss" in text and "-1.25" in text
    assert "N/A" in text


def test_text_cache_never_fills_shared_json_callback_or_tracking_events(tmp_path):
    config = configure(tmp_path)
    received, scalar_events, wandb_events = [], [], []

    class Writer:
        def add_scalar(self, key, value, global_step):
            scalar_events.append((key, value, global_step))

        def close(self):
            pass

    class WandbRun:
        def log(self, metrics):
            wandb_events.append(metrics)

        def finish(self, exit_code):
            pass

    with TrainingDisplay(tmp_path, config, on_progress=received.append) as display:
        display.writer, display.wandb_run = Writer(), WandbRun()
        capture_text(display)
        metrics = {"train/entropy_loss": -1.25, "train/approx_kl": None}
        display({"stage": "learning_metrics", "sampled_steps": 32, "metrics": metrics})
        display({"stage": "completed", "sampled_steps": 128})
    rows = [
        json.loads(line)
        for line in (tmp_path / "logs/events.jsonl").read_text().splitlines()
    ]
    assert received == rows
    assert rows[0]["metrics"] == metrics and "metrics" not in rows[1]
    assert [row for row in scalar_events if row[0].startswith("learner/")] == [
        ("learner/train/entropy_loss", -1.25, 32)
    ]
    assert wandb_events[0]["learner/train/entropy_loss"] == -1.25
    assert not any(key.startswith("learner/") for key in wandb_events[-1])


def test_verbose_diagnostics_use_cached_full_metrics(tmp_path):
    config = configure(tmp_path)
    config.logging.verbose = 2
    with TrainingDisplay(tmp_path, config) as display:
        output = capture_text(display)
        display(
            {
                "stage": "learning_metrics",
                "sampled_steps": 32,
                "metrics": {"train/policy_gradient_loss": -0.0025315582752227783},
            }
        )
        output.seek(0)
        output.truncate()
        display({"stage": "completed", "sampled_steps": 128})
    assert "train/policy_gradient_loss" in output.getvalue()


def test_metric_rendering_preserves_python_numpy_torch_rngs(tmp_path):
    require("torch")
    import numpy as np
    import torch

    config = configure(tmp_path)
    random.seed(30)
    np.random.seed(30)
    torch.manual_seed(30)
    before = (random.getstate(), np.random.get_state(), torch.get_rng_state())
    with TrainingDisplay(tmp_path, config) as display:
        capture_text(display)
        display(
            {
                "stage": "completed",
                "sampled_steps": 32,
                "metrics": {"machine_policy/entropy": 0.25, "agv_policy/entropy": 0.5},
            }
        )
    assert random.getstate() == before[0]
    after_numpy = np.random.get_state()
    assert np.array_equal(after_numpy[1], before[1][1])
    assert (after_numpy[0], *after_numpy[2:]) == (before[1][0], *before[1][2:])
    assert torch.equal(torch.get_rng_state(), before[2])
