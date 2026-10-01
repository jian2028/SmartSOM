"""Tune presentation uses supplied facts, never scheduler or model state."""

import io
import json
import subprocess
import sys
from copy import deepcopy

import pytest
from rich.console import Console

from smartsom.telemetry.dashboard import phase_bar, task_eta
from smartsom.telemetry.monitor import monitor, read_snapshot
from smartsom.telemetry.runtime import DisplayOptions, RuntimeDisplay
from smartsom.telemetry.tuning_dashboard import (
    clean_summary,
    suggestions,
    summary_lines,
)


def summary(stage="calibration", count=8):
    return {
        "stage": stage,
        "calibration": {
            "active_seconds": 600,
            "wall_seconds": 600,
            "remaining_seconds": 0,
            "phase": "validation",
            "limit_seconds": 600,
            "waiting_seconds": 45,
            "candidates": [
                {"id": f"candidate-{i}", "threads": i + 1, "status": "measured"}
                for i in range(6)
            ],
            "measured": 2,
            "recommendation": {"threads": 2, "num_envs": 1},
            "reason": "已测候选中吞吐较高；等待资源时间另计",
        },
        "resources": {
            "cpus_available": 6,
            "memory_available": 8 * 1024**3,
            "external_cpu_load": 1.5,
            "mode": "full",
            "processes": [],
        },
        "entries": [
            {
                "experiment_id": f"trial-{i}",
                "status": "running",
                "threads": 1,
                "requested_cpus": 8,
                "actual_cpus": 2,
                "allocation_epoch": 4,
                "resource_change_reason": "资源请求将在安全边界应用",
                "pending_resize": True,
            }
            for i in range(count)
        ],
    }


def display(width=180, height=50, stage="training", count=8):
    stream = io.StringIO()
    view = RuntimeDisplay(
        DisplayOptions(verbose=False),
        kind="tune",
        console=Console(file=stream, width=width, height=height),
    )
    view.configure_tuning(summary(stage, count))
    with view.batch_updates():
        for i in range(count):
            view.update(
                f"trial-{i}",
                {
                    "stage": "sampling",
                    "physical_ticks": 420,
                    "updates": 2,
                    "workflow": {
                        "mode": "train-evaluate",
                        "algorithm": "PPO" if i % 2 else "DQN",
                        "training_total": 1000,
                        "round_size": 200,
                        "round_total": 5,
                        "training_unit": "physical ticks",
                        "validation_rounds": 2,
                        "validation_every": 2,
                        "validation_cases": 2,
                        "training_seed": 101,
                    },
                },
                total=1000,
                unit="physical ticks",
            )
    return view


def test_completed_phase_duration_is_frozen(monkeypatch):
    view = display(count=1)
    row = view.tasks["trial-0"]
    row.update(
        status="completed", stage="completed", stage_started_at=100, updated_at=105
    )
    monkeypatch.setattr("smartsom.telemetry.dashboard.time.time", lambda: 99999)
    text = phase_bar(view, row).plain
    assert "阶段耗时0m 05s" in text
    assert "已用" not in text


def plain(view):
    output = io.StringIO()
    console = Console(file=output, width=view.console.width, height=view.console.height)
    console.print(view.render())
    return output.getvalue()


@pytest.mark.parametrize(
    "width,height", [(60, 12), (80, 24), (100, 35), (160, 36), (200, 50)]
)
@pytest.mark.parametrize("stage", ["calibration", "waiting_resources", "training"])
def test_fixed_terminal_layout_and_explicit_hidden_trials(width, height, stage):
    view = display(width, height, stage, count=12)
    content = view.render()
    lines = view.console.render_lines(
        content, view.console.options.update(height=None), pad=False
    )
    assert len(lines) == height - 1
    assert all(sum(segment.cell_length for segment in row) <= width for row in lines)
    text = plain(view)
    if stage == "training" and not all(
        f"trial-{index} " in text for index in range(12)
    ):
        assert "hidden trials" in text
    if stage != "training":
        assert "实验卡片" not in text and "trial-0" not in text
    if height >= 24:
        assert "整体流程" in text and "当前阶段" in text
        if stage == "training":
            assert "实验卡片" in text and "正式实验结束 0/12" in text
        else:
            assert "待运行实验 12" in text
        assert "待生效" in text


def test_calibration_completion_is_not_trial_completion_and_wait_is_separate():
    view = display(stage="calibration", count=1)
    assert view.status == "running" and view.tasks["trial-0"]["status"] == "running"
    text = plain(view)
    assert "性能评估" in text and "600/600s" in text
    assert "等待资源 0m 45s" in text and "待运行实验 1" in text
    assert view.overview is None
    view.configure_tuning(summary("training", 1))
    assert view.status == "running"
    assert view.tasks["trial-0"]["completed"] == 420
    assert "已结束" not in plain(view)


@pytest.mark.parametrize("stage", ["validation", "evaluation"])
def test_case_progress_does_not_replace_training_ticks(stage):
    view = display(count=1)
    view.update(
        "trial-0",
        {
            "stage": stage,
            stage + "_finished": 1,
            stage + "_requested": 2,
            stage + "_tick": 50,
            stage + "_tick_limit": 200,
            stage + "_case_active": True,
        },
    )
    row = view.snapshot()["tasks"][0]
    assert row["completed"] == 420 and row["total"] == 1000
    text = plain(view)
    assert "训练预算" in text and "physical ticks: 420/1000" in text
    assert "当前验证" in text if stage == "validation" else "当前评估" in text
    assert "本案例50/200 ticks" in text


def test_compact_terminal_keeps_current_case_bar_and_reports_hidden_trials():
    view = display(width=80, height=24, count=12)
    view.update(
        "trial-0",
        {
            "stage": "validation",
            "validation_finished": 1,
            "validation_requested": 2,
            "validation_tick": 50,
            "validation_tick_limit": 200,
            "validation_case_active": True,
        },
    )
    text = plain(view)
    assert "当前验证" in text and "hidden trials" in text


def test_training_cards_follow_overview_and_stage_detail_at_normal_width():
    view = display(width=124, height=35, stage="training", count=2)
    text = plain(view)
    assert text.index("整体流程") < text.index("当前阶段") < text.index("实验卡片")
    assert "trial-0 · DQN" in text and "训练预算" in text
    assert "实验 / 状态" not in text


def test_compact_card_keeps_ticks_visible_before_resource_note():
    view = display(width=80, height=24, stage="training", count=2)
    text = plain(view)
    assert "420/1,000 physical ticks" in text
    assert "ETA 估算中" in text
    assert "当前采样" in text and "CPU 实际/请求 2/8" in text


def test_each_experiment_eta_uses_its_own_work_and_elapsed_time(monkeypatch):
    view = display(width=160, height=50, count=2)
    monkeypatch.setattr("smartsom.telemetry.dashboard.time.time", lambda: 200)
    first, second = view.tasks["trial-0"], view.tasks["trial-1"]
    first["started_at"] = second["started_at"] = 100
    first["completed"] = 500
    second["completed"] = 100
    assert task_eta(view, first) == "≈ 5m 00s"
    assert task_eta(view, second) == "≈ 31m 40s"
    text = plain(view)
    assert "ETA ≈ 5m 00s" in text and "ETA ≈ 31m 40s" in text


def test_actual_allocation_comes_from_worker_report_not_recommendation():
    view = display(width=124, height=35, count=1)
    view.tuning["resources"]["mode"] = "performance"
    view.tasks["trial-0"]["values"]["runtime_mode"] = (
        "envs=3; sampling_processes=2; threads=4; device=cpu"
    )
    text = plain(view)
    assert (
        "实际运行 CPU · Performance · 并行实验 1 · 每实验环境 3 · 采样进程 2 · 计算线程 4"
        in text
    )
    assert "推荐设置" not in text


def test_calibration_does_not_show_future_training_cards():
    view = display(width=124, height=35, stage="calibration", count=2)
    text = plain(view)
    assert "候选测量" in text and "待运行实验 2" in text
    assert "实验卡片" not in text and "trial-0" not in text


def test_requested_resource_change_never_fakes_applied_allocation():
    view = display(count=1)
    original = summary("training", 1)
    view.configure_tuning(original)
    original["entries"][0]["actual_cpus"] = 8
    saved = view.snapshot()["tuning"]["entries"][0]
    assert saved["actual_cpus"] == 2 and saved["requested_cpus"] == 8
    assert saved["allocation_epoch"] == 4 and saved["pending_resize"] is True
    text = plain(view)
    assert "CPU 请求 8 / 实际 2" in text and "待生效" in text


def test_calibration_tasks_are_not_counted_as_formal_experiments():
    view = display(stage="calibration", count=0)
    view.update("candidate-0", {"status": "completed"}, final=True)
    assert "待运行实验 0 · 已结束 0" in plain(view)


def test_tune_does_not_reuse_single_experiment_budget_for_batch_overview():
    view = display(stage="calibration", count=2)
    view.configure({"mode": "training", "training_total": 1000})
    view.publish(force=True)
    assert view.overview is None


def test_only_unprotected_apps_are_suggested_while_waiting_for_resources():
    value = summary("waiting_resources", 0)
    value["resources"]["processes"] = [
        {"pid": 1, "name": "SmartSOM-old", "cpu_cores": 4, "protected": False},
        {"pid": 2, "name": "WindowServer", "cpu_cores": 3, "protected": False},
        {"pid": 3, "name": "Python", "cpu_cores": 2, "protected": True},
        {"pid": 4, "name": "Chrome", "cpu_cores": 1, "protected": False},
        {"pid": 5, "name": "Unknown", "cpu_cores": 2},
    ]
    assert [row["name"] for row in suggestions(value)] == ["Chrome"]
    view = display(stage="waiting_resources", count=0)
    view.configure_tuning(value)
    assert "Chrome PID 4" in plain(view)
    assert "SmartSOM-old" not in plain(view) and "WindowServer" not in plain(view)
    value["stage"] = "training"
    assert suggestions(value) == []
    value["stage"] = "waiting_resources"
    value["resources"]["external_cpu_load"] = 0
    assert suggestions(value) == []


def test_unknown_values_are_na_and_process_commands_are_not_archived():
    view = display(count=0)
    value = {
        "stage": "preflight",
        "resources": {
            "processes": [
                {"pid": 42, "name": "[literal]", "cmdline": "secret --token=123"}
            ]
        },
    }
    view.configure_tuning(value)
    saved = view.snapshot()["tuning"]
    assert saved["resources"]["processes"] == [{"pid": 42, "name": "[literal]"}]
    assert "secret" not in json.dumps(saved)
    assert "N/A" in plain(view)


@pytest.mark.parametrize(
    "patch",
    [
        {"calibration": {"active_seconds": float("nan")}},
        {"resources": {"cpus_available": -1}},
        {"resources": {"memory_available": True}},
        {"resources": {"processes": [{"name": "App\n--argument"}]}},
        {"entries": [{"experiment_id": "one", "pending_resize": []}]},
        {"stage": "\x1b[31mtraining"},
        {"stage": None},
        {"calibration": {"candidates": True}},
    ],
)
def test_invalid_tuning_snapshot_is_rejected_before_monitoring(tmp_path, patch):
    view = display(count=1)
    view.bind(tmp_path)
    path = tmp_path / "logs/progress.json"
    snapshot = json.loads(path.read_text())
    snapshot["tuning"] = {**snapshot["tuning"], **patch}
    path.write_text(json.dumps(snapshot))
    with pytest.raises(ValueError, match="tuning"):
        read_snapshot(tmp_path)


def test_snapshot_round_trip_and_json_monitor_are_read_only_and_have_no_ansi(
    tmp_path, capsys
):
    view = display(count=1)
    view.bind(tmp_path)
    snapshot = read_snapshot(tmp_path)
    restored = RuntimeDisplay(kind="tune", readonly=True)
    restored.from_snapshot(snapshot)
    assert restored.snapshot() == snapshot
    before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    assert monitor(tmp_path, once=True, options=DisplayOptions(format="json")) == 0
    output = capsys.readouterr().err
    assert "\x1b" not in output and json.loads(output)["tuning"] == snapshot["tuning"]
    assert before == {
        path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()
    }
    old = deepcopy(snapshot)
    old.pop("tuning")
    restored.from_snapshot(old)
    assert restored.tuning is None and "tuning" not in restored.snapshot()


def test_alternate_screen_exit_retains_recommendation_and_actual_resources(monkeypatch):
    monkeypatch.setenv("TERM", "xterm-256color")
    stream = io.StringIO()
    stream.isatty = lambda: True
    view = RuntimeDisplay(
        kind="tune",
        console=Console(
            file=stream,
            force_terminal=True,
            force_interactive=True,
            width=120,
            height=35,
        ),
    )
    view.configure_tuning(summary("training", 1))
    view.start()
    assert "\x1b[?1049h" in stream.getvalue()
    assert "\x1b[2K" not in stream.getvalue()
    view.finish("completed")
    view.close()
    text = stream.getvalue()
    assert "\x1b[?1049l" in text
    assert "推荐设置" in text and "CPU 请求 8 / 实际 2" in text


def test_uncalibrated_fallback_is_labelled_in_dashboard():
    view = display(stage="calibration_incomplete", count=1)
    state = summary("calibration_incomplete", 1)
    state["calibration"]["calibrated"] = False
    view.configure_tuning(state)
    assert "起始设置（未经校准）" in plain(view)
    assert any("起始设置（未经校准）" in row for row in summary_lines(state))


def test_candidate_profile_uses_readable_compact_fields():
    view = display(width=115, stage="calibration", count=1)
    state = summary("calibration", 1)
    state["calibration"]["profile"] = {
        "threads": 1,
        "num_envs": 2,
        "sampling_processes": 2,
        "concurrency": 1,
        "device": "cpu",
    }
    view.configure_tuning(state)
    rendered = plain(view)
    assert "线程 1 · 环境 2 · 采样进程 2 · 实验并行 1 · cpu" in rendered


def test_tuning_display_import_does_not_import_optional_frameworks():
    command = "import sys; import smartsom.telemetry.tuning_dashboard; assert not any(name in sys.modules for name in ('ray', 'torch'))"
    result = subprocess.run(
        [sys.executable, "-c", command], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_summary_normalizes_only_a_detached_copy():
    value = summary()
    value["resources"]["processes"] = [{"name": "App", "full_command": "private"}]
    assert clean_summary(value)["resources"]["processes"] == [{"name": "App"}]
    assert value["resources"]["processes"][0]["full_command"] == "private"


def test_bad_tuning_restore_preserves_last_recorded_display():
    view = display(count=1)
    before = deepcopy(view.snapshot())
    bad = deepcopy(before)
    bad["name"] = "bad replacement"
    bad["tuning"]["resources"]["cpus_available"] = float("inf")
    with pytest.raises(ValueError, match="tuning"):
        view.from_snapshot(bad)
    assert view.snapshot() == before
