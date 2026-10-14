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
    view.paging = True  # exercise an interactive viewport
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


@pytest.mark.parametrize("width,height", [(80, 24), (124, 50), (200, 60)])
@pytest.mark.parametrize("mode", ["fixed", "online"])
def test_live_resource_notes_do_not_move_trial_cards_or_change_page_bounds(
    width, height, mode
):
    view = display(width, height, count=4)
    state = summary("running", 4)
    state.update(manual=True, concurrency_limit=4, active=4)
    state["calibration"] = {"level": "off", "calibrated": False}
    for e in state["entries"]:
        e["pending_resize"] = False
        e.pop("resource_change_reason")
    if mode == "online":
        state.pop("manual")
        state["calibration"]["level"] = "online"
        state["online"] = {"g": {"active": 4, "limit": 4, "ceiling": 4}}
        for e in state["entries"]:
            e["calibration_group"] = "g"
    view.configure_tuning(state)

    def geometry():
        text = plain(view).splitlines()
        return (
            [i for i, line in enumerate(text) if "trial-0" in line],
            list(view.task_pages.bounds),
        )

    before = geometry()
    assert before[0]
    fluctuating = deepcopy(state)
    fluctuating["stage"] = "waiting_resources"
    fluctuating["resources"]["processes"] = [
        {"pid": 999, "name": "Other editor", "cpu_cores": 1.5, "protected": False}
    ]
    fluctuating["throughput"] = {"reason": "warming execution window"}
    fluctuating["resources"]["owned_cpu_reason"] = (
        "owned process first CPU sample unavailable"
    )
    fluctuating["entries"][0].update(
        pending_resize=True, resource_change_reason="waiting for CPU sample"
    )
    view.configure_tuning(fluctuating)
    assert geometry() == before
    view.configure_tuning(state)
    assert geometry() == before


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
        assert "第 " in text and "下一页" in text
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


def test_online_feedback_is_not_presented_as_completed_calibration():
    from smartsom.telemetry.timeline import detail

    view = display(count=1)
    current = summary("training", 1)
    current["calibration"]["level"] = "online"
    current["online"] = {
        "group": {
            "limit": 2,
            "ceiling": 8,
            "best_concurrency": 1,
            "reason": "testing one more experiment",
        }
    }
    view.configure_tuning(current)
    text = plain(view)
    assert "性能 · 在线调整" in text and "在线并发上限 2/8" in text
    assert "性能评估" not in text and "候选测量" not in text
    assert "起始设置（未经校准）" not in text
    assert (
        text.index("当前阶段") < text.index("性能 · 在线调整") < text.index("实验卡片")
    )
    assert "真实更新" in detail(view, "calibration")


@pytest.mark.parametrize(
    "width,height", [(60, 12), (80, 24), (100, 35), (160, 36), (200, 50)]
)
@pytest.mark.parametrize(
    "stage", ["running", "waiting_resources", "stage_20", "completed", "failed"]
)
def test_online_execution_card_fits_and_uses_live_group_facts(width, height, stage):
    view = display(width, height, count=12)
    if stage == "stage_20":
        view.kind = "batch-directory"
    state = summary(stage, 12)
    state["calibration"] = {"level": "online", "calibrated": False}
    state["calibration_status"] = "online"
    state["batch_training_active"] = True
    state["online"] = {
        "inactive": {
            "active": 0,
            "limit": 1,
            "ceiling": 5,
            "reason": "warming up one real experiment",
        },
        "active": {
            "active": 2,
            "limit": 3,
            "ceiling": 8,
            "best_concurrency": 2,
            "latest_throughput": 180.5,
            "measured_concurrency": 2,
            "reason": "stable useful throughput; testing one more experiment",
        },
    }
    view.configure_tuning(state)
    if stage in {"completed", "failed"}:
        view.status = stage
    lines = view.console.render_lines(
        view.render(), view.console.options.update(height=None), pad=False
    )
    assert len(lines) == height - 1
    assert all(sum(segment.cell_length for segment in row) <= width for row in lines)
    text = plain(view)
    assert "性能评估" not in text and "候选测量" not in text and "0/0s" not in text
    assert "性能 · 在线调整" in text
    assert "180.5 ticks/s" in text
    if height > 12:
        assert "在线并发上限 3/8" in text
        assert "实际运行 2" in text and "实验卡片" in text
        assert "（2 个实验合计）" in text
        assert "trial-0" in text


def test_online_text_summary_omits_probe_budget_and_preserves_all_groups():
    state = summary("waiting_resources", 1)
    state["calibration"] = {"level": "online"}
    state["online"] = {
        str(i): {"active": 0, "limit": 1, "ceiling": 2} for i in range(4)
    }
    rows = summary_lines(state)
    assert not any("性能评估" in row or "已测候选" in row for row in rows)
    assert sum("在线并发上限" in row for row in rows) == 4


@pytest.mark.parametrize("field", ["active", "limit", "latest_throughput"])
def test_online_negative_counts_are_rejected(field):
    with pytest.raises(ValueError, match="tuning count"):
        clean_summary({"online": {"group": {field: -1}}})


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
    assert "当前验证" in text and "下一页" in text


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
    assert "ETA 采集中" in text
    assert "当前采样" in text and "CPU 实际/请求 2/8" in text


def test_compact_failed_card_shows_worker_reason():
    view = display(width=80, height=24, stage="training", count=1)
    view.update(
        "trial-0",
        {
            "status": "failed",
            "stage": "failed",
            "reason": "worker implementation differs",
        },
        final=True,
    )
    assert "worker implementation differs" in plain(view)


def test_each_experiment_eta_uses_measured_phase_rate_and_ignores_display_elapsed(
    monkeypatch,
):
    from smartsom.telemetry.estimates import remaining_service

    view = display(width=160, height=50, count=2)
    first, second = view.tasks["trial-0"], view.tasks["trial-1"]
    first["started_at"], second["started_at"] = 100, 199
    plan = {"mode": "training", "training_total": 1000, "round_size": 1000}
    estimates = {}
    for task, completed, rate in [(first, 500, 1), (second, 750, 2)]:
        task["completed"] = task["values"]["physical_ticks"] = completed
        estimates[task["id"]] = remaining_service(
            task,
            plan,
            {"phases": {"training": {"rate": rate}, "saving": {"mean_seconds": 1}}},
            reference={"finalization": {"mean_seconds": 2}},
        )
    view.execution = {"tasks": estimates}
    monkeypatch.setattr("smartsom.telemetry.dashboard.time.time", lambda: 200)
    assert task_eta(view, first) == "≈9分钟"
    assert task_eta(view, second) == "≈3分钟"
    monkeypatch.setattr("smartsom.telemetry.dashboard.time.time", lambda: 999999)
    assert task_eta(view, first) == "≈9分钟"
    text = plain(view)
    assert "ETA ≈9分钟" in text and "ETA ≈3分钟" in text


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


@pytest.mark.parametrize("width,height", [(80, 24), (160, 50)])
def test_runtime_control_transitions_preserve_trial_geometry(width, height):
    view = display(width, height, count=12)
    view.controlling = True
    state = summary("training", 12)
    geometry = []
    for active, target, status, reason in [
        (4, 4, "running", None),
        (6, 2, "draining", "memory unavailable; waiting for reserve"),
        (0, 2, "paused", None),
    ]:
        state["runtime_control"] = dict(
            active=active,
            limit=target,
            status=status,
            blocking_reason=reason,
            mode="manual",
        )
        view.configure_tuning(state)
        rows = view.console.render_lines(
            view.render(), view.console.options.update(height=None), pad=False
        )
        assert len(rows) == height - 1
        text = ["".join(segment.text for segment in row) for row in rows]
        geometry.append(
            (
                [i for i, row in enumerate(text) if "trial-" in row],
                list(view.task_pages.bounds),
            )
        )
        assert any("目标" in row and "实际" in row and "manual" in row for row in text)
    assert geometry[0] == geometry[1] == geometry[2]


def test_noninteractive_runtime_summary_includes_control_and_blocking_facts():
    rows = summary_lines(
        {
            "runtime_control": {
                "mode": "adaptive",
                "limit": 6,
                "active": 2,
                "status": "running",
                "blocking_reason": "insufficient memory",
            }
        }
    )
    text = "\n".join(rows)
    assert "自适应上限 6" in text
    assert "实际运行 2" in text
    assert "insufficient memory" in text
    assert "只读 monitor" in text


def test_runtime_remove_waits_for_tombstone_and_details_preserve_geometry():
    view = display(160, 50, count=2)
    state = summary("training", 2)
    state["runtime_control"] = dict(
        limit=2,
        active=2,
        mode="manual",
        status="running",
        tasks={"trial-0": {"status": "removing"}, "trial-1": {"status": "running"}},
        blocking_reason="internal detail",
    )
    view.configure_tuning(state)
    before = plain(view)
    assert "trial-0" in before and "removing" in before
    assert "internal detail" in before
    before_bounds = list(view.task_pages.bounds)
    before_lines = [i for i, row in enumerate(before.splitlines()) if "trial-" in row]
    view.runtime_details = True
    after = plain(view)
    assert "internal detail" in after
    assert before_lines == [
        i for i, row in enumerate(after.splitlines()) if "trial-" in row
    ]
    assert before_bounds == view.task_pages.bounds
    state["runtime_control"]["tasks"]["trial-0"]["status"] = "removed"
    view.configure_tuning(state)
    removed = plain(view)
    assert "trial-0" not in removed
    assert "已移除 1" in removed


def test_runtime_hides_file_aggregates_but_keeps_running_and_removing_children():
    view = display(160, 50, count=0)
    state = summary("training", 0)
    state["runtime_control"] = dict(
        limit=2,
        active=2,
        mode="manual",
        status="running",
        tasks={
            "source-file": {"status": "running"},
            "source-file__entry-1": {"status": "running", "file_id": "source-file"},
            "source-file__entry-2": {"status": "removing", "file_id": "source-file"},
            "source-file__entry-3": {"status": "removed", "file_id": "source-file"},
        },
    )
    view.configure_tuning(state)
    text = plain(view)
    assert "source-file" not in view.task_pages.ids
    assert "完成 0/3" in text
    assert "source-file · running" not in text
    assert "source-file__entry-1" in text
    assert "source-file__entry-2" in text
    assert "source-file__entry-3" not in text
    assert "已移除 1" in text


@pytest.mark.parametrize("self_file_ids", [False, True])
def test_runtime_independent_ids_with_double_underscore_remain_selectable(
    self_file_ids,
):
    view = display(160, 50, count=0)
    tasks = {"foo": {"status": "running"}, "foo__bar": {"status": "queued"}}
    if self_file_ids:
        for key, task in tasks.items():
            task["file_id"] = key
    view.configure_tuning(
        {
            "runtime_control": {
                "limit": 2,
                "active": 1,
                "mode": "manual",
                "status": "running",
                "tasks": tasks,
            }
        }
    )
    text = plain(view)
    assert view.task_pages.ids == ["foo", "foo__bar"]
    assert "完成 0/2" in text and "运行 1" in text and "排队 1" in text
    view.task_pages.anchor = "foo"
    assert "▶ foo" in plain(view)
    view.task_pages.anchor = "foo__bar"
    assert "▶ foo__bar" in plain(view)


@pytest.mark.parametrize(
    "stage,kind,phase,prefix",
    [
        ("validation", None, "validation", "validation"),
        ("validating", None, "validation", "validation"),
        ("evaluation", None, "evaluation", "evaluation"),
        ("evaluating", None, "evaluation", "evaluation"),
        ("test", None, "evaluation", "evaluation"),
        ("evaluation", "initial", "initial", "evaluation"),
        ("evaluating", "rule-control", "rule", "evaluation"),
        ("test", "random-control", "random", "evaluation"),
    ],
)
def test_runtime_rate_and_progress_use_actual_evaluation_phase(
    stage, kind, phase, prefix
):
    from smartsom.telemetry.tuning_dashboard import _runtime_phase_facts

    view = display(80, 24, count=1)
    view.configure_tuning(
        {
            "runtime_control": {
                "limit": 1,
                "active": 1,
                "mode": "manual",
                "status": "running",
            }
        }
    )
    row = view.tasks["trial-0"]
    row.update(
        stage=stage,
        completed=900,
        total=1000,
        execution={"phases": {"training": {"rate": 999}, phase: {"rate": 3}}},
    )
    row["values"].update(
        evaluation_kind=kind,
        ticks_per_second=888,
        **{prefix + "_finished": 1, prefix + "_requested": 4},
    )
    _, done, total, rate, unit = _runtime_phase_facts(row)
    assert (done, total, rate, unit) == (1, 4, 3, "ticks/s")
    rendered = plain(view)
    assert "速度 3 ticks/s" in rendered and "25%" in rendered
    assert "速度 999" not in rendered and "速度 888" not in rendered
    row["execution"]["phases"].pop(phase)
    assert "速度 未记录" in plain(view)


@pytest.mark.parametrize(
    "stage",
    [
        "updating",
        "optimizing",
        "learning_metrics",
        "saving",
        "checkpointing",
        "checkpoint",
        "exporting",
        "initializing",
        "restoring",
        "loading",
        "restored",
        "compiling",
        "finalizing",
        "closing",
        "cleanup",
        "unknown",
    ],
)
def test_runtime_non_sampling_stages_do_not_borrow_training_fraction_or_rate(stage):
    from smartsom.telemetry.tuning_dashboard import _runtime_phase_facts

    view = display(80, 24, count=1)
    view.configure_tuning(
        {
            "runtime_control": {
                "limit": 1,
                "active": 1,
                "mode": "manual",
                "status": "running",
            }
        }
    )
    row = view.tasks["trial-0"]
    row.update(
        stage=stage,
        completed=900,
        total=1000,
        execution={"phases": {"training": {"rate": 999}}},
    )
    row["values"]["ticks_per_second"] = 888
    assert _runtime_phase_facts(row)[1:4] == (None, None, None)
    rendered = plain(view)
    assert "进度未记录" in rendered and "速度 未记录" in rendered
    assert "90%" not in rendered and "速度 999" not in rendered


@pytest.mark.parametrize("stage", ["training", "running", "sampling"])
@pytest.mark.parametrize(
    "unit,speed_unit",
    [
        ("physical ticks", "ticks/s"),
        ("environment steps", "steps/s"),
        ("adapter decisions", "decisions/s"),
        ("sampling decisions", "decisions/s"),
    ],
)
def test_runtime_training_counters_are_cumulative_and_rate_units_are_explicit(
    stage, unit, speed_unit
):
    from smartsom.telemetry.tuning_dashboard import _runtime_phase_facts

    row = {
        "stage": stage,
        "status": "running",
        "unit": unit,
        "completed": 900,
        "total": 1000,
        "execution": {"phases": {"training": {"rate": 999}}},
    }
    label, done, total, rate, recorded_unit = _runtime_phase_facts(row)
    assert "训练累计" in label
    assert (done, total, recorded_unit) == (900, 1000, speed_unit)
    assert rate == (None if stage == "sampling" else 999)
    row["status"] = "queued"
    assert _runtime_phase_facts(row)[1:4] == (None, None, None)
    row["status"] = "completed"
    assert _runtime_phase_facts(row)[3] is None
