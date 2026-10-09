"""Runtime presentation contracts independent of learner or simulator internals."""

import io
import json
from types import SimpleNamespace

import pytest
from rich.console import Console

from smartsom.config.experiment import LoggingOptions, apply_overrides, load_preset
from smartsom.experiments.cli import _parser, main
from smartsom.telemetry.monitor import monitor, read_snapshot
from smartsom.telemetry.runtime import (
    CURRENT,
    DisplayOptions,
    RuntimeDisplay,
    bind,
    emit,
    operation,
    worker_output,
)


def display(tmp_path, **kwargs):
    stream = io.StringIO()
    result = RuntimeDisplay(
        DisplayOptions(**kwargs),
        console=Console(file=stream, force_terminal=False, width=80),
    )
    result.bind(tmp_path)
    return result, stream


def test_summary_cadence_does_not_throttle_snapshot_or_force_micro_stages(
    tmp_path, monkeypatch
):
    now = [0.0]
    monkeypatch.setattr("smartsom.telemetry.runtime.time.monotonic", lambda: now[0])
    view, stream = display(tmp_path)
    stream.seek(0)
    stream.truncate()
    for index in range(1, 30):
        now[0] = index
        view.update(
            "learner",
            {
                "stage": "sampling" if index % 2 else "learning_metrics",
                "sampled_steps": index,
            },
            total=100,
            unit="adapter decisions",
        )
    assert stream.getvalue() == ""
    saved = json.loads((tmp_path / "logs/progress.json").read_text())
    assert saved["tasks"][0]["completed"] == 29
    now[0] = 30
    view.update("learner", {"stage": "sampling", "sampled_steps": 30})
    assert stream.getvalue().count("sampled_steps=30") == 1
    view.phase("evaluation")
    assert "evaluation" in stream.getvalue()


def test_no_verbose_keeps_evidence_and_explicit_debug_is_independent(tmp_path):
    view, stream = display(
        tmp_path, verbose=False, debug=True, format="json", progress="on"
    )
    view.start()
    assert view.live is None
    view.update(
        "task",
        {
            "stage": "learning_metrics",
            "metrics": {"private_diagnostic": 1.234567890123},
        },
    )
    view.finish("completed")
    lines = [json.loads(line) for line in stream.getvalue().splitlines()]
    assert len(lines) == 1 and lines[0]["type"] == "debug"
    assert lines[0]["event"]["metrics"]["private_diagnostic"] == 1.234567890123
    assert "\x1b" not in stream.getvalue() and "\r" not in stream.getvalue()
    assert (tmp_path / "logs/runtime.log").exists()
    assert (
        json.loads((tmp_path / "logs/debug.jsonl").read_text())["metrics"][
            "private_diagnostic"
        ]
        == 1.234567890123
    )


def test_dynamic_roles_missing_counters_and_unknown_budget(tmp_path):
    view, _ = display(tmp_path)
    view.update(
        "task",
        {
            "stage": "learning_metrics",
            "metrics": {"quality/entropy_loss": -0.25, "buffer/approx_kl": None},
        },
    )
    snapshot = view.snapshot()["tasks"][0]
    assert "ppo_updates" not in snapshot["values"] and "total" not in snapshot
    assert set(snapshot["learner"]) == {"quality/entropy_loss", "buffer/approx_kl"}
    view.console.print(view.render())
    assert "machine_policy" not in view.text_summary()


def test_nested_training_evaluation_owns_one_session_and_one_final(
    tmp_path, monkeypatch
):
    started = []
    original = RuntimeDisplay.start
    monkeypatch.setattr(
        RuntimeDisplay, "start", lambda self: (started.append(self), original(self))
    )

    @operation("training")
    def training():
        bind(tmp_path)
        emit(
            "training",
            {"stage": "sampling", "sampled_steps": 100},
            total=100,
            unit="decisions",
        )
        assert CURRENT.get().status == "running"
        return SimpleNamespace(status="completed")

    @operation("evaluation")
    def evaluation():
        assert CURRENT.get() is started[0]
        return SimpleNamespace(status="completed")

    @operation("train-evaluate")
    def combined():
        training()
        assert CURRENT.get().status == "running"
        evaluation()
        return SimpleNamespace(status="completed")

    combined(display_options={"verbose": False})
    assert len(started) == 1 and CURRENT.get() is None
    assert read_snapshot(tmp_path)["status"] == "completed"
    assert (tmp_path / "logs/runtime.log").read_text().count(
        "train-evaluate: completed"
    ) <= 1


@pytest.mark.parametrize(
    "status",
    ["failed", "interrupted", "early_stopped", "pruned", "completed_with_failures"],
)
def test_final_statuses_are_not_success_by_sampling_percentage(tmp_path, status):
    view, _ = display(tmp_path)
    view.update(
        "task", {"stage": "saving", "sampled_steps": 100}, total=100, unit="decisions"
    )
    assert view.snapshot()["status"] == "running"
    view.finish(status)
    assert read_snapshot(tmp_path)["status"] == status


def test_returned_batch_failure_remains_visible_after_monitor_control_overlay(tmp_path):
    @operation("batch-directory")
    def failed_batch():
        bind(tmp_path)
        return {"status": "failed", "error": "worker implementation differs"}

    failed_batch(display_options={"verbose": False})
    assert (
        json.loads((tmp_path / "logs/progress.json").read_text())["notice"]
        == "worker implementation differs"
    )
    snapshot = read_snapshot(tmp_path)
    assert snapshot["status"] == "failed"
    assert snapshot["notice"].startswith("worker implementation differs")
    assert "Control: failed" in snapshot["notice"]


def test_worker_output_does_not_reach_parent_terminal(tmp_path, capsys):
    @worker_output("attempt")
    def worker(attempt):
        print("full diagnostic")

        @operation("training")
        def train():
            bind(attempt)
            emit("child", {"stage": "sampling", "sampled_steps": 5})

        train()

    worker(tmp_path)
    captured = capsys.readouterr()
    assert captured.out == captured.err == ""
    assert "full diagnostic" in (tmp_path / "worker.log").read_text()
    assert read_snapshot(tmp_path)["status"] == "completed"


def test_monitor_is_read_only_and_ignores_truncated_update(
    tmp_path, monkeypatch, capsys
):
    view, _ = display(tmp_path, verbose=False)
    view.update("task", {"stage": "sampling", "sampled_steps": 7})
    view.publish(force=True)
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    assert monitor(tmp_path, once=True, options=DisplayOptions(format="json")) == 0
    assert (
        json.loads(capsys.readouterr().err)["tasks"][0]["values"]["sampled_steps"] == 7
    )
    assert before == {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    calls = [0]

    def sleep(_):
        calls[0] += 1
        if calls[0] == 1:
            (tmp_path / "logs/progress.json").write_text("{")
        else:
            raise KeyboardInterrupt

    monkeypatch.setattr("smartsom.telemetry.monitor.time.sleep", sleep)
    assert monitor(tmp_path, options=DisplayOptions(every_seconds=0.00001)) == 0
    assert "last valid update" in capsys.readouterr().err


def test_monitor_rejects_experiments_and_reads_old_mainline_without_creating_files(
    tmp_path,
):
    path = tmp_path / "run.json"
    path.write_text(json.dumps({"kind": "smartsom.base-learning/v1"}))
    with pytest.raises(ValueError, match="unsupported"):
        monitor(tmp_path, once=True)
    path.write_text(
        json.dumps({"schema": "smartsom.production-run/v1", "status": "completed"})
    )
    assert monitor(tmp_path, once=True) == 0
    assert list(tmp_path.iterdir()) == [path]


def test_all_runtime_commands_accept_display_options():
    for command, positional in [
        ("run", []),
        ("train", []),
        ("train-evaluate", []),
        ("evaluate", ["run"]),
        ("resume", ["run"]),
        ("batch", ["study"]),
        ("batch-train", []),
        ("search", []),
        ("monitor", ["run"]),
    ]:
        args = _parser().parse_args(
            [
                command,
                *positional,
                "--no-debug",
                "--no-verbose",
                "--progress",
                "off",
                "--summary-interval",
                "12",
                "--log-format",
                "json",
            ]
        )
        assert args.debug is False and args.verbose == 0 and args.summary_interval == 12


def test_legacy_debug_alias_respects_explicit_override():
    assert LoggingOptions(verbose=2).debug
    assert not LoggingOptions(verbose=2, debug=False).debug
    config = apply_overrides(load_preset("test"), [("logging.verbose", 2)])
    assert config.logging.debug
    config = apply_overrides(
        load_preset("test"), [("logging.verbose", 2), ("logging.debug", False)]
    )
    assert not config.logging.debug


def test_json_legacy_warning_only_once_and_run_has_no_tick_events(tmp_path, capsys):
    assert (
        main(
            [
                "run",
                "--preset",
                "test",
                "--output-root",
                str(tmp_path),
                "--verbose",
                "2",
                "--no-debug",
                "--log-format",
                "json",
            ]
        )
        == 0
    )
    captured = capsys.readouterr()
    lines = [json.loads(line) for line in captured.err.splitlines()]
    assert sum(line.get("type") == "warning" for line in lines) == 1
    assert all("events" not in line for line in lines)
    assert "completed makespan=" in captured.out
    assert "\x1b" not in captured.err


@pytest.mark.parametrize("width,height", [(52, 24), (100, 35), (40, 18)])
def test_narrow_many_task_layout_keeps_bottom_progress_visible(width, height):
    console = Console(file=io.StringIO(), width=width, height=height)
    view = RuntimeDisplay(DisplayOptions(verbose=False), console=console)
    view.total_tasks = 20
    for index in range(20):
        view.update(
            f"trial-{index:02}",
            {"stage": "sampling", "sampled_steps": index},
            total=100,
            unit="adapter decisions",
        )
    lines = console.render_lines(
        view.render(), console.options.update(height=None), pad=False
    )
    assert len(lines) <= height
    text = "\n".join("".join(segment.text for segment in line) for line in lines)
    assert "Finished tasks" in text
    assert "other active tasks" in text


def test_backend_debug_is_json_and_file_is_plain(tmp_path, capsys):
    from smartsom.telemetry.runtime import backend_diagnostics

    view = RuntimeDisplay(DisplayOptions(debug=True, format="json", verbose=False))
    view.bind(tmp_path)
    token = CURRENT.set(view)
    try:
        with backend_diagnostics():
            import sys

            assert sys.stdout.encoding.lower().replace("-", "") == "utf8"
            assert sys.stdout.writable()
            print("\x1b[33mframework warning\x1b[0m")
    finally:
        CURRENT.reset(token)
    line = json.loads(capsys.readouterr().err)
    assert line == {"type": "backend", "message": "framework warning"}
    assert (tmp_path / "logs/backend.log").read_text() == "framework warning\n"


def test_backend_diagnostics_unicode_with_non_utf8_default(tmp_path, monkeypatch):
    from pathlib import Path

    from smartsom.telemetry.runtime import backend_diagnostics

    original_open = Path.open

    def locale_open(path, *args, **kwargs):
        if path.name == "backend.log":
            kwargs.setdefault("encoding", "cp1252")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", locale_open)
    view = RuntimeDisplay(DisplayOptions(verbose=False))
    view.bind(tmp_path)
    with backend_diagnostics(view):
        print("工厂诊断")
    assert (tmp_path / "logs/backend.log").read_text(encoding="utf-8") == "工厂诊断\n"


def test_new_replication_does_not_reuse_previous_seed_metrics(tmp_path):
    view, _ = display(tmp_path, verbose=False)
    view.update(
        "trial",
        {
            "stage": "learning_metrics",
            "display_run": "seed1",
            "sampled_steps": 100,
            "ppo_updates": 4,
            "metrics": {"quality/entropy": 0.25},
        },
    )
    view.update(
        "trial", {"stage": "sampling", "display_run": "seed2", "sampled_steps": 1}
    )
    row = view.snapshot()["tasks"][0]
    assert row["completed"] == 1
    assert "ppo_updates" not in row["values"] and row["learner"] == {}


def test_known_target_without_sample_count_does_not_show_zero_percent(tmp_path):
    view, _ = display(tmp_path, verbose=False)
    view.update("task", {"stage": "preparing"}, total=100, unit="adapter decisions")
    stream = io.StringIO()
    Console(file=stream, width=80, height=30).print(view.render())
    assert "N/A/100" in stream.getvalue()
    assert "0%" not in stream.getvalue()


@pytest.mark.parametrize("status", ["failed", "interrupted", "completed_with_failures"])
def test_failures_still_visible_with_no_verbose(tmp_path, status):
    view, stream = display(tmp_path, verbose=False, format="json")
    view.update("worker", {"stage": status, "status": status}, final=True)
    lines = [json.loads(line) for line in stream.getvalue().splitlines()]
    assert len(lines) == 1 and lines[0]["tasks"][0]["status"] == status
    view.finish(status)
    assert json.loads(stream.getvalue().splitlines()[-1])["status"] == status


def test_monitor_rejects_partial_structure_before_replacing_good_snapshot(tmp_path):
    view, _ = display(tmp_path, verbose=False)
    view.update("worker", {"stage": "sampling", "sampled_steps": 7})
    view.publish(force=True)
    good = read_snapshot(tmp_path)
    for bad in ({**good, "tasks": [None]}, {**good, "tasks": [{"id": "x"}]}):
        (tmp_path / "logs/progress.json").write_text(json.dumps(bad))
        with pytest.raises(ValueError, match="invalid"):
            read_snapshot(tmp_path)


def test_cli_json_errors_stay_machine_readable(tmp_path, capsys):
    assert main(["monitor", str(tmp_path), "--once", "--log-format", "json"]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert json.loads(captured.err)["type"] == "error"


def test_long_task_name_does_not_hide_progress_unit(tmp_path):
    view, stream = display(tmp_path)
    view.update(
        "long-run-name-" * 10,
        {"stage": "simulation", "tick": 7},
        total=100,
        unit="physical ticks",
    )
    view.console.width = 52
    view.console.print(view.render())
    assert "ticks ·" in stream.getvalue()


@pytest.mark.parametrize(
    "status",
    [
        "completed",
        "ineligible",
        "failed",
        "pruned",
        "interrupted",
        "early_stopped",
        "truncated",
    ],
)
def test_finished_tasks_are_separate_from_successes(tmp_path, status):
    view, stream = display(tmp_path)
    view.total_tasks = 2
    for task in ("first", "second"):
        view.update(task, {"stage": status, "status": status}, final=True)
    view.finish("completed" if status == "completed" else "finished_with_failures")
    assert view.task_counts() == (2, 2 if status == "completed" else 0)
    view.console.print(view.render())
    assert "Tasks ended: 2/2" in stream.getvalue()
    assert "100%" in stream.getvalue()
    assert (
        "successful: 2" in stream.getvalue()
        if status == "completed"
        else "successful: 0" in stream.getvalue()
    )


def test_interrupt_does_not_count_unstarted_tasks_as_finished(tmp_path):
    view, _ = display(tmp_path, verbose=False)
    view.total_tasks = 4
    for task, status in (
        ("done", "completed"),
        ("active", "running"),
        ("later", "pending"),
        ("queued", "queued"),
    ):
        view.update(task, {"stage": status, "status": status}, final=True)
    view.finish("interrupted")
    assert view.task_counts() == (2, 1)
    assert view.tasks["later"]["status"] == "pending"
    assert view.tasks["queued"]["status"] == "queued"
    assert read_snapshot(tmp_path)["tasks"] == list(view.tasks.values())


def test_retry_reopens_a_finished_task_without_double_counting(tmp_path):
    view, _ = display(tmp_path, verbose=False)
    view.total_tasks = 1
    view.update("trial", {"stage": "failed", "status": "failed"}, final=True)
    assert view.task_counts() == (1, 0)
    view.update("trial", {"stage": "sampling", "sampled_steps": 3})
    assert view.task_counts() == (0, 0)
    view.update("trial", {"stage": "completed", "status": "completed"}, final=True)
    assert view.task_counts() == (1, 1)


@pytest.mark.parametrize(
    "bad_field",
    [
        {"total": "unknown"},
        {"completed": -1},
        {"total": float("nan")},
        {"learner": []},
        {"completed": True},
        {"started_at": -1},
    ],
)
def test_monitor_rejects_unrenderable_progress_before_accepting_it(tmp_path, bad_field):
    view, _ = display(tmp_path, verbose=False)
    view.update("task", {"stage": "sampling", "sampled_steps": 5})
    snapshot = view.snapshot()
    assert snapshot["tasks"][0]["started_at"] > 0
    snapshot["tasks"][0].update(bad_field)
    (tmp_path / "logs/progress.json").write_text(json.dumps(snapshot))
    with pytest.raises(ValueError, match="invalid"):
        read_snapshot(tmp_path)


@pytest.mark.parametrize(
    "bad", [[], None, {"updated_at": float("nan")}, {"total_tasks": "two"}]
)
def test_monitor_rejects_invalid_snapshot_envelope(tmp_path, bad):
    view, _ = display(tmp_path, verbose=False)
    snapshot = {**view.snapshot(), **bad} if isinstance(bad, dict) else bad
    (tmp_path / "logs/progress.json").write_text(json.dumps(snapshot))
    with pytest.raises(ValueError, match="invalid"):
        read_snapshot(tmp_path)


@pytest.mark.parametrize("width", [32, 40, 52, 60, 80])
def test_compact_cards_preserve_status_and_budget_with_long_names(width):
    console = Console(file=io.StringIO(), width=width, height=18)
    view = RuntimeDisplay(DisplayOptions(verbose=False), console=console)
    view.total_tasks = 2
    for task in ("first-long-task-" * 8, "second-long-task-" * 8):
        view.update(
            task,
            {
                "stage": "completed_with_failures",
                "status": "completed_with_failures",
                "sampled_steps": 1024,
            },
            total=1024,
            unit="sampling decisions",
            final=True,
        )
    lines = console.render_lines(
        view.render(), console.options.update(height=None), pad=False
    )
    text = "\n".join("".join(s.text for s in line) for line in lines)
    assert len(lines) <= 18
    assert text.count("completed_with_failures") == 2
    assert text.count("sample dec: 1,024/1,024") == 2


def test_dashboard_hides_waiting_rows_and_limits_recent_results(monkeypatch):
    monkeypatch.setattr("smartsom.telemetry.runtime.time.time", lambda: 100.0)
    stream = io.StringIO()
    view = RuntimeDisplay(
        DisplayOptions(verbose=False),
        console=Console(file=stream, width=100, height=40),
    )
    view.total_tasks = 12
    for index in range(6):
        view.update(
            f"ended-{index}",
            {"status": "completed", "stage": "completed", "sampled_steps": 100},
            total=100,
            unit="sampling decisions",
            final=True,
        )
    for index in range(4):
        view.update(
            f"waiting-{index}", {"status": "pending", "stage": "pending"}, final=True
        )
    for index in range(2):
        view.update(
            f"active-{index}",
            {"stage": "sampling", "sampled_steps": 50},
            total=100,
            unit="sampling decisions",
        )
    view.console.print(view.render())
    text = stream.getvalue()
    assert "Active 2 · waiting 4" in text
    assert "waiting-" not in text and "ended-0" not in text
    assert all(f"ended-{i}" in text for i in (3, 4, 5))
    assert "3 older results" in text
    assert text.count("50%") == 3  # Two live tasks and the overall ended count.


def test_truncated_episode_is_a_result_not_a_stalled_progress_bar(tmp_path):
    view, stream = display(tmp_path, verbose=False)
    view.update(
        "episode",
        {
            "status": "truncated",
            "stage": "truncated",
            "tick": 20,
            "reason": "Decision budget exhausted",
            "display_values": {"decisions": 128, "decision_limit": 128},
        },
        total=1000,
        unit="physical ticks",
        final=True,
    )
    view.console.print(view.render())
    text = stream.getvalue()
    assert "Decision budget exhausted" in text
    assert "decisions 128/128" in text and "20/1,000" in text
    assert "2%" not in text


def test_saving_target_is_not_a_completed_task():
    stream = io.StringIO()
    view = RuntimeDisplay(
        DisplayOptions(verbose=False),
        console=Console(file=stream, width=100, height=30),
    )
    view.total_tasks = 2
    view.update(
        "saving-model",
        {"stage": "saving", "sampled_steps": 100},
        total=100,
        unit="sampling decisions",
    )
    view.console.print(view.render())
    text = stream.getvalue()
    assert "Sampling target reached; task still running" in text
    assert "Tasks ended: 0/2" in text and "100%" not in text


def test_evaluation_summary_does_not_mix_training_success_with_episode_success():
    stream = io.StringIO()
    view = RuntimeDisplay(
        DisplayOptions(verbose=False),
        console=Console(file=stream, width=100, height=35),
    )
    view.update(
        "model",
        {"status": "completed", "stage": "completed", "sampled_steps": 100},
        total=100,
        unit="environment steps",
        final=True,
    )
    view.update(
        "evaluation",
        {
            "status": "completed_with_failures",
            "stage": "completed_with_failures",
            "display_values": {
                "evaluation_finished": 5,
                "evaluation_requested": 5,
                "evaluation_completed": 0,
            },
        },
        final=True,
    )
    view.console.print(view.render())
    text = stream.getvalue()
    assert "Training: completed" in text
    assert "Evaluation successes 0/5" in text
    assert "Successful 1" not in text


def test_v3_ticks_dqn_and_validation_remain_distinct(tmp_path):
    view, stream = display(tmp_path, verbose=False)
    view.console = Console(file=stream, width=160, height=50)
    view.update(
        "training",
        {
            "stage": "sampling",
            "physical_ticks": 1024,
            "algorithm": "DQN",
            "optimization_steps": 72,
        },
        total=16384,
        unit="physical ticks",
    )
    view.update(
        "training",
        {
            "stage": "validation",
            "validation_finished": 2,
            "validation_requested": 5,
            "validation_tick": 100,
            "validation_tick_limit": 4096,
        },
    )
    row = view.snapshot()["tasks"][0]
    assert row["completed"] == 1024 and row["total"] == 16384
    assert row["values"]["validation_finished"] == 2
    assert row["values"]["optimization_steps"] == 72
    view.console.print(view.render())
    assert "1,024/16,384" in stream.getvalue()
    assert "PPO updates" not in stream.getvalue()


def test_parallel_queue_and_final_states_do_not_count_as_active(tmp_path):
    view, stream = display(tmp_path, verbose=False)
    view.console = Console(file=stream, width=180, height=65)
    view.total_tasks = 24
    for index in range(8):
        view.update(
            str(index),
            {
                "stage": "validation",
                "physical_ticks": 1024,
                "algorithm": "DQN",
                "optimization_steps": 10,
                "validation_finished": 2,
                "validation_requested": 5,
                "context": "H=h0 V=low travel=auto envs=1 sampling=sequential",
            },
            total=16384,
            unit="physical ticks",
        )
    for index in range(8, 24):
        view.update(
            str(index),
            {"stage": "queued", "status": "queued"},
            total=16384,
            unit="physical ticks",
        )
    view.console.print(view.render())
    text = stream.getvalue()
    assert "Active 8" in text and "waiting 16" in text
    assert "validation 2/5" in text and "DQN optimizer 10" in text
    assert "H=h0 V=low" in text
    view.update("0", {"stage": "saving", "physical_ticks": 16384})
    assert view.task_counts() == (0, 0)
    view.update("0", {"stage": "completed", "status": "completed"}, final=True)
    assert view.task_counts() == (1, 1)


def test_study_poll_batches_all_workers_and_skips_unchanged_frames(
    tmp_path, monkeypatch
):
    now = [0.0]
    monkeypatch.setattr("smartsom.telemetry.runtime.time.monotonic", lambda: now[0])
    view, _ = display(tmp_path, verbose=False)
    view.kind = "study"
    frames = []
    view.live = SimpleNamespace(
        update=lambda *args, **kwargs: frames.append(view.snapshot())
    )
    with view.batch_updates():
        for index in range(8):
            view.update(
                str(index),
                {"stage": "sampling", "physical_ticks": 256},
                total=16384,
                unit="physical ticks",
            )
            assert not frames
    assert len(frames) == 1 and len(frames[0]["tasks"]) == 8
    saved = read_snapshot(tmp_path)
    assert len(saved["tasks"]) == 8
    now[0] = 2.0
    with view.batch_updates():
        for index in range(8):
            view.update(str(index), {"stage": "sampling", "physical_ticks": 256})
    assert len(frames) == 1
    now[0] = 2.1
    view.update("0", {"stage": "sampling", "physical_ticks": 257})
    assert len(frames) == 2
    now[0] = 2.2
    view.update("1", {"stage": "sampling", "physical_ticks": 258})
    assert len(frames) == 2
    now[0] = 3.2
    view.publish()
    assert len(frames) == 3


def test_study_batch_preserves_forced_failure_snapshot(tmp_path):
    view, _ = display(tmp_path, verbose=False)
    with view.batch_updates():
        view.update(
            "worker",
            {"stage": "sampling", "physical_ticks": 10},
            total=100,
            unit="physical ticks",
        )
        view.update(
            "worker",
            {"stage": "failed", "status": "failed", "reason": "worker stopped"},
            final=True,
        )
    assert read_snapshot(tmp_path)["tasks"][0]["status"] == "failed"
    assert "worker stopped" in (tmp_path / "logs/runtime.log").read_text()


@pytest.mark.parametrize("width,height", [(52, 24), (100, 35), (180, 65)])
def test_study_layout_has_fixed_height_across_phases(width, height):
    console = Console(file=io.StringIO(), width=width, height=height)
    view = RuntimeDisplay(DisplayOptions(verbose=False), console=console, kind="study")
    view.total_tasks = 24
    sizes = []
    for stage in ("starting", "sampling", "validation", "saving", "completed"):
        with view.batch_updates():
            for index in range(8):
                view.update(
                    str(index),
                    {
                        "stage": stage,
                        "status": stage if stage == "completed" else "running",
                        "physical_ticks": 1024,
                        "validation_finished": 2,
                        "validation_requested": 5,
                        "validation_tick": 100,
                        "validation_tick_limit": 4096,
                        "context": "PPO | H=h0 V=low travel=auto | envs=1 sampling=sequential validation=sequential",
                        "group_statistics": "machine=20/20/4 buffer=10/10/4 dispatcher=30/30/4",
                    },
                    total=16384,
                    unit="physical ticks",
                    final=stage == "completed",
                )
        lines = console.render_lines(
            view.render(), console.options.update(height=None), pad=False
        )
        sizes.append(len(lines))
        assert len(lines) <= height
        text = "\n".join("".join(segment.text for segment in row) for row in lines)
        if width >= 100 and stage == "validation":
            assert "Training ticks (%)" in text
            assert "1,024/16,384" in text
            assert "Validation" in text
            assert "2/5; 100/4,096" in text
    assert len(set(sizes)) == 1


def test_study_terminal_overwrites_without_erasing_lines(monkeypatch):
    monkeypatch.setenv("TERM", "xterm-256color")
    stream = io.StringIO()
    stream.isatty = lambda: True
    console = Console(
        file=stream,
        force_terminal=True,
        force_interactive=True,
        width=100,
        height=35,
        legacy_windows=False,
    )
    view = RuntimeDisplay(console=console, kind="study")
    view.start()
    assert "\x1b[?1049h" in stream.getvalue()
    stream.seek(0)
    stream.truncate()
    with view.batch_updates():
        for index in range(8):
            view.update(
                str(index),
                {"stage": "sampling", "physical_ticks": 256},
                total=16384,
                unit="physical ticks",
            )
    first = stream.getvalue()
    assert "\x1b[H" in first
    assert "\x1b[2K" not in first and "\x1b[2J" not in first
    # Eight worker updates produced one terminal frame.
    assert first.count("\x1b[H") == 1
    view.finish("completed")
    view.close()
    assert "\x1b[?1049l" in stream.getvalue()
    assert "study: completed [completed]" in stream.getvalue()


def test_study_dashboard_labels_columns_and_overall_eta():
    console = Console(file=io.StringIO(), width=180, height=40)
    view = RuntimeDisplay(DisplayOptions(verbose=False), console=console, kind="study")
    view.total_tasks = 24
    view.overview = {
        "work_completed": 100,
        "work_total": 1000,
        "elapsed_seconds": 30,
        "eta_seconds": 270,
    }
    view.update(
        "h0_low_dqn_auto",
        {
            "stage": "validation",
            "physical_ticks": 1024,
            "validation_finished": 1,
            "validation_requested": 5,
            "validation_tick": 100,
            "validation_tick_limit": 4096,
            "study_case": {
                "algorithm": "DQN",
                "H": "h0",
                "V": "low",
                "travel": "auto",
                "last_activity": 10,
            },
        },
        total=16384,
        unit="physical ticks",
    )
    lines = console.render_lines(
        view.render(), console.options.update(height=None), pad=False
    )
    text = "\n".join("".join(s.text for s in line) for line in lines)
    assert "Overall workflow (budget weighted)" in text and "10.0%" in text
    assert "ETA ~ 4m 30s" in text and "Elapsed 0m 30s" in text
    assert all(
        label in text
        for label in (
            "Algorithm",
            "H / V",
            "Travel",
            "Phase",
            "Training ticks (%)",
            "Cases ended; current tick",
        )
    )
    assert "DQN" in text and "h0 / low" in text and "1/5; 100/4,096" in text
    assert "Groups decisions/samples/optimizer" not in text and "pid=" not in text


def test_monitor_rejects_nonfinite_overall_eta(tmp_path):
    view, _ = display(tmp_path, verbose=False)
    snapshot = view.snapshot()
    snapshot["overview"] = {"eta_seconds": float("inf")}
    (tmp_path / "logs/progress.json").write_text(json.dumps(snapshot))
    with pytest.raises(ValueError, match="invalid runtime overview"):
        read_snapshot(tmp_path)
