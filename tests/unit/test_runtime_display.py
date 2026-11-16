"""Runtime presentation contracts independent of learner or simulator internals."""

import io
import json
from types import SimpleNamespace

import pytest
from rich.console import Console

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


def test_frozen_case_streams_reach_runtime_facts_and_clear_on_training(tmp_path):
    from smartsom.telemetry.dashboard import facts

    view, _ = display(tmp_path, verbose=False, format="json")
    workflow = dict(mode="train-evaluate", algorithm="DQN", training_seed=101)
    streams = dict(
        seed_semantics="data_seed",
        data_seed=7617270868981535538,
        environment_seed=303001,
        policy_root=503001,
    )
    view.update(
        "learner", dict(stage="periodic_evaluation", workflow=workflow, **streams)
    )
    row = view.tasks["learner"]
    assert all(row["values"][key] == value for key, value in streams.items())
    assert "Data 7617270868981535538 Env 303001 Policy 503001" in facts(view, row)[-1]
    saved = json.loads((tmp_path / "logs/progress.json").read_text())
    assert all(
        saved["tasks"][0]["values"][key] == value for key, value in streams.items()
    )
    view.update(
        "learner",
        dict(
            stage="sampling",
            physical_ticks=16,
            workflow=workflow,
            display_values=streams,
        ),
    )
    assert not set(streams) & row["values"].keys()
    assert "Seed 101" in facts(view, row)[-1] and "Data " not in facts(view, row)[-1]


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
        json.dumps({"schema": "smartsom.production-run/v2", "status": "completed"})
    )
    assert monitor(tmp_path, once=True) == 0
    assert list(tmp_path.iterdir()) == [path]


def test_all_runtime_commands_accept_display_options():
    for command, positional in [
        ("run", []),
        ("resume", ["run"]),
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
            "stage": "periodic_evaluation",
            "periodic_evaluation_finished": 2,
            "periodic_evaluation_requested": 5,
            "periodic_evaluation_tick": 100,
            "periodic_evaluation_tick_limit": 4096,
        },
    )
    row = view.snapshot()["tasks"][0]
    assert row["completed"] == 1024 and row["total"] == 16384
    assert row["values"]["periodic_evaluation_finished"] == 2
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
                "stage": "periodic_evaluation",
                "physical_ticks": 1024,
                "algorithm": "DQN",
                "optimization_steps": 10,
                "periodic_evaluation_finished": 2,
                "periodic_evaluation_requested": 5,
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
    assert "periodic_evaluation 2/5" in text and "DQN optimizer 10" in text
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
    for stage in ("starting", "sampling", "periodic_evaluation", "saving", "completed"):
        with view.batch_updates():
            for index in range(8):
                view.update(
                    str(index),
                    {
                        "stage": stage,
                        "status": stage if stage == "completed" else "running",
                        "physical_ticks": 1024,
                        "periodic_evaluation_finished": 2,
                        "periodic_evaluation_requested": 5,
                        "periodic_evaluation_tick": 100,
                        "periodic_evaluation_tick_limit": 4096,
                        "context": "PPO | H=h0 V=low travel=auto | envs=1 sampling=sequential periodic_evaluation=sequential",
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
        if width >= 100 and stage == "periodic_evaluation":
            assert "Training ticks (%)" in text
            assert "1,024/16,384" in text
            assert "Periodic eval" in text
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
            "stage": "periodic_evaluation",
            "physical_ticks": 1024,
            "periodic_evaluation_finished": 1,
            "periodic_evaluation_requested": 5,
            "periodic_evaluation_tick": 100,
            "periodic_evaluation_tick_limit": 4096,
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
    assert "ETA ~ estimating" in text and "Elapsed 0m 30s" in text
    assert "ETA ~ 4m 30s" not in text  # no unsafe aggregate-speed fallback
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


def test_monitor_terminal_eof_detaches_without_control(tmp_path, monkeypatch):
    from smartsom.telemetry import monitor as module

    view, _ = display(tmp_path)
    monkeypatch.setattr(module, "read_snapshot", lambda _: view.snapshot())
    monkeypatch.setattr(module, "_key", lambda *_: "")
    assert (
        monitor(tmp_path, controlling=True, options=DisplayOptions(progress="off")) == 0
    )


def test_readonly_arrow_detail_navigation_never_submits_controls(tmp_path, monkeypatch):
    from smartsom.experiments import runtime_control
    from smartsom.telemetry import monitor as module

    view, _ = display(tmp_path)
    view.kind, view.stage = "tune", "training"
    view.configure_tuning(
        {
            "runtime_control": {
                "limit": 2,
                "mode": "manual",
                "status": "running",
                "tasks": {"a": {"status": "running"}, "b": {"status": "queued"}},
            }
        }
    )
    monkeypatch.setattr(module, "read_snapshot", lambda _: view.snapshot())
    keys = iter(["down", "\r", "down", "\x1b", "d"])
    monkeypatch.setattr(module, "_key", lambda *_: next(keys))
    states = []

    def publish(self, **kwargs):
        self.task_pages.ids = ["a", "b"]
        self.task_pages.anchor = self.task_pages.anchor or "a"
        states.append(
            (
                self.task_pages.anchor,
                getattr(self, "runtime_detail_task", None),
                getattr(self, "runtime_detail_offset", 0),
            )
        )

    monkeypatch.setattr(RuntimeDisplay, "publish", publish)
    monkeypatch.setattr(
        runtime_control,
        "submit",
        lambda *a, **k: pytest.fail("read-only navigation wrote a control command"),
        raising=False,
    )
    assert (
        monitor(tmp_path, controlling=False, options=DisplayOptions(progress="off"))
        == 0
    )
    assert ("b", "b", 0) in states
    assert ("b", "b", 1) in states
    assert states[-1][0:2] == ("b", None)
    assert not (tmp_path / "control").exists()


def test_validation_modes_merge_observed_returns_without_inventing_missing(tmp_path):
    view, _ = display(tmp_path)
    view.update(
        "trial",
        {
            "stage": "periodic_evaluation",
            "periodic_evaluation_results": {
                "argmax": {"mean_return": -2.25, "return_samples": 2, "update": 3}
            },
        },
    )
    view.update(
        "trial",
        {
            "stage": "periodic_evaluation",
            "periodic_evaluation_results": {
                "sampling": {"mean_return": -4.0, "return_samples": 2, "update": 3}
            },
        },
    )
    view.update(
        "trial",
        {
            "stage": "periodic_evaluation",
            "periodic_evaluation_results": {
                "argmax": {"mean_return": -1.25, "return_samples": 2, "update": 4}
            },
        },
    )
    results = next(row for row in view.snapshot()["tasks"] if row["id"] == "trial")[
        "values"
    ]["periodic_evaluation_results"]
    assert results["argmax"]["update"] == 4
    assert results["sampling"]["update"] == 3
    assert results["argmax"]["mean_return"] == -1.25


def test_existing_learner_diagnostics_and_saved_checkpoint_reach_snapshot(tmp_path):
    view, _ = display(tmp_path)
    diagnostics = {
        "groups": {
            "machine": {
                "metrics": {
                    "td_loss": {
                        "mean": 0.25,
                        "weight": 32,
                        "unavailable_minibatches": 0,
                    }
                }
            }
        }
    }
    view.update(
        "trial",
        {
            "stage": "training",
            "learner_diagnostics": diagnostics,
            "checkpoint_path": "checkpoints/update-4",
            "checkpoint_update": 4,
        },
    )
    values = next(row for row in view.snapshot()["tasks"] if row["id"] == "trial")[
        "values"
    ]
    assert values["learner_diagnostics"] == diagnostics
    assert values["checkpoint_path"] == "checkpoints/update-4"
    assert values["checkpoint_update"] == 4


def test_runtime_monitor_stale_warning_uses_fixed_view_slot(tmp_path, monkeypatch):
    from smartsom.telemetry import monitor as module

    view, _ = display(tmp_path)
    view.kind, view.stage = "tune", "training"
    view.configure_tuning(
        {
            "runtime_control": {
                "status": "running",
                "tasks": {"trial": {"status": "pausing"}},
            }
        }
    )
    snapshot = view.snapshot()
    snapshot["updated_at"] = 100
    monkeypatch.setattr(module, "read_snapshot", lambda _: snapshot)
    clock = [0]
    monkeypatch.setattr(module.time, "time", lambda: 100 + clock[0])
    keys = iter([(None, 7), (None, 12), ("d", 13)])

    def key(*_):
        value, clock[0] = next(keys)
        return value

    monkeypatch.setattr(module, "_key", key)
    states = []
    monkeypatch.setattr(
        RuntimeDisplay,
        "publish",
        lambda self, **kwargs: states.append(
            (self.notice, getattr(self, "observation_notice", None))
        ),
    )
    assert monitor(tmp_path, options=DisplayOptions(progress="off")) == 0
    assert (None, "观测过期 7s · 进程状态未知") in states
    assert (None, "观测过期 12s · 进程状态未知") in states
    assert not (tmp_path / "control").exists()


@pytest.mark.parametrize("controlling", [False, True])
def test_search_help_and_detail_navigation_consume_keys_without_controls(
    tmp_path, monkeypatch, controlling
):
    from smartsom.experiments import runtime_control
    from smartsom.telemetry import monitor as module

    view, _ = display(tmp_path)
    view.kind, view.stage = "tune", "training"
    view.configure_tuning(
        {
            "runtime_control": {
                "status": "running",
                "tasks": {
                    "工厂甲": {"status": "failed"},
                    "工厂乙": {"status": "failed"},
                    "other": {"status": "running"},
                },
            }
        }
    )
    snapshot = view.snapshot()
    monkeypatch.setattr(module, "read_snapshot", lambda _: snapshot)
    keys = iter(
        [
            "/",
            "D",
            "P",
            "+",
            " ",
            "\x1b",
            "?",
            "D",
            "P",
            "+",
            " ",
            "\x1b",
            "e",
            "n",
            "\r",
            "pagedown",
            "r",
            "G",
            "\x1b",
            "/",
            "工",
            "厂",
            "甲",
            "left",
            "right",
            "up",
            "down",
            "\r",
            "d",
        ]
    )
    monkeypatch.setattr(module, "_key", lambda *_: next(keys))
    states = []

    def publish(self, **kwargs):
        self.console = Console(file=io.StringIO(), width=80, height=24)
        self.paging = True
        requested_offset = getattr(self, "runtime_detail_offset", 0)
        self.render()
        states.append(
            (
                self.task_pages.scope,
                self.task_pages.query,
                self.task_pages.anchor,
                getattr(self, "runtime_detail_raw", False),
                requested_offset,
                getattr(self, "runtime_detail_offset_page_size", 0),
            )
        )

    monkeypatch.setattr(RuntimeDisplay, "publish", publish)
    monkeypatch.setattr(
        runtime_control,
        "submit",
        lambda *a, **k: pytest.fail("navigation submitted a command"),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_control,
        "removal_target",
        lambda *a, **k: pytest.fail("help opened removal"),
        raising=False,
    )
    assert (
        monitor(
            tmp_path, controlling=controlling, options=DisplayOptions(progress="off")
        )
        == 0
    )
    assert any(state[3] for state in states)
    # PageDown uses the previous page's visible rows. Keeping a heading with
    # its first value can give the next page a different capacity.
    assert any(
        current[4] == previous[4] + previous[5] and previous[5] > 1 and not current[3]
        for previous, current in zip(states, states[1:])
    )
    assert states[-1][:3] == ("failed", "工厂甲", "工厂甲")
    assert not (tmp_path / "control").exists()


def test_native_progress_units_keep_inflight_separate():
    from smartsom.telemetry.current_progress import presentation

    row, total, unit = presentation(
        {
            "physical_ticks": 120,
            "episode_tick": 40,
            "episodes": 2,
            "episode_budget": 8,
            "completed": 17,
        }
    )
    assert (row["completed"], total, unit) == (2, 8, "episodes")
    assert row["committed_physical_ticks"] == 120
    assert row["inflight_episode_tick"] == 40
    row, total, unit = presentation(
        {"phase": "evaluation", "completed_cases": 3, "total_cases": 8, "case_index": 4}
    )
    assert (row["completed"], total, unit) == (3, 8, "cases")
    row, _, _ = presentation({"losses": {"custom_loss": 0.5}})
    assert row["learner_diagnostics"] == {}


def test_native_presentation_reaches_snapshot_without_stale_budget(tmp_path):
    from smartsom.telemetry.current_progress import presentation

    view, _ = display(tmp_path)
    row, total, unit = presentation(
        {"episodes": 2, "episode_budget": 8, "physical_ticks": 100, "episode_tick": 40}
    )
    view.update("run", row, total=total, unit=unit)
    task = view.snapshot()["tasks"][0]
    assert (task["completed"], task["total"], task["unit"]) == (2, 8, "episodes")
    assert task["values"]["inflight_episode_tick"] == 40
    row, total, unit = presentation({"phase": "evaluation", "completed_cases": 1})
    view.update("run", row, total=total, unit=unit)
    task = view.snapshot()["tasks"][0]
    assert task["completed"] == 1 and task["total"] is None


def test_directory_publish_preserves_real_child_progress(tmp_path):
    from smartsom.experiments.author_batch import _publish

    view, _ = display(tmp_path)
    child = tmp_path / "experiments/one"
    child.mkdir(parents=True)
    (child / "progress.json").write_text(
        json.dumps(
            {
                "phase": "train",
                "status": "training",
                "seed": 7,
                "episodes": 2,
                "episode_budget": 8,
                "physical_ticks": 120,
                "episode_tick": 40,
            }
        )
    )
    token = CURRENT.set(view)
    try:
        _publish(
            tmp_path,
            {
                "files": [
                    {
                        "id": "one",
                        "stage": 0,
                        "child": "experiments/one",
                        "task": "train",
                    }
                ]
            },
            {"stage": "running", "files": {"one": {"status": "running"}}},
        )
    finally:
        CURRENT.reset(token)
    task = read_snapshot(tmp_path)["tasks"][0]
    assert (task["completed"], task["total"], task["unit"]) == (2, 8, "episodes")
    assert task["values"]["committed_physical_ticks"] == 120
    assert task["values"]["inflight_episode_tick"] == 40


def test_internal_validation_uses_parent_row_without_completing_training(tmp_path):
    from smartsom.config.base_diagnostic import BaseWorkload, TrackingOptions
    from smartsom.config.base_layout import base_factory
    from smartsom.experiments.reservation_runtime import episode, publish_progress
    from smartsom.policies.rule import BaseRule
    from smartsom.telemetry.experiment_tracking import ExperimentTracking
    from tests.helpers.algorithm import algorithm_options

    child = tmp_path / "periodic_evaluation/episode-1/sample"
    child.mkdir(parents=True)
    view, _ = display(tmp_path)
    tracker = ExperimentTracking(
        tmp_path, TrackingOptions(mode="disabled"), job_type="train"
    )
    tracker.start({"source": {}, "inputs": {}}, {})
    tracker.status["url"] = "https://wandb.ai/entity/project/runs/example"
    token = CURRENT.set(view)
    try:
        tracker.log(
            {
                "phase": "train",
                "episodes": 1,
                "episode_budget": 2,
                "physical_ticks": 120,
            }
        )
        (tmp_path / "progress.json").write_text(
            json.dumps(
                {
                    "phase": "train",
                    "status": "training",
                    "episodes": 1,
                    "episode_budget": 2,
                    "physical_ticks": 120,
                }
            )
        )
        tracker.log(
            {
                "phase": "periodic_evaluation",
                "kind": "summary",
                "selection": "sample",
                "methods": {},
            }
        )
        episode(
            base_factory("small"),
            BaseWorkload(
                schema="smartsom.base-workload/v1",
                orders=4,
                initial_orders=2,
                mean_interarrival=8,
            ),
            algorithm_options(method="rule"),
            BaseRule("rule", 51),
            71,
            "normal",
            30,
            max_ticks=20,
            progress_interval=0,
            on_progress=lambda event: publish_progress(
                child,
                {
                    **event,
                    "phase": "periodic_evaluation",
                    "status": "evaluating",
                    "completed_cases": 0,
                    "total_cases": 4,
                    "tracking": tracker.status,
                },
                display_root=tmp_path,
            ),
        )
        assert view.snapshot()["tasks"][0]["values"]["case_tick"] == 20
        publish_progress(
            child,
            {
                "phase": "periodic_evaluation",
                "status": "completed",
                "completed_cases": 4,
                "total_cases": 4,
                "episode_tick": 60,
                "tracking": tracker.status,
            },
            display_root=tmp_path,
        )
        snapshot = view.snapshot()
        assert len(snapshot["tasks"]) == 1
        row = snapshot["tasks"][0]
        assert row["status"] == "running" and row["stage"] == "periodic_evaluation"
        assert (row["completed"], row["total"], row["unit"]) == (4, 4, "cases")
        assert row["values"]["case_tick"] == 60
        assert row["values"]["physical_ticks"] == 120
        assert row["values"]["wandb_url"] == tracker.status["url"]
        assert (
            json.loads((tmp_path / "progress.json").read_text())["status"]
            == "evaluating"
        )
        assert (
            json.loads((child / "progress.json").read_text())["status"] == "completed"
        )
    finally:
        CURRENT.reset(token)
        tracker.close()


@pytest.mark.parametrize("trigger", ["q", "interrupt"])
@pytest.mark.parametrize("choice,delete", [("1", False), ("\r", False), ("2", True)])
def test_exit_menu_stops_with_explicit_retention(
    tmp_path, monkeypatch, trigger, choice, delete
):
    from smartsom.experiments import control
    from smartsom.telemetry import monitor as module

    view, _ = display(tmp_path)
    keys = iter([trigger, choice])
    calls, notices = [], []
    monkeypatch.setattr(module.sys.stdin, "isatty", lambda: False)
    monkeypatch.setattr(control, "read", lambda _: None)

    def key(*_):
        value = next(keys)
        if value == "interrupt":
            raise KeyboardInterrupt
        return value

    def stop(root, **kwargs):
        calls.append((root, kwargs))
        return {"remaining": [], "deleted": kwargs["delete"]}

    monkeypatch.setattr(module, "_key", key)
    monkeypatch.setattr(control, "stop", stop)
    monkeypatch.setattr(
        RuntimeDisplay, "publish", lambda self, **kw: notices.append(self.notice or "")
    )
    assert (
        monitor(tmp_path, controlling=True, options=DisplayOptions(progress="off")) == 0
    )
    assert calls == [
        (tmp_path, {"timeout": 120, "delete": delete, "expected_owner_id": None})
    ]
    assert any("1 / Enter" in n and "W&B" in n for n in notices)
    assert any(("已停止并删除" if delete else "产物已保留") in n for n in notices)


@pytest.mark.parametrize("keys", [["q", "\x1b", "d"], ["q", "interrupt", "d"], ["d"]])
def test_exit_cancel_and_detach_do_not_stop(tmp_path, monkeypatch, keys):
    from smartsom.experiments import control
    from smartsom.telemetry import monitor as module

    display(tmp_path)
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    iterator = iter(keys)

    def key(*_):
        value = next(iterator)
        if value == "interrupt":
            raise KeyboardInterrupt
        return value

    monkeypatch.setattr(module.sys.stdin, "isatty", lambda: False)
    monkeypatch.setattr(module, "_key", key)
    monkeypatch.setattr(control, "stop", lambda *a, **k: pytest.fail("no stop consent"))
    assert (
        monitor(tmp_path, controlling=True, options=DisplayOptions(progress="off")) == 0
    )
    assert before == {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}


@pytest.mark.parametrize("outcome", ["timeout", "ownership"])
def test_exit_failure_is_visible_and_retains_files(tmp_path, monkeypatch, outcome):
    from smartsom.experiments import control
    from smartsom.telemetry import monitor as module

    display(tmp_path)
    keys, notices = iter(["q", "2"]), []
    monkeypatch.setattr(module.sys.stdin, "isatty", lambda: False)
    monkeypatch.setattr(module, "_key", lambda *_: next(keys))

    def stop(*a, **k):
        if outcome == "ownership":
            raise ValueError("ownership changed")
        return {"remaining": [123], "deleted": False}

    monkeypatch.setattr(control, "stop", stop)
    monkeypatch.setattr(
        RuntimeDisplay, "publish", lambda self, **kw: notices.append(self.notice or "")
    )
    assert (
        monitor(tmp_path, controlling=True, options=DisplayOptions(progress="off")) == 1
    )
    assert (tmp_path / "logs/progress.json").is_file()
    assert any(
        ("超时" if outcome == "timeout" else "ownership changed") in n for n in notices
    )


@pytest.mark.parametrize("key,offline", [("y", True), ("n", False)])
def test_foreground_tracking_menu_records_only_explicit_choice(
    tmp_path, monkeypatch, key, offline
):
    from smartsom.experiments import control, foreground
    from smartsom.telemetry import monitor as module

    display(tmp_path)
    directory = tmp_path / "control/foreground-prompts"
    directory.mkdir(parents=True)
    control.write_json(directory / "session.json", {"id": "session"})
    control.write_json(
        directory / "request.request.json",
        {"id": "request", "session": "session", "error": "cloud unavailable"},
    )
    keys = iter([key, "d"])
    monkeypatch.setattr(module.sys.stdin, "isatty", lambda: False)
    monkeypatch.setattr(module, "_key", lambda *_: next(keys))
    assert (
        monitor(tmp_path, controlling=True, options=DisplayOptions(progress="off")) == 0
    )
    assert json.loads((directory / "request.answer.json").read_text()) == {
        "session": "session",
        "offline": offline,
    }
    assert foreground.pending_prompt(tmp_path) is None
