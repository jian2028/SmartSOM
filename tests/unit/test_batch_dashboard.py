"""Parent batch progress remains live across rule and training stages."""

import json
from io import StringIO
from types import SimpleNamespace

import pytest
from rich.console import Console

from smartsom.experiments.author_batch import _publish, _run_tune
from smartsom.telemetry.runtime import CURRENT, RuntimeDisplay
from smartsom.telemetry.timeline import _calibration, _formal


def test_parent_projects_rule_case_progress_into_three_stage_dashboard(tmp_path):
    child = tmp_path / "experiments/control_zero"
    (child / "entries/entry-0001/logs").mkdir(parents=True)
    (child / "batch.json").write_text(
        json.dumps({"entries": {"entry-0001": {"status": "running"}}})
    )
    progress = child / "entries/entry-0001/logs/progress.json"
    progress.write_text(
        json.dumps(
            {
                "tasks": [
                    {
                        "values": {
                            "evaluation_finished": 3,
                            "evaluation_requested": 5,
                            "evaluation_case_active": True,
                            "evaluation_tick": 1941,
                            "evaluation_tick_limit": 4096,
                        }
                    }
                ]
            }
        )
    )
    plan = {
        "files": [
            {
                "id": "control_zero",
                "task": "evaluate",
                "stage": 10,
                "child": "experiments/control_zero",
                "entry_ids": ["entry-0001"],
            }
        ]
    }
    state = {
        "stage": "stage_10",
        "files": {"control_zero": {"status": "running"}},
    }
    view = RuntimeDisplay(kind="batch-directory", quiet=True)
    view.total_tasks = 1
    view.preflight = {"status": "passed", "checks_done": 1, "checks_total": 1}
    token = CURRENT.set(view)
    try:
        _publish(tmp_path, plan, state)
    finally:
        CURRENT.reset(token)
    values = view.tasks["control_zero"]["values"]
    assert values["evaluation_finished"] == 3
    assert values["evaluation_tick"] == 1941
    assert _formal(view, "formal")[1] == (3 + 1941 / 4096) / 5
    output = StringIO()
    console = Console(file=output, width=110, color_system=None)
    console.print(view.render())
    text = output.getvalue()
    assert "整体流程" in text
    assert "规则对照" in text
    assert "3/5" in text
    assert "1941/4096" in text
    assert "当前阶段" in text


def test_batch_monitor_heartbeat_refreshes_without_task_state_change():
    view = RuntimeDisplay(kind="batch-directory", quiet=True, readonly=True)
    view.updated_at = 100.1
    before = view._frame_state()
    view.updated_at = 101.1
    assert view._frame_state() != before
    ordinary = RuntimeDisplay(kind="run", quiet=True, readonly=True)
    ordinary.updated_at = 100.1
    before = ordinary._frame_state()
    ordinary.updated_at = 101.1
    assert ordinary._frame_state() == before


def test_preflight_does_not_mark_scheduled_calibration_complete():
    view = RuntimeDisplay(kind="batch-directory", quiet=True, readonly=True)
    view.preflight = {
        "status": "running",
        "checks_done": 8,
        "checks_total": 10,
    }
    view.tuning = {
        "stage": "preflight",
        "calibration_status": "queued",
        "calibration": {"level": "quick"},
    }
    assert _calibration(view, "preflight")[0] == "未开始"
    output = StringIO()
    Console(file=output, width=90, color_system=None).print(view.render())
    text = output.getvalue()
    assert "性能评估" in text and "未开始" in text
    assert "实验执行" in text
    assert "规则对照＋训练评估" not in text


def test_explicit_calibration_skip_is_distinct_from_completion():
    view = RuntimeDisplay(kind="batch-directory", quiet=True, readonly=True)
    view.tuning = {
        "stage": "stage_10",
        "calibration_status": "skipped",
        "calibration": {"level": "off"},
    }
    assert _calibration(view, "formal")[0] == "跳过"
    output = StringIO()
    Console(file=output, width=90, color_system=None).print(view.render())
    assert "已跳过" in output.getvalue()


def test_interactive_batch_launch_shows_preparation_before_attach(monkeypatch):
    from rich.console import Console

    from smartsom.experiments import author_batch, cli
    from smartsom.telemetry import monitor

    events = []

    class Indicator:
        def __enter__(self):
            events.append("spinner")
            return self

        def __exit__(self, *_args):
            events.append("spinner_closed")

        def update(self, message):
            events.append(message)

    monkeypatch.setattr(
        cli,
        "sys",
        SimpleNamespace(
            stdin=SimpleNamespace(isatty=lambda: True),
            stdout=SimpleNamespace(isatty=lambda: True),
            stderr=SimpleNamespace(isatty=lambda: True),
        ),
    )
    monkeypatch.setattr(Console, "status", lambda *_a, **_k: Indicator())
    monkeypatch.setattr(
        author_batch,
        "compile_directory",
        lambda *_a, **_k: events.append("compiled") or object(),
    )
    monkeypatch.setattr(
        author_batch,
        "run",
        lambda *_a, **_k: (
            events.append("launched")
            or {"background": True, "directory": "/tmp/test", "status": "running"}
        ),
    )
    monkeypatch.setattr(monitor, "monitor", lambda *_a, **_k: events.append("attached"))
    assert cli.main(["batch-run", "/tmp/test"]) == 0
    assert events[0:2] == ["spinner", "compiled"]
    assert events.index("spinner_closed") < events.index("attached")


def test_failed_tune_worker_reason_reaches_parent_and_rich(tmp_path, monkeypatch):
    from smartsom.experiments import author_batch

    tune = tmp_path / "performance/tune"
    tune.mkdir(parents=True)
    (tune / "batch.json").write_text(
        json.dumps(
            {
                "status": "failed",
                "entries": {
                    "train__entry-0001": {
                        "status": "failed",
                        "failure": {
                            "message": "remote traceback\nValueError: worker implementation differs from frozen batch"
                        },
                    }
                },
            }
        )
    )
    plan = {
        "files": [
            {
                "id": "train",
                "task": "train-evaluate",
                "stage": 20,
                "entry_ids": ["entry-0001"],
            }
        ]
    }
    state = {
        "status": "failed",
        "stage": "failed",
        "tune_directory": "performance/tune",
        "files": {"train": {"status": "failed"}},
    }

    class Process:
        exitcode = 1

        def start(self):
            pass

        def is_alive(self):
            return False

        def join(self):
            pass

    monkeypatch.setattr(
        author_batch.multiprocessing,
        "get_context",
        lambda *_: SimpleNamespace(Process=lambda **_: Process()),
    )
    view = RuntimeDisplay(kind="batch-directory", quiet=True)
    token = CURRENT.set(view)
    try:
        _publish(tmp_path, plan, state)
        with pytest.raises(
            RuntimeError,
            match="train__entry-0001: ValueError: worker implementation differs",
        ):
            _run_tune(tmp_path, plan, state, recommend_only=False)
        view.finish("failed", error="ValueError: worker implementation differs")
    finally:
        CURRENT.reset(token)
    assert view.tasks["train__entry-0001"]["reason"].startswith("ValueError:")
    output = StringIO()
    Console(file=output, width=110, color_system=None).print(view.render())
    assert "worker implementation differs" in output.getvalue()
