"""Parent batch progress remains live across rule and training stages."""

import json
from io import StringIO
from types import SimpleNamespace

import pytest
from rich.console import Console

from smartsom.experiments.author_batch import _publish, _run_tune
from smartsom.telemetry.runtime import CURRENT, RuntimeDisplay
from smartsom.telemetry.timeline import _calibration
from smartsom.telemetry.timeline import render as render_timeline


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
    Console(file=output, width=90, color_system=None).print(render_timeline(view))
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
    Console(file=output, width=90, color_system=None).print(render_timeline(view))
    assert "已跳过" in output.getvalue()


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
