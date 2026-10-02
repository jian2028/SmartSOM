"""Parent batch progress remains live across rule and training stages."""

from io import StringIO

from rich.console import Console

from smartsom.telemetry.runtime import RuntimeDisplay
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
