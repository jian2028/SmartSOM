"""Preflight accounting and the shared phase display contract."""

import json

from rich.console import Console

from smartsom.experiments import preflight
from smartsom.telemetry.runtime import DisplayOptions, RuntimeDisplay
from smartsom.telemetry.timeline import detail, render


def _plan(level="full"):
    return {
        "schema": "smartsom.author-plan/v1",
        "experiment": {"execution": {"preflight": level, "preflight_coverage": "each"}},
        "entries": [
            {
                "id": f"entry-{index}",
                "snapshot": f"inputs/entry-{index}",
                "sources": {"factory": "same", "algorithm": "same"},
                "H": {"level": "same"},
                "workload": {"selected_level": "mid"},
                "task": "train-evaluate",
            }
            for index in range(2)
        ],
    }


def test_full_smoke_switches_to_representative_without_counting_skip_as_pass(
    tmp_path, monkeypatch
):
    import smartsom.experiments.composable as composable

    monkeypatch.setattr(composable, "prepared_from_run", lambda *_: object())
    monkeypatch.setattr(composable, "verify_prepared_rules", lambda *_: None)
    monkeypatch.setattr(
        preflight,
        "smoke",
        lambda *_args, **_kwargs: {
            "status": "passed",
            "ticks": 1,
            "exercised": ["machine"],
            "uncovered": [],
        },
    )
    monkeypatch.setattr(preflight, "_coverage", lambda *_: "representative")
    result = preflight.run(tmp_path, _plan())
    assert result["status"] == "passed"
    assert result["checks_done"] == result["checks_total"] == 3
    assert result["smoke_done"] == 1
    assert result["smoke_skipped"] == 1
    assert result["entries"]["entry-1"]["status"] == "skipped"
    assert json.loads((tmp_path / "preflight.json").read_text()) == result


def test_quick_preflight_does_not_run_smoke(tmp_path, monkeypatch):
    import smartsom.experiments.composable as composable

    monkeypatch.setattr(composable, "prepared_from_run", lambda *_: object())
    monkeypatch.setattr(composable, "verify_prepared_rules", lambda *_: None)
    monkeypatch.setattr(
        preflight,
        "smoke",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("smoke")),
    )
    result = preflight.run(tmp_path, _plan("quick"))
    assert result["smoke_total"] == 0
    assert result["status"] == "passed"


def test_timeline_keeps_calibration_wall_budget_independent_from_candidates():
    view = RuntimeDisplay(DisplayOptions(), kind="tune")
    view.console = Console(width=120, force_terminal=False, color_system=None)
    view.preflight = {
        "status": "passed",
        "checks_done": 3,
        "checks_total": 3,
        "smoke_total": 2,
        "smoke_done": 1,
        "smoke_skipped": 1,
    }
    view.tuning = {
        "stage": "calibrating",
        "calibration": {"wall_seconds": 20, "limit_seconds": 180, "measured": 1},
    }
    with view.console.capture() as captured:
        view.console.print(render(view))
    text = captured.get()
    assert "✓ 预检" in text and "已完成" in text
    assert "▶ 性能评估" in text and "进行中" in text
    assert "20/180s" in detail(view, "calibration")
    assert "11.1%" in text


def test_timeline_moves_to_formal_wait_after_recommendation():
    view = RuntimeDisplay(DisplayOptions(), kind="tune")
    view.console = Console(width=120, force_terminal=False, color_system=None)
    view.tuning = {
        "stage": "waiting_resources",
        "calibration": {"recommendation": {"group": {"threads": 1}}},
    }
    view.tasks["entry"] = {
        "id": "entry",
        "status": "queued",
        "stage": "queued",
        "unit": "physical ticks",
        "completed": 0,
        "total": 16,
        "values": {},
    }
    with view.console.capture() as captured:
        view.console.print(render(view))
    text = captured.get()
    assert "▶ 性能评估" in text and "进行中" in text
    assert "训练＋最终评估" in text and "未开始" in text
    assert "0.0%" in text


def test_timeline_marks_final_evaluation_from_finished_cases():
    view = RuntimeDisplay(DisplayOptions(), kind="tune")
    view.console = Console(width=120, force_terminal=False, color_system=None)
    view.status = "completed"
    view.tuning = {"stage": "completed", "calibration": {}}
    view.tasks["entry"] = {
        "id": "entry",
        "status": "completed",
        "stage": "completed",
        "unit": "physical ticks",
        "completed": 16,
        "total": 16,
        "values": {"evaluation_finished": 1, "evaluation_requested": 1},
    }
    with view.console.capture() as captured:
        view.console.print(render(view))
    text = captured.get()
    assert "训练＋最终评估" in text and "已完成" in text
    assert "最终评估 1/1 案例" in detail(view, "formal")


def test_formal_overview_averages_parallel_experiments_and_keeps_evaluation_pending():
    view = RuntimeDisplay(DisplayOptions(), kind="tune")
    view.console = Console(width=120, force_terminal=False, color_system=None)
    view.tuning = {"stage": "training", "calibration": {"recommendation": {}}}
    for name, ticks in (("first", 100), ("second", 0)):
        view.tasks[name] = {
            "id": name,
            "status": "running",
            "stage": "sampling",
            "unit": "physical ticks",
            "completed": ticks,
            "total": 100,
            "values": {
                "workflow": {"mode": "train-evaluate", "evaluation_cases": 2},
                "evaluation_finished": 0,
                "evaluation_requested": 2,
            },
        }
    with view.console.capture() as captured:
        view.console.print(render(view))
    text = captured.get()
    assert "▶ 训练＋最终评估" in text and "进行中" in text
    assert "25.0%" in text
    assert "训练 100/200 ticks · 最终评估 0/4 案例" in detail(view, "formal")

    view.tasks["first"]["stage"] = "evaluation"
    view.tasks["first"]["values"]["evaluation_finished"] = 1
    with view.console.capture() as captured:
        view.console.print(render(view))
    text = captured.get()
    assert "▶ 训练＋最终评估" in text and "进行中" in text
    assert "37.5%" in text


def test_short_terminal_retains_all_three_overview_bars():
    view = RuntimeDisplay(DisplayOptions(), kind="tune")
    view.console = Console(width=60, height=12, force_terminal=False, color_system=None)
    view.preflight = {"status": "running", "checks_done": 1, "checks_total": 2}
    view.tuning = {"stage": "preflight", "calibration": {}}
    with view.console.capture() as captured:
        view.console.print(render(view))
    text = captured.get()
    assert text.count("▶ 预检") == 1
    assert text.count("○ 性能评估") == 1
    assert text.count("○ 训练＋最终评估") == 1
    assert "▶ 预检" in text
    with view.console.capture() as captured:
        view.console.print(view.render())
    assert "▶ 预检" in captured.get()
    assert "性能评估" in captured.get()
    assert "训练＋最终评估" in captured.get()


def test_failed_calibration_does_not_appear_completed():
    view = RuntimeDisplay(DisplayOptions(), kind="tune")
    view.console = Console(width=120, force_terminal=False, color_system=None)
    view.status = "failed"
    view.tuning = {
        "stage": "failed",
        "calibration": {"wall_seconds": 5, "limit_seconds": 10, "measured": 0},
    }
    with view.console.capture() as captured:
        view.console.print(render(view))
    assert "性能评估" in captured.get() and "已停止" in captured.get()


def test_overview_columns_and_panel_borders_align_at_multiple_widths():
    from rich.cells import cell_len

    for width in (60, 80, 120, 200):
        view = RuntimeDisplay(DisplayOptions(), kind="tune")
        view.console = Console(width=width, force_terminal=False, color_system=None)
        view.preflight = {"status": "passed", "checks_done": 1, "checks_total": 1}
        view.tuning = {
            "stage": "calibrating",
            "calibration": {"wall_seconds": 10, "limit_seconds": 20},
        }
        with view.console.capture() as captured:
            view.console.print(render(view))
        lines = captured.get().splitlines()
        assert all(cell_len(row) == width for row in lines)
        assert lines[0].startswith("╭") and lines[0].endswith("╮")
        assert lines[-1].startswith("╰") and lines[-1].endswith("╯")
        stages = [
            line
            for line in lines
            if any(name in line for name in ("预检", "性能评估", "训练＋最终评估"))
        ]
        assert len(stages) == 3
        for status in ("已完成", "进行中", "未开始"):
            assert sum(status in line for line in stages) == 1
        assert (
            len(
                {
                    cell_len(line.split(status)[0])
                    for line, status in zip(stages, ("已完成", "进行中", "未开始"))
                }
            )
            == 1
        )
        assert len({cell_len(line.rsplit(" ", 1)[0]) for line in stages}) == 1
