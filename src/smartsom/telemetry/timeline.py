"""Evidence-backed stages; isolated calibration appears only when applicable."""

from rich.panel import Panel
from rich.progress import BarColumn, Progress
from rich.table import Table

from smartsom.telemetry.dashboard import line

BRIGHT = {
    "preflight": "#b39aff",
    "calibration": "#f0a940",
    "formal": "#70b7ee",
    "evaluation": "#85cca8",
}
COMPLETE = "#7d948e"
PENDING = "#555b66"
FAILURE = "#e78787"


def online_mode(summary):
    return (
        summary.get("calibration_status") == "online"
        or (summary.get("calibration") or {}).get("level") == "online"
    )


def has_calibration_stage(view):
    tuning = view.tuning or {}
    if online_mode(tuning):
        return False
    if tuning.get("calibration_status") in {"skipped", "not_applicable"}:
        return False
    level = (tuning.get("calibration") or {}).get("level")
    if level == "off":
        return False
    # Missing levels in legacy Tune snapshots retain their original workflow.
    return level in {"quick", "full"} or view.kind == "tune" or bool(tuning)


def _current(view):
    preflight = view.preflight or {}
    if preflight.get("status") == "running":
        return "preflight"
    tuning = view.tuning or {}
    if tuning.get("stage") in {
        "calibration",
        "calibrating",
        "waiting_resources",
        "waiting_for_resources",
    }:
        if not has_calibration_stage(view):
            return "formal"
        return "calibration"
    if view.status not in {
        "completed",
        "recommended",
        "failed",
        "stopped",
        "force_stopped",
    }:
        return "formal"
    return None


def _evaluation_active(view):
    if (view.workflow or {}).get("mode") == "evaluation":
        return True
    return any(
        row.get("stage") in {"evaluation", "test"}
        or row.get("values", {}).get("evaluation_case_active")
        for row in view.tasks.values()
    )


def _training_active(view):
    return any(
        row.get("status") not in {"completed", "failed", "stopped"}
        and row.get("stage")
        in {
            "sampling",
            "optimizing",
            "updating",
            "learning_metrics",
            "validation",
            "saving",
        }
        for row in view.tasks.values()
    )


def _candidate_count(value):
    if isinstance(value, list):
        return len(value)
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _preflight(view, current):
    state = view.preflight or {}
    if not state:
        return "未记录", None, "尚无预检记录"
    checks_done = state.get("checks_done", 0)
    checks_total = state.get("checks_total", 0)
    smoke_done = state.get("smoke_done", 0)
    smoke_skipped = state.get("smoke_skipped", 0)
    smoke_total = state.get("smoke_total", 0)
    count = checks_done + smoke_done + smoke_skipped
    total = checks_total + smoke_total
    detail = (
        f"检查 {checks_done}/{checks_total} · 烟测 {smoke_done}/{smoke_total}"
        f" · 跳过 {smoke_skipped}"
    )
    fraction = min(1.0, count / total) if total else 0.0
    if state.get("status") == "failed":
        return "失败", fraction, detail
    if state.get("status") == "passed":
        return "已完成", 1.0, detail
    return ("进行中" if current == "preflight" else "未开始"), fraction, detail


def _calibration(view, current):
    if view.kind != "tune" and view.tuning is None:
        return "跳过", None, "本次没有性能评估"
    recorded_status = (view.tuning or {}).get("calibration_status")
    calibration = (view.tuning or {}).get("calibration", {})
    if recorded_status == "online" or calibration.get("level") == "online":
        return "在线调整", None, "不运行独立性能测试；根据真实更新的总吞吐调整并发"
    if recorded_status in {"skipped", "not_applicable"}:
        return "跳过", None, "使用冻结的固定资源配置，未运行性能评估"
    spent = calibration.get("wall_seconds")
    limit = calibration.get("limit_seconds")
    measured = calibration.get("measured")
    candidates = _candidate_count(calibration.get("candidates"))
    if isinstance(spent, (int, float)) and isinstance(limit, (int, float)):
        detail = (
            f"实测候选 {measured if measured is not None else '待测'}"
            f"/{candidates if candidates is not None else '未记录'}"
            f" · 墙钟 {spent:.0f}/{limit:.0f}s"
        )
    else:
        detail = "等待校准数据"
    elapsed_fraction = (
        min(1.0, max(0.0, spent / limit))
        if isinstance(spent, (int, float))
        and isinstance(limit, (int, float))
        and limit > 0
        else 0.0
    )
    if current == "calibration":
        if (view.tuning or {}).get("stage") in {
            "waiting_resources",
            "waiting_for_resources",
        }:
            return "等待资源", 1.0, detail
        return "进行中 · 预算使用", elapsed_fraction, detail
    stage = (view.tuning or {}).get("stage")
    if view.status == "failed" or stage == "failed":
        return "失败", elapsed_fraction, detail
    if (
        recorded_status == "completed"
        or calibration.get("recommendation")
        or view.status == "recommended"
        or stage == "calibration_complete"
    ):
        return "已完成", 1.0, detail + " · 阶段完成，预算可剩余"
    return "未开始", 0.0, detail


def _formal(view, current):
    if view.status == "recommended":
        return "未执行", 0.0, "本次只给出资源建议，没有启动训练"
    rows = [
        row
        for row in view.tasks.values()
        if row.get("unit") == "physical ticks"
        or (row.get("values", {}).get("workflow") or {}).get("mode")
        in {"train", "train-evaluate", "evaluation"}
    ]
    # Directory batches also expose file-level status rows. Count only the
    # scientific entries, so those file rows do not dilute formal progress.
    expected = (
        len(rows)
        if view.kind == "batch-directory"
        else max(len(rows), view.total_tasks or 0)
    )
    if expected == 0:
        status = "进行中" if current == "formal" else "未开始"
        return status, None, "等待正式实验数据"
    contributions = []
    train_done = train_total = eval_done = eval_total = 0
    finished = 0
    for row in rows:
        values = row.get("values", {})
        workflow = values.get("workflow") or view.workflow or {}
        mode = workflow.get("mode", "train-evaluate")
        has_train = mode not in {"evaluation", "run"}
        has_eval = mode not in {"train", "training"}
        done = row.get("completed") or 0
        total = row.get("total") or workflow.get("training_total") or 0
        requested = (
            values.get("evaluation_requested") or workflow.get("evaluation_cases") or 0
        )
        evaluated = values.get("evaluation_finished") or 0
        case_tick = values.get("evaluation_tick") or 0
        case_limit = values.get("evaluation_tick_limit") or 0
        case_fraction = (
            min(1.0, max(0.0, case_tick / case_limit))
            if values.get("evaluation_case_active") and case_limit > 0
            else 0.0
        )
        if has_train:
            train_done += done
            train_total += total
        if has_eval:
            eval_done += evaluated
            eval_total += requested
        fractions = []
        if has_train:
            fractions.append(min(1.0, done / total) if total else 0.0)
        if has_eval:
            fractions.append(
                min(1.0, (evaluated + case_fraction) / requested) if requested else 0.0
            )
        contributions.append(sum(fractions) / len(fractions) if fractions else 0.0)
        finished += row.get("status") == "completed"
    progress = sum(contributions) / expected
    details = []
    if train_total:
        details.append(f"训练 {train_done}/{train_total} ticks")
    if eval_total:
        label = "评估" if view.kind == "batch-directory" else "最终评估"
        details.append(f"{label} {eval_done}/{eval_total} 案例")
    details.append(f"实验完成 {finished}/{expected}")
    detail = " · ".join(details)
    if current in {"preflight", "calibration"}:
        return "未开始", 0.0, "正式实验尚未启动"
    if view.status == "completed" and finished == expected:
        return "已完成", 1.0, detail
    if view.status in {"failed", "stopped", "force_stopped"}:
        return "失败或停止", progress, detail
    if current == "formal":
        if rows and all(row.get("status") in {"queued", "pending"} for row in rows):
            return "待启动", 0.0, detail
        if _evaluation_active(view) and _training_active(view):
            activity = "训练及评估中"
        elif _evaluation_active(view):
            activity = "规则对照中" if view.kind == "batch-directory" else "最终评估中"
        else:
            activity = "训练中"
        return (
            activity,
            progress,
            detail,
        )
    return "未开始", progress, detail


def detail(view, stage):
    current = _current(view)
    return {
        "preflight": _preflight,
        "calibration": _calibration,
        "formal": _formal,
    }[stage](view, current)[2]


def render(view):
    current = _current(view)
    stages = [
        ("preflight", "预检", _preflight(view, current)),
        (
            "formal",
            "实验执行" if view.kind == "batch-directory" else "训练＋最终评估",
            _formal(view, current),
        ),
    ]
    if has_calibration_stage(view):
        stages.insert(1, ("calibration", "性能评估", _calibration(view, current)))
    grid = Table.grid(expand=True, padding=(0, 1))
    grid.add_column(width=16, no_wrap=True)
    grid.add_column(width=8, no_wrap=True)
    grid.add_column(ratio=1, no_wrap=True)
    grid.add_column(width=6, justify="right", no_wrap=True)
    for stage, name, (status, fraction, _detail) in stages:
        active = stage == current
        marker = "▶" if active else "✓" if status == "已完成" else "○"
        if active:
            color = (
                BRIGHT["evaluation"]
                if stage == "formal"
                and _evaluation_active(view)
                and not _training_active(view)
                else BRIGHT[stage]
            )
        elif status in {"失败", "失败或停止"}:
            color = FAILURE
        elif status == "已完成":
            color = COMPLETE
        else:
            color = PENDING
        display_status = (
            "进行中"
            if active and status not in {"失败", "失败或停止"}
            else "已完成"
            if status == "已完成"
            else "已跳过"
            if status == "跳过"
            else "未开始"
            if status in {"未开始", "待启动", "未执行", "未记录"}
            else "已停止"
        )
        progress = Progress(
            BarColumn(
                bar_width=None,
                complete_style=color,
                finished_style=color,
                style="grey37",
            ),
            expand=True,
        )
        progress.add_task("", total=1, completed=fraction or 0)
        grid.add_row(
            line(f"{marker} {name}", color),
            line(display_status, color),
            progress,
            line(f"{fraction:.1%}" if fraction is not None else "N/A", color),
        )
    if view.status in {"failed", "stopped", "force_stopped"}:
        border = FAILURE
    elif (
        current == "formal" and _evaluation_active(view) and not _training_active(view)
    ):
        border = BRIGHT["evaluation"]
    else:
        border = BRIGHT.get(current, COMPLETE)
    return Panel(
        grid,
        title=line("整体流程", "bold"),
        border_style=border,
        padding=(0, 1),
    )
