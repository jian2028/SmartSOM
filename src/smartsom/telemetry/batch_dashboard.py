"""Stage details for a V4 directory batch outside its Tune training stage."""

import time

from rich.console import Group
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TaskProgressColumn
from rich.text import Text

from smartsom.telemetry.task_pages import render_panel
from smartsom.telemetry.timeline import online_mode
from smartsom.telemetry.timeline import render as render_timeline


def render(view):
    if (view.tuning or {}).get("runtime_control"):
        from smartsom.telemetry.tuning_dashboard import runtime_render

        return runtime_render(view)
    rows = [row for row in view.tasks.values() if row.get("unit") == "entries"]
    rows = [row for _, row in view.task_pages.ordered([(None, row) for row in rows])]
    cards, compact = [], []
    for row in rows:
        values = row.get("values", {})
        evaluation = (values.get("workflow") or {}).get("mode") == "evaluation"
        finished = values.get("evaluation_finished") or 0
        requested = values.get("evaluation_requested") or 0
        tick = values.get("evaluation_tick") or 0
        limit = values.get("evaluation_tick_limit") or 0
        active = bool(values.get("evaluation_case_active"))
        facts = (
            f"已完成 {finished}/{requested if 'evaluation_target' in values else requested or '待记录'} 案例"
            + (f" · 当前 {tick}/{limit} tick" if active else "")
            if evaluation
            else f"条目 {row.get('completed', 0)}/{row.get('total', 0)}"
        )
        progress = Progress(
            BarColumn(bar_width=None, complete_style="#85cca8"),
            TaskProgressColumn(),
            expand=True,
        )
        progress.add_task(
            "",
            total=requested or 1 if evaluation else row.get("total") or 1,
            completed=(finished + (min(1.0, tick / limit) if active and limit else 0))
            if evaluation
            else row.get("completed", 0),
        )
        heading = Text(
            f"{row['name']} · {row['status']}",
            style="#85cca8" if row["status"] == "running" else "dim",
        )
        cumulative = [Text(""), Text("")] if evaluation else []
        if "evaluation_historical_completed" in values:
            cumulative = [
                Text(
                    f"历史完成 {values['evaluation_historical_completed']} · 本次完成 {values.get('evaluation_current_completed', 0)}"
                ),
                Text(
                    f"剩余 {values.get('evaluation_remaining', 'N/A')} · 总目标 {values.get('evaluation_target', 'N/A')}"
                ),
            ]
        cards.append(
            Panel(
                Group(heading, Text(facts), *cumulative, progress),
                padding=(0, 1),
                border_style="#85cca8",
            )
        )
        compact.append(
            Group(
                heading,
                Text(facts),
                *(cumulative if "evaluation_historical_completed" in values else []),
            )
        )
    from smartsom.telemetry.execution_metrics import batch_text

    header = [
        render_timeline(view),
        Text(batch_text(view), style="cyan", no_wrap=True, overflow="ellipsis"),
    ]
    short = [Text(f"实验执行 · {view.stage} · 共 {len(rows)} 个任务", style="bold")]
    if not cards:
        header.append(Text("等待当前阶段进度…", style="dim"))
    if (view.tuning or {}).get("native_stage"):
        from smartsom.telemetry.tuning_dashboard import execution_card, execution_lines

        header.append(execution_card(view.tuning))
        short.extend(
            Text(line, style="cyan", no_wrap=True, overflow="ellipsis")
            for line in execution_lines(view.tuning)[:2]
        )
    elif online_mode(view.tuning or {}):
        from smartsom.telemetry.tuning_dashboard import (
            performance_card,
            performance_lines,
        )

        header.append(performance_card(view.tuning, condensed=True))
        short += [
            Text("性能 · 在线调整", style="cyan"),
            *[
                Text(line, style="cyan")
                for line in performance_lines(view.tuning, condensed=True)
            ],
        ]
    from smartsom.telemetry.tuning_dashboard import control_footer, runtime_lines

    if (view.tuning or {}).get("runtime_control"):
        header.extend(
            Text(value, style="cyan", no_wrap=True, overflow="ellipsis")
            for value in runtime_lines(view.tuning, controlling=view.controlling)
        )
        short.extend(
            Text(value, style="cyan", no_wrap=True, overflow="ellipsis")
            for value in runtime_lines(
                view.tuning, controlling=view.controlling, condensed=True
            )
        )
    if view.console.height > 12:
        header.append(
            Text(
                f"最近记录 {time.strftime('%H:%M:%S', time.localtime(view.updated_at))}",
                style="dim",
            )
        )
    short.append(
        Text(batch_text(view), style="cyan", no_wrap=True, overflow="ellipsis")
    )
    return render_panel(
        view,
        header=header,
        short_header=short,
        ids=[row["id"] for row in rows],
        cards=cards,
        compact_cards=compact,
        title="当前阶段 · 规则对照"
        if view.stage.startswith("stage_")
        else "批次状态与结果",
        footer=control_footer(view, view.tuning or {})
        + "d 离开界面 · Ctrl+C×2 "
        + ("安全停止任务" if view.controlling else "关闭监控"),
        border_style="#85cca8",
    )
