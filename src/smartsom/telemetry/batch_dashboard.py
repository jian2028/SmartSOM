"""Stage details for a V4 directory batch outside its Tune training stage."""

import time

from rich.console import Group
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TaskProgressColumn
from rich.text import Text

from smartsom.telemetry.timeline import render as render_timeline


def render(view):
    rows = [
        row
        for row in view.tasks.values()
        if row.get("unit") == "entries"
        and (row.get("values", {}).get("workflow") or {}).get("mode") == "evaluation"
    ]
    details = []
    for row in rows:
        values = row.get("values", {})
        finished = values.get("evaluation_finished") or 0
        requested = values.get("evaluation_requested") or 0
        tick = values.get("evaluation_tick") or 0
        limit = values.get("evaluation_tick_limit") or 0
        active = bool(values.get("evaluation_case_active"))
        progress = Progress(
            BarColumn(bar_width=None, complete_style="#85cca8"),
            TaskProgressColumn(),
            expand=True,
        )
        progress.add_task(
            "",
            total=requested or 1,
            completed=finished + (min(1.0, tick / limit) if active and limit else 0),
        )
        details.extend(
            [
                Text(
                    f"{row['name']} · {row['status']} · "
                    f"已完成 {finished}/{requested or '待记录'} 案例"
                    + (f" · 当前 {tick}/{limit} tick" if active else ""),
                    style="#85cca8" if row["status"] == "running" else "dim",
                ),
                progress,
            ]
        )
    if view.stage in {"completed", "failed", "stopped", "force_stopped"}:
        for row in view.tasks.values():
            if row.get("unit") == "entries" and row not in rows:
                details.append(
                    Text(
                        f"{row['name']} · {row['status']} · "
                        f"条目 {row.get('completed', 0)}/{row.get('total', 0)}"
                    )
                )
    if not details:
        details.append(Text("等待当前阶段进度…", style="dim"))
    if view.notice:
        details.append(
            Text(str(view.notice), style="red" if view.stage == "failed" else "dim")
        )
    details.append(
        Text(
            f"最近记录 {time.strftime('%H:%M:%S', time.localtime(view.updated_at))}",
            style="dim",
        )
    )
    return Group(
        render_timeline(view),
        Panel(
            Group(*details),
            title=(
                "当前阶段 · 规则对照"
                if view.stage.startswith("stage_")
                else "批次状态与结果"
            ),
            border_style="#85cca8",
        ),
    )
