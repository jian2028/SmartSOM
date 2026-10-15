"""View-only adaptive Rich grids, stable task anchors and height pagination."""

from dataclasses import dataclass, field

from rich.table import Table

MIN_CARD_WIDTH = 48
GAP = 2


def grid(cards, columns):
    table = Table.grid(expand=True, padding=(0, 1), pad_edge=False)
    for _ in range(columns):
        table.add_column(ratio=1)
    for start in range(0, len(cards), columns):
        row = cards[start : start + columns]
        table.add_row(*row, *([""] * (columns - len(row))))
    return table


@dataclass
class TaskPages:
    anchor: str | None = None
    index: int = 0
    pending: int = 0
    bounds: list = field(default_factory=list)
    ids: list = field(default_factory=list)
    columns: int = 1
    order: list = field(default_factory=list)

    def ordered(self, pairs):
        by_id = {row["id"]: (entry, row) for entry, row in pairs}
        self.order = [key for key in self.order if key in by_id] + [
            key for key in by_id if key not in self.order
        ]
        return [by_id[key] for key in self.order]

    def move(self, direction):
        self.pending += direction

    def choose(self, console, ids, cards, *, width, height):
        self.columns = min(
            max(1, len(cards)), max(1, (width + GAP) // (MIN_CARD_WIDTH + GAP))
        )
        cell_width = max(1, (width - GAP * (self.columns - 1)) // self.columns)
        row_heights = [
            max(
                len(
                    console.render_lines(
                        card,
                        console.options.update(width=cell_width, height=None),
                        pad=False,
                    )
                )
                for card in cards[start : start + self.columns]
            )
            for start in range(0, len(cards), self.columns)
        ]
        bounds, start, used = [], 0, 0
        for row_index, row_height in enumerate(row_heights):
            offset = row_index * self.columns
            if used and used + row_height > height:
                bounds.append((start, offset))
                start, used = offset, 0
            used += row_height
        if cards:
            bounds.append((start, len(cards)))
        self.bounds, self.ids = bounds, list(ids)
        if not bounds:
            self.index, self.anchor, self.pending = 0, None, 0
            return [], (0, 0)
        if self.anchor in ids:
            anchor_index = ids.index(self.anchor)
            self.index = next(
                i for i, (a, b) in enumerate(bounds) if a <= anchor_index < b
            )
        self.index = max(0, min(len(bounds) - 1, self.index + self.pending))
        moved = bool(self.pending)
        self.pending = 0
        a, b = bounds[self.index]
        if moved or self.anchor not in ids:
            self.anchor = ids[a]
        return cards[a:b], (a, b)

    @property
    def label(self):
        if not self.bounds:
            return "第 1/1 页 · 任务 0/0"
        a, b = self.bounds[self.index]
        return f"第 {self.index + 1}/{len(self.bounds)} 页 · 任务 {a + 1}–{b}/{len(self.ids)} · {self.columns} 列"


def render_panel(
    view,
    *,
    header,
    ids,
    cards,
    compact_cards,
    short_header,
    title,
    subtitle=None,
    footer="",
    border_style=None,
):
    """Measure cells and keep every task reachable in interactive viewports."""
    from rich import box
    from rich.console import Group
    from rich.panel import Panel
    from rich.text import Text

    paged = view.paging
    width, height = max(1, view.console.width - 4), max(3, view.console.height - 1)
    columns = min(max(1, len(cards)), max(1, (width + GAP) // (MIN_CARD_WIDTH + GAP)))
    cell_width = max(1, (width - GAP * (columns - 1)) // columns)
    items = list(header)
    notices = (
        [Text(str(view.notice), style="yellow", no_wrap=True, overflow="ellipsis")]
        if view.notice
        else []
    )
    footer_text = Text(footer, style="dim", no_wrap=True, overflow="ellipsis")
    if paged:
        if width < 90:
            for long, short in (
                ("c 修改总并发", "c 并发"),
                ("d 离开界面", "d 离开"),
                ("安全停止任务", "停止"),
                ("关闭监控", "退出"),
            ):
                footer = footer.replace(long, short)
        footer_text = Text(
            (
                ("[ 上页 · ] 下一页 · " if width < 90 else "[ 上一页 · ] 下一页 · ")
                if cards
                else ""
            )
            + footer,
            style="dim",
            no_wrap=True,
            overflow="ellipsis",
        )

        def available_space():
            overhead = (
                len(
                    view.console.render_lines(
                        Group(
                            *items,
                            *notices,
                            *([Text("页码")] if cards else []),
                            *([footer_text] if footer else []),
                        ),
                        view.console.options.update(width=width, height=None),
                        pad=False,
                    )
                )
                + 2
            )
            return height - overhead

        available = available_space()
        required = 1 if cards else 0
        if height <= 12 or available < (4 if cards else 0):
            items = list(short_header)
            available = available_space()
        if available < required:
            items = []
            available = available_space()
            if available < required and notices:
                return Panel(
                    Group(*notices),
                    title=title,
                    subtitle=subtitle,
                    box=box.SQUARE,
                    border_style=border_style or "",
                    padding=(0, 1),
                    height=height,
                )
            available = max(required, available)

    if paged and cards:

        def measured(card):
            return len(
                view.console.render_lines(
                    card,
                    view.console.options.update(width=cell_width, height=None),
                    pad=False,
                )
            )

        if any(measured(card) > available for card in cards):
            cards = compact_cards
        if any(measured(card) > available for card in cards):
            from smartsom.telemetry.dashboard import task_eta

            cards = [
                Text(
                    identity
                    + " · ETA "
                    + task_eta(
                        view,
                        view.tasks.get(identity, {"id": identity, "status": "queued"}),
                    ),
                    no_wrap=True,
                    overflow="ellipsis",
                )
                for identity in ids
            ]
        visible, _ = view.task_pages.choose(
            view.console, ids, cards, width=width, height=available
        )
        items += [
            *notices,
            grid(visible, view.task_pages.columns),
            Text(
                "实验卡片 · " + view.task_pages.label,
                style="bold",
                no_wrap=True,
                overflow="ellipsis",
            ),
            footer_text,
        ]
    else:
        items.extend(notices)
        if cards:
            items += [
                Text(f"实验卡片 · 全部 {len(cards)} 个任务", style="bold"),
                grid(cards, columns),
            ]
        if footer:
            items.append(footer_text)
    return Panel(
        Group(*items),
        title=title,
        subtitle=subtitle,
        box=box.SQUARE,
        border_style=border_style or "",
        padding=(0, 1),
        height=height if paged else None,
    )
