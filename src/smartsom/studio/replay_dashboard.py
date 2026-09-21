"""Recording-only overview and a dependency-free throughput chart."""

from PySide6.QtCore import QEvent, QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QStackedWidget,
    QTabBar,
    QToolButton,
    QToolTip,
    QVBoxLayout,
    QWidget,
)

from smartsom.studio.replay_evidence import outside_count, qualified, submitted
from smartsom.studio.replay_inspector import order_label
from smartsom.studio.workspace_style import ReplaySelector


class ThroughputChart(QWidget):
    """Tick-exact readout over a separately selected display and rate window."""

    def __init__(self, evidence, passing=False):
        super().__init__()
        self.evidence, self.tick, self.window = evidence, 0, 100
        self.visible_window = 500
        self.passing = passing
        self.hover_tick = None
        self.pinned = False
        self.setMouseTracking(True)
        self.setMinimumHeight(110)
        self.setToolTip(
            "Click to pin a recorded tick; move to inspect. Values stop at the displayed tick."
        )

    @property
    def start_tick(self):
        return max(0, self.tick - self.visible_window) if self.visible_window else 0

    def value_at(self, tick):
        return (
            self.evidence.passing_rates[tick]
            if self.passing
            else self.evidence.throughput(tick, self.window)
        )

    def formatted_value(self, tick):
        value = self.value_at(tick)
        if value is None:
            return "—"
        return f"{value:.1%}" if self.passing else f"{value:.3f} jobs/tick"

    def plot_rect(self):
        return QRectF(44, 30, max(1, self.width() - 58), max(1, self.height() - 55))

    def tick_at(self, x):
        rect = self.plot_rect()
        ratio = max(0, min(1, (x - rect.left()) / rect.width()))
        return round(self.start_tick + ratio * (self.tick - self.start_tick))

    def update_tick(self, tick):
        if tick != self.tick:
            self.hover_tick = None
            self.pinned = False
        self.tick = tick
        self.update()

    def mouseMoveEvent(self, event):
        if not self.pinned:
            self.hover_tick = self.tick_at(event.position().x())
            self.update()
        if self.hover_tick is not None:
            QToolTip.showText(
                event.globalPosition().toPoint(),
                f"tick {self.hover_tick} · {self.formatted_value(self.hover_tick)}",
                self,
            )

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.hover_tick = self.tick_at(event.position().x())
            self.pinned = not self.pinned
            self.update()
        super().mousePressEvent(event)

    def leaveEvent(self, event):
        if not self.pinned:
            self.hover_tick = None
            self.update()
        super().leaveEvent(event)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.plot_rect()
        painter.fillRect(rect, QColor("#f1f5f7"))
        painter.setPen(QColor("#536a77"))
        painter.drawText(
            QRectF(0, 0, self.width(), 23),
            Qt.AlignmentFlag.AlignRight,
            self.formatted_value(self.tick),
        )
        painter.setPen(QColor("#c9d6de"))
        painter.drawLine(rect.bottomLeft(), rect.bottomRight())
        points = [(t, self.value_at(t)) for t in range(self.start_tick, self.tick + 1)]
        known = [(t, y) for t, y in points if y is not None]
        maximum = (
            1.0 if self.passing else max(0.01, max((y for _, y in known), default=0))
        )
        painter.setPen(QColor("#536a77"))
        painter.drawText(
            QRectF(0, rect.top() - 8, 42, 20),
            Qt.AlignmentFlag.AlignLeft,
            "100%" if self.passing else f"{maximum:.2f}",
        )
        painter.drawText(
            QRectF(rect.left(), rect.bottom() + 2, 90, 20),
            Qt.AlignmentFlag.AlignLeft,
            str(self.start_tick),
        )
        painter.drawText(
            QRectF(rect.right() - 100, rect.bottom() + 2, 100, 20),
            Qt.AlignmentFlag.AlignRight,
            f"{self.tick} tick",
        )
        if not known:
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, "No recorded value")
            return

        def position(t, y):
            return QPointF(
                rect.left()
                + (t - self.start_tick)
                / max(1, self.tick - self.start_tick)
                * rect.width(),
                rect.bottom() - y / maximum * rect.height(),
            )

        path = QPainterPath()
        connected = False
        for t, y in points:
            if y is None:
                connected = False
                continue
            point = position(t, y)
            if connected:
                path.lineTo(point)
            else:
                path.moveTo(point)
            connected = True
        color = QColor("#536bab" if self.passing else "#287c74")
        painter.setPen(QPen(color, 2))
        painter.drawPath(path)
        painter.setBrush(color)
        # A single recorded sample is still visible.
        if len(known) == 1:
            painter.drawEllipse(position(*known[0]), 3, 3)
        if (
            self.hover_tick is not None
            and self.start_tick <= self.hover_tick <= self.tick
        ):
            value = self.value_at(self.hover_tick)
            x = position(self.hover_tick, 0).x()
            painter.setPen(QPen(QColor("#758994"), 1, Qt.PenStyle.DashLine))
            painter.drawLine(QPointF(x, rect.top()), QPointF(x, rect.bottom()))
            if value is not None:
                painter.setPen(QPen(color, 1))
                painter.drawEllipse(position(self.hover_tick, value), 4, 4)
            painter.setPen(QColor("#294758"))
            label = f"{self.hover_tick}: {self.formatted_value(self.hover_tick)}"
            painter.drawText(
                rect.adjusted(6, 3, -6, -3),
                Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft,
                label,
            )


class MachineModeChart(QWidget):
    """Cumulative empirical mode frequencies, one event per processing start."""

    COLORS = {"slow": "#90a0bf", "normal": "#648d9e", "fast": "#536bab"}

    def __init__(self, player):
        super().__init__()
        self.player = player
        self.tick = 0
        self.names = {
            m.machine_id: m.name or m.machine_id for m in player.factory.machines
        }
        self.modes = list(
            dict.fromkeys(
                [
                    "slow",
                    "normal",
                    "fast",
                    *[
                        mode
                        for snapshot in player.evidence.mode_counts
                        for counts in snapshot.values()
                        for mode in counts
                    ],
                ]
            )
        )
        self.setFixedHeight(62 + 28 * len(self.names))
        self.setToolTip(
            "Cumulative successful processing starts up to this frame. Each start counts once regardless of duration. No starts: —. Hover this chart for counts."
        )

    def update_tick(self, tick):
        self.tick = tick
        self.update()

    def event(self, event):
        if event.type() == QEvent.Type.ToolTip:
            index = (event.pos().y() - 62) // 28
            if 0 <= index < len(self.names):
                key = list(self.names)[index]
                counts = self.player.evidence.mode_counts[self.tick].get(key, {})
                total = sum(counts.values())
                lines = [f"{self.names[key]} ({key}) · tick {self.tick}"]
                for mode in self.modes:
                    count = counts.get(mode, 0)
                    rate = f"{count / total:.1%}" if total else "—"
                    lines.append(f"{mode.title()}: {rate} · {count} starts")
                lines.append(
                    f"Total: {total} successful starts"
                    if self.player.evidence.has_output_events
                    else "Start events unavailable"
                )
                lines.append("Choice frequency, not processing time share")
                QToolTip.showText(event.globalPos(), "\n".join(lines), self)
                return True
        return super().event(event)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        font = QFont(painter.font())
        font.setPixelSize(13)
        painter.setFont(font)
        painter.setPen(QColor("#294550"))
        painter.drawText(
            QRectF(0, 0, self.width(), 28),
            Qt.AlignmentFlag.AlignVCenter,
            "Machine speed selection",
        )
        x = 0
        for mode in self.modes:
            label = mode.title()
            painter.fillRect(
                QRectF(x, 38, 8, 8), QColor(self.COLORS.get(mode, "#8979a3"))
            )
            painter.setPen(QColor("#526873"))
            painter.drawText(
                QRectF(x + 12, 33, 75, 18), Qt.AlignmentFlag.AlignVCenter, label
            )
            x += max(72, painter.fontMetrics().horizontalAdvance(label) + 20)
        rows = self.player.evidence.mode_counts[self.tick]
        left = (
            min(
                140,
                max(
                    38,
                    max(
                        (
                            painter.fontMetrics().horizontalAdvance(name)
                            for name in self.names.values()
                        ),
                        default=38,
                    ),
                ),
            )
            + 12
        )
        right = self.width() - 8
        for index, (key, name) in enumerate(self.names.items()):
            counts = rows.get(key, {})
            total = sum(counts.values())
            y = 62 + 28 * index
            painter.setPen(QColor("#294550"))
            painter.drawText(
                QRectF(0, y, left - 12, 21),
                Qt.AlignmentFlag.AlignVCenter,
                painter.fontMetrics().elidedText(
                    name,
                    Qt.TextElideMode.ElideRight,
                    left - 12,
                ),
            )
            painter.fillRect(
                QRectF(left, y, max(1, right - left), 20), QColor("#e7edf1")
            )
            if not total:
                painter.drawText(
                    QRectF(left, y, max(1, right - left), 20),
                    Qt.AlignmentFlag.AlignCenter,
                    "—",
                )
                continue
            x = left
            for mode in self.modes:
                count = counts.get(mode, 0)
                if not count:
                    continue
                width = (right - left) * count / total
                painter.fillRect(
                    QRectF(x, y, width, 20), QColor(self.COLORS.get(mode, "#8979a3"))
                )
                label = f"{count / total:.0%}"
                if width >= painter.fontMetrics().horizontalAdvance(label) + 8:
                    painter.setPen(QColor("#183440" if mode == "slow" else "#ffffff"))
                    painter.drawText(
                        QRectF(x, y, width, 20), Qt.AlignmentFlag.AlignCenter, label
                    )
                x += width


class ReplayDashboard(QWidget):
    def __init__(self, player):
        super().__init__()
        self.player = player
        self.setObjectName("runAnalysis")
        self.summary = QLabel()
        self.summary.setWordWrap(True)
        self.summary.setStyleSheet(
            "font-size: 15px; padding: 12px 18px; color: #264854; background: white;"
        )
        self.chart = ThroughputChart(player.evidence)
        self.quality_chart = ThroughputChart(player.evidence, passing=True)
        self.mode_chart = MachineModeChart(player)
        self.window_selector = ReplaySelector()
        self.window_selector.setAccessibleName("Throughput calculation window")
        for ticks in (20, 100, 500):
            self.window_selector.addItem(f"Window: {ticks} ticks", ticks)
        self.window_selector.setCurrentIndex(1)
        self.window_selector.setToolTip(
            "Qualified deliveries in the preceding W ticks / min(W, current tick)."
        )
        self.range_selector = ReplaySelector()
        self.range_selector.setAccessibleName("Chart display range")
        for title, ticks in (
            ("Show: last 100", 100),
            ("Show: last 500", 500),
            ("Show: all to now", None),
        ):
            self.range_selector.addItem(title, ticks)
        self.range_selector.setCurrentIndex(1)
        self.range_selector.setToolTip(
            "Visible history for both trend charts, ending at the displayed tick."
        )
        self.inventory = QLabel()
        self.inventory.setWordWrap(True)
        self.inventory.setStyleSheet("padding: 6px; color: #536a77;")
        self.station_stats = QLabel()
        self.station_stats.setWordWrap(True)
        self.station_stats.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.station_stats.setStyleSheet("padding: 10px; color: #536a77;")
        self.jobs = QListWidget()
        self.jobs.setMinimumHeight(60)
        self.jobs.setToolTip(
            "Recorded in-factory orders. Click to locate the holder on the map."
        )
        self.jobs.itemClicked.connect(self.locate)
        self.events = QListWidget()
        self.events.setToolTip(
            "Latest 200 recorded events through the displayed tick. Double-click to seek. Hidden quality is omitted."
        )
        self.events.itemDoubleClicked.connect(
            lambda item: player.seek(item.data(Qt.ItemDataRole.UserRole))
        )
        performance = QWidget()
        self.plots = QHBoxLayout(performance)
        self.plots.setContentsMargins(10, 10, 10, 8)
        self.plots.setSpacing(24)
        self.plot_panels = []
        for title, chart in (
            ("Throughput", self.chart),
            ("Cumulative output passing rate", self.quality_chart),
        ):
            plot_panel = QWidget()
            column = QVBoxLayout(plot_panel)
            column.setContentsMargins(0, 0, 0, 0)
            column.setSpacing(2)
            heading = QLabel(title)
            heading.setFixedHeight(26)
            column.addWidget(heading)
            if chart is self.chart:
                controls = QHBoxLayout()
                controls.setSpacing(6)
                controls.addWidget(self.window_selector)
                controls.addWidget(self.range_selector)
                column.addLayout(controls)
            else:
                self.range_label = QLabel("Showing last 500 ticks")
                self.range_label.setStyleSheet("color: #536a77; font-size: 12px;")
                self.range_label.setFixedHeight(36)
                column.addWidget(self.range_label)
            column.addWidget(chart, 1)
            self.plots.addWidget(plot_panel, 1)
            self.plot_panels.append(plot_panel)
        resources = QWidget()
        self.resource_layout = QHBoxLayout(resources)
        self.mode_scroll = QScrollArea()
        self.mode_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.mode_scroll.setWidgetResizable(True)
        self.mode_scroll.setWidget(self.mode_chart)
        self.mode_scroll.setMinimumWidth(280)
        self.mode_scroll.setStyleSheet(
            "QScrollArea { background: transparent; border: none; }"
        )
        self.mode_chart.setAutoFillBackground(False)
        self.mode_scroll.viewport().setAutoFillBackground(False)
        self.resource_layout.addWidget(self.mode_scroll, 1)
        self.resource_layout.addWidget(self.station_stats, 1)
        orders = QWidget()
        inventory_column = QVBoxLayout(orders)
        inventory_column.addWidget(self.inventory)
        inventory_column.addWidget(QLabel("In-factory orders · click to locate"))
        inventory_column.addWidget(self.jobs, 1)
        self.overview = QWidget()
        self.overview_layout = QHBoxLayout(self.overview)
        self.overview_layout.setSpacing(24)
        self.overview_layout.setContentsMargins(10, 10, 10, 8)
        self.body = QStackedWidget()
        for page in (self.overview, performance, resources, orders, self.events):
            self.body.addWidget(page)
        self.tabs = QTabBar()
        self.tabs.setObjectName("analysisTabs")
        self.tabs.setExpanding(False)
        self.tabs.setDrawBase(False)
        for label in ("Overview", "Output", "Resources", "Orders", "Events"):
            self.tabs.addTab(label)
        self.tabs.currentChanged.connect(self.set_analysis_view)
        self.collapse = QToolButton()
        self.collapse.setArrowType(Qt.ArrowType.DownArrow)
        self.collapse.setFixedHeight(24)
        self.collapse.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self.collapse.setToolTip("Collapse analysis")
        self.collapse.setAccessibleName("Collapse analysis")
        self.collapse.setStyleSheet(
            "QToolButton { border: none; border-top: 1px solid #dce4e9; background: #f3f6f8; padding: 0; } QToolButton:hover { background: #deedf5; }"
        )
        self.collapse.setCheckable(True)
        self.collapse.toggled.connect(self.set_collapsed)
        self.header_widget = QWidget()
        header = QHBoxLayout(self.header_widget)
        header.setContentsMargins(12, 0, 12, 0)
        header.addWidget(QLabel("RUN ANALYSIS"))
        header.addWidget(self.tabs)
        header.addStretch()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 4)
        layout.setSpacing(0)
        layout.addWidget(self.collapse)
        layout.addWidget(self.header_widget)
        layout.addWidget(self.body, 1)
        self.setMinimumHeight(245)
        self.row = None
        self.window_selector.currentIndexChanged.connect(self.change_chart_settings)
        self.range_selector.currentIndexChanged.connect(self.change_chart_settings)
        self.set_analysis_view(0)

    def set_analysis_view(self, index):
        # Keep one instance of each chart; navigation never changes playback.
        for panel in self.plot_panels:
            (self.overview_layout if index == 0 else self.plots).addWidget(panel, 1)
            panel.show()
        if index == 0:
            self.overview_layout.addWidget(self.mode_scroll, 1)
        else:
            self.resource_layout.insertWidget(0, self.mode_scroll, 1)
        self.mode_scroll.show()
        self.body.setCurrentIndex(index)

    def set_collapsed(self, collapsed):
        splitter = self.parentWidget()
        if collapsed and isinstance(splitter, QSplitter):
            self.expanded_sizes = splitter.sizes()
        self.body.setVisible(not collapsed)
        self.header_widget.setVisible(not collapsed)
        self.collapse.setArrowType(
            Qt.ArrowType.UpArrow if collapsed else Qt.ArrowType.DownArrow
        )
        label = "Expand analysis" if collapsed else "Collapse analysis"
        self.collapse.setToolTip(label)
        self.collapse.setAccessibleName(label)
        self.setMinimumHeight(0 if collapsed else 245)
        self.setMaximumHeight(28 if collapsed else 16777215)
        if isinstance(splitter, QSplitter):
            if collapsed:
                sizes = splitter.sizes()
                index = splitter.indexOf(self)
                sizes[index - 1] += sizes[index] - 28
                sizes[index] = 28
                splitter.setSizes(sizes)
            elif hasattr(self, "expanded_sizes"):
                splitter.setSizes(self.expanded_sizes)

    def change_chart_settings(self):
        self.chart.window = self.window_selector.currentData()
        visible_window = self.range_selector.currentData()
        for chart in (self.chart, self.quality_chart):
            chart.visible_window = visible_window
            chart.hover_tick = None
            chart.pinned = False
            chart.update()
        self.range_label.setText(
            f"Showing last {visible_window} ticks"
            if visible_window
            else "Showing all recorded ticks to now"
        )
        if self.row:
            self.update_row(self.row)

    def locate(self, item):
        owner = item.data(Qt.ItemDataRole.UserRole)
        self.player.workspace.select_resource(owner)
        if owner in self.player.scene.entity_items:
            self.player.view.centerOn(self.player.scene.entity_items[owner])

    def update_row(self, row):
        self.row = row
        state, tick = row["state"], row["tick"]
        self.mode_chart.update_tick(tick)
        self.player.state_layer.output_window = self.chart.window
        self.player.state_layer.update()
        for buffer in self.player.factory.buffers:
            if buffer.role == "system_input":
                evidence = self.player.evidence
                waiting = evidence.input_waiting(buffer.buffer_id, state)
                progress = evidence.input_progress(buffer.buffer_id, tick)
                arrival = (
                    f"Next recorded release: +{progress[2]} at tick {tick + progress[1]}\nRemaining: {progress[1]} / {progress[0]} ticks"
                    if progress
                    else "No further arrival recorded for this input"
                )
                self.player.scene.entity_items[buffer.buffer_id].setToolTip(
                    f"{buffer.name} · {buffer.buffer_id}\nOutside waiting: {waiting if waiting is not None else 'Unavailable'}\n{arrival}"
                )
            if buffer.role == "system_output":
                count, rate, speed = self.player.evidence.output_metrics(
                    buffer.buffer_id, tick, self.chart.window
                )
                rate_label = f"{rate:.1%}" if rate is not None else "—"
                speed_label = f"{speed:.3f}" if speed is not None else "—"
                self.player.scene.entity_items[buffer.buffer_id].setToolTip(
                    f"{buffer.name} · {buffer.buffer_id}\nQualified deliveries: {count if count is not None else 'Unavailable'}\nOutput passing rate: {rate_label}\nThroughput: {speed_label} jobs/tick · W={self.chart.window}"
                )
        good, attempts = qualified(state), submitted(state)
        rate = f"{good / attempts:.1%}" if attempts else "—"
        throughput = self.player.evidence.throughput(tick, self.chart.window)
        speed = f"{throughput:.3f} jobs/tick" if throughput is not None else "—"
        self.summary.setText(
            f"Qualified deliveries   {good}     |     Output passing rate   {rate}     |     Throughput   {speed}"
        )
        self.events.clear()
        from bisect import bisect_right

        recorded = self.player.evidence.recorded_events
        end = bisect_right(recorded, tick, key=lambda event: event[0])
        for when, kind, identities in reversed(recorded[max(0, end - 200) : end]):
            self.events.addItem(f"{when}   {kind}   {identities}")
            self.events.item(self.events.count() - 1).setData(
                Qt.ItemDataRole.UserRole, when
            )
        if not self.events.count():
            self.events.addItem("No recorded events through this tick")
            self.events.item(0).setFlags(Qt.ItemFlag.NoItemFlags)
        self.summary.setToolTip(
            "Output passing rate = qualified deliveries / output submissions. Inspection results are separate."
        )
        count = outside_count(state)
        scrap = (
            state["metrics"].get("pre_output_scrap", 0)
            if "completed" in state
            else state["metrics"].get("preoutput_scrap")
        )
        inputs = {
            b.buffer_id for b in self.player.factory.buffers if b.role == "system_input"
        }
        outputs = {
            b.buffer_id
            for b in self.player.factory.buffers
            if b.role == "system_output"
        }
        input_count = sum(
            len(jobs)
            for owner in inputs
            for jobs in state["storage"].get(owner, {}).values()
        )
        present = [
            (jid, job)
            for jid, job in state["jobs"].items()
            if job.get("location") in self.player.scene.entity_items
            and job.get("location") not in outputs
        ]
        wip = state["metrics"].get("wip", len(present))
        self.inventory.setText(
            f"Outside {count if count is not None else 'Unavailable'} · Input {input_count}\nIn factory {wip} · Disposed {format(scrap, 'g') if scrap is not None else 'Unavailable'}"
        )
        previous = self.jobs.currentItem().text() if self.jobs.currentItem() else None
        self.jobs.clear()
        for jid, job in present:
            owner = job["location"]
            resource = self.player.workspace.resources.get(owner)
            self.jobs.addItem(
                f"Order {order_label(jid)}  →  {getattr(resource, 'name', owner)}"
            )
            item = self.jobs.item(self.jobs.count() - 1)
            item.setData(Qt.ItemDataRole.UserRole, owner)
            item.setToolTip(f"{jid} · {owner}")
            if item.text() == previous:
                self.jobs.setCurrentItem(item)
        station_lines = ["Inspection stations"]
        for station in self.player.factory.inspection_stations:
            key = station.inspection_station_id
            busy = self.player.evidence.station_busy[tick].get(key, 0)
            inspected = self.player.evidence.inspected[tick].get(key, 0)
            known = "historical_frame" not in row or bool(
                getattr(self.player.playback, "events", None)
            )
            label = (
                f"{inspected} inspected · {busy / tick:.1%} busy"
                if known and tick
                else "No elapsed ticks"
                if known
                else "Statistics unavailable"
            )
            station_lines.append(f"{station.name}: {label}")
        self.station_stats.setText("\n\n".join(station_lines))
        for chart in (self.chart, self.quality_chart):
            chart.update_tick(tick)
