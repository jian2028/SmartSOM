"""Studio-style, read-only workspace around the existing replay canvas."""

from PySide6.QtCore import QEvent, QPoint, QSize, Qt, QTimer
from PySide6.QtGui import (
    QAction,
    QActionGroup,
    QColor,
    QIcon,
    QKeySequence,
    QPainter,
    QPen,
    QPixmap,
    QShortcut,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMenu,
    QSizePolicy,
    QSplitter,
    QStyle,
    QTabWidget,
    QTextBrowser,
    QToolBar,
    QToolButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from smartsom.domain.factory_design import entity_id
from smartsom.studio.items import COLORS
from smartsom.studio.performance_panel import TaskPerformancePanel
from smartsom.studio.properties import PropertyTree, type_label
from smartsom.studio.replay_dashboard import ReplayDashboard
from smartsom.studio.replay_decisions import DecisionInspector
from smartsom.studio.replay_evidence import (
    displayed_quality,
    movement_conflicts,
    outside_count,
    resource_conflicts,
    scenario_reference,
)
from smartsom.studio.replay_inspector import RuntimeInspector
from smartsom.studio.replay_model import ReplayIndex
from smartsom.studio.replay_timeline import EventTimeline
from smartsom.studio.workspace_style import (
    REPLAY_STYLE,
    STYLE,
    ReplaySelector,
    apply_light_palette,
    apply_workspace_theme,
    panel,
)


class ReplayWorkspace(QWidget):
    def __init__(self, player):
        super().__init__(player)
        self.player = player
        self.selected_id = None
        self.selected_job = None
        self.dark = False
        self.replay_index = ReplayIndex(player.playback, player.factory)
        player.state_layer.replay_index = self.replay_index
        self.row = None
        self.resources = {None: player.factory}
        self.tree_items = {}
        self.selecting = False
        self.mode = "standard"
        self.narrow = False
        self.drawer_open = False
        self.responsive_ready = False
        player.resize(1420, 860)
        player.setMinimumSize(1000, 620)
        apply_light_palette(player)
        player.setStyleSheet(STYLE + REPLAY_STYLE)
        self.resource_panel, left = panel("RESOURCES")
        self.filter = QLineEdit()
        self.filter.setObjectName("resourceFilter")
        self.filter.setPlaceholderText("Find a resource…")
        self.filter.setClearButtonEnabled(True)
        search = QWidget()
        search_layout = QHBoxLayout(search)
        search_layout.setContentsMargins(10, 0, 10, 10)
        search_layout.addWidget(self.filter)
        left.addWidget(search)
        self.resource_tree = QTreeWidget()
        self.resource_tree.setObjectName("resourceTree")
        self.resource_tree.setHeaderHidden(True)
        self.resource_tree.setIndentation(14)
        self.resource_tree.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        left.addWidget(self.resource_tree, 1)
        self._populate_resources()
        self.resource_tree.currentItemChanged.connect(self._tree_selected)
        self.filter.textChanged.connect(self._filter_resources)
        player.scene.entity_selected.connect(self.select_resource)

        self.property_panel, right = panel("REPLAY INSPECTOR")
        self.property_panel.setObjectName("replayInspectorPanel")
        self.drawer_close = QToolButton()
        self.drawer_close.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_TitleBarCloseButton)
        )
        self.drawer_close.setAccessibleName("Close inspector")
        self.drawer_close.setToolTip("Close inspector · Escape")
        self.drawer_close.clicked.connect(lambda: self.set_drawer(False))
        self.drawer_close.hide()
        close_row = QHBoxLayout()
        close_row.addStretch()
        close_row.addWidget(self.drawer_close)
        right.insertLayout(0, close_row)
        self.title = QLabel(player.factory.name)
        self.title.setObjectName("propertyTitle")
        self.title.setWordWrap(True)
        self.title.setStyleSheet(
            "font-size: 17px; font-weight: 600; padding: 2px 12px 4px;"
        )
        self.subtitle = QLabel("Factory overview · read-only")
        self.subtitle.setWordWrap(True)
        self.subtitle.setStyleSheet("color: #536a77; padding: 0 12px 12px;")
        self.dashboard = ReplayDashboard(player)
        self.timeline = EventTimeline(self.replay_index)
        self.timeline.seek_requested.connect(player.seek)
        self.timeline.events_selected.connect(self.show_events)
        self.dashboard.body.addWidget(self.timeline)
        self.dashboard.tabs.addTab("Timeline")
        self.dashboard.tabs.moveTab(5, 0)
        self.dashboard.tabs.currentChanged.disconnect()
        self.dashboard.tabs.currentChanged.connect(self.change_analysis)
        self.dashboard.tabs.setCurrentIndex(0)
        self.change_analysis(0)
        self.back_global = QToolButton()
        self.back_global.setText("← Global performance")
        self.back_global.clicked.connect(lambda: self.select_resource(None))
        right.addWidget(self.back_global)
        right.addWidget(self.title)
        right.addWidget(self.subtitle)
        self.performance_panel = TaskPerformancePanel(
            player.evidence, scenario_reference(player.playback)
        )
        right.addWidget(self.performance_panel)
        self.inspector = QTabWidget()
        self.inspector.setObjectName("replayInspector")
        self.inspector.setUsesScrollButtons(True)
        self.inspector.setElideMode(Qt.TextElideMode.ElideNone)
        self.inspector.tabBar().setExpanding(False)
        self.runtime_inspector = RuntimeInspector()
        self.design_tree = PropertyTree()
        self.inspector.addTab(self.runtime_inspector, "State")
        self.decisions = DecisionInspector(player.playback)
        self.related_events = QListWidget()
        self.related_events.setWordWrap(True)
        self.related_events.itemDoubleClicked.connect(
            lambda item: player.seek(item.data(Qt.ItemDataRole.UserRole))
        )
        self.job_detail = QTextBrowser()
        self.inspector.addTab(self.decisions, "Decisions")
        self.inspector.addTab(self.related_events, "Events")
        self.inspector.addTab(self.job_detail, "Orders")
        self.inspector.addTab(self.design_tree, "Design")
        self.inspector.addTab(player.details, "Raw frame")
        right.addWidget(self.inspector, 1)
        right.addStretch()

        self.map_panel = QWidget()
        map_layout = QVBoxLayout(self.map_panel)
        map_layout.setContentsMargins(0, 0, 0, 0)
        map_layout.setSpacing(0)
        heading = QLabel(f"{player.factory.name} · Replay")
        heading.setStyleSheet("background: white; padding: 5px 8px;")
        heading.setTextFormat(Qt.TextFormat.PlainText)
        self.map_tools = QToolBar("Map tools")
        self.map_tools.setObjectName("mapTools")
        self.map_tools.setMovable(False)
        self.map_tools.addWidget(heading)
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.map_tools.addWidget(spacer)
        self.resources_action = QAction("Resources", self)
        self.resources_action.setCheckable(True)
        self.resources_action.toggled.connect(self.set_resources_visible)
        self.view_controls = QToolBar("View controls", player.view)
        self.view_controls.setObjectName("replayViewControls")
        self.view_controls.setMovable(False)
        self.view_controls.setStyleSheet(
            "QToolBar { background: #ffffff; border: 1px solid #c9d6de; border-radius: 6px; padding: 2px; }"
        )
        self.map_viewport = player.view.viewport()
        self.map_viewport.installEventFilter(self)
        fit = self.view_controls.addAction("Fit map")
        fit.setObjectName("fitMapAction")
        pixmap = QPixmap(40, 40)
        pixmap.setDevicePixelRatio(2)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QPen(QColor("#426577"), 1.6))
        for x, y, dx, dy in (
            (3, 3, 1, 1),
            (17, 3, -1, 1),
            (3, 17, 1, -1),
            (17, 17, -1, -1),
        ):
            painter.drawLine(x, y, x + 4 * dx, y)
            painter.drawLine(x, y, x, y + 4 * dy)
        painter.end()
        fit.setIcon(QIcon(pixmap))
        fit.setToolTip("Fit map · Show the entire factory")
        self.view_controls.widgetForAction(fit).setToolButtonStyle(
            Qt.ToolButtonStyle.ToolButtonIconOnly
        )
        fit.triggered.connect(player.view.fit_map)
        self.screenshot_action = self.view_controls.addAction("Screenshot")
        self.screenshot_action.setToolTip("Copy the current map view to the clipboard")
        self.screenshot_action.triggered.connect(self.copy_map_screenshot)
        self.inspector_action = self.map_tools.addAction("Inspector")
        self.inspector_action.setCheckable(True)
        self.inspector_action.toggled.connect(self.set_drawer)
        self.inspector_action.setVisible(False)
        map_layout.addWidget(self.map_tools)
        map_layout.addWidget(player.view, 1)
        self.outside = QLabel()
        self.outside.setWordWrap(True)
        self.outside.setStyleSheet(
            "background: #edf3f7; color: #335365; padding: 10px 16px;"
        )
        self.outside.setTextFormat(Qt.TextFormat.PlainText)
        map_layout.addWidget(self.outside)
        hint = QLabel(
            "   Click to inspect  ·  Wheel to zoom  ·  Middle/Space-drag to pan"
        )
        hint.setStyleSheet("color: #536a77; padding: 6px;")
        player.view.setToolTip(
            hint.text()
            + "\nAGV identity rings; red ring = recorded conflict. Dashed destinations show AGV IDs. Trail is recorded history, not a plan."
        )
        hint.deleteLater()
        self.canvas_splitter = QSplitter(Qt.Orientation.Vertical)
        self.canvas_splitter.addWidget(self.map_panel)
        self.canvas_splitter.setCollapsible(0, False)
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.setObjectName("workspaceSplitter")
        self.splitter.installEventFilter(self)
        for widget in (self.resource_panel, self.canvas_splitter, self.property_panel):
            self.splitter.addWidget(widget)
        self.splitter.setSizes([180, 950, 410])
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setCollapsible(1, False)
        self.resource_panel.setMinimumWidth(160)
        self.property_panel.setMinimumWidth(380)
        self.property_panel.setMaximumWidth(450)
        self.end_label = QLabel(f"{player.playback.last_tick} ticks")
        self.analysis_splitter = QSplitter(Qt.Orientation.Vertical)
        self.resource_rail = QToolButton(self.map_panel)
        self.resource_rail.setObjectName("resourceRail")
        self.resource_rail.setFixedSize(26, 76)
        self.map_panel.installEventFilter(self)
        self.resource_rail.clicked.connect(self.resources_action.toggle)
        self.set_resources_visible(False)
        # The analysis pane belongs below the map, not below the inspector.
        self.analysis_splitter.addWidget(self.map_panel)
        self.analysis_splitter.addWidget(self.dashboard)
        self.canvas_splitter.addWidget(self.analysis_splitter)
        self.analysis_splitter.setCollapsible(0, False)
        self.analysis_splitter.setCollapsible(1, False)
        self.analysis_splitter.setStretchFactor(0, 1)
        self.analysis_splitter.setSizes([535, 210])
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(1)
        layout.addWidget(self.dashboard.summary)
        layout.addWidget(self.splitter, 1)
        self._toolbar()
        layout.insertWidget(1, self.playback_controls)
        self.dashboard.window_selector.currentIndexChanged.connect(
            self.refresh_inspector
        )
        self.resource_panel.hide()
        self.zoom = QLabel("100%")
        self.zoom.setObjectName("zoomLabel")
        player.statusBar().addPermanentWidget(self.zoom)
        player.view.zoom_changed.connect(
            lambda value: self.zoom.setText(f"{value:.0%}")
        )
        self.select_resource(None)
        self.responsive_ready = True
        self.escape_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Escape), self)
        self.escape_shortcut.activated.connect(self.clear_selection)
        QTimer.singleShot(0, self.sync_responsive)

    def set_resources_visible(self, visible):
        self.resource_panel.setVisible(visible)
        self.resources_action.blockSignals(True)
        self.resources_action.setChecked(visible)
        self.resources_action.blockSignals(False)
        self.resource_rail.setArrowType(
            Qt.ArrowType.LeftArrow if visible else Qt.ArrowType.RightArrow
        )
        label = "Collapse resources" if visible else "Expand resources"
        self.resource_rail.setToolTip(label)
        self.resource_rail.setAccessibleName(label)
        self.position_resource_handle()

    def position_resource_handle(self):
        self.resource_rail.move(
            0, max(0, (self.map_panel.height() - self.resource_rail.height()) // 2)
        )
        self.resource_rail.raise_()

    def _toolbar(self):
        player = self.player
        toolbar = QToolBar("Replay workspace", player)
        toolbar.setObjectName("mainToolbar")
        toolbar.setMovable(False)
        player.addToolBar(toolbar)
        brand = QLabel(
            "SmartSOM  <span style='font-weight:400;color:#738794'>Replay</span>"
        )
        brand.setStyleSheet("font-size: 18px; font-weight: 700; padding-right: 20px;")
        toolbar.addWidget(brand)
        group = QActionGroup(player)
        group.setExclusive(True)
        self.layouts = {}
        for mode, label in (("standard", "Standard"), ("focus", "Focus map")):
            action = QAction(label, player)
            action.setObjectName(f"{mode}LayoutAction")
            action.setCheckable(True)
            action.setChecked(mode == "standard")
            action.triggered.connect(lambda checked, m=mode: self.set_layout(m))
            group.addAction(action)
            toolbar.addAction(action)
            self.layouts[mode] = action
        toolbar.addSeparator()
        more = QToolButton()
        more.setText("More")
        more.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        menu = QMenu(more)
        layers = menu.addMenu("Layers")
        self.ports_action = layers.addAction("Interaction points")
        self.ports_action.setObjectName("portsLayerAction")
        self.ports_action.setCheckable(True)
        self.view_controls.addAction(self.ports_action)
        self.ports_action.setToolTip("Show / hide interaction points")
        for action, kind in (
            (self.screenshot_action, "camera"),
            (self.ports_action, "ports"),
        ):
            pixmap = QPixmap(40, 40)
            pixmap.setDevicePixelRatio(2)
            pixmap.fill(Qt.GlobalColor.transparent)
            painter = QPainter(pixmap)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            pen = QPen(QColor("#64889b"), 1.6)
            if kind == "ports":
                pen.setStyle(Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.drawRect(
                3, 5 if kind == "camera" else 3, 14, 12 if kind == "camera" else 14
            )
            if kind == "camera":
                painter.drawEllipse(7, 8, 6, 6)
                painter.drawLine(6, 3, 11, 3)
            painter.end()
            action.setIcon(QIcon(pixmap))
        for action in self.view_controls.actions():
            button = self.view_controls.widgetForAction(action)
            button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
            button.setIconSize(QSize(20, 20))
            button.setFixedSize(34, 34)
            button.setAccessibleName(action.text())
        self.view_controls.adjustSize()
        self.position_view_controls()
        self.ports_action.setChecked(player.scene.ports_visible)
        self.ports_action.toggled.connect(
            lambda visible: player.scene.set_layer("ports", visible)
        )
        menu.addAction("Raw frame", self.show_raw_frame)
        presentation = menu.addMenu("Presentation")
        preview = presentation.addAction("Charging preview")
        preview.setCheckable(True)
        preview.setToolTip(
            "Presentation demo only; this recording has no charging execution."
        )
        preview.setEnabled(bool(player.factory.chargers))
        preview.toggled.connect(self._charging_preview)
        more.setMenu(menu)
        toolbar.addWidget(more)
        toolbar.addSeparator()
        label = QLabel("Read-only playback")
        toolbar.addWidget(label)
        self.theme_action = toolbar.addAction("Dark theme")
        self.theme_action.setCheckable(True)
        self.theme_action.toggled.connect(self.set_theme)
        source = str(
            getattr(
                player.playback, "root", getattr(player.playback, "source", "Recording")
            )
        )
        label.setToolTip(source)
        controls = QToolBar("Playback controls", player)
        controls.setObjectName("playbackToolbar")
        controls.setMovable(False)
        self.playback_controls = controls
        self.canvas_splitter.insertWidget(0, controls)
        controls.setMaximumHeight(48)
        for widget in (
            player.back_button,
            player.pause_button,
            player.step_button,
            player.speed,
        ):
            controls.addWidget(widget)
        controls.addSeparator()
        self.marker_selector = ReplaySelector()
        self.marker_selector.setAccessibleName("Jump marker category")
        for key, label in (
            ("movement", "Movement conflict"),
            ("delivery", "Qualified delivery"),
            ("resource", "Pickup/drop conflict"),
            ("scrap", "Scrap"),
            ("inspection", "Inspection"),
            ("outage", "Machine outage"),
            ("all", "All events"),
        ):
            self.marker_selector.addItem(f"Jump: {label}", key)
        self.marker_selector.setToolTip(
            "Category for the jump arrows; recorded markers only."
        )
        self.marker_selector.currentIndexChanged.connect(lambda _: self.sync_markers())
        self.map_tools.addWidget(self.marker_selector)
        self.jump_back = QToolButton()
        self.jump_back.setText("◀|")
        self.jump_back.setAccessibleName("Previous marker")
        self.jump_back.setToolTip("Seek to the previous marker of this category")
        self.jump_back.clicked.connect(lambda: self.jump(-1))
        self.jump_forward = QToolButton()
        self.jump_forward.setText("|▶")
        self.jump_forward.setAccessibleName("Next marker")
        self.jump_forward.setToolTip("Seek to the next marker of this category")
        self.jump_forward.clicked.connect(lambda: self.jump(1))
        controls.addWidget(self.jump_back)
        controls.addWidget(self.jump_forward)
        self.marker_label = QLabel()
        self.marker_label.setStyleSheet("color: #536a77; padding: 0 8px;")
        self.marker_label.setVisible(False)
        self.map_tools.addWidget(self.marker_label)
        self.object_filter = ReplaySelector()
        self.object_filter.addItems(["All objects", "Selected object"])
        self.object_filter.setAccessibleName("Event object filter")
        self.object_filter.currentIndexChanged.connect(self.sync_markers)
        self.map_tools.addWidget(self.object_filter)
        self.trail_selector = ReplaySelector()
        for title, length in (
            ("Trail: off", 0),
            ("Trail: 12", 12),
            ("Trail: 30", 30),
            ("Trail: 60", 60),
        ):
            self.trail_selector.addItem(title, length)
        self.trail_selector.setCurrentIndex(1)
        self.trail_selector.currentIndexChanged.connect(self.refresh_inspector)
        self.map_tools.addWidget(self.trail_selector)
        fit_time = self.map_tools.addAction("Fit time")
        fit_time.triggered.connect(self.timeline.fit)
        controls.addSeparator()
        controls.addWidget(player.tick_label)
        player.slider.setMinimumWidth(60)
        controls.addWidget(player.slider)
        controls.addWidget(self.end_label)

    def marker_ticks(self):
        category = self.marker_selector.currentData()
        return sorted(
            {
                e.tick
                for e in self.replay_index.filtered(
                    self.filter_owner(), None if category == "all" else category
                )
            }
        )

    def jump(self, direction):
        """Seek to the neighbouring recorded marker, without wrapping around."""
        ticks = self.marker_ticks()
        current = self.row["tick"] if self.row else 0
        candidates = [
            t for t in ticks if (t > current if direction > 0 else t < current)
        ]
        if candidates:
            self.player.seek(min(candidates) if direction > 0 else max(candidates))

    def sync_markers(self):
        """Keep the jump arrows and their count honest about recorded evidence."""
        self.timeline.set_frame(
            self.row["tick"] if self.row else 0, self.filter_owner()
        )
        ticks = self.marker_ticks()
        current = self.row["tick"] if self.row else 0
        self.jump_back.setEnabled(any(t < current for t in ticks))
        self.jump_forward.setEnabled(any(t > current for t in ticks))
        seen = sum(1 for t in ticks if t <= current)
        self.marker_label.setText(f"{seen}/{len(ticks)}" if ticks else "none recorded")

    def _charging_preview(self, enabled):
        self.player.state_layer.charging_preview = (
            {next(iter(self.player.state_layer.state["agvs"]))}
            if enabled and self.player.state_layer.state.get("agvs")
            else set()
        )
        self.player.state_layer.update()

    def set_layout(self, mode):
        if mode == self.mode:
            return
        self.mode = mode
        if mode == "focus":
            self.standard_sizes = self.splitter.sizes()
            self.resources_were_visible = not self.resource_panel.isHidden()
            self.set_resources_visible(False)
            self.property_panel.hide()
            self.dashboard.hide()
        else:
            self.property_panel.setVisible(not self.narrow or self.drawer_open)
            self.dashboard.show()
            self.set_resources_visible(self.resources_were_visible)
            self.splitter.setSizes(self.standard_sizes)
        self.layouts[mode].setChecked(True)

    def _populate_resources(self):
        root = QTreeWidgetItem(self.resource_tree, [self.player.factory.name])
        root.setExpanded(True)
        self.tree_items[None] = root
        collections = (
            ("Machines", "machine", "machines"),
            ("Buffers", "buffer", "buffers"),
            ("Inspection", "inspection", "inspection_stations"),
            ("Scrap bins", "scrap", "scrap_bins"),
            ("Chargers", "charger", "chargers"),
            ("Ports", "port", "ports"),
            ("AGVs", "agv", "agvs"),
        )
        for label, kind, field in collections:
            resources = getattr(self.player.factory, field)
            group = QTreeWidgetItem(root, [f"{label}  {len(resources)}"])
            group.setExpanded(kind in ("machine", "agv"))
            group.setFlags(group.flags() & ~Qt.ItemFlag.ItemIsSelectable)
            for resource in resources:
                key = entity_id(resource)
                item = QTreeWidgetItem(group, [resource.name])
                item.setData(0, Qt.ItemDataRole.UserRole, key)
                item.setToolTip(0, key)
                item.setForeground(0, QColor(COLORS[kind][1]))
                self.tree_items[key] = item
                self.resources[key] = resource

    def _filter_resources(self, text):
        query = text.casefold().strip()
        for key, item in self.tree_items.items():
            if key is not None:
                item.setHidden(query not in f"{item.text(0)} {key}".casefold())
        root = self.tree_items[None]
        for index in range(root.childCount()):
            group = root.child(index)
            group.setHidden(
                all(group.child(n).isHidden() for n in range(group.childCount()))
            )
            if query:
                group.setExpanded(True)

    def _tree_selected(self, item, previous):
        if item:
            self.select_resource(item.data(0, Qt.ItemDataRole.UserRole))

    def select_resource(self, key):
        if self.selecting:
            return
        self.selected_job = None
        self.selecting = True
        try:
            selected = key if key in self.resources else None
            self.selected_id = selected
            resource = self.resources[self.selected_id]
            self.player.scene.select_entity(self.selected_id)
            item = self.tree_items[self.selected_id]
            self.resource_tree.setCurrentItem(item)
            parent = item.parent()
            while parent:
                parent.setExpanded(True)
                parent = parent.parent()
            self.title.setText(
                resource.name if self.selected_id else "Global performance"
            )
            identity = self.selected_id or self.player.factory.factory_id
            if (
                identity.startswith("machine_")
                and identity.removeprefix("machine_").isdigit()
            ):
                identity = f"M{int(identity.removeprefix('machine_'))} · {identity}"
            self.subtitle.setText(f"{type_label(resource)} · {identity}")
            self.design_tree.show_resource(resource)
            self.performance_panel.setVisible(selected is None)
            self.back_global.setVisible(selected is not None)
            self.inspector.setVisible(selected is not None)
            self._update_state()
            self.sync_markers()
            if self.narrow:
                self.set_drawer(selected is not None)
        finally:
            self.selecting = False

    def update_row(self, row):
        self.row = row
        self.dashboard.update_row(row)
        state, tick = row["state"], row["tick"]
        self.performance_panel.update_tick(tick, self.dashboard.chart.window)
        self.sync_markers()
        count = outside_count(state)
        upcoming = self.player.evidence.next_arrival(tick)
        arrival = (
            f"Next new demand: tick {upcoming[0]} · in {upcoming[0] - tick} ticks · {upcoming[1]} orders"
            if upcoming
            else "No further arrivals in recording"
        )
        self.outside.setText(
            f"Outside waiting: {count if count is not None else 'Unavailable'}  ·  {arrival} · From recording"
        )
        queue = state.get("queue")
        self.outside.setToolTip(
            "\n".join(str(jid) for jid in queue)
            if queue is not None
            else "Historical recording stores the outside count only; job identities unavailable."
        )
        self._update_state()

    def _state_values(self):
        row, key = self.row, self.selected_id
        if row is None:
            return {"status": "Waiting for a recorded frame"}
        state = row.get("historical_frame", row["state"])
        values = {"recorded_tick": row["tick"]}
        if key is None:
            return {
                "selection": "Click a machine, AGV, buffer or job holder to inspect its recorded state"
            }
        resource = self.resources[key]
        current = "completed" in row["state"]
        tick = row["tick"]
        if hasattr(resource, "scrap_bin_id"):
            values["cumulative_disposed"] = (
                row["state"]["metrics"].get(f"scrap:{key}", 0)
                if current
                else self.player.evidence.disposals[tick].get(key, 0)
                if getattr(self.player.playback, "events", None)
                else "Unavailable"
            )
        if getattr(resource, "role", None) == "system_output":
            slots = row["state"]["storage"].get(key, {})
            values["qualified_deliveries"] = (
                sum(
                    row["state"]["jobs"].get(jid, {}).get("quality") == "PASS"
                    for jobs in slots.values()
                    for jid in jobs
                )
                if current
                else self.player.evidence.outputs[tick].get(key, 0)
                if getattr(self.player.playback, "events", None)
                else "Unavailable"
            )
            values["resident_inventory"] = (
                sum(len(jobs) for jobs in slots.values())
                if current
                else "Unavailable (historical output consumes jobs)"
            )
        if getattr(resource, "role", None) == "system_input":
            values["outside_waiting"] = self.player.evidence.input_waiting(
                key, row["state"]
            )
        if getattr(resource, "role", None) == "system_output":
            _, rate, throughput = self.player.evidence.output_metrics(
                key, tick, self.dashboard.chart.window
            )
            values["output_passing_rate"] = f"{rate:.1%}" if rate is not None else "—"
            values["throughput"] = (
                f"{throughput:.3f} jobs/tick" if throughput is not None else "—"
            )
        if key in row["state"]["stations"]:
            tick = row["tick"]
            evidence = self.player.evidence
            has_events = "historical_frame" not in row or bool(
                getattr(self.player.playback, "events", None)
            )
            values["inspection_statistics"] = {
                "inspected_jobs": evidence.inspected[tick].get(key, 0)
                if has_events
                else "Unavailable",
                "busy_ticks": evidence.station_busy[tick].get(key, 0)
                if has_events
                else "Unavailable",
                "utilization": f"{evidence.station_busy[tick].get(key, 0) / tick:.1%}"
                if tick and has_events
                else "—",
            }
        if key in row["state"]["agvs"]:
            values["movement_conflict"] = key in movement_conflicts(row)
            values["resource_conflict"] = key in resource_conflicts(row)
            if "historical_frame" not in row:
                values["action"] = dict(
                    row.get("actions", {}).get(
                        "movers", row.get("actions", {}).get("agvs", [])
                    )
                ).get(key, "Unavailable")
                values["rejection"] = row.get("rejections", {}).get(f"agv:{key}")
        for section in (
            "agvs",
            "machines",
            "stations",
            "inspections",
            "storage",
            "stores",
            "rankings",
        ):
            if key in state.get(section, {}):
                values[section] = state[section][key]
        if key in row["state"]["machines"]:
            values["display_mode"] = (
                row["state"]["machines"][key].get("mode") or "Unavailable"
            )
        jobs = {
            jid: {field: value for field, value in job.items() if field != "defective"}
            for jid, job in state.get("jobs", {}).items()
            if job.get("holder", job.get("location")) == key
        }
        if jobs:
            for jid, task in (
                row["state"].get("stations", {}).get(key, {}).get("jobs", {}).items()
            ):
                if jid in jobs:
                    jobs[jid].update(
                        inspection_phase=task["status"],
                        remaining_ticks=task["remaining"],
                    )
            values["jobs"] = jobs
        if len(values) == 1:
            values["status"] = "No recorded runtime state for this resource"
        return values

    def _update_state(self):
        if self.row:
            self.decisions.update_row(self.row, self.selected_id, self.selected_job)
            self.refresh_events()
            layer = self.player.state_layer
            layer.selected_agv = (
                self.selected_id
                if self.selected_id in self.row["state"].get("agvs", {})
                else None
            )
            layer.trail_length = (
                self.trail_selector.currentData()
                if hasattr(self, "trail_selector")
                else 12
            )
            layer.trail_points = (
                self.replay_index.history(
                    layer.selected_agv, self.row["tick"], layer.trail_length
                )
                if layer.selected_agv
                else []
            )
            layer.update()
            if self.selected_job:
                self.update_job()
            if self.dark:
                apply_workspace_theme(self.player, True)
        self.runtime_inspector.update_state(
            self.row,
            self.selected_id,
            self._state_values(),
            self.resources.get(self.selected_id),
        )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.responsive_ready:
            self.sync_responsive()

    def position_view_controls(self):
        viewport = self.player.view.viewport()
        self.view_controls.adjustSize()
        self.view_controls.move(
            max(0, viewport.geometry().right() - self.view_controls.width() - 12),
            viewport.geometry().top() + 12,
        )
        self.view_controls.raise_()

    def copy_map_screenshot(self):
        # Copy only the viewport at its current zoom and pan; controls stay out.
        self.view_controls.hide()
        try:
            image = self.player.view.viewport().grab().toImage()
            QApplication.clipboard().setImage(image)
            self.player.statusBar().showMessage(
                f"Copied map screenshot · {image.width()} × {image.height()} px", 5000
            )
        finally:
            self.view_controls.show()

    def eventFilter(self, watched, event):
        if watched is self.map_viewport:
            if event.type() in (QEvent.Type.Resize, QEvent.Type.Show):
                self.position_view_controls()
            return super().eventFilter(watched, event)
        if watched is self.map_panel and event.type() in (
            QEvent.Type.Resize,
            QEvent.Type.Show,
        ):
            self.position_resource_handle()
        if watched is self.splitter and event.type() == QEvent.Type.Resize:
            self.position_drawer()
        return super().eventFilter(watched, event)

    def sync_responsive(self):
        if not self.responsive_ready:
            return
        narrow = self.width() < 1200
        if narrow != self.narrow:
            self.narrow = narrow
            self.property_panel.hide()
            if narrow:
                self.property_panel.setParent(self)
                self.drawer_open = False
                self.wide_analysis_expanded = not self.dashboard.collapse.isChecked()
                self.dashboard.collapse.setChecked(False)
                self.dashboard.setMinimumHeight(210)
                self.analysis_splitter.setSizes([max(200, self.height() - 330), 210])
                self.dashboard.collapse.setChecked(True)
            else:
                self.dashboard.setMinimumHeight(210)
                self.splitter.addWidget(self.property_panel)
                self.property_panel.setVisible(self.mode == "standard")
                self.splitter.setSizes([180, max(300, self.width() - 480), 300])
                if getattr(self, "wide_analysis_expanded", False):
                    self.dashboard.collapse.setChecked(False)
                self.drawer_open = False
            self.drawer_close.setVisible(narrow)
            self.inspector_action.setVisible(narrow)
            self.inspector_action.blockSignals(True)
            self.inspector_action.setChecked(False)
            self.inspector_action.blockSignals(False)
        self.position_drawer()

    def position_drawer(self):
        if not self.narrow:
            return
        top = self.splitter.mapTo(self, QPoint(0, 0)).y()
        width = min(390, self.width())
        self.property_panel.setGeometry(
            self.width() - width, top, width, max(100, self.height() - top)
        )
        if self.drawer_open:
            self.property_panel.raise_()

    def set_drawer(self, visible):
        if not self.narrow:
            return
        self.drawer_open = bool(visible)
        self.property_panel.setVisible(self.drawer_open and self.mode == "standard")
        self.inspector_action.blockSignals(True)
        self.inspector_action.setChecked(self.drawer_open)
        self.inspector_action.blockSignals(False)
        self.position_drawer()

    def filter_owner(self):
        if hasattr(self, "object_filter") and self.object_filter.currentIndex() == 1:
            return self.selected_job or self.selected_id or "__no_selection__"
        return None

    def change_analysis(self, index):
        if index == 0:
            self.dashboard.set_analysis_view(0)
            self.dashboard.body.setCurrentWidget(self.timeline)
        else:
            self.dashboard.set_analysis_view(index - 1)

    def refresh_inspector(self, *args):
        if self.row:
            self.performance_panel.update_tick(
                self.row["tick"], self.dashboard.chart.window
            )
            self._update_state()

    def set_theme(self, dark):
        self.dark = bool(dark)
        apply_workspace_theme(self.player, self.dark)
        self.timeline.dark = self.dark
        self.timeline.update()
        self.theme_action.setText("Light theme" if dark else "Dark theme")

    def clear_selection(self):
        self.select_resource(None)
        if self.narrow:
            self.set_drawer(False)

    def refresh_events(self):
        if not self.row:
            return
        self.related_events.clear()
        events = [
            e
            for e in self.replay_index.filtered(self.selected_job or self.selected_id)
            if e.tick <= self.row["tick"]
        ][-100:]
        for event in reversed(events):
            self.related_events.addItem(f"Tick {event.tick} · {event.detail}")
            self.related_events.item(self.related_events.count() - 1).setData(
                Qt.ItemDataRole.UserRole, event.tick
            )

    def show_raw_frame(self):
        self.performance_panel.hide()
        self.back_global.show()
        self.inspector.show()
        self.inspector.setCurrentWidget(self.player.details)
        if self.narrow:
            self.set_drawer(True)

    def show_events(self, events):
        if not events:
            return
        self.performance_panel.hide()
        self.back_global.show()
        self.related_events.clear()
        for event in events:
            self.related_events.addItem(f"Tick {event.tick} · {event.detail}")
            self.related_events.item(self.related_events.count() - 1).setData(
                Qt.ItemDataRole.UserRole, event.tick
            )
        self.inspector.show()
        self.inspector.setCurrentWidget(self.related_events)
        if self.narrow:
            self.set_drawer(True)

    def select_job(self, job):
        if not self.row or job not in self.row["state"].get("jobs", {}):
            return
        owner = self.row["state"]["jobs"][job].get("location")
        self.select_resource(owner)
        self.selected_job = job
        self.performance_panel.hide()
        self.back_global.show()
        self.inspector.show()
        self.inspector.setCurrentWidget(self.job_detail)
        self.update_job()
        self.decisions.update_row(self.row, self.selected_id, self.selected_job)
        self.sync_markers()
        self.refresh_events()

    def update_job(self):
        from html import escape

        job = dict(self.row["state"].get("jobs", {}).get(self.selected_job, {}))
        if "quality" in job:
            job["quality"] = displayed_quality(
                self.row["state"], self.selected_job, job.get("location")
            )
        self.title.setText("Order " + self.selected_job)
        lines = [
            f"<b>{escape(str(k))}</b>: {escape(str(v))}"
            for k, v in job.items()
            if k not in ("defective", "risk")
        ]
        self.job_detail.setHtml("<br>".join(lines) or "Not present in this frame")
