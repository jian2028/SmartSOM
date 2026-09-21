"""Studio-style, read-only workspace around the existing replay canvas."""

from PySide6.QtCore import QEvent, QPoint, Qt, QTimer
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
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QSizePolicy,
    QSplitter,
    QStyle,
    QTabWidget,
    QToolBar,
    QToolButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from smartsom.domain.factory_design import entity_id
from smartsom.studio.items import COLORS
from smartsom.studio.properties import PropertyTree, type_label
from smartsom.studio.replay_dashboard import ReplayDashboard
from smartsom.studio.replay_evidence import (
    movement_conflicts,
    outside_count,
    resource_conflicts,
)
from smartsom.studio.replay_inspector import RuntimeInspector
from smartsom.studio.workspace_style import (
    REPLAY_STYLE,
    STYLE,
    apply_light_palette,
    panel,
)


class ReplayWorkspace(QWidget):
    def __init__(self, player):
        super().__init__(player)
        self.player = player
        self.selected_id = None
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

        self.property_panel, right = panel("Selected object")
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
        right.addWidget(self.title)
        right.addWidget(self.subtitle)
        self.inspector = QTabWidget()
        self.inspector.setObjectName("replayInspector")
        self.runtime_inspector = RuntimeInspector()
        self.design_tree = PropertyTree()
        self.inspector.addTab(self.runtime_inspector, "State")
        self.inspector.addTab(self.design_tree, "Design")
        self.inspector.addTab(player.details, "Raw frame")
        right.addWidget(self.inspector, 1)

        self.map_panel = QWidget()
        map_layout = QVBoxLayout(self.map_panel)
        map_layout.setContentsMargins(0, 0, 0, 0)
        map_layout.setSpacing(0)
        heading = QLabel(f"{player.factory.name} · Replay")
        heading.setStyleSheet("background: white; padding: 11px 16px;")
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
        fit = self.map_tools.addAction("Fit map")
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
        self.map_tools.widgetForAction(fit).setToolButtonStyle(
            Qt.ToolButtonStyle.ToolButtonIconOnly
        )
        fit.triggered.connect(player.view.fit_map)
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
        player.view.setToolTip(hint.text())
        hint.deleteLater()
        self.canvas_splitter = QSplitter(Qt.Orientation.Vertical)
        self.canvas_splitter.addWidget(self.map_panel)
        self.canvas_splitter.setCollapsible(0, False)
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.setObjectName("workspaceSplitter")
        self.splitter.installEventFilter(self)
        for widget in (self.resource_panel, self.canvas_splitter, self.property_panel):
            self.splitter.addWidget(widget)
        self.splitter.setSizes([180, 940, 300])
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setCollapsible(1, False)
        self.resource_panel.setMinimumWidth(160)
        self.property_panel.setMinimumWidth(260)
        self.property_panel.setMaximumWidth(450)
        self.end_label = QLabel(f"{player.playback.last_tick} ticks")
        self.analysis_splitter = QSplitter(Qt.Orientation.Vertical)
        self.resource_rail = QToolButton(self.map_panel)
        self.resource_rail.setObjectName("resourceRail")
        self.resource_rail.setFixedSize(26, 76)
        self.map_panel.installEventFilter(self)
        self.resource_rail.clicked.connect(self.resources_action.toggle)
        self.set_resources_visible(False)
        self.analysis_splitter.addWidget(self.splitter)
        self.analysis_splitter.addWidget(self.dashboard)
        self.analysis_splitter.setCollapsible(0, False)
        self.analysis_splitter.setCollapsible(1, False)
        self.analysis_splitter.setStretchFactor(0, 1)
        self.analysis_splitter.setSizes([475, 270])
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(1)
        layout.addWidget(self.dashboard.summary)
        layout.addWidget(self.analysis_splitter, 1)
        self._toolbar()
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
        self.escape_shortcut.activated.connect(lambda: self.set_drawer(False))
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
        presentation = menu.addMenu("Presentation")
        preview = presentation.addAction("Charging preview")
        preview.setCheckable(True)
        preview.setToolTip(
            "Presentation demo only; this recording has no charging execution."
        )
        preview.toggled.connect(self._charging_preview)
        more.setMenu(menu)
        toolbar.addWidget(more)
        toolbar.addSeparator()
        label = QLabel("Read-only playback")
        toolbar.addWidget(label)
        player.addToolBarBreak()
        controls = QToolBar("Playback controls", player)
        controls.setObjectName("playbackToolbar")
        controls.setMovable(False)
        player.addToolBar(controls)
        for widget in (
            player.back_button,
            player.pause_button,
            player.step_button,
            player.speed,
        ):
            controls.addWidget(widget)
        controls.addSeparator()
        controls.addWidget(player.tick_label)
        player.slider.setMinimumWidth(160)
        controls.addWidget(player.slider)
        controls.addWidget(self.end_label)

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
                resource.name if self.selected_id else "Select an object"
            )
            identity = self.selected_id or self.player.factory.factory_id
            if (
                identity.startswith("machine_")
                and identity.removeprefix("machine_").isdigit()
            ):
                identity = f"M{int(identity.removeprefix('machine_'))} · {identity}"
            self.subtitle.setText(f"{type_label(resource)} · {identity}")
            self.design_tree.show_resource(resource)
            self._update_state()
            if self.narrow:
                self.set_drawer(selected is not None)
        finally:
            self.selecting = False

    def update_row(self, row):
        self.row = row
        self.dashboard.update_row(row)
        state, tick = row["state"], row["tick"]
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
                values["action"] = dict(row.get("actions", {}).get("agvs", [])).get(
                    key, "Unavailable"
                )
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
            values["jobs"] = jobs
        if len(values) == 1:
            values["status"] = "No recorded runtime state for this resource"
        return values

    def _update_state(self):
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

    def eventFilter(self, watched, event):
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
                self.dashboard.collapse.setChecked(True)
            else:
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
        width = min(360, self.width())
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
