"""Typed, transactional frame drawing editor for Studio."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from smartsom.config.drawing_state import (
    DrawingAGV,
    DrawingBuffer,
    DrawingJob,
    DrawingMachine,
    DrawingState,
    DrawingStation,
)
from smartsom.studio.canvas import FactoryScene, FactoryView
from smartsom.studio.drawing_state import attach_drawing, locations, validate_drawing
from smartsom.studio.workspace_style import WorkspaceSelector


def number(value, maximum=100000, decimals=None):
    widget = QSpinBox() if decimals is None else QDoubleSpinBox()
    widget.setRange(0, maximum)
    if decimals is not None:
        widget.setDecimals(decimals)
    widget.setValue(value)
    return widget


def choice(options, value):
    widget = WorkspaceSelector()
    for label, data in options:
        widget.addItem(label, data)
    index = widget.findData(value)
    if index >= 0:
        widget.setCurrentIndex(index)
    return widget


def yes_no(value):
    return choice([("No", False), ("Yes", True)], value)


class DrawingStateDialog(QDialog):
    def __init__(self, document, parent=None):
        super().__init__(parent)
        self.design = document.design
        self.drawing = document.authoring.drawing_state
        self.setWindowTitle("State · Studio")
        self.resize(1180, 750)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("State at this tick"))
        row = QHBoxLayout()
        row.addWidget(QLabel("Tick"))
        self.tick = number(self.drawing.tick, 1000000000)
        row.addWidget(self.tick)
        preview = QPushButton("Preview changes")
        preview.clicked.connect(self.preview)
        row.addWidget(preview)
        row.addStretch()
        layout.addLayout(row)
        splitter = QSplitter()
        self.tabs = QTabWidget()
        splitter.addWidget(self.tabs)
        scene = FactoryScene(self.design)
        attach_drawing(scene, self.drawing)
        self.view = FactoryView(scene)
        scene.setParent(self.view)
        splitter.addWidget(self.view)
        splitter.setSizes([720, 460])
        layout.addWidget(splitter, 1)
        self.tables = {}
        self.build_tables()
        self.error = QLabel()
        self.error.setWordWrap(True)
        self.error.setStyleSheet("color: #b33f36;")
        layout.addWidget(self.error)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Apply
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Apply).clicked.connect(
            self.apply
        )
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def table(self, name, columns):
        table = QTableWidget(0, len(columns))
        table.setHorizontalHeaderLabels(columns)
        table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )
        table.horizontalHeader().setStretchLastSection(True)
        table.verticalHeader().hide()
        self.tables[name] = table
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(table)
        self.tabs.addTab(page, name)
        return table, layout

    def add_row(self, table, key, widgets):
        row = table.rowCount()
        table.insertRow(row)
        label = QTableWidgetItem(key)
        label.setFlags(label.flags() & ~Qt.ItemFlag.ItemIsEditable)
        table.setItem(row, 0, label)
        for column, widget in enumerate(widgets, 1):
            table.setCellWidget(row, column, widget)
        table.setRowHeight(row, 34)

    def build_tables(self):
        t, _ = self.table(
            "Machines", ["Machine", "Status", "Speed", "Total ticks", "Remaining"]
        )
        for m in self.design.machines:
            d = self.drawing.machines.get(m.machine_id, DrawingMachine())
            modes = list(
                dict.fromkeys(
                    [
                        "slow",
                        "normal",
                        "fast",
                        *[q.quality_mode_id for q in m.quality_modes],
                    ]
                )
            )
            self.add_row(
                t,
                m.machine_id,
                [
                    choice(
                        [
                            (s, s)
                            for s in ("IDLE", "READY", "PROCESSING", "BLOCKED", "DOWN")
                        ],
                        d.status,
                    ),
                    choice([(s, s) for s in modes], d.mode),
                    number(d.total),
                    number(d.remaining),
                ],
            )
        t, layout = self.table(
            "Jobs", ["", "Order", "Attempt", "Quality", "Location / slot"]
        )
        self.job_locations = [
            (f"{owner} / {slot}" if slot else owner, (owner, slot))
            for owner, slot in locations(self.design)
        ]
        for job in self.drawing.jobs:
            self.add_job(job)
        actions = QHBoxLayout()
        add = QPushButton("Add job")
        add.setEnabled(bool(self.job_locations))
        add.clicked.connect(lambda: self.add_job())
        remove = QPushButton("Remove selected job")
        remove.clicked.connect(lambda: t.removeRow(t.currentRow()))
        actions.addWidget(add)
        actions.addWidget(remove)
        actions.addStretch()
        layout.addLayout(actions)
        t, _ = self.table("AGVs", ["AGV", "X", "Y", "State", "Conflict with"])
        for a in self.design.agvs:
            d = self.drawing.agvs.get(
                a.agv_id, DrawingAGV(x=a.initial_cell.x, y=a.initial_cell.y)
            )
            self.add_row(
                t,
                a.agv_id,
                [
                    number(d.x, self.design.grid.width - 1),
                    number(d.y, self.design.grid.height - 1),
                    choice(
                        [
                            ("Normal", "NORMAL"),
                            ("Conflict", "CONFLICT"),
                            ("Charging", "CHARGING"),
                        ],
                        d.status,
                    ),
                    choice(
                        [("Unspecified", None)]
                        + [
                            (f"{other.name} · {other.agv_id}", other.agv_id)
                            for other in self.design.agvs
                            if other.agv_id != a.agv_id
                        ],
                        d.conflict_with,
                    ),
                ],
            )
        for row in range(t.rowCount()):
            status, partner = t.cellWidget(row, 3), t.cellWidget(row, 4)
            partner.setEnabled(status.currentData() == "CONFLICT")
            status.currentIndexChanged.connect(
                lambda _, status=status, partner=partner: partner.setEnabled(
                    status.currentData() == "CONFLICT"
                )
            )
        t, _ = self.table(
            "Buffers",
            [
                "Buffer",
                "Outside waiting",
                "Arrival interval",
                "Remaining",
                "Good delivered",
                "Passing %",
                "Jobs / tick",
            ],
        )
        for b in self.design.buffers:
            d = self.drawing.buffers.get(b.buffer_id, DrawingBuffer())
            widgets = [
                number(d.waiting),
                number(d.total),
                number(d.remaining),
                number(d.delivered),
                number(d.passing_percent, 100, 1),
                number(d.throughput, 100000, 3),
            ]
            for index, widget in enumerate(widgets):
                widget.setEnabled(
                    b.role == ("system_input" if index < 3 else "system_output")
                )
            self.add_row(t, b.buffer_id, widgets)
        t, _ = self.table(
            "Inspection", ["Station", "Inspecting jobs", "Total ticks", "Remaining"]
        )
        for q in self.design.inspection_stations:
            d = self.drawing.stations.get(q.inspection_station_id, DrawingStation())
            self.add_row(
                t,
                q.inspection_station_id,
                [yes_no(d.inspecting), number(d.total), number(d.remaining)],
            )
        t, _ = self.table("Disposal", ["Scrap bin", "Disposed jobs"])
        for b in self.design.scrap_bins:
            self.add_row(
                t,
                b.scrap_bin_id,
                [number(self.drawing.disposed.get(b.scrap_bin_id, 0))],
            )

    def add_job(self, job=None):
        t = self.tables["Jobs"]
        if job is None:
            order = (
                max(
                    (t.cellWidget(r, 1).value() for r in range(t.rowCount())), default=0
                )
                + 1
            )
            owner, slot = self.job_locations[0][1]
            job = DrawingJob(order=order, owner=owner, slot=slot)
        order, attempt = number(job.order), number(job.attempt)
        order.setMinimum(1)
        attempt.setMinimum(1)
        options = list(self.job_locations)
        if (job.owner, job.slot) not in [data for _, data in options]:
            options.append(
                (f"Missing: {job.owner} / {job.slot}", (job.owner, job.slot))
            )
        self.add_row(
            t,
            "",
            [
                order,
                attempt,
                choice([(s, s) for s in ("UNKNOWN", "PASS", "FAIL")], job.quality),
                choice(options, (job.owner, job.slot)),
            ],
        )

    def read(self):
        def rows(name):
            t = self.tables[name]
            for r in range(t.rowCount()):
                values = []
                for c in range(1, t.columnCount()):
                    w = t.cellWidget(r, c)
                    values.append(
                        w.currentData() if isinstance(w, QComboBox) else w.value()
                    )
                yield t.item(r, 0).text(), values

        drawing = DrawingState(
            tick=self.tick.value(),
            machines={
                k: DrawingMachine(status=v[0], mode=v[1], total=v[2], remaining=v[3])
                for k, v in rows("Machines")
            },
            jobs=tuple(
                DrawingJob(
                    order=v[0], attempt=v[1], quality=v[2], owner=v[3][0], slot=v[3][1]
                )
                for _, v in rows("Jobs")
            ),
            agvs={
                k: DrawingAGV(
                    x=v[0],
                    y=v[1],
                    conflict=v[2] == "CONFLICT",
                    charging=v[2] == "CHARGING",
                    conflict_with=v[3] if v[2] == "CONFLICT" else None,
                )
                for k, v in rows("AGVs")
            },
            buffers={
                k: DrawingBuffer(
                    waiting=v[0],
                    total=v[1],
                    remaining=v[2],
                    delivered=v[3],
                    passing_percent=v[4],
                    throughput=v[5],
                )
                for k, v in rows("Buffers")
            },
            stations={
                k: DrawingStation(inspecting=v[0], total=v[1], remaining=v[2])
                for k, v in rows("Inspection")
            },
            disposed={k: v[0] for k, v in rows("Disposal")},
        )
        validate_drawing(self.design, drawing)
        return drawing

    def preview(self):
        try:
            drawing = self.read()
        except ValueError as exc:
            self.error.setText(str(exc))
            return False
        old = self.view.scene()
        scene = FactoryScene(self.design, self.view)
        attach_drawing(scene, drawing)
        self.view.setScene(scene)
        self.view.fit_map()
        old.deleteLater()
        self.error.clear()
        return True

    def apply(self):
        if self.preview():
            self.drawing = self.read()
            self.accept()
