"""Public recorded state grouped by resource task, never simulator execution."""

from html import escape

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QPainter
from PySide6.QtWidgets import (
    QLabel,
    QProgressBar,
    QScrollArea,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from smartsom.studio.properties import field_label
from smartsom.studio.replay_evidence import job_marks
from smartsom.studio.state_layer import FactoryStateLayer

# smartsom.spatial-run/v1: frozen spatial/models.py AGV_ACTIONS.
HISTORICAL_AGV_ACTIONS = ("UP", "DOWN", "LEFT", "RIGHT", "INTERACT", "WAIT")


def text(value):
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, float):
        return f"{value:g}"
    return escape(str(value))


def order_label(jid):
    if not jid:
        return "—"
    order, attempt = job_marks(jid)
    return f"{order} · Attempt {attempt}"


def table(rows):
    return (
        '<table width="100%" cellspacing="0" cellpadding="7">'
        + "".join(
            f'<tr><td style="color:#536a77">{text(label)}</td>'
            f'<td align="right">{text(value)}</td></tr>'
            for label, value in rows
        )
        + "</table>"
    )


class RemainingTicks(QProgressBar):
    """Use the same tick depletion renderer as the factory machines."""

    def paintEvent(self, event):
        painter = QPainter(self)
        FactoryStateLayer.segments(
            painter, QRectF(self.rect()), self.maximum(), self.value()
        )


class RuntimeInspector(QScrollArea):
    def __init__(self):
        super().__init__()
        self.setWidgetResizable(True)
        self.setFrameShape(QScrollArea.Shape.NoFrame)
        self.setStyleSheet("QScrollArea, QWidget { background: white; }")
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(14, 18, 14, 12)
        layout.setSpacing(12)
        self.status = QLabel()
        self.status.setWordWrap(True)
        self.status.setStyleSheet("color: #216787; font-size: 14px;")
        self.summary = QLabel()
        self.summary.setWordWrap(True)
        self.progress_title = QLabel("Remaining processing")
        self.progress = RemainingTicks()
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(9)
        self.timing = QLabel()
        self.timing.setWordWrap(True)
        self.timing.setStyleSheet("color: #536a77; font-size: 12px;")
        self.feedback_title = QLabel("Execution")
        self.feedback_title.setStyleSheet("color: #536a77; font-size: 12px;")
        self.feedback = QLabel()
        self.feedback.setWordWrap(True)
        self.job_toggle = QToolButton()
        self.job_toggle.setText("Order details")
        self.job_toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.job_toggle.setCheckable(True)
        self.job_toggle.setArrowType(Qt.ArrowType.RightArrow)
        self.job_toggle.toggled.connect(self.toggle_jobs)
        self.jobs = QLabel()
        self.jobs.setWordWrap(True)
        self.jobs.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.jobs.hide()
        for widget in (
            self.status,
            self.summary,
            self.progress_title,
            self.progress,
            self.timing,
            self.feedback_title,
            self.feedback,
            self.job_toggle,
            self.jobs,
        ):
            layout.addWidget(widget)
        layout.addStretch()
        self.setWidget(content)
        self.key = None

    def toggle_jobs(self, checked):
        self.jobs.setVisible(checked)
        self.job_toggle.setArrowType(
            Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow
        )

    def update_state(self, row, key, values, resource=None):
        if key != self.key:
            self.key = key
            self.job_toggle.setChecked(False)
            self.verticalScrollBar().setValue(0)
        state = row["state"] if row else {}
        machine = state.get("machines", {}).get(key)
        station = state.get("stations", {}).get(key)
        agv = state.get("agvs", {}).get(key)
        progress_data = machine if machine and machine.get("job") else station
        has_progress = bool(
            progress_data and (progress_data.get("job") or progress_data.get("batch"))
        )
        for widget in (self.progress_title, self.progress, self.timing):
            widget.setVisible(has_progress)
        if has_progress:
            remaining = progress_data.get("remaining", 0)
            total = progress_data.get(
                "total", progress_data.get("elapsed", 0) + remaining
            )
            if station is not None:
                total = progress_data.get(
                    "total", getattr(resource, "inspection_ticks", total)
                )
            self.progress_title.setText(
                "Remaining inspection"
                if station is not None
                else "Remaining processing"
            )
            self.progress.setRange(0, max(1, total))
            self.progress.setValue(remaining)
            self.progress.setAccessibleName(f"Remaining {remaining} of {total} ticks")
            self.timing.setText(
                f"Remaining {remaining} / {total} ticks · Elapsed {max(0, total - remaining)}"
            )
        self.feedback.clear()
        if machine is not None:
            mode = machine.get("mode") or "—"
            self.status.setText(
                f"{text(machine.get('status', 'Unknown')).title()} · {text(mode).title()}"
            )
            self.summary.setText(
                table([("Current order", order_label(machine.get("job")))])
            )
            raw = row.get("historical_frame", {}).get("machines", {}).get(key, {})
            rejection = row.get("rejections", {}).get(f"machine:{key}")
            outcome = raw.get("previous_outcome")
            if rejection or outcome:
                self.feedback.setText(
                    table(
                        [
                            (
                                "Rejection" if rejection else "Previous outcome",
                                (rejection or outcome).replace("_", " ").title(),
                            )
                        ]
                    )
                )
        elif agv is not None:
            conflict = values.get("movement_conflict") or values.get(
                "resource_conflict"
            )
            self.status.setText(
                f"{'Carrying' if agv.get('job') else 'Empty'} · {'Conflict' if conflict else 'No conflict'}"
            )
            cell = agv.get("cell")
            battery = agv.get("battery")
            battery_spec = getattr(resource, "battery", None)
            capacity = float(battery_spec.energy_capacity) if battery_spec else 100
            rows = [
                ("Position", f"({cell[0]}, {cell[1]})" if cell else None),
                ("Current order", order_label(agv.get("job"))),
                (
                    "Battery",
                    f"{battery / capacity:.0%}"
                    if battery is not None
                    else "Not recorded",
                ),
            ]
            self.summary.setText(table(rows))
            historical = "historical_frame" in row
            action = agv.get("previous_action") if historical else values.get("action")
            if historical and type(action) is int:
                action = (
                    HISTORICAL_AGV_ACTIONS[action]
                    if 0 <= action < len(HISTORICAL_AGV_ACTIONS)
                    else f"Unknown ({action})"
                )
            action = (
                str(action).replace("_", " ").title()
                if action is not None
                else "Not recorded"
            )
            outcome = (
                agv.get("previous_outcome") if historical else values.get("rejection")
            )
            feedback = [
                ("Previous action" if historical else "Recorded action", action)
            ]
            if outcome:
                feedback.append(
                    (
                        "Previous outcome" if historical else "Rejection",
                        str(outcome).replace("_", " ").title(),
                    )
                )
            self.feedback.setText(table(feedback))
        elif key in state.get("storage", {}):
            slots = state["storage"][key]
            inventory = sum(len(jobs) for jobs in slots.values())
            storage = getattr(resource, "storage", None)
            design_slots = getattr(storage, "slots", getattr(resource, "slots", ()))
            capacity = getattr(
                storage, "capacity", sum(s.capacity for s in design_slots)
            )
            self.status.setText(
                str(station.get("status", "Idle")).title()
                if station is not None
                else "Inventory"
            )
            rows = [
                ("Resident orders", values.get("resident_inventory", inventory)),
                ("Capacity", capacity if capacity is not None else "Unlimited"),
            ]
            if station is not None:
                rows.append(("Current batch", len(station.get("batch", []))))
                rows.extend(
                    (field_label(k), v)
                    for k, v in values.get("inspection_statistics", {}).items()
                )
            for field in (
                "outside_waiting",
                "qualified_deliveries",
                "output_passing_rate",
                "throughput",
            ):
                if field in values:
                    rows.append((field_label(field), values[field]))
            self.summary.setText(table(rows))
        else:
            self.status.setText("Recorded state" if key else "Select an object")
            rows = [
                (field_label(k), v)
                for k, v in values.items()
                if k not in ("recorded_tick", "id", "jobs", "defective")
                and not isinstance(v, (dict, list, tuple))
            ]
            self.summary.setText(
                table(rows)
                if key
                else "Click a machine, AGV or buffer to inspect its recorded state."
            )
        self.feedback_title.setVisible(bool(self.feedback.text()))
        self.feedback.setVisible(bool(self.feedback.text()))
        job_values = values.get("jobs", {})
        self.job_toggle.setVisible(bool(job_values))
        self.jobs.setText(
            "<br><br>".join(
                f"<b>Order {text(order_label(jid))}</b>"
                + table(
                    [
                        ("Job ID", jid),
                        *[
                            (field_label(k), v)
                            for k, v in job.items()
                            if k not in ("defective", "risk", "id")
                            and not isinstance(v, (dict, list, tuple))
                        ],
                    ]
                )
                for jid, job in job_values.items()
            )
        )
        self.jobs.setVisible(bool(job_values) and self.job_toggle.isChecked())
