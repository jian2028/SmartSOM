"""Task performance beside the map: whole run, trailing window, declared bound.

Read-only presentation of recorded evidence. The values come from
`smartsom.trace.performance`, the same definitions evaluation reports.
"""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QGridLayout, QLabel, QToolButton, QWidget

from smartsom.studio.replay_evidence import performance_rows

HEADINGS = ("", "Run", "Recent", "Reference")
LABEL_STYLE = "color: #536a77;"
VALUE_STYLE = "color: #264854; font-weight: 600;"
BOUND_STYLE = "color: #7d8f99;"
BOUND_TIP = (
    "Declared reference computed from work content and machine capability. "
    "It is a bound, not a target, and no controller is expected to reach it."
)


class TaskPerformancePanel(QWidget):
    """Throughput, passing rate and tardiness for the displayed tick."""

    def __init__(self, evidence, reference=None):
        super().__init__()
        self.evidence = evidence
        self.reference = reference
        self.setObjectName("taskPerformance")
        grid = QGridLayout(self)
        grid.setContentsMargins(12, 2, 12, 10)
        heading = QLabel("TASK PERFORMANCE")
        heading.setStyleSheet("color: #7d8f99; font-size: 11px; letter-spacing: 0.4px;")
        grid.addWidget(heading, 0, 0, 1, 4)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(3)
        self.window_heading = QLabel(HEADINGS[2])
        for column, text in enumerate(HEADINGS):
            title = self.window_heading if column == 2 else QLabel(text)
            title.setStyleSheet("color: #7d8f99; font-size: 11px;")
            grid.addWidget(
                title,
                1,
                column,
                Qt.AlignmentFlag.AlignLeft
                if column == 0
                else Qt.AlignmentFlag.AlignRight,
            )
        self.cells = {}
        for row, entry in enumerate(performance_rows(evidence.performance, 0, 1), 2):
            name = QLabel(entry["label"])
            name.setStyleSheet(LABEL_STYLE)
            grid.addWidget(name, row, 0)
            for column, key, style in (
                (1, "total", VALUE_STYLE),
                (2, "recent", VALUE_STYLE),
                (3, "bound", BOUND_STYLE),
            ):
                value = QLabel("—")
                value.setStyleSheet(style)
                value.setWordWrap(True)
                value.setAccessibleName(f"{entry['label']} {HEADINGS[column]}")
                value.setAlignment(Qt.AlignmentFlag.AlignRight)
                if column == 3:
                    value.setToolTip(self.bound_tooltip())
                grid.addWidget(value, row, column)
                self.cells[entry["label"], key] = value
        self.reference_note = QLabel(
            self.bound_tooltip() + "\nMissing evidence is shown as Not recorded."
        )
        self.reference_note.setWordWrap(True)
        self.reference_note.setStyleSheet(
            "color: #536a77; font-size: 11px; padding-top: 12px;"
        )
        self.reference_note.hide()
        self.reference_toggle = QToolButton()
        self.reference_toggle.setText("Reference assumptions")
        self.reference_toggle.setCheckable(True)
        self.reference_toggle.toggled.connect(self.reference_note.setVisible)
        grid.addWidget(self.reference_toggle, 8, 0, 1, 4)
        grid.addWidget(self.reference_note, 9, 0, 1, 4)
        grid.setColumnStretch(0, 1)
        grid.setColumnMinimumWidth(1, 54)
        grid.setColumnMinimumWidth(2, 54)

    def bound_tooltip(self):
        if not (self.reference and self.reference.get("available")):
            return "No declared reference for this recording."
        return "\n".join([BOUND_TIP, "Assumptions:", *self.reference["assumptions"]])

    def update_tick(self, tick, window):
        self.window_heading.setText(f"Last {window}")
        rows = performance_rows(self.evidence.performance, tick, window, self.reference)
        for entry in rows:
            for key in ("total", "recent", "bound"):
                value = entry[key]
                if value == "—":
                    value = "Not recorded"
                    performance = self.evidence.performance
                    start = max(0, tick - window) if key == "recent" else 0
                    if key == "bound" and entry["label"] == "Passing rate":
                        value = "Not applicable"
                    elif key != "bound":
                        if (
                            entry["label"] == "Throughput"
                            and tick == 0
                            and performance.qualified[tick] is not None
                        ):
                            value = "No elapsed ticks"
                        elif entry["label"] == "Passing rate":
                            submitted = performance.submitted[tick]
                            previous = (
                                performance.submitted[start] if key == "recent" else 0
                            )
                            if (
                                submitted is not None
                                and previous is not None
                                and submitted == previous
                                and performance.qualified[tick] is not None
                            ):
                                value = "No submissions"
                        elif (
                            key == "recent"
                            and tick == 0
                            and entry["label"] in ("Total tardiness", "Tardy jobs")
                            and performance.identified
                        ):
                            value = "No elapsed ticks"
                self.cells[entry["label"], key].setText(value)
