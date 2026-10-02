"""Recorded decisions in the same read-only inspector used by Replay."""

from html import escape

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QProgressBar, QScrollArea, QVBoxLayout, QWidget

from smartsom.studio.replay_model import decision_rows


class DecisionInspector(QScrollArea):
    def __init__(self, recording):
        super().__init__()
        self.recording = recording
        self.setWidgetResizable(True)
        self.setFrameShape(QScrollArea.Shape.NoFrame)
        self.content = QWidget()
        self.content.setStyleSheet("background: #ffffff;")
        self.column = QVBoxLayout(self.content)
        self.column.setContentsMargins(12, 12, 12, 12)
        self.setWidget(self.content)
        self.signature = None
        self.rows = []

    def update_row(self, row, owner, job=None):
        signature = (row.get("tick"), owner, job)
        if signature == self.signature:
            return
        self.signature = signature
        while self.column.count():
            item = self.column.takeAt(0)
            if item.widget():
                item.widget().hide()
                item.widget().deleteLater()
        provider = getattr(self.recording, "manifest", {}).get("provider")
        self.rows = decision_rows(row, owner, provider, job) if owner or job else []

        def label(message, rich=False):
            widget = QLabel(message)
            widget.setTextFormat(
                Qt.TextFormat.RichText if rich else Qt.TextFormat.PlainText
            )
            widget.setWordWrap(True)
            widget.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            self.column.addWidget(widget)

        if not self.rows:
            label(f"No recorded decision for this object in frame {row['tick']}.")
            label(
                "This frame may be between decisions, or the recording may omit decision evidence."
            )
        for decision in self.rows:
            label(f"{decision['role'].title()} · {decision['owner']}")
            label(
                f"Decision at {decision['tick']} → result frame {decision['result_tick']}"
                + (f" · {decision['stage']}" if decision["stage"] else "")
            )
            label(decision["score_kind"])
            for choice in decision["choices"]:
                legal = (
                    "legal"
                    if choice["legal"] is True
                    else "masked"
                    if choice["legal"] is False
                    else "legality not recorded"
                )
                selected = "✓ " if choice["selected"] else ""
                value = choice["score"]
                score = (
                    f" · {value:.4g}"
                    if value is not None
                    and not decision["score_kind"].startswith("Policy probability")
                    else ""
                )
                line = f"{selected}{choice['label']} · {legal}{score}"
                label(
                    f"<b>{escape(line)}</b>" if choice["selected"] else line,
                    choice["selected"],
                )
                if value is not None and decision["score_kind"].startswith(
                    "Policy probability"
                ):
                    bar = QProgressBar()
                    bar.setRange(0, 10000)
                    bar.setValue(round(value * 10000))
                    bar.setFormat(f"{value:.1%}")
                    bar.setAccessibleName(
                        f"{choice['label']} policy probability {value:.1%}"
                    )
                    self.column.addWidget(bar)
        self.column.addStretch()
