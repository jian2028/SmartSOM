"""Maintain exclusive modes when activated through native accessibility."""

from PySide6.QtCore import QRectF, QSize, Qt, QTimer
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import QAbstractButton


class BufferDisplaySwitch(QAbstractButton):
    """Checked means individual cells; unchecked means a counted stack."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setCheckable(True)
        self.setAccessibleName("Buffer display: grid or stack")
        self.setToolTip("On: individual cells · Off: stacked jobs")
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def sizeHint(self):
        return QSize(135, 30)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#397a93" if self.isChecked() else "#a5b5be"))
        painter.drawRoundedRect(QRectF(2, 6, 34, 18), 9, 9)
        painter.setBrush(QColor("white"))
        painter.drawEllipse(QRectF(20 if self.isChecked() else 4, 8, 14, 14))
        painter.setPen(QColor("#294758"))
        painter.drawText(
            QRectF(45, 0, 90, 30),
            Qt.AlignmentFlag.AlignVCenter,
            "Grid" if self.isChecked() else "Stack",
        )
        if self.hasFocus():
            painter.setPen(QColor("#397a93"))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRoundedRect(QRectF(0, 3, 38, 24), 5, 5)


def keep_exclusive_selection(action):
    # Accessibility may directly uncheck the current mode. Restore it after Qt
    # finishes updating the group, unless a newer selection has superseded it.
    def toggled(checked):
        group = action.actionGroup()
        generation = (group.property("selectionGeneration") or 0) + 1
        group.setProperty("selectionGeneration", generation)
        if checked:
            return

        def restore():
            if (
                action.isEnabled()
                and group.property("selectionGeneration") == generation
                and group.checkedAction() is None
            ):
                action.setChecked(True)

        QTimer.singleShot(0, action, restore)

    action.toggled.connect(toggled)
