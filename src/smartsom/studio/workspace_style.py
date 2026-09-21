"""Shared window chrome; factory graphics and their colors live separately."""

from PySide6.QtCore import QPointF
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPalette, QPen
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QLabel,
    QListView,
    QStyledItemDelegate,
    QStyleFactory,
    QVBoxLayout,
)

STYLE = """
QMainWindow, QDialog { background: #f3f6f8; color: #273e4d; }
QWidget { font-size: 12px; color: #273e4d; }
QWidget:disabled { color: #84949f; }
QToolBar { background: #ffffff; border: 0; border-bottom: 1px solid #dce4e9;
    spacing: 7px; padding: 8px 10px; }
QToolButton { padding: 6px 10px; border-radius: 4px; color: #294758; }
QToolButton:hover { background: #edf3f7; }
QToolButton:checked { background: #deedf5; color: #155f86; }
QToolBar QLabel { padding: 0 5px; }
QTreeWidget { background: #ffffff; border: 0; outline: 0; color: #2f4858; }
QTreeWidget::item { padding: 5px 2px; border: 0; }
QTreeWidget::item:selected { background: #e2eff6; color: #164e6d; }
QTreeWidget::item:hover { background: #f0f5f8; }
QHeaderView::section { background: #f7f9fb; color: #70828f; font-size: 10px;
    font-weight: 600; border: 0; padding: 8px 7px; text-align: left; }
QLineEdit { background: #f8fafc; border: 1px solid #dce4e9; border-radius: 5px;
    padding: 7px; color: #273e4d; }
QLineEdit:focus { border-color: #77a7bd; }
QComboBox { background: #ffffff; color: #273e4d; border: 1px solid #dce4e9; border-radius: 4px;
    padding: 5px 9px; min-width: 92px; }
QComboBox QAbstractItemView { background: white; color: #273e4d; }
QTabWidget { background: #eaf0f4; }
QTabWidget::pane { border: 0; }
QTabBar { background: #eaf0f4; }
QTabBar::tab { background: #eaf0f4; color: #617887; padding: 11px 16px;
    border-right: 1px solid #dce4e9; }
QTabBar::tab:selected { background: #ffffff; color: #214b62; }
QTabBar::tab:hover { background: #f2f6f9; }
QSplitter::handle { background: #dce4e9; }
QSplitter::handle:horizontal { width: 1px; }
QSplitter::handle:vertical { height: 1px; }
QStatusBar { background: #ffffff; border-top: 1px solid #dce4e9; color: #6d808c; }
QPushButton { background: #ffffff; color: #2f5265; border: 1px solid #d1dfe7;
    border-radius: 5px; padding: 7px 14px; }
QPushButton:hover { background: #edf4f8; }
QPushButton:default { background: #216787; color: #ffffff; border-color: #216787; }
QPushButton:disabled { background: #f6f8fa; color: #8c9ba5; border-color: #e2e9ed; }
QWidget#propertyEditor, QWidget#propertyForm { background: #ffffff; }
QWidget#propertyEditor QLabel { background: transparent; }
QGroupBox { background: #f8fafc; border: 1px solid #dce4e9; border-radius: 5px;
    margin-top: 12px; padding: 12px 6px 6px; }
QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; }
QTableWidget { background: white; alternate-background-color: #f8fafc;
    gridline-color: #e1e8ed; border: 1px solid #dce4e9; }
QRadioButton { padding: 10px 2px; }
"""


def apply_light_palette(widget):
    """Keep this light workspace readable under a dark system appearance."""
    palette = QPalette(widget.palette())
    for role, color in {
        QPalette.ColorRole.Window: "#f3f6f8",
        QPalette.ColorRole.WindowText: "#273e4d",
        QPalette.ColorRole.Base: "#ffffff",
        QPalette.ColorRole.AlternateBase: "#f7f9fb",
        QPalette.ColorRole.Text: "#273e4d",
        QPalette.ColorRole.Button: "#ffffff",
        QPalette.ColorRole.ButtonText: "#273e4d",
        QPalette.ColorRole.Highlight: "#deedf5",
        QPalette.ColorRole.HighlightedText: "#155f86",
        QPalette.ColorRole.PlaceholderText: "#84949f",
        QPalette.ColorRole.ToolTipBase: "#ffffff",
        QPalette.ColorRole.ToolTipText: "#273e4d",
    }.items():
        palette.setColor(role, QColor(color))
    for role in (
        QPalette.ColorRole.WindowText,
        QPalette.ColorRole.Text,
        QPalette.ColorRole.ButtonText,
    ):
        palette.setColor(QPalette.ColorGroup.Disabled, role, QColor("#84949f"))
    widget.setPalette(palette)


def panel(title, subtitle=None):
    widget = QFrame()
    widget.setStyleSheet("QFrame { background: #ffffff; }")
    layout = QVBoxLayout(widget)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(0)
    heading = QLabel(title)
    heading.setStyleSheet(
        "font-size: 11px; font-weight: 700; color: #547080; padding: 14px 12px 8px;"
    )
    layout.addWidget(heading)
    if subtitle:
        detail = QLabel(subtitle)
        detail.setWordWrap(True)
        detail.setStyleSheet("color: #84949f; padding: 0 12px 12px;")
        layout.addWidget(detail)
    return widget, layout


class SelectorDelegate(QStyledItemDelegate):
    def sizeHint(self, option, index):
        size = super().sizeHint(option, index)
        size.setHeight(32)
        return size


class ReplaySelector(QComboBox):
    """Owned Fusion style and explicit list view avoid native macOS popup chrome."""

    def __init__(self):
        super().__init__()
        self._selector_style = QStyleFactory.create("Fusion")
        self._selector_style.setParent(self)
        self.setStyle(self._selector_style)
        self.setView(QListView(self))
        self.view().setStyle(self._selector_style)
        self.setItemDelegate(SelectorDelegate(self))
        self.setStyleSheet("""
            QComboBox { background: white; color: #294758; border: 1px solid #c8d9e3;
                border-radius: 5px; padding: 7px 30px 7px 12px; min-width: 110px; }
            QComboBox:hover, QComboBox:focus { border-color: #508bab; background: #f4f9fc; }
            QComboBox::drop-down { subcontrol-origin: padding; subcontrol-position: top right;
                width: 24px; border: 0; }
            QComboBox QAbstractItemView { background: white; color: #294758;
                border: 1px solid #c8d9e3; padding: 4px; outline: 0;
                selection-background-color: #deedf5; selection-color: #155f86; }
            QComboBox QAbstractItemView::item { min-height: 30px; padding: 3px 10px; }
        """)

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QPen(QColor("#426577"), 1.5))
        center = QPointF(self.width() - 15, self.height() / 2)
        path = QPainterPath(center + QPointF(-4, -2))
        path.lineTo(center + QPointF(0, 2))
        path.lineTo(center + QPointF(4, -2))
        painter.drawPath(path)


# Replay-only overrides: keep the authoring Studio unchanged.
REPLAY_STYLE = """
QWidget { font-size: 13px; color: #294758; }
QLabel { background: transparent; }
QHeaderView::section { color: #536a77; font-size: 12px; }
QStatusBar { color: #536a77; font-size: 12px; }
QFrame#replayInspectorPanel { border-left: 1px solid #c8d9e3; }
QToolButton#resourceRail { background: #f4f8fa; border: 1px solid #c8d9e3;
    border-left: 0; border-top-right-radius: 5px;
    border-bottom-right-radius: 5px; padding: 0; }
QToolButton#resourceRail:hover { background: #deedf5; }
QToolButton#resourceRail:focus { border: 1px solid #508bab; }
QToolBar#mapTools { padding: 0 8px; spacing: 5px; }
QToolBar#mapTools QToolButton { padding: 5px 10px; }
QTabWidget, QTabBar { background: white; }
QTabBar#analysisTabs { background: transparent; border: none; }
QTabWidget::pane { border: none; }
QTabBar::tab { background: #edf2f5; color: #536a77; border: 1px solid transparent;
    border-radius: 5px; padding: 7px 12px; margin: 3px 2px; }
QTabBar::tab:selected { background: #deedf5; color: #155f86; border-color: #b7d2e1; }
QTabBar::tab:hover { background: #e5eff5; }
QSplitter::handle:vertical { height: 3px; background: #dce4e9; }
QSplitter::handle:hover { background: #8bb1c4; }
QListWidget { background: white; color: #294758; border: 1px solid #dce4e9; }
QListWidget::item { padding: 5px; }
QListWidget::item:selected { background: #deedf5; color: #155f86; }
"""
