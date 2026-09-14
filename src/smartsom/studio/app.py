"""Explicit entry point for the optional Qt desktop application."""

from collections.abc import Sequence
from pathlib import Path


def main(paths: Sequence[str | Path] = ()) -> int:
    """Show Studio, owning the event loop only when creating the application."""
    from PySide6.QtCore import QSettings, QTimer
    from PySide6.QtWidgets import QApplication

    from smartsom.studio.window import StudioWindow

    existing = QApplication.instance()
    if existing is not None and not isinstance(existing, QApplication):
        raise RuntimeError("SmartSOM Studio requires a QApplication.")
    app = existing or QApplication([])
    app.setApplicationName("SmartSOM Studio")
    app.setOrganizationName("SmartSOM")
    window = StudioWindow(settings=QSettings("SmartSOM", "Studio"))
    if paths:
        for path in paths:
            window.open_path(path)
    window.show()

    def startup():
        if window.editor.recovery.entries():
            window.editor.recover_dialog()
        if not paths and not window.documents:
            window.new_dialog()

    QTimer.singleShot(0, startup)
    # Keep windows alive when an existing Qt application owns the event loop.
    windows = getattr(app, "_smartsom_studio_windows", None)
    if windows is None:
        windows = []
        app._smartsom_studio_windows = windows
    windows.append(window)
    window.destroyed.connect(
        lambda: windows.remove(window) if window in windows else None
    )
    return int(app.exec()) if existing is None else 0
