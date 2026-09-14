"""Maintain exclusive modes when activated through native accessibility."""

from PySide6.QtCore import QTimer


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
