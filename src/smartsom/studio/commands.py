"""Qt undo commands publish already-validated immutable document snapshots."""

from PySide6.QtGui import QUndoCommand

from smartsom.studio.editing import checked


class DesignCommand(QUndoCommand):
    def __init__(
        self, document, candidate, label, refresh, selection=None, authoring=None
    ):
        super().__init__(label)
        self.document = document
        self.before = document.design
        self.after = checked(candidate)
        self.before_authoring = document.authoring
        self.after_authoring = document.authoring if authoring is None else authoring
        self.before_selection = tuple(document.selected_ids)
        self.after_selection = tuple(
            selection if selection is not None else document.selected_ids
        )
        self.refresh = refresh

    def _publish(self, design, selection, authoring):
        self.document.design = design
        self.document.authoring = authoring
        self.document.selected_ids = selection
        self.document.selected_id = selection[0] if selection else None
        self.refresh(self.document)

    def redo(self):
        self._publish(self.after, self.after_selection, self.after_authoring)

    def undo(self):
        self._publish(self.before, self.before_selection, self.before_authoring)
