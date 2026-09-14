"""Coordinate authoring UI, document transactions, persistence and validation."""

import tempfile
from dataclasses import replace
from pathlib import Path

from PySide6.QtCore import QMimeData, QObject, QStandardPaths, Qt, QTimer
from PySide6.QtGui import QAction, QActionGroup, QKeySequence, QPen, QUndoStack
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QInputDialog,
    QLabel,
    QListWidget,
    QMenu,
    QMessageBox,
    QPushButton,
    QToolBar,
    QToolButton,
    QVBoxLayout,
)

from smartsom.config.factory_design import (
    FactoryDesignFile,
    load_factory_design,
    save_factory_design,
)
from smartsom.domain.factory_design import (
    FactoryDesign,
    SlotStorage,
    iter_resources,
    target_cell,
    target_owner_id,
)
from smartsom.studio import editing
from smartsom.studio.canvas import FactoryScene
from smartsom.studio.commands import DesignCommand
from smartsom.studio.controls import keep_exclusive_selection
from smartsom.studio.dialogs import (
    BindingsDialog,
    ExportDialog,
    SlotsDialog,
    TemplateSaveDialog,
)
from smartsom.studio.export import export_map
from smartsom.studio.interaction import MapInteraction
from smartsom.studio.items import BINDING_FILL, CELL_SIZE
from smartsom.studio.persistence import (
    RecoveryStore,
    TemplateCatalog,
    file_digest,
)
from smartsom.studio.properties import PropertyEditor, ValueField


class StudioEditor(QObject):
    MIME = "application/x-smartsom-factory-selection"

    def __init__(self, window, data_dir=None):
        super().__init__(window)
        self.window = window
        self.active_document = None
        self.refreshing = False
        self.selecting_many = None
        self.draft_dialog = None
        self.draft_document = None
        self.draft_identifier = None
        self.draft_overlays = []
        self._temporary = None
        if data_dir is None:
            if window._settings is None:
                self._temporary = tempfile.TemporaryDirectory(prefix="smartsom-studio-")
                data_dir = self._temporary.name
            else:
                data_dir = QStandardPaths.writableLocation(
                    QStandardPaths.StandardLocation.AppDataLocation
                )
        self.catalog = TemplateCatalog(data_dir)
        self.recovery = RecoveryStore(data_dir)
        self.properties = PropertyEditor()
        window.property_layout.addWidget(self.properties, 1)
        self.properties.hide()
        self.properties.apply_requested.connect(self.apply_properties)
        self.properties.cancel_requested.connect(self.present_selection)
        self.properties.operation_requested.connect(self.operation)
        self._build_actions()
        self.timer = QTimer(self)
        self.timer.setInterval(60_000)
        self.timer.timeout.connect(self.snapshot_all)
        self.timer.start()

    @property
    def document(self):
        return self.window.current_document

    def status(self, text):
        self.window.status_message.setText(text)

    def error(self, title, exception):
        self.status(str(exception))
        QMessageBox.warning(self.window, title, str(exception))

    def _build_actions(self):
        w = self.window
        self.toolbar = QToolBar("Editing", w)
        self.toolbar.setObjectName("editingToolbar")
        w.addToolBarBreak()
        w.addToolBar(self.toolbar)
        self.toolbar.setMovable(False)
        self.file_menu = next(
            a.menu() for a in w.menuBar().actions() if a.text() == "File"
        )
        self.edit_menu = w.menuBar().addMenu("Edit")
        self.actions = {}

        def action(key, label, callback, shortcut=None, menu=None, toolbar=False):
            a = QAction(label, w)
            a.setObjectName(key + "Action")
            a.triggered.connect(callback)
            if shortcut:
                a.setShortcut(QKeySequence(shortcut))
            (menu or self.edit_menu).addAction(a)
            if toolbar:
                self.toolbar.addAction(a)
            self.actions[key] = a
            return a

        action(
            "save",
            "Save",
            lambda: self.save(),
            QKeySequence.StandardKey.Save,
            self.file_menu,
            True,
        )
        action(
            "saveAs",
            "Save As…",
            lambda: self.save(as_new=True),
            QKeySequence.StandardKey.SaveAs,
            self.file_menu,
        )
        action(
            "saveTemplate",
            "Save as Template…",
            self.save_as_template,
            menu=self.file_menu,
        )
        action(
            "templates", "Manage templates…", self.manage_templates, menu=self.file_menu
        )
        action("exportMap", "Export map…", self.export_dialog, menu=self.file_menu)
        action(
            "recovery",
            "Recover unsaved designs…",
            self.recover_dialog,
            menu=self.file_menu,
        )
        self.toolbar.addSeparator()
        action(
            "undo",
            "Undo",
            lambda: self.undo(-1),
            QKeySequence.StandardKey.Undo,
            toolbar=True,
        )
        action(
            "redo",
            "Redo",
            lambda: self.undo(1),
            QKeySequence.StandardKey.Redo,
            toolbar=True,
        )
        self.toolbar.addSeparator()
        self.mode_group = QActionGroup(w)
        self.mode_actions = {}
        for enabled, label in ((False, "Browse"), (True, "Edit")):
            a = QAction(label, w)
            a.setObjectName(label.lower() + "ModeAction")
            a.setCheckable(True)
            a.toggled.connect(
                lambda checked, value=enabled: self.set_mode(value) if checked else None
            )
            self.mode_group.addAction(a)
            self.toolbar.addAction(a)
            keep_exclusive_selection(a)
            self.mode_actions[enabled] = a
        self.toolbar.addSeparator()
        self.add_button = QToolButton()
        self.add_button.setObjectName("addResourceButton")
        self.add_button.setText("Add")
        menu = QMenu(self.add_button)
        for kind in editing.COLLECTIONS:
            a = menu.addAction(
                "AGV" if kind == "agv" else kind.replace("_", " ").capitalize()
            )
            a.triggered.connect(lambda checked=False, name=kind: self.choose_tool(name))
        menu.addSeparator()
        for kind, label in (
            ("obstacle_paint", "Paint obstacles"),
            ("obstacle_rectangle", "Rectangle obstacle"),
            ("obstacle_erase", "Erase obstacles"),
        ):
            a = menu.addAction(label)
            a.triggered.connect(lambda checked=False, name=kind: self.choose_tool(name))
        self.add_button.setMenu(menu)
        self.add_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.toolbar.addWidget(self.add_button)
        action("select", "Select", self.cancel_tool, toolbar=True)
        action("rotate", "Rotate 90°", self.rotate_selection, toolbar=True)
        action(
            "delete",
            "Delete",
            self.delete_selection,
            QKeySequence.StandardKey.Delete,
            toolbar=True,
        )
        action("copy", "Copy", self.copy_selection, QKeySequence.StandardKey.Copy)
        action(
            "cut",
            "Cut",
            lambda: self.copy_selection(cut=True),
            QKeySequence.StandardKey.Cut,
        )
        action("paste", "Paste", self.paste_selection, QKeySequence.StandardKey.Paste)
        self.continuous_action = QAction("Continuous placement", w)
        self.continuous_action.setCheckable(True)
        self.toolbar.addAction(self.continuous_action)
        self.tool_label = QLabel("Browse")
        self.toolbar.addWidget(self.tool_label)

    def attach(self, document):
        if document.undo_stack is not None:
            return
        document.undo_stack = QUndoStack(self)
        document.undo_stack.indexChanged.connect(
            lambda _, doc=document: self.update_titles(doc)
        )
        document.undo_stack.cleanChanged.connect(
            lambda _, doc=document: self.update_titles(doc)
        )
        document.interaction = MapInteraction(self, document)

    def detach(self, document):
        if getattr(document.interaction, "detached", False):
            return
        document.interaction.detached = True
        document.view.viewport().removeEventFilter(document.interaction)
        document.view.removeEventFilter(document.interaction)
        document.undo_stack.indexChanged.disconnect()
        document.undo_stack.cleanChanged.disconnect()

    def before_switch(self, index):
        target = (
            self.window.documents[index]
            if 0 <= index < len(self.window.documents)
            else None
        )
        if (
            self.active_document is not None
            and self.active_document is not target
            and not self.refreshing
        ):
            tabs = self.window.tabs
            tabs.blockSignals(True)
            tabs.setCurrentIndex(self.window.documents.index(self.active_document))
            tabs.blockSignals(False)
            if not self.resolve_pending():
                if self.active_document in self.window.documents:
                    tabs = self.window.tabs
                    tabs.blockSignals(True)
                    tabs.setCurrentIndex(
                        self.window.documents.index(self.active_document)
                    )
                    tabs.blockSignals(False)
                return False
            self.active_document.interaction.cancel()
            tabs.blockSignals(True)
            tabs.setCurrentIndex(index)
            tabs.blockSignals(False)
        self.active_document = target
        if target is not None:
            self.attach(target)
        return True

    def before_select(self, identifier):
        if self.refreshing:
            return True
        doc = self.document
        desired = (
            self.selecting_many
            if self.selecting_many is not None
            else ((identifier,) if identifier else ())
        )
        if doc and desired != doc.selected_ids and not self.resolve_pending():
            doc.scene.select_entities(doc.selected_ids)
            self.window._selecting = True
            self.window.resource_tree.setCurrentItem(
                self.window._tree_items.get(doc.selected_id)
            )
            self.window._selecting = False
            return False
        return True

    def select_many(self, identifiers):
        doc = self.document
        identifiers = tuple(dict.fromkeys(identifiers))
        if identifiers != doc.selected_ids and not self.resolve_pending():
            return False
        self.selecting_many = identifiers
        try:
            self.window.select_entity(identifiers[0] if identifiers else None)
        finally:
            self.selecting_many = None
        return True

    def present_selection(self):
        doc = self.document
        if doc is None:
            self.properties.hide()
            self.window.property_tree.show()
            self.update_actions()
            return
        self.window.property_tree.setVisible(not doc.edit_mode)
        self.properties.setVisible(doc.edit_mode)
        doc.footer.setText(
            "   Shift-click to select multiple  ·  Drag to move  ·  Add to place  ·  Esc to cancel"
            if doc.edit_mode
            else "   Hover for machine group  ·  Click to inspect  ·  Wheel to zoom  ·  Middle/Space-drag to pan"
        )
        if doc.edit_mode:
            values = [editing.resource(doc.design, i) for i in doc.selected_ids]
            value = (
                values[0]
                if values and all(type(v) is type(values[0]) for v in values)
                else None
                if values
                else doc.design
            )
            self.properties.show_resource(
                value, doc.design, max(1, len(values)), values
            )
        else:
            self.properties.pending = False
        doc.interaction.handles()
        self.update_actions()

    def update_actions(self):
        doc = self.document
        editing_enabled = (
            doc is not None and doc.edit_mode and self.draft_dialog is None
        )
        selected = bool(doc and doc.selected_ids)
        for mode, action in self.mode_actions.items():
            action.setEnabled(doc is not None)
            action.setChecked(bool(doc and doc.edit_mode) == mode)
        self.add_button.setEnabled(editing_enabled)
        self.continuous_action.setEnabled(editing_enabled)
        for key in ("select", "paste"):
            self.actions[key].setEnabled(editing_enabled)
        for key in ("rotate", "delete", "cut"):
            self.actions[key].setEnabled(editing_enabled and selected)
        self.actions["copy"].setEnabled(selected and self.draft_dialog is None)
        for key in ("save", "saveAs", "saveTemplate", "exportMap"):
            self.actions[key].setEnabled(doc is not None)
        for key, can, text in (
            (
                "undo",
                doc.undo_stack.canUndo() if doc else False,
                doc.undo_stack.undoText() if doc else "",
            ),
            (
                "redo",
                doc.undo_stack.canRedo() if doc else False,
                doc.undo_stack.redoText() if doc else "",
            ),
        ):
            self.actions[key].setEnabled(editing_enabled and can)
            self.actions[key].setText(key.capitalize() + (" " + text if text else ""))
        self.tool_label.setText(
            (
                doc.interaction.tool.replace("_", " ").capitalize()
                if doc and doc.edit_mode
                else "Browse"
            )
            + (" · draft" if self.draft_dialog else "")
        )
        self.properties.setEnabled(self.draft_dialog is None)
        self.window.binding_combo.setEnabled(
            doc is not None and self.draft_dialog is None
        )

    def set_mode(self, enabled):
        doc = self.document
        if doc is None:
            return False
        if doc.edit_mode == enabled:
            return True
        if doc.edit_mode != enabled and not self.resolve_pending():
            self.update_actions()
            return False
        doc.edit_mode = enabled
        doc.interaction.cancel()
        self.present_selection()
        self.status(
            "Edit · select, drag, or use Add"
            if enabled
            else "Browse · editing is disabled"
        )
        return True

    def resolve_pending(self):
        if self.refreshing:
            return True
        if self.draft_dialog is not None:
            box = QMessageBox(self.window)
            box.setWindowTitle("Unapplied map edits")
            box.setText("Apply this slot/binding edit before continuing?")
            box.setStandardButtons(
                QMessageBox.StandardButton.Apply
                | QMessageBox.StandardButton.Discard
                | QMessageBox.StandardButton.Cancel
            )
            choice = box.exec()
            if choice == QMessageBox.StandardButton.Cancel:
                return False
            if choice == QMessageBox.StandardButton.Apply:
                self.draft_dialog.apply_requested.emit()
                if self.draft_dialog is not None:
                    return False
            else:
                self.draft_dialog.reject()
        if not self.properties.pending:
            return True
        box = QMessageBox(self.window)
        box.setWindowTitle("Unapplied properties")
        box.setText("Apply these property changes before continuing?")
        box.setStandardButtons(
            QMessageBox.StandardButton.Apply
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel
        )
        choice = box.exec()
        if choice == QMessageBox.StandardButton.Apply:
            return self.apply_properties()
        if choice == QMessageBox.StandardButton.Discard:
            self.present_selection()
            return True
        return False

    def confirm_impact(self, lines):
        if not lines:
            return True
        box = QMessageBox(self.window)
        box.setWindowTitle("Confirm affected objects and bindings")
        box.setText(
            "Apply these changes? One Undo restores the complete previous design."
        )
        box.setInformativeText(
            "\n".join(lines[:8])
            + (f"\n… and {len(lines) - 8} more changes" if len(lines) > 8 else "")
        )
        box.setDetailedText("\n".join(lines))
        box.setStandardButtons(
            QMessageBox.StandardButton.Apply | QMessageBox.StandardButton.Cancel
        )
        box.setDefaultButton(QMessageBox.StandardButton.Cancel)
        return box.exec() == QMessageBox.StandardButton.Apply

    def commit(self, candidate, label, *, selection=None, confirm=False):
        doc = self.document
        if doc is None or not doc.edit_mode:
            return False
        try:
            editing.checked(candidate)
            if candidate == doc.design:
                return True
            if confirm and not self.confirm_impact(
                editing.impact(doc.design, candidate)
            ):
                return False
            doc.undo_stack.push(
                DesignCommand(doc, candidate, label, self.refresh_document, selection)
            )
            self.status(label + " · applied")
            return True
        except (ValueError, TypeError, ArithmeticError) as exc:
            self.properties.error.setText(str(exc))
            self.status(str(exc))
            return False

    def refresh_document(self, doc):
        self.refreshing = True
        try:
            old = doc.scene
            center = doc.view.mapToScene(doc.view.viewport().rect().center())
            transform = doc.view.transform()
            fit = doc.view._fit_to_map
            doc.interaction.clear_overlays()
            scene = FactoryScene(doc.design, doc.view)
            for layer in ("grid", "names", "ports"):
                scene.set_layer(layer, getattr(old, layer + "_visible"))
            scene.binding_mode = old.binding_mode
            doc.scene = scene
            doc.view.setScene(scene)
            valid = set(scene.entity_items)
            doc.selected_ids = tuple(i for i in doc.selected_ids if i in valid)
            doc.selected_id = doc.selected_ids[0] if doc.selected_ids else None
            scene.entity_selected.connect(
                lambda identifier, d=doc: (
                    self.window.select_entity(identifier)
                    if d is self.document
                    else None
                )
            )
            old.deleteLater()
            if doc is self.document:
                self.window._document_changed(self.window.tabs.currentIndex())
            scene.select_entities(doc.selected_ids)
            doc.view.setTransform(transform)
            doc.view.centerOn(center)
            if fit:
                doc.view.fit_map()
            self.update_titles(doc)
        finally:
            self.refreshing = False

    def update_titles(self, doc):
        if doc not in self.window.documents:
            return
        title = (
            doc.title
            + (" · Template" if doc.saved_as_template else "")
            + (" •" if doc.modified else "")
        )
        index = self.window.documents.index(doc)
        self.window.tabs.setTabText(index, title)
        self.window.tabs.setTabToolTip(
            index,
            str(doc.source_path)
            if doc.source_path
            else f"{doc.design.name} · Unsaved"
            + (f" · from {doc.origin.name}" if doc.origin else ""),
        )
        if doc is self.document:
            self.window.setWindowTitle(f"{title} — SmartSOM Studio")
            self.update_actions()

    def apply_properties(self):
        doc = self.active_document or self.document
        if doc is None or not doc.edit_mode:
            return False
        try:
            values = self.properties.values()
            old = self.properties.resource
            if old is None:
                return True
            if isinstance(old, FactoryDesign):
                candidate = editing.checked(replace(doc.design, **values))
            else:
                candidate = doc.design
                slot_capacity = values.pop("capacity_per_slot", None)
                for identifier in doc.selected_ids:
                    r = editing.resource(candidate, identifier)
                    changed = replace(r, **values)
                    if slot_capacity is not None:
                        changed = editing.with_slots(
                            changed,
                            tuple(
                                replace(slot, capacity=slot_capacity)
                                for slot in editing.slots_of(r)
                            ),
                        )
                    candidate = editing.update_resource(candidate, r, changed)
            pending = self.properties.pending
            self.properties.pending = False
            if not self.commit(candidate, "Edit properties", confirm=True):
                self.properties.pending = pending
                return False
            self.present_selection()
            return True
        except (ValueError, TypeError, ArithmeticError) as exc:
            self.properties.error.setText(str(exc))
            return False

    def undo(self, direction):
        if self.document and self.document.edit_mode and self.resolve_pending():
            self.cancel_tool()
            (
                self.document.undo_stack.undo
                if direction < 0
                else self.document.undo_stack.redo
            )()

    def choose_tool(self, tool):
        if self.document and self.document.edit_mode and self.resolve_pending():
            self.document.interaction.cancel()
            self.document.interaction.tool = tool
            self.document.view.setCursor(Qt.CursorShape.CrossCursor)
            self.update_actions()
            self.status(
                "Click for 1 × 1, or drag a rectangle; Esc cancels"
                if tool in editing.COLLECTIONS
                else tool.replace("_", " ")
            )

    def cancel_tool(self):
        if self.draft_dialog is not None:
            self.draft_dialog.reject()
        if self.document:
            self.document.interaction.cancel()
        self.update_actions()

    def rotate_selection(self):
        if self.document and self.document.edit_mode and self.resolve_pending():
            try:
                self.commit(
                    editing.rotate(self.document.design, self.document.selected_ids),
                    "Rotate selection",
                )
            except (ValueError, TypeError) as exc:
                self.status(str(exc))

    def delete_selection(self):
        if self.document and self.document.edit_mode and self.resolve_pending():
            candidate = editing.delete(self.document.design, self.document.selected_ids)
            self.commit(candidate, "Delete selection", selection=(), confirm=True)

    def copy_selection(self, checked=False, *, cut=False):
        doc = self.document
        if doc is None or not doc.selected_ids or not self.resolve_pending():
            return
        values = editing.copy_selection(doc.design, doc.selected_ids)
        copied = replace(
            doc.design,
            **{
                collection: tuple(r for r in values if isinstance(r, cls))
                for collection, cls, _ in editing.COLLECTIONS.values()
            },
        )
        copied = replace(copied, grid=replace(copied.grid, blocked_cells=()))
        payload = FactoryDesignFile(
            schema="smartsom.factory/v2", factory=copied
        ).model_dump_json(by_alias=True)
        mime = QMimeData()
        mime.setData(self.MIME, payload.encode())
        mime.setText(f"SmartSOM selection · {len(values)} objects")
        QApplication.clipboard().setMimeData(mime)
        if cut:
            self.delete_selection()
        else:
            self.status(
                f"Copied {len(values)} objects; external references are cleared on paste"
            )

    def paste_selection(self):
        if (
            not self.document
            or not self.document.edit_mode
            or not self.resolve_pending()
        ):
            return
        mime = QApplication.clipboard().mimeData()
        if not mime or not mime.hasFormat(self.MIME):
            self.status("Copy a Studio selection before pasting")
            return
        try:
            copied = FactoryDesignFile.model_validate_json(
                bytes(mime.data(self.MIME))
            ).factory
            self.choose_tool("paste")
            self.document.interaction.copied = iter_resources(copied)
            self.status("Click to paste the selection; Esc cancels")
        except (ValueError, TypeError) as exc:
            self.error("Cannot paste selection", exc)

    def operation(self, operation):
        doc = self.document
        if not doc or not doc.edit_mode or not self.resolve_pending():
            return
        identifier = doc.selected_id
        value = editing.resource(doc.design, identifier) if identifier else doc.design
        if operation == "rename":
            old_id = identifier or doc.design.factory_id
            new_id, accepted = QInputDialog.getText(
                self.window,
                "Rename ID",
                "New ID (all references update together)",
                text=old_id,
            )
            if accepted:
                try:
                    candidate = editing.rename(doc.design, identifier, new_id)
                    self.commit(
                        candidate,
                        "Rename ID",
                        selection=(new_id,) if identifier else (),
                    )
                except (ValueError, TypeError) as exc:
                    self.error("Cannot rename ID", exc)
            return
        if operation == "storage" and isinstance(value.storage, SlotStorage):
            dialog = QDialog(self.window)
            dialog.setWindowTitle("Convert to capacity pool")
            layout = QVBoxLayout(dialog)
            layout.addWidget(
                QLabel(
                    "Review the new total capacity. Old slot bindings will be removed."
                )
            )
            field = ValueField(
                "capacity", sum(s.capacity for s in value.storage.slots), doc.design
            )
            layout.addWidget(field)
            error = QLabel()
            layout.addWidget(error)
            buttons = QDialogButtonBox(
                QDialogButtonBox.StandardButton.Apply
                | QDialogButtonBox.StandardButton.Cancel
            )

            def apply():
                try:
                    candidate = editing.convert_storage(
                        doc.design, identifier, "pool", field.value()
                    )
                    if self.commit(candidate, "Convert to pool", confirm=True):
                        dialog.accept()
                except (ValueError, TypeError) as exc:
                    error.setText(str(exc))

            buttons.button(QDialogButtonBox.StandardButton.Apply).clicked.connect(apply)
            buttons.rejected.connect(dialog.reject)
            layout.addWidget(buttons)
            dialog.exec()
            return
        if operation in ("slots", "storage"):
            converting = operation == "storage"
            if converting:
                candidate = editing.convert_storage(doc.design, identifier, "slots", 1)
                value = editing.resource(candidate, identifier)
            dialog = SlotsDialog(value, self.window, converting=converting)
            dialog.cell_selected.connect(self.window._highlight_cell)

            def apply_slots():
                try:
                    slots, renames = dialog.values()
                    candidate = editing.set_slots(
                        doc.design, identifier, slots, renames
                    )
                    if self.commit(
                        candidate,
                        "Edit slots" if not converting else "Convert to slots",
                        confirm=True,
                    ):
                        dialog.accept()
                except (ValueError, TypeError) as exc:
                    dialog.error.setText(str(exc))

            dialog.apply_requested.connect(apply_slots)
        elif operation == "bindings":
            dialog = BindingsDialog(value, self.window)

            def apply_bindings():
                try:
                    changed = replace(
                        editing.resource(doc.design, identifier),
                        bindings=dialog.values(),
                    )
                    candidate = editing.checked(
                        editing.replace_resources(doc.design, {identifier: changed})
                    )
                    if self.commit(candidate, "Edit bindings"):
                        dialog.accept()
                except (ValueError, TypeError) as exc:
                    dialog.error.setText(str(exc))

            dialog.apply_requested.connect(apply_bindings)
            dialog.draft_changed.connect(self.show_draft_bindings)
        else:
            return
        self.cancel_tool()
        self.draft_dialog, self.draft_document, self.draft_identifier = (
            dialog,
            doc,
            identifier,
        )
        doc.scene.edit_gesture = True
        doc.interaction.clear_overlays()
        dialog.finished.connect(self.end_draft)
        dialog.show()
        if isinstance(dialog, BindingsDialog):
            self.show_draft_bindings()
        self.update_actions()

    def draft_click(self, item, cell):
        if isinstance(self.draft_dialog, SlotsDialog):
            self.draft_dialog.click_cell(cell)
        elif isinstance(self.draft_dialog, BindingsDialog) and item is not None:
            target = editing.target_at(self.draft_document.design, item.entity_id, cell)
            if target is not None:
                self.draft_dialog.toggle_target(target)

    def clear_draft_overlays(self):
        for item in self.draft_overlays:
            try:
                if item.scene():
                    item.scene().removeItem(item)
            except RuntimeError:
                pass
        self.draft_overlays = []

    def show_draft_bindings(self):
        if not isinstance(self.draft_dialog, BindingsDialog):
            return
        from PySide6.QtCore import QRectF

        self.clear_draft_overlays()
        scene = self.draft_document.scene
        for item in scene._binding_highlights.values():
            item.hide()
        for binding in self.draft_dialog.values():
            target = binding.target
            cell = target_cell(self.draft_document.design, target)
            if cell is None:
                continue
            if hasattr(target, "slot_id"):
                rect = QRectF(
                    cell.x * CELL_SIZE + 1,
                    cell.y * CELL_SIZE + 1,
                    CELL_SIZE - 2,
                    CELL_SIZE - 2,
                )
            else:
                owner = scene.entity_items[target_owner_id(target)]
                rect = owner.mapRectToScene(owner.shape().boundingRect()).adjusted(
                    1, 1, -1, -1
                )
            item = scene.addRect(rect, QPen(Qt.PenStyle.NoPen), BINDING_FILL)
            item.setZValue(2)
            item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
            self.draft_overlays.append(item)

    def end_draft(self, *_):
        dialog = self.draft_dialog
        doc = self.draft_document
        self.clear_draft_overlays()
        self.draft_dialog = self.draft_document = self.draft_identifier = None
        if doc:
            doc.scene.edit_gesture = False
            doc.scene.slot_highlight.hide()
            doc.scene.update_bindings(doc.selected_id)
        if dialog:
            dialog.deleteLater()
        if doc:
            doc.interaction.handles()
        self.update_actions()

    def save_to(
        self, doc, path, *, expected_digest=None, as_template=False, origin=None
    ):
        path = Path(path).expanduser().resolve()
        for other in self.window.documents:
            if other is not doc and other.source_path == path:
                raise ValueError(
                    "This file is already open in another tab. Close that tab before replacing it."
                )
        digest = save_factory_design(path, doc.design, expected_digest=expected_digest)
        # Publish the successful disk write before optional catalog bookkeeping.
        doc.source_path, doc.source_digest, doc.title = path, digest, path.name
        doc.saved_as_template = as_template
        if origin is not None:
            doc.origin = replace(origin, target_digest=digest)
        doc.undo_stack.setClean()
        self.update_titles(doc)
        try:
            self.recovery.remove(doc.recovery_id)
            if as_template and doc.origin and doc.origin.builtin:
                self.catalog.use_local(doc.origin.builtin)
        except (OSError, ValueError) as exc:
            self.error("Factory saved; local bookkeeping needs attention", exc)
        self.status(f"Saved {path}")
        return True

    def save(self, checked=False, *, as_new=False):
        if not self.resolve_pending():
            return False
        doc = self.document
        if doc is None:
            return False
        origin = None
        as_template = doc.saved_as_template
        if not as_new and doc.source_path is None and doc.origin:
            choice = TemplateSaveDialog(doc.origin, self.window)
            if choice.exec() != QDialog.DialogCode.Accepted:
                return False
            if choice.update_template.isChecked():
                origin = doc.origin
                path, digest, as_template = (
                    origin.target_path,
                    origin.target_digest,
                    True,
                )
            else:
                as_new = True
        elif not as_new and doc.source_path:
            path, digest = doc.source_path, doc.source_digest
        else:
            as_new = True
        if as_new:
            name, _ = QFileDialog.getSaveFileName(
                self.window,
                "Save factory design",
                str(doc.source_path or "factory.yaml"),
                "Factory YAML (*.yaml *.yml)",
            )
            if not name:
                return False
            path = Path(name).expanduser().resolve()
            if not path.suffix:
                path = path.with_suffix(".yaml")
            digest = doc.source_digest if path == doc.source_path else None
            as_template = False
        try:
            if as_new and path != doc.source_path:
                digest = file_digest(path)
            if origin and origin.builtin:
                path.parent.mkdir(parents=True, exist_ok=True)
            return self.save_to(
                doc,
                path,
                expected_digest=digest,
                as_template=as_template,
                origin=origin,
            )
        except (ValueError, OSError, TypeError) as exc:
            self.status(str(exc))
            box = QMessageBox(self.window)
            box.setWindowTitle("Could not save factory design")
            box.setText(str(exc))
            save_as = box.addButton("Save As…", QMessageBox.ButtonRole.AcceptRole)
            reload_button = box.addButton(
                "Reload and discard changes", QMessageBox.ButtonRole.DestructiveRole
            )
            reload_button.setEnabled(doc.source_path is not None)
            box.addButton(QMessageBox.StandardButton.Cancel)
            box.exec()
            if box.clickedButton() is save_as:
                return self.save(as_new=True)
            if box.clickedButton() is reload_button:
                try:
                    design, new_digest = load_factory_design(doc.source_path)
                    doc.design, doc.source_digest = design, new_digest
                    doc.undo_stack.clear()
                    self.refresh_document(doc)
                    return False
                except (OSError, ValueError, TypeError) as reload_error:
                    self.error("Cannot reload", reload_error)
            return False

    def save_as_template(self):
        if self.save(as_new=True):
            doc = self.document
            try:
                self.catalog.register(doc.source_path)
                doc.saved_as_template = True
                self.update_titles(doc)
            except (OSError, ValueError) as exc:
                self.error("File saved, but template registration failed", exc)

    def can_close(self, doc):
        if not self.resolve_pending():
            return False
        if not doc.modified:
            self.recovery.remove(doc.recovery_id)
            return True
        box = QMessageBox(self.window)
        box.setWindowTitle("Unsaved factory design")
        box.setText(f"Save changes to {doc.title} before closing?")
        box.setStandardButtons(
            QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel
        )
        box.setDefaultButton(QMessageBox.StandardButton.Save)
        choice = box.exec()
        if choice == QMessageBox.StandardButton.Cancel:
            return False
        if choice == QMessageBox.StandardButton.Save:
            if doc is not self.document:
                self.window.tabs.setCurrentIndex(self.window.documents.index(doc))
                if doc is not self.document:
                    return False
            if not self.save():
                return False
        self.recovery.remove(doc.recovery_id)
        return True

    def snapshot_all(self):
        for doc in self.window.documents:
            try:
                if doc.modified:
                    self.recovery.snapshot(doc)
                else:
                    self.recovery.remove(doc.recovery_id)
            except (OSError, ValueError) as exc:
                self.status(f"Recovery snapshot failed: {exc}")

    def recover_dialog(self):
        entries = self.recovery.entries()
        if not entries:
            self.status("No recovery snapshots available")
            return
        paths = {}
        failures = []
        for path in entries:
            try:
                design, _ = load_factory_design(path)
                paths[f"{design.name} · {path.name}"] = path
            except (OSError, ValueError, TypeError) as exc:
                failures.append(f"{path.name}: {exc}")
        if failures:
            self.error("Some recovery snapshots could not be read", "\n".join(failures))
        if not paths:
            return
        label, accepted = QInputDialog.getItem(
            self.window,
            "Recover unsaved design",
            "Restore as a new document (source file is unchanged)",
            list(paths),
            editable=False,
        )
        if accepted:
            try:
                design, _ = load_factory_design(paths[label])
                doc = self.window.add_design(design, title="Recovered " + design.name)
                if doc is None:
                    return
                doc.undo_stack.resetClean()
                self.recovery.snapshot(doc)
                self.recovery.remove(paths[label].stem)
                self.update_titles(doc)
            except (OSError, ValueError, TypeError) as exc:
                self.error("Cannot restore design", exc)

    def export_dialog(self):
        if self.document is None or not self.resolve_pending():
            return
        dialog = ExportDialog(self.document, self.window)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        suffix = dialog.format.currentText().lower()
        path, _ = QFileDialog.getSaveFileName(
            self.window,
            "Export factory map",
            str(
                (self.document.source_path or Path("factory.yaml")).with_suffix(
                    "." + suffix
                )
            ),
            f"{suffix.upper()} image (*.{suffix})",
        )
        if path:
            if not Path(path).suffix:
                path += "." + suffix
            try:
                export_map(
                    path,
                    self.document.design,
                    cell_pixels=dialog.resolution.value(),
                    **dialog.options(),
                )
                self.status(f"Exported {path}")
            except (OSError, ValueError) as exc:
                self.error("Cannot export map", exc)

    def manage_templates(self):
        dialog = QDialog(self.window)
        dialog.setWindowTitle("Factory templates")
        dialog.resize(600, 380)
        layout = QVBoxLayout(dialog)
        entries = QListWidget()
        layout.addWidget(entries)

        def refresh():
            entries.clear()
            records = self.catalog.records()
            for n in (1, 2):
                entries.addItem(
                    f"Template {n}"
                    + (" · Local" if n in records["local"] else " · Original")
                )
            for path in records["files"]:
                entries.addItem(path)

        def selected():
            row = entries.currentRow()
            return (
                row + 1
                if row in (0, 1)
                else entries.currentItem().text()
                if row >= 0
                else None
            )

        def new():
            target = selected()
            if target is not None:
                try:
                    self.window.new_template(target) if isinstance(
                        target, int
                    ) else self.window.new_from_path(target)
                    dialog.accept()
                except (OSError, ValueError) as exc:
                    self.error("Cannot load template", exc)

        def register():
            path, _ = QFileDialog.getOpenFileName(
                dialog,
                "Register factory YAML template",
                "",
                "Factory YAML (*.yaml *.yml)",
            )
            if path:
                try:
                    self.catalog.register(path)
                    refresh()
                except (OSError, ValueError) as exc:
                    self.error("Cannot register template", exc)

        def remove():
            target = selected()
            if target is None:
                return
            try:
                if isinstance(target, int):
                    self.catalog.restore_original(target)
                else:
                    self.catalog.forget(target)
                refresh()
            except (OSError, ValueError) as exc:
                self.error("Cannot update template records", exc)

        for label, callback in (
            ("New from selected template", new),
            ("Register existing YAML…", register),
            ("Restore original / remove record (keeps file)", remove),
        ):
            button = QPushButton(label)
            button.clicked.connect(callback)
            layout.addWidget(button)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        try:
            refresh()
            dialog.exec()
        except (OSError, ValueError) as exc:
            self.error("Cannot read template catalog", exc)
