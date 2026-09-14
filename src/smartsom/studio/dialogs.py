"""Native editor dialogs; modeless slot/binding editors cooperate with the map."""

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from smartsom.domain.factory_design import (
    Cell,
    ChargerTarget,
    InspectionStationDesign,
    PortBinding,
    ScrapBinTarget,
    SlotDesign,
    operation_type_key,
    target_owner_id,
)
from smartsom.studio import editing
from smartsom.studio.editing import default_binding, next_id, slots_of
from smartsom.studio.export import export_scene, render_image


class OperationCatalogDialog(QDialog):
    """One unapplied catalog transaction, including its editing preference."""

    def __init__(self, design, mode, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Factory operation types")
        self.resize(460, 440)
        self.design = design
        layout = QVBoxLayout(self)
        self.mode = QComboBox()
        self.mode.addItem("Automatic · expand for new machines", "auto")
        self.mode.addItem("Manual · keep my catalog", "manual")
        self.mode.setCurrentIndex(0 if mode == "auto" else 1)
        self.mode.currentIndexChanged.connect(self.change_mode)
        layout.addWidget(self.mode)
        self.types = QListWidget()
        layout.addWidget(self.types)
        row = QHBoxLayout()
        self.add_button = QPushButton("Add operation type")
        self.remove_button = QPushButton("Delete selected type")
        self.add_button.clicked.connect(self.add_type)
        self.remove_button.clicked.connect(self.remove_type)
        row.addWidget(self.add_button)
        row.addWidget(self.remove_button)
        layout.addLayout(row)
        hint = QLabel(
            "Unused types may remain. Change machine selections before deleting a type in use."
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)
        self.error = QLabel()
        self.error.setWordWrap(True)
        layout.addWidget(self.error)
        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Apply
            | QDialogButtonBox.StandardButton.Cancel
        )
        self.buttons.button(QDialogButtonBox.StandardButton.Apply).clicked.connect(
            self.accept
        )
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        self.refresh()

    def refresh(self):
        self.types.clear()
        for identifier in sorted(self.design.operation_types, key=operation_type_key):
            count = sum(identifier in m.operation_types for m in self.design.machines)
            item = QListWidgetItem(
                f"{identifier.replace('_', ' ').capitalize()} · {count} {'machine' if count == 1 else 'machines'}"
            )
            item.setData(Qt.ItemDataRole.UserRole, identifier)
            self.types.addItem(item)
        if self.types.count():
            self.types.setCurrentRow(self.types.count() - 1)

    def change_mode(self):
        if self.mode.currentData() == "auto":
            self.design = editing.auto_operation_catalog(self.design)
            self.refresh()

    def add_type(self):
        self.design = editing.add_operation_type(self.design)
        self.mode.setCurrentIndex(1)
        self.error.clear()
        self.refresh()

    def remove_type(self):
        item = self.types.currentItem()
        if item is None:
            return
        try:
            self.design = editing.remove_operation_type(
                self.design, item.data(Qt.ItemDataRole.UserRole)
            )
        except ValueError as exc:
            self.error.setText(str(exc))
            return
        self.mode.setCurrentIndex(1)
        self.error.clear()
        self.refresh()


class MachineCapabilitiesDialog(QDialog):
    def __init__(self, catalog, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Choose machine capabilities")
        self.resize(420, 350)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Select one or more existing operation types."))
        self.types = QListWidget()
        for identifier in sorted(catalog, key=operation_type_key):
            item = QListWidgetItem(identifier.replace("_", " ").capitalize())
            item.setData(Qt.ItemDataRole.UserRole, identifier)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Unchecked)
            self.types.addItem(item)
        layout.addWidget(self.types)
        self.error = QLabel()
        layout.addWidget(self.error)
        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

    def values(self):
        return tuple(
            self.types.item(i).data(Qt.ItemDataRole.UserRole)
            for i in range(self.types.count())
            if self.types.item(i).checkState() == Qt.CheckState.Checked
        )

    def accept(self):
        if not self.values():
            self.error.setText("Select at least one operation type.")
            return
        super().accept()


class TemplateSaveDialog(QDialog):
    def __init__(self, origin, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Save template changes")
        self.resize(480, 230)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(f"Where should changes from {origin.name} be saved?"))
        self.new_file = QRadioButton("Save as a new factory file")
        self.new_file.setChecked(True)
        self.update_template = QRadioButton(
            f"Update {origin.name}" + (" · local version" if origin.builtin else "")
        )
        layout.addWidget(self.new_file)
        layout.addWidget(self.update_template)
        detail = QLabel(
            "Future designs use the updated template. Existing designs stay unchanged.\n"
            + (
                "The bundled original remains available through Restore original."
                if origin.builtin
                else str(origin.target_path)
            )
        )
        detail.setWordWrap(True)
        layout.addWidget(detail)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)


class DraftDialog(QDialog):
    apply_requested = Signal()

    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(False)
        self.resize(540, 410)
        self.layout_body = QVBoxLayout(self)
        self.help = QLabel()
        self.help.setWordWrap(True)
        self.layout_body.addWidget(self.help)
        self.table = QTableWidget()
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.layout_body.addWidget(self.table, 1)
        self.error = QLabel()
        self.error.setWordWrap(True)
        self.error.setStyleSheet("color: #a55549")
        self.layout_body.addWidget(self.error)
        self.actions = QHBoxLayout()
        self.layout_body.addLayout(self.actions)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Apply
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Apply).clicked.connect(
            self.apply_requested
        )
        buttons.rejected.connect(self.reject)
        self.layout_body.addWidget(buttons)


class SlotsDialog(DraftDialog):
    cell_selected = Signal(object)

    def __init__(self, value, parent=None, *, converting=False):
        super().__init__(
            ("Convert to Slots · " if converting else "Edit slots · ") + value.name,
            parent,
        )
        self.value_resource = value
        self.original_ids = (
            {s.slot_id for s in slots_of(value)} if not converting else set()
        )
        self.help.setText(
            "Click a cell on the map to select or add a slot. Changes stay here until Apply.\n"
            + (
                "Conversion removes incompatible pool bindings; confirm the new capacities before applying."
                if converting
                else "Unused cells may remain empty. IDs remain stable when cells move."
            )
        )
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(
            ["SLOT ID", "LOCAL X", "LOCAL Y", "CAPACITY"]
        )
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        for slot in slots_of(value):
            self.append_slot(
                slot, slot.slot_id if slot.slot_id in self.original_ids else None
            )
        self.table.currentCellChanged.connect(self._select_row)
        self.table.cellChanged.connect(self._summary)
        for title, callback in (
            ("Add slot", self.add_next),
            ("Remove selected", self.remove_selected),
            ("Rename slot ID…", self.rename_slot),
        ):
            button = QPushButton(title)
            button.clicked.connect(callback)
            self.actions.addWidget(button)
        self._summary()

    def append_slot(self, slot, original=None):
        row = self.table.rowCount()
        self.table.insertRow(row)
        for col, text in enumerate(
            (slot.slot_id, slot.local_cell.x, slot.local_cell.y, slot.capacity)
        ):
            item = QTableWidgetItem(str(text))
            if (
                col == 0
                or col == 3
                and isinstance(self.value_resource, InspectionStationDesign)
            ):
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            if col == 0:
                item.setData(Qt.ItemDataRole.UserRole, original)
            self.table.setItem(row, col, item)

    def values(self):
        slots, renames = [], {}
        for row in range(self.table.rowCount()):
            identifier = self.table.item(row, 0).text()
            x, y, capacity = (int(self.table.item(row, i).text()) for i in (1, 2, 3))
            slots.append(SlotDesign(identifier, Cell(x, y), capacity))
            original = self.table.item(row, 0).data(Qt.ItemDataRole.UserRole)
            if original and original != identifier:
                renames[original] = identifier
        return tuple(slots), renames

    def click_cell(self, cell):
        f = self.value_resource.footprint
        x, y = cell.x - f.x, cell.y - f.y
        if not (0 <= x < f.width and 0 <= y < f.height):
            return
        try:
            slots, _ = self.values()
        except (ValueError, TypeError) as exc:
            self.error.setText(str(exc))
            return
        for row, slot in enumerate(slots):
            if slot.local_cell == Cell(x, y):
                self.table.selectRow(row)
                return
        identifier = next_id("slot", self.original_ids | {s.slot_id for s in slots})
        self.append_slot(SlotDesign(identifier, Cell(x, y)))
        self.table.selectRow(self.table.rowCount() - 1)
        self._summary()

    def add_next(self):
        try:
            slots, _ = self.values()
        except (ValueError, TypeError) as exc:
            self.error.setText(str(exc))
            return
        used = {s.local_cell for s in slots}
        f = self.value_resource.footprint
        for y in range(f.height):
            for x in range(f.width):
                if Cell(x, y) not in used:
                    self.click_cell(Cell(f.x + x, f.y + y))
                    return
        self.error.setText(
            "Every cell already has a slot. Resize the resource to add space."
        )

    def remove_selected(self):
        for row in sorted(
            {i.row() for i in self.table.selectedIndexes()}, reverse=True
        ):
            self.table.removeRow(row)
        self._summary()

    def rename_slot(self):
        row = self.table.currentRow()
        if row < 0:
            return
        item = self.table.item(row, 0)
        value, accepted = QInputDialog.getText(
            self,
            "Rename slot ID",
            "New ID (references update on Apply)",
            text=item.text(),
        )
        if accepted:
            item.setText(value)

    def _select_row(self, row, *_):
        if row < 0:
            return
        try:
            x, y = (int(self.table.item(row, col).text()) for col in (1, 2))
            f = self.value_resource.footprint
            self.cell_selected.emit(Cell(f.x + x, f.y + y))
        except (ValueError, AttributeError):
            pass

    def _summary(self, *_):
        try:
            slots, _ = self.values()
            self.error.setText(
                f"{len(slots)} slots · total capacity {sum(s.capacity for s in slots)}"
            )
        except (ValueError, TypeError, AttributeError):
            self.error.setText(
                "Enter valid integer coordinates and positive slot capacities."
            )


class BindingsDialog(DraftDialog):
    draft_changed = Signal()

    def __init__(self, port, parent=None):
        super().__init__(f"Edit bindings · {port.name}", parent)
        self.help.setText(
            "Click target slots or resources on the map to add/remove them.\nEach target has its own operation permissions. Apply commits the whole selection."
        )
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(["TARGET", "OPERATIONS"])
        self.table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch
        )
        self.table.setColumnWidth(1, 150)
        for binding in port.bindings:
            self.append_binding(binding)
        remove = QPushButton("Remove selected")
        remove.clicked.connect(self.remove_selected)
        self.actions.addWidget(remove)

    def append_binding(self, binding):
        row = self.table.rowCount()
        self.table.insertRow(row)
        target = binding.target
        label = target_owner_id(target) + (
            " / " + target.slot_id
            if hasattr(target, "slot_id")
            else " · whole resource"
        )
        item = QTableWidgetItem(label)
        item.setData(Qt.ItemDataRole.UserRole, target)
        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        self.table.setItem(row, 0, item)
        operations = QComboBox()
        if isinstance(target, ChargerTarget):
            operations.addItem("Charge", ("charge",))
        elif isinstance(target, ScrapBinTarget):
            operations.addItem("Drop off", ("drop_off",))
        else:
            for label, values in (
                ("Pickup + drop off", ("pickup", "drop_off")),
                ("Pickup", ("pickup",)),
                ("Drop off", ("drop_off",)),
            ):
                operations.addItem(label, values)
        for index in range(operations.count()):
            if set(operations.itemData(index)) == set(binding.operations):
                operations.setCurrentIndex(index)
        operations.currentIndexChanged.connect(self.draft_changed)
        self.table.setCellWidget(row, 1, operations)

    def values(self):
        return tuple(
            PortBinding(
                self.table.item(row, 0).data(Qt.ItemDataRole.UserRole),
                tuple(self.table.cellWidget(row, 1).currentData()),
            )
            for row in range(self.table.rowCount())
        )

    def toggle_target(self, target):
        for row in range(self.table.rowCount()):
            if self.table.item(row, 0).data(Qt.ItemDataRole.UserRole) == target:
                self.table.removeRow(row)
                self.draft_changed.emit()
                return
        self.append_binding(default_binding(target))
        self.draft_changed.emit()

    def remove_selected(self):
        for row in sorted(
            {i.row() for i in self.table.selectedIndexes()}, reverse=True
        ):
            self.table.removeRow(row)
        self.draft_changed.emit()


class ExportDialog(QDialog):
    def __init__(self, document, parent=None):
        super().__init__(parent)
        self.document = document
        self.setWindowTitle("Export factory map")
        self.resize(520, 550)
        layout = QVBoxLayout(self)
        self.preview = QLabel()
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumHeight(230)
        layout.addWidget(self.preview)
        form = QFormLayout()
        self.format = QComboBox()
        self.format.addItems(["PNG", "SVG"])
        form.addRow("Format", self.format)
        self.resolution = QSpinBox()
        self.resolution.setRange(1, 200)
        self.resolution.setValue(40)
        form.addRow("Pixels per cell", self.resolution)
        self.boxes = {}
        for name, checked in (("grid", True), ("numbers", False), ("ports", True)):
            box = QCheckBox(name.capitalize())
            box.setChecked(checked)
            box.toggled.connect(self.refresh)
            form.addRow(box)
            self.boxes[name] = box
        self.bindings = QComboBox()
        for text, data in (
            ("None", "none"),
            ("Selected object's bindings", "selected"),
            ("All bindings", "all"),
        ):
            self.bindings.addItem(text, data)
        self.bindings.currentIndexChanged.connect(self.refresh)
        form.addRow("Bindings", self.bindings)
        layout.addLayout(form)
        self.details = QLabel(
            "Full map. Selection, handles, hover and placement previews are excluded."
        )
        self.details.setWordWrap(True)
        layout.addWidget(self.details)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.refresh()

    def options(self):
        return {
            **{k: w.isChecked() for k, w in self.boxes.items()},
            "bindings": self.bindings.currentData(),
            "selected": self.document.selected_id,
        }

    def refresh(self):
        scene = export_scene(self.document.design, **self.options())
        try:
            size = max(
                self.document.design.grid.width, self.document.design.grid.height
            )
            image = render_image(scene, max(1, min(40, 440 // size)))
            self.preview.setPixmap(
                QPixmap.fromImage(image).scaled(
                    450,
                    240,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
        finally:
            scene.deleteLater()
