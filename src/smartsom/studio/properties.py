"""Browse properties and typed, explicitly applied authoring forms."""

import re
from collections.abc import Mapping
from dataclasses import fields, is_dataclass, replace
from decimal import Decimal
from enum import Enum

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLayout,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from smartsom.domain.factory_design import (
    BatteryDesign,
    BufferDesign,
    FactoryDesign,
    InspectionStationDesign,
    PoolStorage,
    PortDesign,
    QualityMode,
    SlotStorage,
    world_cell,
)


def field_label(value):
    if value == "operation_types":
        return "Supported operations"
    return (
        value.replace("_", " ").capitalize().replace("Agv", "AGV").replace(" id", " ID")
    )


def type_label(value):
    name = type(value).__name__
    labels = {
        "FactoryDesign": "Factory",
        "GridDesign": "Grid",
        "PoolStorage": "Capacity pool",
        "SlotStorage": "Slot storage",
        "PortBinding": "Binding",
        "BufferSlotTarget": "Buffer slot",
        "InspectionSlotTarget": "Inspection slot",
        "AGVDesign": "AGV",
        "InspectionStationDesign": "Inspection station",
    }
    if name in labels:
        return labels[name]
    name = name.removesuffix("Design").removesuffix("Target")
    return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", name).capitalize()


def object_fields(value):
    if is_dataclass(value):
        return [(field.name, getattr(value, field.name)) for field in fields(value)]
    if hasattr(type(value), "model_fields"):
        return [(name, getattr(value, name)) for name in type(value).model_fields]
    if isinstance(value, Mapping):
        return list(value.items())
    return None


class PropertyTree(QTreeWidget):
    cell_selected = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("propertyTree")
        self.setColumnCount(2)
        self.setHeaderLabels(["PROPERTY", "VALUE"])
        self.setAlternatingRowColors(False)
        self.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.setIndentation(14)
        self.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Interactive)
        self.header().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.setColumnWidth(0, 132)
        self._footprint = None
        self._summary_mode = False
        self.itemClicked.connect(self._item_clicked)

    def show_resource(self, resource):
        self.clear()
        self._footprint = getattr(resource, "footprint", None)
        self._summary_mode = isinstance(resource, FactoryDesign)
        values = object_fields(resource)
        if values is None:
            return
        for key, value in values:
            self._append(self, str(key), value)
        self.expandToDepth(0)

    def _append(self, parent, key, value):
        item = QTreeWidgetItem(parent, [field_label(key), ""])
        item.setToolTip(0, field_label(key))
        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        item.setData(0, Qt.ItemDataRole.UserRole + 1, key)
        nested = object_fields(value)
        if nested is not None:
            item.setText(
                1, type_label(value) if not isinstance(value, Mapping) else "Object"
            )
            for child_key, child in nested:
                self._append(item, str(child_key), child)
            if hasattr(value, "local_cell") and self._footprint is not None:
                cell = world_cell(self._footprint, value.local_cell)
                item.setData(0, Qt.ItemDataRole.UserRole, cell)
                item.setToolTip(
                    0, f"Select to highlight cell ({cell.x}, {cell.y}) on the map"
                )
                QTreeWidgetItem(item, ["World cell", f"({cell.x}, {cell.y})"])
        elif key == "operation_types":
            item.setText(1, ", ".join(field_label(v) for v in value) or "Unspecified")
            item.setToolTip(
                0, "Processing categories supported by this machine; not a job route."
            )
        elif isinstance(value, (tuple, list)):
            item.setText(1, f"{len(value)} items")
            if self._summary_mode:
                item.setToolTip(
                    1,
                    "Choose a resource in the resource tree to inspect its properties.",
                )
                return item
            for index, child in enumerate(value):
                name = next(
                    (
                        str(getattr(child, attr))
                        for attr in ("slot_id", "mode_id", "quality_mode_id", "port_id")
                        if hasattr(child, attr)
                    ),
                    str(index + 1),
                )
                self._append(item, name, child)
        elif isinstance(value, Enum):
            item.setText(1, str(value.value))
        elif value is None:
            if key.endswith("capacity"):
                label = "Unlimited"
            elif key == "battery":
                label = "Not modeled"
            elif key.endswith("_id") or key == "owner":
                label = "Unassigned"
            else:
                label = "Not set"
            item.setText(1, label)
        elif isinstance(value, bool):
            item.setText(1, "True" if value else "False")
        elif isinstance(value, Decimal):
            item.setText(1, format(value, "f"))
        else:
            item.setText(1, str(value))
        item.setToolTip(1, item.text(1))
        return item

    def _item_clicked(self, item, column):
        cell = item.data(0, Qt.ItemDataRole.UserRole)
        if cell is not None:
            self.cell_selected.emit(cell)

    def reveal_field(self, path):
        parts = str(path or "").replace("[", ".").replace("]", "").split(".")
        stack = [self.topLevelItem(i) for i in range(self.topLevelItemCount())]
        while stack:
            item = stack.pop(0)
            if item.data(0, Qt.ItemDataRole.UserRole + 1) in parts:
                parent = item.parent()
                while parent:
                    parent.setExpanded(True)
                    parent = parent.parent()
                self.setCurrentItem(item)
                self.scrollToItem(item)
                return
            stack.extend(item.child(i) for i in range(item.childCount()))


class ValueField(QWidget):
    """Native typed controls, recursively composed for capability groups."""

    changed = Signal()

    def __init__(self, key, value, design, parent=None):
        super().__init__(parent)
        self.key, self.original, self.design = key, value, design
        self.children_fields = {}
        self.rows = []
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        if key == "battery":
            self.enabled_box = QCheckBox("Model battery")
            self.enabled_box.setChecked(value is not None)
            self.nested = ValueField(
                "battery_parameters", value or BatteryDesign(), design
            )
            self.nested.setEnabled(value is not None)
            self.enabled_box.toggled.connect(self.nested.setEnabled)
            self.enabled_box.toggled.connect(self.changed)
            self.nested.changed.connect(self.changed)
            layout.addWidget(self.enabled_box)
            layout.addWidget(self.nested)
        elif key == "operation_types":
            choices = dict.fromkeys(
                [f"operation_{n}" for n in range(1, 5)]
                + [
                    kind
                    for machine in design.machines
                    for kind in machine.operation_types
                ]
                + list(value)
            )
            for operation_type in choices:
                box = QCheckBox(field_label(operation_type))
                box.setChecked(operation_type in value)
                box.toggled.connect(self.changed)
                layout.addWidget(box)
                self.children_fields[operation_type] = box
            hint = QLabel(
                "Select all supported categories. Machines may share categories. None selected means unspecified."
            )
            hint.setWordWrap(True)
            layout.addWidget(hint)
        elif key == "allowed_headings":
            for heading in ("north", "east", "south", "west"):
                box = QCheckBox(heading.capitalize())
                box.setChecked(heading in value)
                box.toggled.connect(self.changed)
                layout.addWidget(box)
                self.children_fields[heading] = box
        elif key == "quality_modes":
            self.row_layout = QVBoxLayout()
            layout.addLayout(self.row_layout)
            for mode in value:
                self.add_mode(mode)
            button = QPushButton("Add quality mode")
            button.clicked.connect(self.new_mode)
            layout.addWidget(button)
        elif is_dataclass(value):
            form = QFormLayout()
            form.setFieldGrowthPolicy(
                QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
            )
            form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
            layout.addLayout(form)
            for f in fields(value):
                if f.name == "mode":
                    continue
                if f.name == "blocked_cells":
                    form.addRow(
                        "Obstacles",
                        QLabel(
                            f"{len(value.blocked_cells)} cells · use obstacle tools"
                        ),
                    )
                    continue
                if f.name == "rotation":
                    form.addRow("Rotation", QLabel(f"{value.rotation}° · use Rotate"))
                    continue
                child = ValueField(f.name, getattr(value, f.name), design)
                child.changed.connect(self.changed)
                self.children_fields[f.name] = child
                form.addRow(field_label(f.name), child)
        elif key == "machine_id" or key in ("role", "initial_heading"):
            self.control = QComboBox()
            if key == "machine_id":
                self.control.addItem("Unassigned", None)
                for machine in design.machines:
                    self.control.addItem(
                        f"{machine.name} · {machine.machine_id}", machine.machine_id
                    )
            else:
                choices = (
                    (
                        "storage",
                        "machine_pre",
                        "machine_post",
                        "system_input",
                        "system_output",
                    )
                    if key == "role"
                    else ("north", "east", "south", "west")
                )
                for choice in choices:
                    self.control.addItem(choice.replace("_", " ").capitalize(), choice)
            self.control.setCurrentIndex(self.control.findData(value))
            self.control.currentIndexChanged.connect(self.changed)
            layout.addWidget(self.control)
        else:
            self.control = QLineEdit("" if value is None else str(value))
            self.control.setMinimumWidth(100)
            self.control.setObjectName(f"edit_{key}")
            self.control.textChanged.connect(self.changed)
            if key == "capacity":
                self.unlimited = QCheckBox("Unlimited")
                self.unlimited.setChecked(value is None)
                self.control.setText(str(value if value is not None else 0))
                self.control.setEnabled(value is not None)
                self.unlimited.toggled.connect(
                    lambda checked: self.control.setEnabled(not checked)
                )
                self.unlimited.toggled.connect(self.changed)
                layout.addWidget(self.unlimited)
            if key == "parallel_capacity":
                self.control.setPlaceholderText("Positive integer or max")
            layout.addWidget(self.control)

    def add_mode(self, mode):
        row = QGroupBox("Quality mode")
        layout = QVBoxLayout(row)
        field = ValueField("quality_mode", mode, self.design)
        field.changed.connect(self.changed)
        layout.addWidget(field)
        remove = QPushButton("Remove mode")
        remove.clicked.connect(lambda: self.remove_mode(row))
        layout.addWidget(remove)
        self.rows.append((row, field))
        self.row_layout.addWidget(row)

    def remove_mode(self, row):
        self.rows = [(w, f) for w, f in self.rows if w is not row]
        row.setParent(None)
        row.deleteLater()
        self.changed.emit()

    def new_mode(self):
        self.add_mode(
            QualityMode(f"mode_{len(self.rows) + 1:03d}", Decimal(1), Decimal(0))
        )
        self.changed.emit()

    def value(self):
        if self.key == "battery":
            return self.nested.value() if self.enabled_box.isChecked() else None
        if self.key in ("allowed_headings", "operation_types"):
            return tuple(
                k for k, box in self.children_fields.items() if box.isChecked()
            )
        if self.key == "quality_modes":
            return tuple(f.value() for _, f in self.rows)
        if is_dataclass(self.original):
            return replace(
                self.original, **{k: f.value() for k, f in self.children_fields.items()}
            )
        if isinstance(self.control, QComboBox):
            return self.control.currentData()
        value = self.control.text().strip()
        if self.key == "capacity" and self.unlimited.isChecked():
            return None
        if self.key == "parallel_capacity":
            return "max" if value == "max" else int(value)
        if isinstance(self.original, Decimal):
            return Decimal(value)
        if isinstance(self.original, int) or self.key == "capacity":
            return int(value)
        return value


class PropertyEditor(QWidget):
    apply_requested = Signal()
    cancel_requested = Signal()
    operation_requested = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("propertyEditor")
        self.pending = False
        self.changed_fields = set()
        self.resource = None
        self.inputs = {}
        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        scroll = QScrollArea()
        self.scroll = scroll
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self.body = QWidget()
        self.body.setObjectName("propertyForm")
        self.form = QFormLayout(self.body)
        self.form.setSizeConstraint(QLayout.SizeConstraint.SetMinAndMaxSize)
        self.form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        self.form.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
        )
        scroll.setWidget(self.body)
        outer.addWidget(scroll, 1)
        self.error = QLabel()
        self.error.setWordWrap(True)
        self.error.setStyleSheet("color: #a55549;")
        outer.addWidget(self.error)
        row = QHBoxLayout()
        self.apply_button = QPushButton("Apply")
        self.apply_button.setObjectName("applyPropertiesButton")
        self.apply_button.setDefault(True)
        self.apply_button.clicked.connect(self.apply_requested)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.cancel_requested)
        row.addWidget(self.apply_button)
        row.addWidget(self.cancel_button)
        outer.addLayout(row)

    def _changed(self, key):
        self.pending = True
        self.changed_fields.add(key)
        self.apply_button.setEnabled(True)
        self.cancel_button.setEnabled(True)

    def show_resource(self, value, design, count=1, selected=()):
        while self.form.rowCount():
            self.form.removeRow(0)
        self.resource = value
        self.inputs = {}
        self.operation_buttons = {}
        self.pending = False
        self.changed_fields = set()
        self.error.clear()
        self.apply_button.setEnabled(False)
        self.cancel_button.setEnabled(False)
        if value is None:
            self.form.addRow(
                QLabel(f"{count} mixed objects selected. Use Move, Rotate or Delete.")
            )
            return
        identifier = getattr(value, "factory_id", None)
        from smartsom.domain.factory_design import entity_id

        identifier = identifier or entity_id(value)
        self.form.addRow(
            "ID",
            QLabel(
                identifier
                if count == 1
                else f"{count} objects · only changed fields apply"
            ),
        )
        excluded = {
            "slots",
            "bindings",
            "machines",
            "buffers",
            "inspection_stations",
            "scrap_bins",
            "chargers",
            "agvs",
            "ports",
        }
        for field in fields(value):
            key = field.name
            if key in excluded or key.endswith("_id") and key != "machine_id":
                continue
            if key == "machine_id" and not isinstance(value, BufferDesign):
                continue
            if count > 1 and key in (
                "name",
                "footprint",
                "initial_cell",
                "cell",
                "machine_id",
                "role",
            ):
                continue
            if (
                count > 1
                and key == "storage"
                and any(type(r.storage) is not type(value.storage) for r in selected)
            ):
                self.form.addRow(
                    "Storage", QLabel("Mixed Pool / Slots · edit separately")
                )
                continue
            if key == "storage" and not isinstance(value.storage, PoolStorage):
                self.form.addRow(
                    "Storage", QLabel(f"Slots · {len(value.storage.slots)} slots")
                )
                if count > 1 and isinstance(value.storage, SlotStorage):
                    capacity = next(
                        (s.capacity for r in selected for s in r.storage.slots), 1
                    )
                    editor = ValueField("capacity_per_slot", capacity, design)
                    editor.changed.connect(lambda: self._changed("capacity_per_slot"))
                    self.inputs["capacity_per_slot"] = editor
                    self.form.addRow("Capacity per existing slot", editor)
                continue
            editor = ValueField(key, getattr(value, key), design)
            if count > 1 and key in ("quality_modes", "battery", "operation_types"):
                detail = QLabel(
                    "Apply replaces this complete parameter group on each selected object."
                )
                detail.setWordWrap(True)
                self.form.addRow(detail)
            editor.changed.connect(lambda k=key: self._changed(k))
            self.inputs[key] = editor
            if is_dataclass(getattr(value, key)) or key in (
                "battery",
                "quality_modes",
                "allowed_headings",
                "operation_types",
            ):
                self.form.addRow(QLabel(field_label(key)))
                self.form.addRow(editor)
            else:
                self.form.addRow(field_label(key), editor)
        if count == 1:
            operations = [("Rename ID…", "rename")]
            if isinstance(value, (BufferDesign, InspectionStationDesign)):
                if not isinstance(value, BufferDesign) or not isinstance(
                    value.storage, PoolStorage
                ):
                    operations.append(("Edit slots…", "slots"))
            if isinstance(value, BufferDesign):
                operations.append(("Change storage mode…", "storage"))
            if isinstance(value, PortDesign):
                operations.append(("Edit bindings…", "bindings"))
            for label, key in operations:
                button = QPushButton(label)
                self.operation_buttons[key] = button
                button.clicked.connect(
                    lambda checked=False, op=key: self.operation_requested.emit(op)
                )
                self.form.addRow(button)

    def reveal_field(self, path):
        key = str(path or "").split(".")[0].split("[")[0]
        widget = self.inputs.get(key) or self.operation_buttons.get(key)
        if widget is None:
            return
        self.scroll.ensureWidgetVisible(widget)
        controls = widget.findChildren(QLineEdit) or widget.findChildren(QComboBox)
        (controls[0] if controls else widget).setFocus()

    def values(self):
        result = {}
        for key in self.changed_fields:
            try:
                result[key] = self.inputs[key].value()
            except (ValueError, TypeError, ArithmeticError) as exc:
                raise ValueError(f"{field_label(key)}: {exc}") from exc
        return result
