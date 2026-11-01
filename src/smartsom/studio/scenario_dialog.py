"""Scenario defaults, inherited overrides and job rules as ordinary Qt forms."""

from copy import deepcopy

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from smartsom.config.codec import primitive
from smartsom.config.scenario_authoring import ScenarioFile, merge

LABELS = {
    "processing_rate_multiplier": "Processing speed multiplier",
    "quality_modes": "Processing modes (time / defect probability)",
    "job_capacity": "Maximum jobs carried",
    "move_cells_per_tick": "Movement speed (cells / tick)",
    "initial_heading": "Initial heading",
    "mean_interval_ticks": "Mean arrival interval (ticks)",
    "base_ticks": "Base processing work (ticks)",
    "relative_variation": "Relative processing variation (± fraction)",
    "repeat_first_probability": "Probability of repeating the first operation",
    "normal_multiplier": "Normal due-date multiplier (min, max)",
    "urgent_multiplier": "Urgent due-date multiplier (min, max)",
    "scale": "Exponential scale (ticks, before truncation)",
    "lower": "Lower bound (ticks)",
    "upper": "Upper bound (ticks)",
    "breakdown": "Machine breakdown",
    "intervals": "Uptime by fault condition (normal always disables faults)",
    "repair": "Repair duration (truncated exponential)",
    "machine_pre": "Pre buffers",
    "machine_post": "Post buffers",
}


def label(key):
    return LABELS.get(key, key.replace("_", " ").capitalize())


class TableField(QWidget):
    def __init__(self, values, parent=None):
        super().__init__(parent)
        self.original = deepcopy(values)
        self.mapping = bool(values and isinstance(values[0], dict))
        self.columns = list(values[0]) if self.mapping else ["Operation sequence"]
        layout = QVBoxLayout(self)
        self.table = QTableWidget(0, len(self.columns))
        self.table.setHorizontalHeaderLabels([label(k) for k in self.columns])
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        for row in values:
            self.add(row)
        layout.addWidget(self.table)
        buttons = QHBoxLayout()
        add, remove = QPushButton("Add row"), QPushButton("Remove selected row")
        add.clicked.connect(
            lambda: self.add({k: "" for k in self.columns} if self.mapping else [])
        )
        remove.clicked.connect(lambda: self.table.removeRow(self.table.currentRow()))
        buttons.addWidget(add)
        buttons.addWidget(remove)
        layout.addLayout(buttons)
        if not self.mapping:
            layout.addWidget(
                QLabel(
                    "One route per row, e.g. A, B, C, D. Routes are sampled uniformly."
                )
            )

    def add(self, values):
        row = self.table.rowCount()
        self.table.insertRow(row)
        data = (
            [values[k] for k in self.columns] if self.mapping else [", ".join(values)]
        )
        for col, value in enumerate(data):
            self.table.setItem(row, col, QTableWidgetItem(str(value)))

    def value(self):
        result = []
        for row in range(self.table.rowCount()):
            values = [
                self.table.item(row, col).text().strip()
                if self.table.item(row, col)
                else ""
                for col in range(self.table.columnCount())
            ]
            if self.mapping and row < len(self.original):
                values = [
                    self.original[row][key]
                    if str(self.original[row][key]) == value
                    else value
                    for key, value in zip(self.columns, values, strict=True)
                ]
            result.append(
                dict(zip(self.columns, values, strict=True))
                if self.mapping
                else [v.strip() for v in values[0].split(",") if v.strip()]
            )
        return result


class ParameterForm(QWidget):
    def __init__(self, values, parent=None):
        super().__init__(parent)
        self.original = deepcopy(values)
        self.fields = {}
        self.initial_numbers = {}
        form = QFormLayout(self)
        for key, value in values.items():
            if key == "battery":
                widget = BatteryField(value)
                form.addRow(label(key), widget)
            elif isinstance(value, dict):
                widget = ParameterForm(value)
                group = QGroupBox(label(key))
                QVBoxLayout(group).addWidget(widget)
                form.addRow(group)
            elif isinstance(value, list) and (
                key == "families" or (value and isinstance(value[0], dict))
            ):
                widget = TableField(value)
                form.addRow(label(key), widget)
            elif isinstance(value, bool):
                widget = QCheckBox()
                widget.setChecked(value)
                form.addRow(label(key), widget)
            elif isinstance(value, int):
                widget = QSpinBox()
                widget.setRange(0, 2_000_000_000)
                widget.setValue(value)
                form.addRow(label(key), widget)
            elif isinstance(value, float):
                widget = QDoubleSpinBox()
                widget.setDecimals(9)
                widget.setRange(0, 1e12)
                widget.setValue(value)
                form.addRow(label(key), widget)
            else:
                widget = QLineEdit(
                    ", ".join(map(str, value))
                    if isinstance(value, (list, tuple))
                    else ""
                    if value is None
                    else str(value)
                )
                if value is None:
                    widget.setPlaceholderText("None (leave empty)")
                form.addRow(label(key), widget)
            widget.setObjectName(key)
            self.fields[key] = widget
            if isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                self.initial_numbers[key] = widget.value()

    def value(self):
        result = {}
        for key, widget in self.fields.items():
            if isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                # Opening/applying a form is not a numeric normalization step.
                result[key] = (
                    self.original[key]
                    if widget.value() == self.initial_numbers[key]
                    else widget.value()
                )
            elif isinstance(widget, (ParameterForm, TableField, BatteryField)):
                result[key] = widget.value()
            elif isinstance(widget, QCheckBox):
                result[key] = widget.isChecked()
            else:
                text = widget.text().strip()
                old = self.original[key]
                if isinstance(old, (list, tuple)):
                    result[key] = [float(v.strip()) for v in text.split(",")]
                elif old is None:
                    if text:
                        raise ValueError(
                            f"{label(key)} is optional; this form currently accepts an empty value only"
                        )
                    result[key] = None
                else:
                    result[key] = text
        return result


class BatteryField(QWidget):
    def __init__(self, value, parent=None):
        super().__init__(parent)
        from dataclasses import asdict

        from smartsom.domain.factory_design import BatteryDesign

        layout = QVBoxLayout(self)
        self.enabled = QCheckBox("Model battery")
        self.enabled.setChecked(value is not None)
        self.form = ParameterForm(value or primitive(asdict(BatteryDesign())))
        self.form.setEnabled(self.enabled.isChecked())
        self.enabled.toggled.connect(self.form.setEnabled)
        layout.addWidget(self.enabled)
        layout.addWidget(self.form)

    def value(self):
        return self.form.value() if self.enabled.isChecked() else None


def scroll(widget):
    area = QScrollArea()
    area.setWidgetResizable(True)
    area.setWidget(widget)
    return area


class ScenarioSettingsDialog(QDialog):
    def __init__(self, scene, parent=None):
        super().__init__(parent)
        self.scene = scene
        self.setWindowTitle("Scenario · shared defaults and job generator")
        self.resize(800, 700)
        layout = QVBoxLayout(self)
        hint = QLabel(
            "Defaults update inherited fields only. Entity overrides remain unchanged. Time is measured in ticks."
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)
        tabs = QTabWidget()
        layout.addWidget(tabs)
        self.forms = {}
        for key, title, values in (
            (
                "machines",
                "Machines / breakdown",
                scene.factory_spec["machines"]["defaults"],
            ),
            (
                "buffers",
                "Pre / Post buffers",
                scene.factory_spec.get("buffer_defaults", {}),
            ),
            ("agvs", "AGVs", scene.factory_spec["agvs"]["defaults"]),
            ("jobs", "Job generator", primitive(scene.jobs)),
        ):
            self.forms[key] = ParameterForm(values)
            tabs.addTab(scroll(self.forms[key]), title)
        preview = QWidget()
        preview_layout = QVBoxLayout(preview)
        controls = QHBoxLayout()
        self.seed = QSpinBox()
        self.seed.setRange(0, 2_000_000_000)
        self.seed.setValue(101)
        self.condition = QComboBox()
        self.condition.addItems(["normal", "low", "medium", "high"])
        self.drain = QSpinBox()
        self.drain.setRange(1, 2_000_000_000)
        self.drain.setValue(2000)
        for title, widget in (
            ("Case seed", self.seed),
            ("Condition", self.condition),
            ("Drain ticks", self.drain),
        ):
            controls.addWidget(QLabel(title))
            controls.addWidget(widget)
        button = QPushButton("Preview generated case")
        button.clicked.connect(self.preview)
        preview_layout.addLayout(controls)
        preview_layout.addWidget(button)
        self.preview_text = QPlainTextEdit()
        self.preview_text.setReadOnly(True)
        preview_layout.addWidget(self.preview_text)
        tabs.addTab(preview, "Case preview")
        self.error = QLabel()
        self.error.setWordWrap(True)
        layout.addWidget(self.error)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Apply
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Apply).clicked.connect(
            self.accept
        )
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def value(self):
        data = primitive(self.scene)
        for key in ("machines", "agvs"):
            data["factory"][key]["defaults"] = self.forms[key].value()
        data["factory"]["buffer_defaults"] = self.forms["buffers"].value()
        data["jobs"] = self.forms["jobs"].value()
        return ScenarioFile.model_validate(data)

    def accept(self):
        try:
            self.result_scene = self.value()
        except (ValueError, TypeError, KeyError) as exc:
            self.error.setText(str(exc))
            return
        super().accept()

    def preview(self):
        from smartsom.config.base_diagnostic import generate_scenario
        from smartsom.config.codec import digest

        try:
            scene = self.value()
            factory, workload = scene.expand()
            # Use the same production compiler's eligibility check, without any
            # algorithm or tracking initialization.
            from smartsom.config.base_diagnostic import validate_base_factory

            validate_base_factory(factory.factory)
            world = generate_scenario(
                factory.factory,
                workload,
                self.seed.value(),
                self.condition.currentText(),
                self.drain.value(),
            )
            lines = [
                f"Case digest: {digest(primitive(world))}",
                f"Orders: {len(world.demands)} · Faults: {len(world.outages)}",
                "Preview only; no simulation or training executed.",
                "",
                "Orders (first 100):",
            ]
            lines += [str(primitive(job)) for job in world.demands[:100]]
            lines += ["", "Faults (first 100):"] + [
                str(primitive(f)) for f in world.outages[:100]
            ]
            self.preview_text.setPlainText("\n".join(lines))
            self.error.clear()
        except (ValueError, TypeError, KeyError) as exc:
            self.error.setText(str(exc))


class OverridesDialog(QDialog):
    """Explicit per-field inheritance, including overrides equal to defaults."""

    def __init__(self, scene, identifier, parent=None):
        super().__init__(parent)
        self.scene, self.identifier = scene, identifier
        self.setWindowTitle(f"{identifier} · defaults and overrides")
        self.resize(750, 650)
        self.section = next(
            (
                key
                for key in ("machines", "agvs")
                if any(i["id"] == identifier for i in scene.factory_spec[key]["items"])
            ),
            None,
        )
        if self.section is None:
            buffer = next(
                b for b in scene.factory_spec["buffers"] if b["buffer_id"] == identifier
            )
            defaults = scene.factory_spec.get("buffer_defaults", {}).get(
                buffer["role"], {}
            )
            if not defaults:
                raise ValueError(
                    "Select a machine, AGV, Pre or Post buffer with defaults"
                )
            self.item = buffer
            overrides = {k: buffer[k] for k in defaults if k in buffer}
        else:
            group = scene.factory_spec[self.section]
            defaults = group["defaults"]
            self.item = next(i for i in group["items"] if i["id"] == identifier)
            overrides = self.item.get("overrides", {})
        layout = QVBoxLayout(self)
        layout.addWidget(
            QLabel("Unchecked = inherit shared default. Checked = explicit override.")
        )
        body = QWidget()
        inner = QVBoxLayout(body)
        self.fields = {}
        effective = merge(defaults, overrides)
        self.original_overrides = deepcopy(overrides)
        self.effective = deepcopy(effective)
        for key in defaults:
            check = QCheckBox(f"Override {label(key)}")
            check.setChecked(key in overrides)
            form = ParameterForm({key: effective[key]})
            form.setEnabled(check.isChecked())
            check.toggled.connect(form.setEnabled)
            inner.addWidget(check)
            inner.addWidget(form)
            self.fields[key] = (check, form)
        layout.addWidget(scroll(body))
        reset = QPushButton("Restore all shared defaults")
        reset.clicked.connect(
            lambda: [check.setChecked(False) for check, _ in self.fields.values()]
        )
        layout.addWidget(reset)
        self.error = QLabel()
        self.error.setWordWrap(True)
        layout.addWidget(self.error)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Apply
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Apply).clicked.connect(
            self.accept
        )
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def value(self):
        data = primitive(self.scene)
        overrides = {}
        for key, (check, form) in self.fields.items():
            if check.isChecked():
                value = form.value()[key]
                if key in self.original_overrides:
                    value = edited_override(
                        self.original_overrides[key], self.effective[key], value
                    )
                overrides[key] = value
        if self.section:
            item = next(
                i
                for i in data["factory"][self.section]["items"]
                if i["id"] == self.identifier
            )
            if overrides:
                item["overrides"] = overrides
            else:
                item.pop("overrides", None)
        else:
            item = next(
                i
                for i in data["factory"]["buffers"]
                if i["buffer_id"] == self.identifier
            )
            for key in self.fields:
                item.pop(key, None)
            item.update(overrides)
        return ScenarioFile.model_validate(data)

    def accept(self):
        try:
            self.result_scene = self.value()
        except (ValueError, TypeError, KeyError) as exc:
            self.error.setText(str(exc))
            return
        super().accept()


def edited_override(original, before, after):
    """Keep omitted nested fields inherited when editing a partial override."""
    if after == before:
        return deepcopy(original)
    if (
        isinstance(original, dict)
        and isinstance(before, dict)
        and isinstance(after, dict)
    ):
        result = deepcopy(original)
        for key, value in after.items():
            if key in original:
                result[key] = edited_override(original[key], before.get(key), value)
            elif value != before.get(key):
                result[key] = edited_override({}, before.get(key), value)
        return result
    return deepcopy(after)
