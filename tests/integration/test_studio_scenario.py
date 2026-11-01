"""Scenario editing through Qt forms, undo, disk and the production compiler."""

import os
from copy import deepcopy
from decimal import Decimal

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QEvent
from PySide6.QtWidgets import QApplication, QMessageBox

from smartsom.config.codec import primitive, read_document
from smartsom.config.scenario_authoring import load_scene
from smartsom.studio import editing
from smartsom.studio.scenario_dialog import OverridesDialog, ScenarioSettingsDialog
from smartsom.studio.window import StudioWindow

pytestmark = pytest.mark.studio


@pytest.fixture
def window(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    app.setQuitOnLastWindowClosed(False)
    window = StudioWindow(data_dir=tmp_path / "ui")

    def unexpected(*args):
        raise AssertionError(f"Unexpected modal: {args}")

    monkeypatch.setattr(QMessageBox, "warning", unexpected)
    yield window
    window.editor.properties.pending = False
    for doc in window.documents:
        doc.undo_stack.setClean()
    window.close()
    window.deleteLater()
    app.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    app.processEvents()


def test_defaults_override_undo_save_reopen_and_case_preview(window, tmp_path):
    doc = window.new_scene()
    original = doc.file
    window.editor.set_mode(True)
    dialog = ScenarioSettingsDialog(doc.file)
    dialog.forms["machines"].fields["processing_rate_multiplier"].setText("1.5")
    dialog.forms["jobs"].fields["orders"].fields["total"].setValue(160)
    scene = dialog.value()
    assert window.editor.commit(scene.factory, "Scenario settings", scenario=scene)
    assert all(
        m.processing_rate_multiplier == Decimal("1.5") for m in doc.design.machines
    )
    assert doc.file.jobs.orders.total == 160
    doc.undo_stack.undo()
    assert doc.file == original
    doc.undo_stack.redo()
    identifier = doc.design.machines[0].machine_id
    override = OverridesDialog(doc.file, identifier)
    check, form = override.fields["processing_rate_multiplier"]
    check.setChecked(True)
    form.fields["processing_rate_multiplier"].setText("0.8")
    result = override.value()
    assert window.editor.commit(result.factory, "Override", scenario=result)
    assert doc.design.machines[0].processing_rate_multiplier == Decimal("0.8")
    assert doc.design.machines[1].processing_rate_multiplier == Decimal("1.5")
    target = tmp_path / "scenario.yaml"
    assert window.editor.save_to(doc, target)
    loaded, _ = load_scene(target)
    assert loaded == doc.file
    machine = read_document(target)["factory"]["machines"]["items"][0]
    assert machine["operation"] == "A"
    assert set(machine["position"]) == {"x", "y"}
    assert "rotation" not in machine and "size" not in machine
    copied = window.new_from_path(target)
    assert copied.file == loaded
    preview = ScenarioSettingsDialog(copied.file)
    preview.preview()
    assert not preview.error.text()
    assert "Orders: 160" in preview.preview_text.toPlainText()
    before = preview.preview_text.toPlainText()
    preview.preview()
    assert preview.preview_text.toPlainText() == before
    for widget in (dialog, override, preview):
        widget.deleteLater()


def test_agv_buffer_defaults_reset_and_invalid_apply_are_atomic(window):
    doc = window.new_scene()
    window.editor.set_mode(True)
    dialog = ScenarioSettingsDialog(doc.file)
    dialog.forms["agvs"].fields["move_cells_per_tick"].setValue(2)
    dialog.forms["buffers"].fields["machine_pre"].fields["storage"].fields[
        "capacity"
    ].setValue(7)
    scene = dialog.value()
    assert window.editor.commit(scene.factory, "Defaults", scenario=scene)
    assert all(a.move_cells_per_tick == 2 for a in doc.design.agvs)
    assert all(
        b.storage.capacity == 7 for b in doc.design.buffers if b.role == "machine_pre"
    )
    override = OverridesDialog(doc.file, doc.design.agvs[0].agv_id)
    check, form = override.fields["move_cells_per_tick"]
    check.setChecked(True)
    form.fields["move_cells_per_tick"].setValue(3)
    scene = override.value()
    assert window.editor.commit(scene.factory, "AGV override", scenario=scene)
    check.setChecked(False)
    scene = override.value()
    assert window.editor.commit(scene.factory, "Restore default", scenario=scene)
    assert doc.design.agvs[0].move_cells_per_tick == 2
    before = deepcopy(primitive(doc.file))
    dialog.forms["jobs"].fields["orders"].fields["initial"].setValue(1000)
    dialog.accept()
    assert dialog.error.text()
    assert primitive(doc.file) == before
    dialog.deleteLater()
    override.deleteLater()


def test_machine_properties_have_coordinates_without_size_or_rotation(window):
    doc = window.new_scene()
    window.editor.set_mode(True)
    window.select_entity(doc.design.machines[0].machine_id)
    footprint = window.editor.properties.inputs["footprint"]
    assert set(footprint.children_fields) == {"x", "y"}
    from dataclasses import replace

    old = doc.design.machines[0]
    enlarged = replace(old, footprint=replace(old.footprint, width=2))
    candidate = editing.replace_resources(doc.design, {old.machine_id: enlarged})
    assert not window.editor.commit(candidate, "Invalid size")
    assert doc.design.machines[0] == old


def test_recovery_retains_jobs_defaults_and_overrides(window):
    doc = window.new_scene("medium")
    path = window.editor.recovery.snapshot(doc)
    loaded, _ = load_scene(path)
    assert loaded == doc.file
    assert loaded.jobs.orders.total == 320
    assert len(loaded.factory.machines) == 20


def test_noop_forms_preserve_precision_and_partial_overrides(window):
    from smartsom.config.scenario_authoring import ScenarioFile

    doc = window.new_scene()
    data = primitive(doc.file)
    data["jobs"]["arrivals"]["mean_interval_ticks"] = 36.12345678912345
    first = data["factory"]["machines"]["items"][0]
    first["overrides"] = {"breakdown": {"enabled": False}}
    scene = ScenarioFile.model_validate(data)
    settings = ScenarioSettingsDialog(scene)
    assert settings.value() == scene
    override = OverridesDialog(scene, first["id"])
    assert override.value() == scene
    form = override.fields["breakdown"][1].fields["breakdown"]
    form.fields["repair"].fields["scale"].setValue(80)
    changed = override.value()
    assert changed.factory_spec["machines"]["items"][0]["overrides"] == {
        "breakdown": {"enabled": False, "repair": {"scale": 80}}
    }
    settings.deleteLater()
    override.deleteLater()


def test_partial_buffer_storage_survives_save_reopen_and_default_edit(window, tmp_path):
    from dataclasses import replace

    from smartsom.config.scenario_authoring import ScenarioFile, save_scene

    data = primitive(window.new_scene().file)
    buffer = next(b for b in data["factory"]["buffers"] if b["role"] == "machine_pre")
    buffer["storage"] = {"mode": "pool"}
    path = tmp_path / "partial.yaml"
    save_scene(path, ScenarioFile.model_validate(data))
    doc = window.new_from_path(path)
    window.editor.set_mode(True)
    assert window.editor.commit(replace(doc.design, name="Unrelated edit"), "Rename")
    saved_path = tmp_path / "saved.yaml"
    assert window.editor.save_to(doc, saved_path)
    reopened = window.new_from_path(saved_path)
    window.editor.set_mode(True)
    raw = primitive(reopened.file)
    actual = next(
        b for b in raw["factory"]["buffers"] if b["buffer_id"] == buffer["buffer_id"]
    )
    assert actual["storage"] == {"mode": "pool"}
    dialog = ScenarioSettingsDialog(reopened.file)
    dialog.forms["buffers"].fields["machine_pre"].fields["storage"].fields[
        "capacity"
    ].setValue(9)
    scene = dialog.value()
    assert window.editor.commit(scene.factory, "Default capacity", scenario=scene)
    assert (
        next(
            b for b in reopened.design.buffers if b.buffer_id == buffer["buffer_id"]
        ).storage.capacity
        == 9
    )
    dialog.deleteLater()
