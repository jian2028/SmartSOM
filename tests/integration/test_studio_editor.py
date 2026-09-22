"""Authoring acceptance through Qt gestures, drafts, disk files and undo."""

import importlib.util
import os
from dataclasses import replace
from decimal import Decimal

import pytest

if importlib.util.find_spec("PySide6") is None:
    if os.environ.get("SMARTSOM_REQUIRE_STUDIO") == "1":
        raise ImportError("Studio acceptance requires the studio extra")
    pytest.skip("optional studio extra is not installed", allow_module_level=True)
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QImage
from PySide6.QtTest import QTest
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QInputDialog,
    QMessageBox,
)

from smartsom.config.factory_design import load_factory_design
from smartsom.domain.factory_design import (
    BufferSlotTarget,
    Cell,
    PortBinding,
    entity_id,
    iter_resources,
)
from smartsom.studio import editing
from smartsom.studio.dialogs import ExportDialog, TemplateSaveDialog
from smartsom.studio.export import export_map
from smartsom.studio.window import StudioWindow

pytestmark = pytest.mark.studio


@pytest.fixture(scope="module")
def app():
    result = QApplication.instance() or QApplication([])
    result.setQuitOnLastWindowClosed(False)
    return result


@pytest.fixture
def window(app, tmp_path, monkeypatch):
    result = StudioWindow(data_dir=tmp_path / "userdata")
    result.resize(1440, 950)
    result.show()
    app.processEvents()

    # Unexpected modal errors must fail acceptance rather than hang the suite.
    def unexpected(*args):
        raise AssertionError(f"Unexpected dialog: {args}")

    monkeypatch.setattr(QMessageBox, "warning", unexpected)
    monkeypatch.setattr(QMessageBox, "exec", unexpected)
    yield result
    if result.editor.draft_dialog:
        result.editor.draft_dialog.reject()
    result.editor.properties.pending = False
    for doc in result.documents:
        doc.undo_stack.setClean()
    result.close()
    result.deleteLater()
    app.clipboard().clear()
    app.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    app.processEvents()


def point(doc, x, y):
    return doc.view.mapFromScene(QPointF(x * 40, y * 40))


def drag(app, doc, start, end):
    QTest.mousePress(
        doc.view.viewport(), Qt.MouseButton.LeftButton, pos=point(doc, *start)
    )
    QTest.mouseMove(doc.view.viewport(), point(doc, *end), delay=2)
    QTest.mouseRelease(
        doc.view.viewport(), Qt.MouseButton.LeftButton, pos=point(doc, *end)
    )
    app.processEvents()


def add(window, kind, x, y, width=1, height=1):
    doc = window.current_document
    candidate, identifier = editing.create_resource(
        doc.design, kind, x, y, width, height
    )
    assert window.editor.commit(candidate, "Add " + kind, selection=(identifier,))
    return identifier


def test_browse_guards_mutations_then_reverse_drag_creates_one_resource(app, window):
    doc = window.new_blank()
    app.processEvents()
    assert not doc.edit_mode and not window.editor.add_button.isEnabled()
    drag(app, doc, (4.5, 4.5), (2.5, 2.5))
    assert not doc.modified and not doc.design.buffers
    window.editor.set_mode(True)
    app.processEvents()
    window.editor.choose_tool("buffer")
    drag(app, doc, (4.5, 4.5), (2.5, 2.5))
    assert len(doc.design.buffers) == 1
    assert editing.rect_of(doc.design.buffers[0]) == (2, 2, 3, 3)
    assert doc.undo_stack.count() == 1 and doc.modified
    assert doc.interaction.tool == "select"
    window.editor.undo(-1)
    assert not doc.design.buffers and not doc.modified
    window.editor.undo(1)
    assert len(doc.design.buffers[0].storage.slots) == 9


def test_properties_reject_invalid_group_and_preserve_draft_on_same_selection(window):
    doc = window.new_blank()
    e = window.editor
    e.set_mode(True)
    identifier = add(window, "agv", 2, 2)
    e.properties.inputs["name"].control.setText("Carrier")
    e.properties.inputs["job_capacity"].control.setText("0")
    window.select_entity(identifier)
    assert e.properties.pending
    before, count = doc.design, doc.undo_stack.count()
    assert not e.apply_properties()
    assert doc.design == before and doc.undo_stack.count() == count
    e.properties.inputs["job_capacity"].control.setText("3")
    assert e.apply_properties()
    assert doc.design.agvs[0].name == "Carrier" and doc.design.agvs[0].job_capacity == 3
    assert doc.undo_stack.count() == count + 1
    e.undo(-1)
    assert doc.design == before


def test_machine_categories_apply_cancel_undo_and_save(window, tmp_path):
    doc = window.new_template()
    editor = window.editor
    editor.set_mode(True)
    window.select_entity("machine_001")
    boxes = editor.properties.inputs["operation_types"].children_fields
    assert boxes["operation_1"].isChecked()
    before = doc.design
    boxes["operation_2"].setChecked(True)
    assert doc.design == before
    editor.properties.cancel_button.click()
    assert (
        not editor.properties.inputs["operation_types"]
        .children_fields["operation_2"]
        .isChecked()
    )
    editor.properties.inputs["operation_types"].children_fields[
        "operation_2"
    ].setChecked(True)
    assert editor.apply_properties()
    assert doc.design.machines[0].operation_types == ("operation_1", "operation_2")
    assert doc.design.machines[1].operation_types == ("operation_2",)
    editor.undo(-1)
    assert doc.design == before
    editor.undo(1)
    path = tmp_path / "shared-operations.yaml"
    assert editor.save_to(doc, path)
    assert load_factory_design(path)[0] == doc.design


@pytest.mark.parametrize(
    "choice",
    [
        QMessageBox.StandardButton.Apply,
        QMessageBox.StandardButton.Discard,
        QMessageBox.StandardButton.Cancel,
    ],
)
def test_tab_switch_resolves_the_old_document_draft(window, monkeypatch, choice):
    first = window.new_blank()
    e = window.editor
    e.set_mode(True)
    second = window.new_blank()
    window.tabs.setCurrentIndex(0)
    e.properties.inputs["name"].control.setText("First edited")
    monkeypatch.setattr(QMessageBox, "exec", lambda self: choice)
    window.tabs.setCurrentIndex(1)
    assert second.design.name == "Untitled Factory" and not second.modified
    if choice == QMessageBox.StandardButton.Cancel:
        assert window.current_document is first and e.properties.pending
    else:
        assert window.current_document is second
        assert first.modified == (choice == QMessageBox.StandardButton.Apply)
        assert first.design.name == (
            "First edited" if first.modified else "Untitled Factory"
        )


def test_resize_preserves_slots_and_undo_restores_deleted_binding(
    app, window, monkeypatch
):
    doc = window.new_blank()
    e = window.editor
    e.set_mode(True)
    buffer = add(window, "buffer", 2, 2, 3, 2)
    port = add(window, "port", 3, 5)
    p = replace(
        doc.design.ports[0],
        bindings=(PortBinding(BufferSlotTarget(buffer, "slot_006")),),
    )
    e.commit(editing.replace_resources(doc.design, {port: p}), "Bind")
    window.select_entity(buffer)
    before = doc.design
    app.processEvents()
    impacts = []
    monkeypatch.setattr(
        e, "confirm_impact", lambda lines: impacts.extend(lines) or True
    )
    drag(app, doc, (5, 3), (4.1, 3))
    assert doc.design.buffers[0].footprint.width == 2
    assert not doc.design.ports[0].bindings
    assert any("slot_006" in line for line in impacts)
    e.undo(-1)
    assert doc.design == before


def test_slot_dialog_map_selection_rename_and_cancel(window, app, monkeypatch):
    doc = window.new_blank()
    e = window.editor
    e.set_mode(True)
    identifier = add(window, "buffer", 2, 2, 2, 2)
    e.operation("slots")
    dialog = e.draft_dialog
    dialog.table.item(0, 3).setText("5")
    before = doc.design
    assert doc.design == before
    dialog.reject()
    assert doc.design == before and e.draft_dialog is None
    e.operation("slots")
    dialog = e.draft_dialog
    e.draft_click(doc.scene.entity_items[identifier], Cell(3, 3))
    assert dialog.table.currentRow() == 3
    monkeypatch.setattr(QInputDialog, "getText", lambda *a, **kw: ("corner", True))
    dialog.rename_slot()
    dialog.table.item(3, 3).setText("5")
    monkeypatch.setattr(e, "confirm_impact", lambda lines: True)
    dialog.apply_requested.emit()
    assert e.draft_dialog is None
    assert doc.design.buffers[0].storage.slots[3].slot_id == "corner"
    assert doc.design.buffers[0].storage.slots[3].capacity == 5
    e.undo(-1)
    assert doc.design == before


def test_bindings_map_clicks_use_precise_slots_without_lines(app, window):
    doc = window.new_blank()
    e = window.editor
    e.set_mode(True)
    buffer = add(window, "buffer", 2, 2, 2, 2)
    port = add(window, "port", 3, 5)
    e.operation("bindings")
    dialog = e.draft_dialog
    e.draft_click(doc.scene.entity_items[buffer], Cell(2, 3))
    e.draft_click(doc.scene.entity_items[buffer], Cell(3, 3))
    assert len(e.draft_overlays) == 2 and not doc.design.ports[0].bindings
    assert all(item.pen().style() == Qt.PenStyle.NoPen for item in e.draft_overlays)
    dialog.table.cellWidget(0, 1).setCurrentIndex(1)
    dialog.apply_requested.emit()
    assert e.draft_dialog is None
    assert [b.target.slot_id for b in doc.design.ports[0].bindings] == [
        "slot_003",
        "slot_004",
    ]
    assert doc.design.ports[0].bindings[0].operations == ("pickup",)
    assert doc.selected_id == port
    e.undo(-1)
    assert not doc.design.ports[0].bindings


def test_agv_body_and_exposed_port_edge_remain_selectable_in_edit(app, window):
    doc = window.new_blank()
    e = window.editor
    e.set_mode(True)
    port = add(window, "port", 2, 2)
    agv = add(window, "agv", 2, 2)
    app.processEvents()
    QTest.mouseClick(
        doc.view.viewport(), Qt.MouseButton.LeftButton, pos=point(doc, 2.5, 2.5)
    )
    assert doc.selected_id == agv
    QTest.mouseClick(
        doc.view.viewport(), Qt.MouseButton.LeftButton, pos=point(doc, 2.05, 2.05)
    )
    assert doc.selected_id == port


def test_group_move_and_clipboard_paste_are_one_undo_each(app, window, monkeypatch):
    monkeypatch.setattr(QMessageBox, "exec", lambda _: QMessageBox.StandardButton.Ok)
    doc = window.new_blank()
    e = window.editor
    e.set_mode(True)
    a = add(window, "machine", 2, 2)
    b = add(window, "buffer", 4, 2)
    e.select_many((a, b))
    app.processEvents()
    before = doc.design
    count = doc.undo_stack.count()
    drag(app, doc, (2.5, 2.5), (2.5, 4.5))
    assert doc.design.machines[0].footprint.y == 4
    assert doc.design.buffers[0].footprint.y == 4
    assert doc.selected_ids == (a, b) and doc.undo_stack.count() == count + 1
    e.copy_selection()
    target = window.new_blank()
    e.set_mode(True)
    app.processEvents()
    e.paste_selection()
    drag(app, target, (6.5, 6.5), (6.5, 6.5))
    assert len(iter_resources(target.design)) == 2
    e.undo(-1)
    assert not iter_resources(target.design)
    window.tabs.setCurrentIndex(0)
    e.undo(-1)
    assert doc.design == before


def test_capability_forms_roundtrip_decimal_and_battery(window, tmp_path):
    doc = window.new_template()
    e = window.editor
    e.set_mode(True)
    for resource in iter_resources(doc.design):
        window.select_entity(entity_id(resource))
        for field in e.properties.inputs.values():
            assert field.value() == field.original
    window.select_entity(doc.design.agvs[0].agv_id)
    battery = e.properties.inputs["battery"]
    battery.enabled_box.setChecked(True)
    battery.nested.children_fields["energy_capacity"].control.setText("220.5")
    assert e.apply_properties()
    assert doc.design.agvs[0].battery.energy_capacity == Decimal("220.5")
    path = tmp_path / "capabilities.yaml"
    e.save_to(doc, path)
    assert load_factory_design(path)[0] == doc.design


def test_save_clean_point_conflict_and_recovery(window, tmp_path):
    doc = window.new_blank()
    e = window.editor
    e.set_mode(True)
    add(window, "machine", 2, 2)
    path = tmp_path / "factory.yaml"
    e.save_to(doc, path)
    saved = doc.design
    add(window, "buffer", 4, 2)
    e.snapshot_all()
    entries = e.recovery.entries()
    assert len(entries) == 1 and load_factory_design(entries[0])[0] == doc.design
    assert load_factory_design(path)[0] == saved
    path.write_text(path.read_text() + "\n# external edit\n")
    with pytest.raises(ValueError):
        e.save_to(doc, path, expected_digest=doc.source_digest)
    assert doc.modified and "external edit" in path.read_text()
    e.undo(-1)
    assert not doc.modified
    e.snapshot_all()
    assert not e.recovery.entries()


def test_template_save_destination_and_local_restore(window, tmp_path, monkeypatch):
    doc = window.new_template()
    e = window.editor
    e.set_mode(True)
    original = doc.design
    e.properties.inputs["name"].control.setText("My local template")
    assert e.apply_properties()

    def update(dialog):
        assert dialog.new_file.isChecked()
        dialog.update_template.setChecked(True)
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(TemplateSaveDialog, "exec", update)
    assert e.save()
    assert doc.source_path == e.catalog.builtin_path(1) and doc.saved_as_template
    child = window.new_template()
    assert child.design.name == "My local template" and not child.edit_mode
    e.catalog.restore_original(1)
    restored = window.new_template()
    assert restored.design == original
    assert child.design.name == "My local template" and doc.source_path.exists()
    # Saving a template copy chooses a regular file and subsequent saves reuse it.
    window.tabs.setCurrentIndex(1)
    e.set_mode(True)
    e.properties.inputs["name"].control.setText("Independent copy")
    e.apply_properties()
    path = tmp_path / "copy.yaml"
    monkeypatch.setattr(
        TemplateSaveDialog, "exec", lambda dialog: QDialog.DialogCode.Accepted
    )
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args: (str(path), ""))
    assert e.save() and child.source_path == path and not child.saved_as_template
    assert e.save() and load_factory_design(path)[0] == child.design


def test_export_full_map_preserves_edit_state(window, tmp_path):
    doc = window.new_template()
    e = window.editor
    e.set_mode(True)
    window.select_entity("machine_001")
    before = (
        doc.design,
        doc.selected_ids,
        doc.scene.names_visible,
        doc.undo_stack.count(),
    )
    preview = ExportDialog(doc, window)
    preview.refresh()
    preview.deleteLater()
    png, svg = tmp_path / "map.png", tmp_path / "map.svg"
    export_map(png, doc.design, cell_pixels=40, grid=False, numbers=False, ports=False)
    export_map(svg, doc.design, cell_pixels=40, bindings="all")
    image = QImage(str(png))
    assert (image.width(), image.height()) == (480, 480)
    assert "<svg" in svg.read_text()
    assert before == (
        doc.design,
        doc.selected_ids,
        doc.scene.names_visible,
        doc.undo_stack.count(),
    )


def test_close_cancel_keeps_document_and_history(window, monkeypatch):
    doc = window.new_blank()
    window.editor.set_mode(True)
    add(window, "machine", 2, 2)
    monkeypatch.setattr(
        QMessageBox, "exec", lambda self: QMessageBox.StandardButton.Cancel
    )
    window.close_document(0)
    assert window.current_document is doc and doc.modified


def test_obstacle_stroke_is_atomic_and_invalid_creation_keeps_tool(app, window):
    doc = window.new_blank()
    e = window.editor
    e.set_mode(True)
    add(window, "machine", 2, 2)
    app.processEvents()
    before = doc.design
    e.choose_tool("obstacle_paint")
    drag(app, doc, (1.5, 2.5), (3.5, 2.5))
    assert doc.design == before
    assert doc.interaction.tool == "obstacle_paint"
    drag(app, doc, (1.5, 4.5), (5.5, 4.5))
    assert len(doc.design.grid.blocked_cells) == 5
    e.undo(-1)
    assert doc.design == before
    e.choose_tool("obstacle_rectangle")
    drag(app, doc, (5.5, 7.5), (3.5, 6.5))
    assert len(doc.design.grid.blocked_cells) == 6
    e.choose_tool("obstacle_erase")
    drag(app, doc, (3.5, 6.5), (5.5, 6.5))
    assert len(doc.design.grid.blocked_cells) == 3


def test_continuous_placement_and_escape_do_not_consume_ids(app, window):
    doc = window.new_blank()
    e = window.editor
    e.set_mode(True)
    e.continuous_action.setChecked(True)
    app.processEvents()
    e.choose_tool("machine")
    drag(app, doc, (2.5, 2.5), (2.5, 2.5))
    assert doc.interaction.tool == "machine"
    drag(app, doc, (4.5, 2.5), (4.5, 2.5))
    assert [m.machine_id for m in doc.design.machines] == ["machine_001", "machine_002"]
    QTest.mouseMove(doc.view.viewport(), point(doc, 6.5, 2.5))
    QTest.keyClick(doc.view, Qt.Key.Key_Escape)
    assert doc.interaction.tool == "select" and len(doc.design.machines) == 2
    assert add(window, "machine", 6, 2) == "machine_003"


def test_same_type_bulk_apply_only_changes_edited_fields(window):
    doc = window.new_blank()
    e = window.editor
    e.set_mode(True)
    first = add(window, "agv", 2, 2)
    second = add(window, "agv", 4, 2)
    e.select_many((first, second))
    e.properties.inputs["job_capacity"].control.setText("4")
    before = doc.design
    assert e.apply_properties()
    assert [a.job_capacity for a in doc.design.agvs] == [4, 4]
    assert [a.initial_cell for a in doc.design.agvs] == [
        a.initial_cell for a in before.agvs
    ]
    assert [a.name for a in doc.design.agvs] == [a.name for a in before.agvs]
    e.undo(-1)
    assert doc.design == before


def test_recovery_restores_a_new_browse_document_with_independent_snapshot(
    window, monkeypatch
):
    doc = window.new_blank()
    e = window.editor
    e.set_mode(True)
    add(window, "machine", 2, 2)
    e.snapshot_all()
    assert e.timer.interval() == 60_000
    original_snapshot = e.recovery.entries()[0]
    monkeypatch.setattr(QInputDialog, "getItem", lambda *args, **kw: (args[3][0], True))
    e.recover_dialog()
    restored = window.current_document
    assert restored is not doc and restored.design == doc.design
    assert restored.source_path is None and restored.modified and not restored.edit_mode
    assert not original_snapshot.exists()
    assert e.recovery.entries()[0].stem == restored.recovery_id


def test_capacity_pool_to_slots_requires_apply_and_preserves_cancel(
    window, monkeypatch
):
    doc = window.new_template()
    e = window.editor
    e.set_mode(True)
    pool = next(b for b in doc.design.buffers if b.storage.mode == "pool")
    window.select_entity(pool.buffer_id)
    before = doc.design
    e.operation("storage")
    assert e.draft_dialog is not None and doc.design == before
    e.draft_dialog.reject()
    assert doc.design == before
    e.operation("storage")
    dialog = e.draft_dialog
    for row in range(dialog.table.rowCount()):
        dialog.table.item(row, 3).setText("3")
    impacts = []
    monkeypatch.setattr(
        e, "confirm_impact", lambda lines: impacts.extend(lines) or True
    )
    dialog.apply_requested.emit()
    changed = editing.resource(doc.design, pool.buffer_id)
    assert changed.storage.mode == "slots"
    assert all(s.capacity == 3 for s in changed.storage.slots)
    assert any("binding" in text for text in impacts)
    e.undo(-1)
    assert doc.design == before


def test_keyboard_typing_does_not_trigger_map_shortcuts(app, window):
    doc = window.new_blank()
    e = window.editor
    e.set_mode(True)
    add(window, "machine", 2, 2)
    field = e.properties.inputs["name"].control
    app.processEvents()
    field.setFocus()
    field.selectAll()
    QTest.keyClicks(field, "factory")
    QTest.keyClick(field, Qt.Key.Key_Backspace)
    assert field.text() == "factor" and len(doc.design.machines) == 1
    assert e.properties.pending


def test_failed_export_does_not_replace_existing_file(window, tmp_path):
    doc = window.new_template()
    path = tmp_path / "map.png"
    path.write_bytes(b"original image")
    with pytest.raises(ValueError):
        export_map(path, doc.design, cell_pixels=100000)
    assert path.read_bytes() == b"original image"
    assert list(tmp_path.glob("*.png")) == [path]


def test_medium_scale_authoring_move_undo_and_save(window, tmp_path, record_property):
    import time

    from test_studio import medium_design

    doc = window.add_design(medium_design())
    e = window.editor
    e.set_mode(True)
    e.select_many(
        (
            "machine_100",
            "buffer_pre_100",
            "buffer_post_100",
            "port_pre_100",
            "port_post_100",
            "agv_100",
        )
    )
    before = doc.design
    start = time.perf_counter()
    assert e.commit(editing.move(doc.design, doc.selected_ids, 0, 3), "Move group")
    elapsed = time.perf_counter() - start
    record_property("medium_edit_seconds", elapsed)
    assert elapsed < 5
    assert len(iter_resources(doc.design)) == 600
    e.undo(-1)
    assert doc.design == before
    path = tmp_path / "large.yaml"
    e.save_to(doc, path)
    assert load_factory_design(path)[0] == before


def test_machine_modes_and_buffer_capacities_support_explicit_bulk_apply(window):
    doc = window.new_blank()
    e = window.editor
    e.set_mode(True)
    a = add(window, "machine", 2, 2)
    b = add(window, "machine", 4, 2)
    e.select_many((a, b))
    e.properties.inputs["quality_modes"].rows[0][1].children_fields[
        "time_scale"
    ].control.setText("0.75")
    assert e.apply_properties()
    assert all(
        m.quality_modes[0].time_scale == Decimal("0.75") for m in doc.design.machines
    )
    a = add(window, "buffer", 2, 5, 2, 2)
    b = add(window, "buffer", 5, 5)
    e.select_many((a, b))
    before = doc.design
    e.properties.inputs["capacity_per_slot"].control.setText("4")
    assert e.apply_properties()
    assert all(s.capacity == 4 for buf in doc.design.buffers for s in buf.storage.slots)
    assert [len(buf.storage.slots) for buf in doc.design.buffers] == [4, 1]
    e.undo(-1)
    assert doc.design == before
    candidate = editing.convert_storage(doc.design, a, "pool", 10)
    candidate = editing.convert_storage(candidate, b, "pool", 20)
    assert e.commit(candidate, "Convert pools")
    e.select_many((a, b))
    e.properties.inputs["storage"].children_fields["capacity"].control.setText("7")
    assert e.apply_properties()
    assert [b.storage.capacity for b in doc.design.buffers] == [7, 7]


def test_discard_before_choosing_tool_resets_the_visible_form(window, monkeypatch):
    window.new_blank()
    e = window.editor
    e.set_mode(True)
    add(window, "machine", 2, 2)
    original = e.properties.inputs["name"].control.text()
    e.properties.inputs["name"].control.setText("Discard this draft")
    monkeypatch.setattr(
        QMessageBox, "exec", lambda self: QMessageBox.StandardButton.Discard
    )
    e.choose_tool("buffer")
    assert not e.properties.pending
    assert e.properties.inputs["name"].control.text() == original
    assert not e.properties.apply_button.isEnabled()


def test_accessible_toolbar_toggle_updates_document_and_preserves_current_draft(
    window, app
):
    doc = window.new_blank()
    e = window.editor
    button = e.toolbar.widgetForAction(e.mode_actions[True])
    button.toggle()  # The native accessibility action toggles, rather than clicks.
    assert doc.edit_mode and e.add_button.isEnabled()
    e.properties.inputs["name"].control.setText("Keep this draft")
    button.toggle()
    app.processEvents()
    assert button.isChecked() and e.properties.pending
    e.mode_actions[True].trigger()
    assert e.properties.pending
    assert e.properties.inputs["name"].control.text() == "Keep this draft"
    e.properties.cancel_button.click()
    e.toolbar.widgetForAction(e.mode_actions[False]).toggle()
    assert not doc.edit_mode and not e.add_button.isEnabled()


def test_accessible_view_toggles_update_the_actual_view(window):
    from PySide6.QtWidgets import QToolBar

    doc = window.new_template()
    toolbar = window.findChild(QToolBar, "mainToolbar")
    toolbar.widgetForAction(window.layer_actions["names"]).toggle()
    assert doc.scene.names_visible
    toolbar.widgetForAction(window.layout_actions["focus"]).toggle()
    assert window.layout_mode == "focus"
    toolbar.widgetForAction(window.layout_actions["standard"]).toggle()
    assert window.layout_mode == "standard"
    window.editor.set_mode(True)
    window.editor.toolbar.widgetForAction(window.editor.continuous_action).toggle()
    assert window.editor.continuous_action.isChecked()


def test_mouse_toolbar_clicks_toggle_once(window, app):
    from PySide6.QtWidgets import QToolBar

    doc = window.new_template()
    toolbar = window.findChild(QToolBar, "mainToolbar")
    names = toolbar.widgetForAction(window.layer_actions["names"])
    for visible in (True, False):
        QTest.mouseClick(names, Qt.MouseButton.LeftButton)
        app.processEvents()
        assert doc.scene.names_visible == visible
        assert names.isChecked() == visible
    for mode in (True, True, False):
        button = window.editor.toolbar.widgetForAction(window.editor.mode_actions[mode])
        QTest.mouseClick(button, Qt.MouseButton.LeftButton)
        app.processEvents()
        assert doc.edit_mode == mode and button.isChecked()


def test_catalog_dialog_guards_references_and_preserves_undo_mode(window):
    from smartsom.config.factory_design import FactoryAuthoring
    from smartsom.studio.dialogs import OperationCatalogDialog

    doc = window.new_template()
    e = window.editor
    e.set_mode(True)
    window.select_entity(None)
    assert "operation_catalog" in e.properties.operation_buttons
    before = doc.file
    dialog = OperationCatalogDialog(doc.design, "auto", window)
    dialog.types.setCurrentRow(0)
    dialog.remove_button.click()
    assert "machine_001" in dialog.error.text()
    assert dialog.design == doc.design
    dialog.add_button.click()
    assert dialog.mode.currentData() == "manual"
    assert dialog.design.operation_types[-1] == "operation_5"
    assert doc.file == before  # Draft has not been applied.
    assert e.commit(
        dialog.design,
        "Edit operation types",
        authoring=FactoryAuthoring(operation_catalog_mode="manual"),
    )
    assert doc.modified and doc.authoring.operation_catalog_mode == "manual"
    e.undo(-1)
    assert doc.file == before and not doc.modified
    e.undo(1)
    assert doc.design.operation_types[-1] == "operation_5"
    assert doc.authoring.operation_catalog_mode == "manual"
    dialog.close()


def test_machine_gestures_default_types_and_manual_selection_are_atomic(
    app, window, monkeypatch
):
    from smartsom.config.factory_design import FactoryAuthoring
    from smartsom.studio.dialogs import MachineCapabilitiesDialog

    doc = window.new_blank()
    e = window.editor
    e.set_mode(True)
    for n in range(4):
        e.choose_tool("machine")
        drag(app, doc, (n * 2 + 0.5, 1.5), (n * 2 + 0.5, 1.5))
    assert doc.design.operation_types == tuple(f"operation_{n}" for n in range(1, 5))
    assert [m.operation_types for m in doc.design.machines] == [
        (t,) for t in doc.design.operation_types
    ]
    e.commit(
        doc.design,
        "Manual catalog",
        authoring=FactoryAuthoring(operation_catalog_mode="manual"),
    )
    before, count = doc.file, doc.undo_stack.count()
    monkeypatch.setattr(
        MachineCapabilitiesDialog, "exec", lambda _: QDialog.DialogCode.Rejected
    )
    e.choose_tool("machine")
    drag(app, doc, (8.5, 1.5), (8.5, 1.5))
    assert doc.file == before and doc.undo_stack.count() == count

    def choose(dialog):
        dialog.types.item(1).setCheckState(Qt.CheckState.Checked)
        dialog.accept()
        return dialog.result()

    monkeypatch.setattr(MachineCapabilitiesDialog, "exec", choose)
    e.choose_tool("machine")
    drag(app, doc, (8.5, 1.5), (8.5, 1.5))
    assert doc.design.machines[-1].operation_types == ("operation_2",)
    assert doc.design.operation_types == before.factory.operation_types
    assert doc.undo_stack.count() == count + 1
    e.undo(-1)
    assert doc.file == before


def test_manual_empty_catalog_cannot_place_machine(app, window):
    from smartsom.config.factory_design import FactoryAuthoring

    doc = window.new_blank()
    e = window.editor
    e.set_mode(True)
    e.commit(
        doc.design,
        "Manual",
        authoring=FactoryAuthoring(operation_catalog_mode="manual"),
    )
    before = doc.file
    e.choose_tool("machine")
    drag(app, doc, (2.5, 2.5), (2.5, 2.5))
    assert doc.file == before
    assert "Add an operation type" in window.status_message.text()


def test_cross_document_paste_import_cancel_and_undo(app, window, monkeypatch):
    source = window.new_blank()
    e = window.editor
    e.set_mode(True)
    a = add(window, "machine", 2, 2)
    e.select_many((a,))
    e.copy_selection()
    target = window.new_blank()
    e.set_mode(True)
    app.processEvents()
    before = target.file
    monkeypatch.setattr(
        QMessageBox, "exec", lambda _: QMessageBox.StandardButton.Cancel
    )
    e.paste_selection()
    drag(app, target, (4.5, 4.5), (4.5, 4.5))
    assert target.file == before and not target.modified
    monkeypatch.setattr(QMessageBox, "exec", lambda _: QMessageBox.StandardButton.Ok)
    e.paste_selection()
    drag(app, target, (4.5, 4.5), (4.5, 4.5))
    assert target.design.operation_types == source.design.operation_types
    assert (
        target.design.machines[0].operation_types
        == source.design.machines[0].operation_types
    )
    assert target.authoring.operation_catalog_mode == "manual"
    assert target.undo_stack.count() == 1
    e.undo(-1)
    assert target.file == before and not target.modified
    e.undo(1)
    assert target.authoring.operation_catalog_mode == "manual"


def test_manual_mode_survives_save_as_template_override_and_recovery(window, tmp_path):
    from smartsom.config.factory_design import (
        FactoryAuthoring,
        load_factory_design_file,
    )

    doc = window.new_template()
    e = window.editor
    e.set_mode(True)
    e.commit(
        editing.add_operation_type(doc.design),
        "Add type",
        authoring=FactoryAuthoring(operation_catalog_mode="manual"),
    )
    expected = doc.file
    path = tmp_path / "saved.yaml"
    assert e.save_to(doc, path)
    assert load_factory_design_file(path)[0] == expected
    assert e.save_to(doc, tmp_path / "save-as.yaml")
    reopened = window.open_path(path)
    assert reopened.file == expected
    assert window.new_from_path(path).file == expected
    e.catalog.builtin_path(1).parent.mkdir(parents=True, exist_ok=True)
    assert e.save_to(
        doc, e.catalog.builtin_path(1), as_template=True, origin=doc.origin
    )
    assert window.new_template(1).file == expected
    recovery_path = e.recovery.snapshot(doc)
    envelope, _ = load_factory_design_file(recovery_path)
    assert envelope == expected
    recovered = window.add_design(envelope.factory, authoring=envelope.authoring)
    assert recovered.file == expected


def test_catalog_mode_can_be_restored_without_reassigning_machines(window):
    from smartsom.studio.dialogs import OperationCatalogDialog

    doc = window.new_template()
    # Two shared categories; no need to remove the unused catalog entries.
    shared = replace(
        doc.design,
        machines=tuple(
            replace(m, operation_types=("operation_1",)) for m in doc.design.machines
        ),
        operation_types=("operation_1", "custom"),
    )
    dialog = OperationCatalogDialog(shared, "manual", window)
    dialog.mode.setCurrentIndex(0)
    assert dialog.design.operation_types == (
        "operation_1",
        "operation_2",
        "operation_3",
        "operation_4",
        "custom",
    )
    assert dialog.design.machines == shared.machines
    dialog.close()


def test_drawing_frame_dialog_roundtrip_undo_and_export(window, app, tmp_path):
    from smartsom.config.drawing_state import DrawingJob
    from smartsom.config.factory_design import load_factory_design_file
    from smartsom.studio.drawing_dialog import DrawingStateDialog
    from smartsom.studio.export import export_scene, render_image

    doc = window.new_template()
    assert not window.editor.actions["drawFrame"].isEnabled()
    window.editor.set_mode(True)
    assert window.editor.actions["drawFrame"].isEnabled()
    dialog = DrawingStateDialog(doc, window)
    machines = dialog.tables["Machines"]
    machines.cellWidget(0, 1).setCurrentText("PROCESSING")
    machines.cellWidget(0, 3).setValue(4)
    machines.cellWidget(0, 4).setValue(2)
    dialog.add_job(DrawingJob(order=7, attempt=2, owner="machine_001"))
    assert dialog.preview()
    assert not doc.authoring.drawing_state.jobs  # Preview is not an edit.
    dialog.apply()
    drawing = dialog.drawing
    assert window.editor.commit(
        doc.design,
        "Edit drawing frame",
        authoring=doc.authoring.model_copy(update={"drawing_state": drawing}),
    )
    state = doc.scene.drawing_layer.state
    assert state["machines"]["machine_001"]["remaining"] == 2
    assert state["machines"]["machine_001"]["job"] == "d000007_a1"
    original = doc.design
    window.editor.undo(-1)
    assert not doc.authoring.drawing_state.jobs
    window.editor.undo(1)
    assert doc.authoring.drawing_state == drawing
    path = tmp_path / "drawing.yaml"
    assert window.editor.save_to(doc, path)
    loaded, _ = load_factory_design_file(path)
    assert loaded.authoring.drawing_state == drawing
    assert loaded.factory == original
    scene = export_scene(loaded.factory, drawing_state=loaded.authoring.drawing_state)
    blank = export_scene(loaded.factory)
    assert render_image(scene) != render_image(blank)
    assert scene.drawing_layer.state == doc.scene.drawing_layer.state
    export_map(tmp_path / "drawing.svg", loaded.factory, drawing_state=drawing)
    assert (tmp_path / "drawing.svg").stat().st_size > 0
    dialog.deleteLater()
    scene.deleteLater()
    blank.deleteLater()


def test_drawing_frame_invalid_apply_is_atomic(window):
    from smartsom.config.drawing_state import DrawingJob
    from smartsom.studio.drawing_dialog import DrawingStateDialog

    doc = window.new_template()
    window.editor.set_mode(True)
    before = doc.authoring
    dialog = DrawingStateDialog(doc, window)
    machines = dialog.tables["Machines"]
    machines.cellWidget(0, 4).setValue(3)
    assert not dialog.preview()
    machines.cellWidget(0, 3).setValue(4)
    dialog.add_job(DrawingJob(order=1, owner="machine_001"))
    dialog.add_job(DrawingJob(order=2, owner="machine_001"))
    assert not dialog.preview()
    assert "only one job" in dialog.error.text()
    assert doc.authoring == before
    dialog.reject()
    dialog.deleteLater()


def test_drawing_all_resources_and_geometry_reconcile(window):
    from smartsom.config.drawing_state import (
        DrawingAGV,
        DrawingBuffer,
        DrawingJob,
        DrawingState,
        DrawingStation,
    )
    from smartsom.studio.drawing_state import reconcile_drawing
    from smartsom.studio.replay_evidence import movement_conflicts

    doc = window.new_template()
    window.editor.set_mode(True)
    factory = doc.design
    station = factory.inspection_stations[0]
    agv = factory.agvs[0]
    b = next(b for b in factory.buffers if b.role == "system_input")
    drawing = DrawingState(
        jobs=(
            DrawingJob(order=1, owner=agv.agv_id),
            DrawingJob(
                order=2,
                owner=station.inspection_station_id,
                slot=station.slots[0].slot_id,
            ),
        ),
        agvs={agv.agv_id: DrawingAGV(x=1, y=1, conflict=True, charging=True)},
        buffers={b.buffer_id: DrawingBuffer(waiting=12, total=5, remaining=2)},
        stations={
            station.inspection_station_id: DrawingStation(
                inspecting=True, total=4, remaining=2
            )
        },
        disposed={factory.scrap_bins[0].scrap_bin_id: 8},
    )
    assert window.editor.commit(
        factory,
        "Draw frame",
        authoring=doc.authoring.model_copy(update={"drawing_state": drawing}),
    )
    layer = doc.scene.drawing_layer
    assert doc.scene.entity_items[agv.agv_id].pos() == QPointF(40, 40)
    assert movement_conflicts(layer.row) == {agv.agv_id}
    assert layer.charging_preview == {agv.agv_id}
    assert layer.evidence.input_progress(b.buffer_id, 0) == (5, 2, 1)
    assert layer.state["stations"][station.inspection_station_id]["batch"] == [
        "d000002_a0"
    ]
    renamed = editing.rename(factory, agv.agv_id, "renamed_agv")
    revised = reconcile_drawing(factory, renamed, drawing, {agv.agv_id: "renamed_agv"})
    assert revised.jobs[0].owner == "renamed_agv"
    assert "renamed_agv" in revised.agvs
    moved = replace(
        factory, agvs=(replace(agv, initial_cell=Cell(2, 2)), *factory.agvs[1:])
    )
    revised = reconcile_drawing(factory, moved, drawing)
    assert revised.agvs[agv.agv_id].x == 2
    assert revised.agvs[agv.agv_id].y == 2


def test_agv_state_is_editable_in_selected_properties(window, tmp_path):
    from smartsom.config.factory_design import load_factory_design_file
    from smartsom.studio.replay_evidence import movement_conflicts

    doc = window.new_template()
    window.editor.set_mode(True)
    a, b = doc.design.agvs[:2]
    window.select_entity(a.agv_id)
    status, partner = window.editor.properties.agv_state_controls
    assert status.currentData() == "NORMAL"
    assert not partner.isEnabled()
    status.setCurrentIndex(status.findData("CONFLICT"))
    assert partner.isEnabled()
    assert partner.findData(a.agv_id) == -1
    partner.setCurrentIndex(partner.findData(b.agv_id))
    assert window.editor.apply_properties()
    state = doc.authoring.drawing_state.agvs[a.agv_id]
    assert state.status == "CONFLICT"
    assert state.conflict_with == b.agv_id
    assert a.agv_id in movement_conflicts(doc.scene.drawing_layer.row)
    window.editor.set_mode(False)
    values = [
        window.property_tree.topLevelItem(i)
        for i in range(window.property_tree.topLevelItemCount())
    ]
    assert any(
        item.text(0) == "Conflict with" and item.text(1) == b.agv_id for item in values
    )
    window.editor.set_mode(True)
    status, partner = window.editor.properties.agv_state_controls
    status.setCurrentIndex(status.findData("NORMAL"))
    assert window.editor.apply_properties()
    assert not doc.authoring.drawing_state.agvs[a.agv_id].conflict
    assert doc.authoring.drawing_state.agvs[a.agv_id].conflict_with is None
    window.editor.undo(-1)
    assert doc.authoring.drawing_state.agvs[a.agv_id].conflict_with == b.agv_id
    path = tmp_path / "conflict.yaml"
    assert window.editor.save_to(doc, path)
    assert (
        load_factory_design_file(path)[0].authoring.drawing_state.agvs[a.agv_id]
        == state
    )


def test_machine_state_properties_browse_edit_and_undo(window, tmp_path):
    from smartsom.config.drawing_state import DrawingMachine
    from smartsom.config.factory_design import load_factory_design_file

    doc = window.new_template()
    editor = window.editor
    editor.set_mode(True)
    drawing = doc.authoring.drawing_state.model_copy(
        update={
            "machines": {
                "machine_002": DrawingMachine(
                    status="DOWN", total=5, remaining=3, mode="slow"
                )
            }
        }
    )
    assert editor.commit(
        doc.design,
        "Set breakdown",
        authoring=doc.authoring.model_copy(update={"drawing_state": drawing}),
    )
    editor.set_mode(False)
    window.select_entity("machine_002")
    root = window.property_tree.topLevelItem(0)
    assert root.text(0) == "Current state"
    assert root.isExpanded()
    values = {
        root.child(i).text(0): root.child(i).text(1) for i in range(root.childCount())
    }
    assert values["Status"] == "Breakdown"
    assert "! = machine breakdown" in values["Indicator"]
    editor.set_mode(True)
    status, mode, total, remaining = editor.properties.machine_state_controls
    assert status.currentData() == "DOWN"
    assert remaining.value() == 3
    status.setCurrentIndex(status.findData("PROCESSING"))
    remaining.setValue(6)
    assert not editor.apply_properties()
    assert doc.authoring.drawing_state.machines["machine_002"].status == "DOWN"
    remaining.setValue(2)
    assert editor.apply_properties()
    state = doc.scene.drawing_layer.state["machines"]["machine_002"]
    assert state["status"] == "PROCESSING"
    assert not state["down"]
    assert state["remaining"] == 2
    editor.undo(-1)
    assert doc.scene.drawing_layer.state["machines"]["machine_002"]["down"]
    path = tmp_path / "machine-state.yaml"
    assert editor.save_to(doc, path)
    assert (
        load_factory_design_file(path)[0]
        .authoring.drawing_state.machines["machine_002"]
        .status
        == "DOWN"
    )


def test_template_3_edit_save_override_restore_and_management(
    window, app, tmp_path, monkeypatch
):
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QListWidget, QPushButton

    from smartsom.config.factory_design import load_factory_design_file
    from smartsom.studio.templates import load_template_file

    doc = window.new_template(3)
    original = load_template_file(3)
    assert doc.design == original.factory and not doc.edit_mode
    window.editor.set_mode(True)
    window.editor.properties.inputs["name"].control.setText("My compact map")
    assert window.editor.apply_properties()
    window.editor.undo(-1)
    assert doc.design == original.factory
    window.editor.undo(1)
    path = tmp_path / "template3-copy.yaml"
    window.editor.save_to(doc, path)
    assert load_factory_design_file(path)[0].factory == doc.design
    assert load_factory_design_file(path)[0].authoring == original.authoring
    assert window.new_from_path(path).design == doc.design

    third = window.new_template(3)
    window.editor.set_mode(True)
    window.editor.properties.inputs["name"].control.setText("Local template 3")
    window.editor.apply_properties()

    def update(dialog):
        dialog.update_template.setChecked(True)
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(TemplateSaveDialog, "exec", update)
    assert window.editor.save()
    assert third.origin.builtin == 3
    assert window.new_template(3).design.name == "Local template 3"
    window.editor.catalog.restore_original(3)
    assert window.new_template(3).design == original.factory
    window.editor.catalog.register(path)
    # The new built-in must not shift the first user file into the built-in branch.
    for row, expected in (
        (2, "Template 3"),
        (3, "Template 4 · Small"),
        (4, "Template 5 · Medium"),
        (5, "Template 6 · Large"),
        (6, "My compact map"),
    ):

        def choose():
            dialog = app.activeModalWidget()
            entries = dialog.findChild(QListWidget)
            assert entries.count() == 7
            entries.setCurrentRow(row)
            next(
                b
                for b in dialog.findChildren(QPushButton)
                if b.text() == "New from selected template"
            ).click()

        QTimer.singleShot(0, choose)
        window.editor.manage_templates()
        assert window.current_document.design.name == expected
        assert not window.current_document.edit_mode
