"""Read-only desktop acceptance using Qt events and complete design documents."""

import importlib.util
import os
import time
from dataclasses import replace

import pytest

if importlib.util.find_spec("PySide6") is None:
    if os.environ.get("SMARTSOM_REQUIRE_STUDIO") == "1":
        raise ImportError("Studio acceptance requires uv sync --locked --extra studio")
    pytest.skip("optional studio extra is not installed", allow_module_level=True)

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QEvent, QPoint, QPointF, QRectF, QSettings, Qt, QTimer
from PySide6.QtGui import QColor, QFocusEvent, QImage, QPainter, QPalette, QWheelEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import (
    QApplication,
    QGraphicsItem,
    QGraphicsRectItem,
    QTabBar,
    QToolBar,
)

from smartsom.config.codec import canonical_json
from smartsom.config.factory_design import load_factory_design, save_factory_design
from smartsom.domain.factory_design import (
    AGVDesign,
    BufferDesign,
    BufferSlotTarget,
    BufferTarget,
    Cell,
    ChargerDesign,
    ChargerTarget,
    FactoryDesign,
    Footprint,
    GridDesign,
    InspectionSlotTarget,
    InspectionStationDesign,
    MachineDesign,
    MachineTarget,
    PoolStorage,
    PortBinding,
    PortDesign,
    ScrapBinDesign,
    ScrapBinTarget,
    SlotDesign,
    SlotStorage,
    validate_factory_design,
)
from smartsom.studio.canvas import FactoryScene
from smartsom.studio.items import CELL_SIZE
from smartsom.studio.templates import load_template_2
from smartsom.studio.window import NewDesignDialog, StudioWindow

pytestmark = pytest.mark.studio


@pytest.fixture(scope="module")
def app():
    application = QApplication.instance() or QApplication([])
    application.setQuitOnLastWindowClosed(False)
    yield application


@pytest.fixture
def window(app):
    instance = StudioWindow()
    instance.resize(1440, 850)
    instance.show()
    app.processEvents()
    yield instance
    instance.close()
    instance.deleteLater()
    app.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    app.processEvents()


def nodes(tree):
    pending = [tree.topLevelItem(i) for i in range(tree.topLevelItemCount())]
    while pending:
        item = pending.pop(0)
        yield item
        pending.extend(item.child(i) for i in range(item.childCount()))


def click_tree_item(app, tree, item):
    parent = item.parent()
    while parent:
        parent.setExpanded(True)
        parent = parent.parent()
    tree.scrollToItem(item)
    app.processEvents()
    QTest.mouseClick(
        tree.viewport(),
        Qt.MouseButton.LeftButton,
        pos=tree.visualItemRect(item).center(),
    )
    app.processEvents()


def test_template_selection_properties_bindings_and_slots(app, window):
    document = window.new_template(2)
    app.processEvents()
    assert len(window.scene.entity_items) == 81
    port = next(p for p in document.design.ports if len(p.bindings) == 4)
    window.select_entity(port.port_id)
    assert document.selected_id == port.port_id
    assert (
        window.resource_tree.currentItem().data(0, Qt.ItemDataRole.UserRole)
        == port.port_id
    )
    assert sum(line.isVisible() for _, _, line in window.scene.binding_items) == 4
    assert any(item.text(1) == port.port_id for item in nodes(window.property_tree))
    assert all(
        not item.flags() & Qt.ItemFlag.ItemIsEditable
        for item in nodes(window.property_tree)
    )

    owner_id = port.bindings[0].target.buffer_id
    window.select_entity(owner_id)
    slot = next(
        item
        for item in nodes(window.property_tree)
        if isinstance(item.data(0, Qt.ItemDataRole.UserRole), Cell)
    )
    cell = slot.data(0, Qt.ItemDataRole.UserRole)
    click_tree_item(app, window.property_tree, slot)
    assert window.scene.slot_highlight.isVisible()
    assert window.scene.slot_highlight.rect().left() == cell.x * CELL_SIZE + 1
    assert window.scene.slot_highlight.rect().top() == cell.y * CELL_SIZE + 1

    machine = window.scene.entity_items["machine_001"]
    point = window.view.mapFromScene(machine.sceneBoundingRect().center())
    QTest.mouseClick(window.view.viewport(), Qt.MouseButton.LeftButton, pos=point)
    app.processEvents()
    assert document.selected_id == "machine_001"
    assert not machine.flags() & QGraphicsItem.GraphicsItemFlag.ItemIsMovable


def test_multi_tabs_templates_and_invalid_open_do_not_change_sources(
    app, window, tmp_path
):
    path = tmp_path / "factory.yaml"
    source = load_template_2()
    save_factory_design(path, source)
    before = path.read_bytes()
    opened = window.open_path(path)
    assert window.open_path(path.parent / "." / path.name) is opened
    assert window.tabs.count() == 1
    fresh = window.new_from_path(path)
    assert fresh is not opened and fresh.source_path is None
    assert fresh.design == opened.design
    assert window.tabs.count() == 2
    window.select_entity("agv_001")
    window.layer_actions["grid"].setChecked(False)
    window.tabs.setCurrentIndex(0)
    assert window.current_document is opened
    assert window.layer_actions["grid"].isChecked()
    window.tabs.setCurrentIndex(1)
    assert fresh.selected_id == "agv_001"
    assert not window.layer_actions["grid"].isChecked()
    bad = tmp_path / "bad.yaml"
    bad.write_text("schema: smartsom.factory/v1\nfactory: {}\n")
    assert window.open_path(bad, show_errors=False) is None
    assert window.current_document is fresh and window.tabs.count() == 2
    assert path.read_bytes() == before
    bar = window.tabs.tabBar()
    close_button = bar.tabButton(0, QTabBar.ButtonPosition.LeftSide) or bar.tabButton(
        0, QTabBar.ButtonPosition.RightSide
    )
    assert close_button is not None
    QTest.mouseClick(close_button, Qt.MouseButton.LeftButton)
    app.processEvents()
    assert window.current_document is fresh and window.tabs.count() == 1


def test_draft_problem_locates_the_resource(app, window, tmp_path):
    design = FactoryDesign(
        "F1",
        "Draft",
        GridDesign(5, 5),
        ports=(PortDesign("P1", "Unbound port", Cell(2, 2)),),
    )
    path = tmp_path / "draft.yaml"
    save_factory_design(path, design)
    window.open_path(path)
    issue = next(
        item
        for item in nodes(window.issue_tree)
        if item.data(0, Qt.ItemDataRole.UserRole) is not None
    )
    click_tree_item(app, window.issue_tree, issue)
    assert window.current_document.selected_id == "P1"
    assert window.scene.entity_items["P1"].isSelected()


def test_factory_overview_does_not_share_resource_id_namespace(app, window):
    design = replace(load_template_2(), factory_id="machine_001")
    assert not validate_factory_design(design)
    document = window.add_design(design)
    app.processEvents()
    root = window.resource_tree.topLevelItem(0)
    assert document.selected_id is None
    assert window.property_title.text() == design.name

    window.select_entity("machine_001")
    assert window.property_title.text() == design.machines[0].name
    click_tree_item(app, window.resource_tree, root)
    assert document.selected_id is None
    assert window.property_title.text() == design.name
    assert not window.scene.selectedItems()

    window.select_entity("machine_001")
    blank_cell = window.view.mapFromScene(QPointF(10.5 * CELL_SIZE, 6.5 * CELL_SIZE))
    QTest.mouseClick(window.view.viewport(), Qt.MouseButton.LeftButton, pos=blank_cell)
    app.processEvents()
    assert document.selected_id is None
    assert window.property_title.text() == design.name
    window.new_blank()
    window.tabs.setCurrentIndex(0)
    assert window.property_title.text() == design.name


def test_optional_resource_values_are_readable(app, window):
    design = load_template_2()
    design = replace(
        design, agvs=(replace(design.agvs[0], battery=None), *design.agvs[1:])
    )
    window.add_design(design)
    window.select_entity("buffer_001")
    assert any(item.text(1) == "Unlimited" for item in nodes(window.property_tree))
    window.select_entity("agv_001")
    assert any(item.text(1) == "Not modeled" for item in nodes(window.property_tree))
    window.select_entity(None)
    machines = next(
        item for item in nodes(window.property_tree) if item.text(0) == "Machines"
    )
    assert machines.text(1) == "8 items" and machines.childCount() == 0


def test_workspace_remains_readable_with_dark_system_palette(app):
    original = app.palette()
    dark = QPalette(original)
    for role in (QPalette.ColorRole.WindowText, QPalette.ColorRole.Text):
        dark.setColor(role, QColor("white"))
    for role in (QPalette.ColorRole.Window, QPalette.ColorRole.Base):
        dark.setColor(role, QColor("#202020"))
    app.setPalette(dark)
    instance = None
    try:
        instance = StudioWindow()
        instance.new_template(2)
        instance.show()
        app.processEvents()
        dialog = NewDesignDialog(instance)
        dialog.show()
        app.processEvents()
        for widget in (
            instance.property_title,
            instance.status_message,
            instance.binding_combo,
            dialog.template,
        ):
            palette = widget.palette()
            assert palette.color(QPalette.ColorRole.WindowText).lightness() < 140
        # Labels are transparent; their own Window role need not be painted.
        for surface in (instance, dialog):
            assert surface.palette().color(QPalette.ColorRole.Window).lightness() > 200
        assert instance.binding_combo.currentText() == "Selected"
        dialog.deleteLater()
    finally:
        if instance is not None:
            instance.close()
        app.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        app.setPalette(original)
        app.processEvents()


def test_navigation_and_layers_preserve_data(app, window):
    document = window.new_template(2)
    app.processEvents()
    numbers = window.layer_actions["names"]
    numbers_button = window.findChild(QToolBar, "mainToolbar").widgetForAction(numbers)
    assert numbers_button.isVisible() and numbers_button.text() == "Show numbers"
    assert not document.scene.names_visible
    assert all(not item.names_visible for item in window.scene.entity_items.values())
    before = canonical_json(document.design)
    window.view.zoom_by(4)
    app.processEvents()
    viewport = window.view.viewport()
    point = viewport.rect().center() + QPoint(25, 25)
    QTest.mouseMove(viewport, point)
    app.processEvents()
    anchor = window.view.mapToScene(point)
    scale = window.view.transform().m11()
    event = QWheelEvent(
        QPointF(point),
        QPointF(viewport.mapToGlobal(point)),
        QPoint(),
        QPoint(0, 120),
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.ScrollUpdate,
        False,
    )
    app.sendEvent(viewport, event)
    app.processEvents()
    assert window.view.transform().m11() > scale
    after_anchor = window.view.mapToScene(point)
    assert abs(after_anchor.x() - anchor.x()) < 4
    assert abs(after_anchor.y() - anchor.y()) < 4

    scrollbar = window.view.horizontalScrollBar()
    before_scroll = scrollbar.value()
    QTest.mousePress(viewport, Qt.MouseButton.MiddleButton, pos=point)
    QTest.mouseMove(viewport, point + QPoint(40, 25))
    QTest.mouseRelease(
        viewport, Qt.MouseButton.MiddleButton, pos=point + QPoint(40, 25)
    )
    assert scrollbar.value() != before_scroll
    for mode in ("all", "none", "selected"):
        window.binding_combo.setCurrentIndex(window.binding_combo.findData(mode))
        app.processEvents()
        assert window.scene.binding_mode == mode
        if mode != "selected":
            assert all(
                line.isVisible() == (mode == "all")
                for _, _, line in window.scene.binding_items
            )
    window.layer_actions["ports"].setChecked(False)
    assert all(
        not window.scene.entity_items[p.port_id].isVisible()
        for p in document.design.ports
    )
    QTest.mouseClick(numbers_button, Qt.MouseButton.LeftButton)
    assert numbers.isChecked() and numbers_button.text() == "Hide numbers"
    assert all(item.names_visible for item in window.scene.entity_items.values())
    window.new_template(2)
    assert not numbers.isChecked() and numbers_button.text() == "Show numbers"
    window.tabs.setCurrentIndex(0)
    assert numbers.isChecked() and numbers_button.text() == "Hide numbers"
    window.set_workspace_layout("focus")
    assert document.scene.names_visible
    QTest.mouseClick(numbers_button, Qt.MouseButton.LeftButton)
    assert not numbers.isChecked() and numbers_button.text() == "Show numbers"
    assert all(not item.names_visible for item in window.scene.entity_items.values())
    window.fit_action.trigger()
    assert canonical_json(document.design) == before


def test_new_dialog_default_and_blank_design(app, window):
    dialog = NewDesignDialog(window)
    assert dialog.template.isChecked()
    assert not dialog.template_2.isChecked()
    dialog.deleteLater()
    default = window.new_template()
    assert default.design.grid == GridDesign(12, 12)
    assert [m.name for m in default.design.machines] == [
        f"Machine {n}" for n in range(1, 5)
    ]
    document = window.new_blank()
    assert document.design.grid == GridDesign(20, 15)
    assert not document.issues
    assert not document.design.machines


def test_new_dialog_opens_all_templates_and_defaults_to_compact(app, window):
    for number, size, count in (
        (1, (12, 12), 4),
        (2, (42, 12), 8),
        (3, (12, 8), 8),
        (4, (19, 11), 8),
        (5, (29, 17), 20),
        (6, (39, 23), 40),
    ):

        def accept_template():
            dialog = app.activeModalWidget()
            if number == 2:
                dialog.template_2.click()
            elif number == 3:
                dialog.template_3.click()
            elif number >= 4:
                getattr(dialog, f"template_{number}").click()
            dialog.accept()

        QTimer.singleShot(0, accept_template)
        window.new_dialog()
        assert (
            window.current_document.design.name.split(" · ")[0] == f"Template {number}"
        )
        assert window.current_document.design.grid == GridDesign(*size)
        assert len(window.current_document.design.machines) == count
        assert window.current_document.source_path is None
        assert not window.current_document.issues
        assert not window.scene.names_visible
    assert window.tabs.count() == 6
    window.tabs.setCurrentIndex(0)
    assert window.current_document.design.name == "Template 1"


def click_layout(app, window, mode):
    toolbar = window.findChild(QToolBar, "mainToolbar")
    QTest.mouseClick(
        toolbar.widgetForAction(window.layout_actions[mode]),
        Qt.MouseButton.LeftButton,
    )
    QTest.qWait(30)
    app.processEvents()


def test_layout_switch_keeps_documents_selection_and_property_navigation(
    app, window, tmp_path
):
    path = tmp_path / "factory.yaml"
    save_factory_design(path, load_template_2())
    source_bytes = path.read_bytes()
    document = window.open_path(path)
    other = window.new_blank()
    window.tabs.setCurrentIndex(0)
    window.select_entity("buffer_004")
    app.processEvents()
    scene, view = window.scene, window.view
    slot = next(
        item
        for item in nodes(window.property_tree)
        if isinstance(item.data(0, Qt.ItemDataRole.UserRole), Cell)
    )
    click_tree_item(app, window.property_tree, slot)
    window.fit_action.trigger()
    QTest.qWait(30)
    baseline_scale = view.transform().m11()
    baseline_width = view.viewport().width()
    before = canonical_json(document.design)

    click_layout(app, window, "focus")
    assert window.layout_mode == "focus"
    assert window.layout_actions["focus"].isChecked()
    assert not window.layout_actions["standard"].isChecked()
    assert window.current_document is document
    assert window.scene is scene and window.view is view
    assert document.selected_id == "buffer_004"
    assert window.property_tree.currentItem() is slot
    assert slot.parent().isExpanded()
    assert scene.slot_highlight.isVisible()
    assert window.resource_panel.isHidden()
    assert window.issues_button.text() == "Design checks · 0 issues"
    assert window.property_panel.parentWidget() is window.canvas_splitter
    assert view.viewport().width() > baseline_width * 1.4
    assert view.transform().m11() > baseline_scale * 1.2
    assert window.property_panel.geometry().top() >= window.workspace_splitter.height()

    window.issues_button.click()
    assert window.issue_panel.isVisible()
    window.issues_button.click()
    window.resources_action.trigger()
    assert window.resource_panel.isVisible()
    window.workspace_splitter.setSizes([285, 1154])
    resource_width = window.resource_panel.width()
    window.resources_action.trigger()
    window.resources_action.trigger()
    assert window.resource_panel.width() == resource_width
    window.resource_filter.setText("machine_003")
    click_tree_item(app, window.resource_tree, window._tree_items["machine_003"])
    assert document.selected_id == "machine_003"
    window.resources_action.trigger()
    assert window.resource_panel.isHidden()

    window.tabs.setCurrentIndex(1)
    assert window.current_document is other and window.layout_mode == "focus"
    window.tabs.setCurrentIndex(0)
    assert document.selected_id == "machine_003"
    click_layout(app, window, "standard")
    assert window.resource_panel.isVisible()
    assert window.property_panel.parentWidget() is window.workspace_splitter
    assert window.issue_panel.isVisible()
    assert canonical_json(document.design) == before
    assert path.read_bytes() == source_bytes
    assert len(window.documents) == 2


def test_layout_switch_preserves_manual_zoom_and_center(app, window):
    window.new_template(2)
    QTest.qWait(30)
    window.view.zoom_by(3)
    window.view.centerOn(QPointF(1000, 220))
    app.processEvents()
    scale = window.view.transform().m11()
    center = window.view.mapToScene(window.view.viewport().rect().center())
    for mode in ("focus", "standard", "focus"):
        click_layout(app, window, mode)
        assert window.view.transform().m11() == pytest.approx(scale)
        after = window.view.mapToScene(window.view.viewport().rect().center())
        assert abs(after.x() - center.x()) < 5
        assert abs(after.y() - center.y()) < 5


def test_hidden_document_fits_new_layout_when_activated(app, window):
    first = window.new_template(2)
    QTest.qWait(30)
    original_scale = first.view.transform().m11()
    window.new_blank()
    click_layout(app, window, "focus")
    window.tabs.setCurrentIndex(0)
    QTest.qWait(30)
    assert first.view.transform().m11() > original_scale
    visible_map = first.view.mapFromScene(first.scene.map_rect).boundingRect()
    assert first.view.viewport().rect().contains(visible_map)


def test_unknown_layout_setting_falls_back_to_standard(app, tmp_path):
    settings = QSettings(str(tmp_path / "studio.ini"), QSettings.Format.IniFormat)
    settings.setValue("workspace/layout", "old-layout")
    settings.setValue("workspace/standard/workspace", "invalid splitter data")
    instance = StudioWindow(settings=settings)
    try:
        instance.show()
        instance.new_template(2)
        QTest.qWait(30)
        assert instance.layout_mode == "standard"
        assert instance.resource_panel.isVisible()
        assert instance.property_panel.isVisible()
    finally:
        instance.close()
        app.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def test_layout_settings_restore_both_modes_without_persisting_documents(app, tmp_path):
    settings_path = str(tmp_path / "studio.ini")
    settings = QSettings(settings_path, QSettings.Format.IniFormat)
    instance = StudioWindow(settings=settings)
    instance.resize(1312, 741)
    instance.show()
    instance.new_template(2)
    QTest.qWait(30)
    instance.workspace_splitter.setSizes([245, 702, 363])
    instance.issues_splitter.setSizes([510, 140])
    standard_sizes = instance.workspace_splitter.sizes()
    click_layout(app, instance, "focus")
    instance.canvas_splitter.setSizes([430, 220])
    focus_sizes = instance.canvas_splitter.sizes()
    click_layout(app, instance, "standard")
    assert instance.workspace_splitter.sizes() == standard_sizes
    click_layout(app, instance, "focus")
    assert instance.canvas_splitter.sizes() == focus_sizes
    instance.resources_action.trigger()
    instance.select_entity("machine_003")
    instance.close()
    app.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    assert settings.status() == QSettings.Status.NoError
    # Older versions remembered the last mode; it must not override the default.
    settings.setValue("workspace/layout", "focus")
    settings.sync()

    restored = StudioWindow(
        settings=QSettings(settings_path, QSettings.Format.IniFormat)
    )
    try:
        restored.resize(1312, 741)
        restored.show()
        QTest.qWait(30)
        assert restored.layout_mode == "standard"
        assert restored.layout_actions["standard"].isChecked()
        assert restored.resource_panel.isVisible()
        assert restored.workspace_splitter.sizes() == standard_sizes
        assert not restored.documents
        restored.new_template(2)
        click_layout(app, restored, "focus")
        assert restored.resource_panel.isHidden()
        assert restored.canvas_splitter.sizes() == focus_sizes
    finally:
        restored.close()
        app.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def medium_design():
    machines, buffers, ports, agvs = [], [], [], []
    for number in range(100):
        x, y = 4 + (number % 10) * 8, 4 + (number // 10) * 8
        machine_id = f"machine_{number + 1:03d}"
        machines.append(
            MachineDesign(
                machine_id,
                f"Machine {number + 1}",
                Footprint(x + 2, y, 2, 2),
                operation_types=(f"operation_{number + 1}",),
            )
        )
        for side, offset, access_x in (("pre", 0, x + 1), ("post", 4, x + 4)):
            buffer_id = f"buffer_{side}_{number + 1:03d}"
            slots = tuple(
                SlotDesign(f"slot_{j + 1:03d}", Cell(j % 2, j // 2)) for j in range(4)
            )
            buffers.append(
                BufferDesign(
                    buffer_id,
                    f"{side.title()} {number + 1}",
                    Footprint(x + offset, y, 2, 2),
                    role=f"machine_{side}",
                    storage=SlotStorage(slots),
                    machine_id=machine_id,
                )
            )
            ports.append(
                PortDesign(
                    f"port_{side}_{number + 1:03d}",
                    f"{side.title()} access {number + 1}",
                    Cell(access_x, y + 2),
                    bindings=tuple(
                        PortBinding(BufferSlotTarget(buffer_id, slot.slot_id))
                        for slot in slots
                    ),
                )
            )
        agvs.append(
            AGVDesign(f"agv_{number + 1:03d}", f"AGV {number + 1}", Cell(x + 1, y + 2))
        )
    return FactoryDesign(
        "scale_001",
        "100-machine preview",
        GridDesign(100, 100),
        operation_types=tuple(f"operation_{n}" for n in range(1, 101)),
        machines=tuple(machines),
        buffers=tuple(buffers),
        ports=tuple(ports),
        agvs=tuple(agvs),
    )


def test_medium_scale_load_select_and_file_roundtrip(
    app, window, tmp_path, record_property
):
    design = medium_design()
    assert not validate_factory_design(design)
    path = tmp_path / "scale.yaml"
    save_factory_design(path, design)
    start = time.perf_counter()
    document = window.open_path(path)
    app.processEvents()
    record_property("load_and_show_seconds", round(time.perf_counter() - start, 4))
    assert len(document.scene.entity_items) == 600
    window.select_entity("machine_100")
    assert document.selected_id == "machine_100"
    window.view.zoom_by(2)
    window.view.fit_map()
    assert load_factory_design(path)[0] == design
    assert document.design == design


def test_orientation_metadata_does_not_rotate_occupied_slots_twice(app, window):
    design = replace(
        FactoryDesign("F1", "Rotation", GridDesign(12, 12)),
        buffers=(
            BufferDesign(
                "B1",
                "Rotated buffer",
                Footprint(4, 5, 3, 2, 90),
                storage=SlotStorage((SlotDesign("S1", Cell(0, 0)),)),
            ),
        ),
    )
    window.add_design(design)
    window.select_entity("B1")
    app.processEvents()
    item = window.scene.entity_items["B1"]
    assert item.pos() == QPointF(4 * CELL_SIZE, 5 * CELL_SIZE)
    assert item.shape().boundingRect().width() == 3 * CELL_SIZE
    slot = next(
        n
        for n in nodes(window.property_tree)
        if isinstance(n.data(0, Qt.ItemDataRole.UserRole), Cell)
    )
    assert slot.data(0, Qt.ItemDataRole.UserRole) == Cell(4, 5)


def hover_map_item(app, view, identifier):
    point = view.mapFromScene(
        view.scene().entity_items[identifier].sceneBoundingRect().center()
    )
    QTest.mouseMove(view.viewport(), QPoint(2, 2))
    QTest.mouseMove(view.viewport(), point, delay=2)
    app.processEvents()
    return point


def highlighted_items(scene):
    return {
        identifier
        for identifier, item in scene.entity_items.items()
        if item.group_highlighted
    }


def test_machine_group_hover_preserves_selection_properties_and_bindings(app, window):
    document = window.new_template()
    app.processEvents()
    window.select_entity("buffer_005")
    scene = document.scene
    scene.highlight_cell(Cell(2, 2))
    selected = document.selected_id
    properties = [(n.text(0), n.text(1)) for n in nodes(window.property_tree)]
    bindings = [line.isVisible() for _, _, line in scene.binding_items]
    slot_rect = scene.slot_highlight.rect()
    before = canonical_json(document.design)
    group = {"machine_001", "buffer_003", "buffer_004", "port_002", "port_003"}
    for identifier in group:
        hover_map_item(app, document.view, identifier)
        assert highlighted_items(scene) == group
        assert scene.highlighted_ids == group
        assert document.selected_id == selected
        assert scene.entity_items[selected].isSelected()
        assert [
            (n.text(0), n.text(1)) for n in nodes(window.property_tree)
        ] == properties
        assert [line.isVisible() for _, _, line in scene.binding_items] == bindings
        assert (
            scene.slot_highlight.isVisible()
            and scene.slot_highlight.rect() == slot_rect
        )
    hover_map_item(app, document.view, "machine_002")
    assert highlighted_items(scene) == {
        "machine_002",
        "buffer_005",
        "buffer_006",
        "port_004",
        "port_005",
    }
    hover_map_item(app, document.view, "buffer_001")
    assert not highlighted_items(scene)
    assert canonical_json(document.design) == before


@pytest.mark.parametrize(
    "cleanup",
    ["leave", "focus", "deactivate", "hide", "tab", "layout", "pan", "zoom", "ports"],
)
def test_group_hover_clears_with_view_lifecycle(app, window, cleanup):
    document = window.new_template()
    app.processEvents()
    scene, view = document.scene, document.view
    point = hover_map_item(app, view, "port_002")
    assert highlighted_items(scene)
    if cleanup == "leave":
        app.sendEvent(view.viewport(), QEvent(QEvent.Type.Leave))
    elif cleanup == "focus":
        app.sendEvent(view, QFocusEvent(QEvent.Type.FocusOut))
    elif cleanup == "deactivate":
        app.sendEvent(view, QEvent(QEvent.Type.WindowDeactivate))
    elif cleanup == "hide":
        view.hide()
    elif cleanup == "tab":
        window.new_blank()
    elif cleanup == "layout":
        window.set_workspace_layout("focus")
    elif cleanup == "pan":
        QTest.mousePress(view.viewport(), Qt.MouseButton.MiddleButton, pos=point)
        QTest.mouseRelease(view.viewport(), Qt.MouseButton.MiddleButton, pos=point)
    elif cleanup == "zoom":
        view.zoom_by(1.2)
    elif cleanup == "ports":
        scene.set_layer("ports", False)
    assert not highlighted_items(scene)
    assert scene.hovered_entity_id is None
    if cleanup == "ports":
        hover_map_item(app, view, "machine_001")
        assert highlighted_items(scene)
        assert all(
            not scene.entity_items[p.port_id].isVisible() for p in document.design.ports
        )


def test_shared_ports_expand_only_direct_machine_groups(app, window):
    design = FactoryDesign(
        "groups",
        "Shared access",
        GridDesign(12, 6),
        operation_types=("operation_1",),
        machines=tuple(
            MachineDesign(
                name, name, Footprint(x, 0, 1, 1), operation_types=("operation_1",)
            )
            for name, x in [("A", 0), ("B", 4), ("C", 8)]
        ),
        buffers=(
            BufferDesign(
                "preA",
                "A slots",
                Footprint(1, 0, 1, 1),
                role="machine_pre",
                machine_id="A",
                storage=SlotStorage((SlotDesign("s", Cell(0, 0)),)),
            ),
            BufferDesign(
                "preB",
                "B pool",
                Footprint(5, 0, 1, 1),
                role="machine_pre",
                machine_id="B",
                storage=PoolStorage(),
            ),
            BufferDesign(
                "public",
                "Public storage",
                Footprint(10, 0, 1, 1),
                storage=PoolStorage(),
            ),
        ),
        ports=(
            PortDesign(
                "AB",
                "Shared AB",
                Cell(2, 2),
                bindings=(
                    PortBinding(BufferSlotTarget("preA", "s")),
                    PortBinding(BufferTarget("preB")),
                ),
            ),
            PortDesign(
                "BC",
                "Shared BC",
                Cell(6, 2),
                bindings=(
                    PortBinding(MachineTarget("B")),
                    PortBinding(MachineTarget("C")),
                ),
            ),
            PortDesign(
                "public_access",
                "Public access",
                Cell(10, 2),
                bindings=(PortBinding(BufferTarget("public")),),
            ),
        ),
        agvs=(AGVDesign("vehicle", "AGV covering BC", Cell(6, 2)),),
    )
    assert not validate_factory_design(design)
    document = window.add_design(design)
    app.processEvents()
    for trigger, expected in [
        ("A", {"A", "preA", "AB"}),
        ("preB", {"B", "preB", "AB", "BC"}),
        ("AB", {"A", "preA", "B", "preB", "AB", "BC"}),
        ("public_access", set()),
        ("vehicle", set()),
    ]:
        hover_map_item(app, document.view, trigger)
        assert highlighted_items(document.scene) == expected
    # A vehicle above a port wins hit testing; hidden ports stay hidden.
    document.scene.entity_items["vehicle"].hide()
    hover_map_item(app, document.view, "BC")
    assert highlighted_items(document.scene) == {"B", "preB", "C", "AB", "BC"}


def test_buffer_shared_boundaries_are_single_and_stronger_than_slot_grid(app):
    design = FactoryDesign(
        "borders",
        "Borders",
        GridDesign(5, 4),
        buffers=(
            BufferDesign(
                "left",
                "Sparse slots",
                Footprint(1, 1, 2, 2),
                storage=SlotStorage(
                    tuple(
                        SlotDesign(str_id, Cell(x, y))
                        for str_id, x, y in [("a", 0, 0), ("b", 1, 0), ("c", 0, 1)]
                    )
                ),
            ),
            BufferDesign(
                "upper",
                "Upper neighbor",
                Footprint(3, 1, 1, 1),
                storage=SlotStorage((SlotDesign("a", Cell(0, 0)),)),
            ),
            BufferDesign(
                "lower", "Pool neighbor", Footprint(3, 2, 1, 1), storage=PoolStorage()
            ),
        ),
    )
    scene = FactoryScene(design)

    def render():
        result = QImage(200, 160, QImage.Format.Format_ARGB32)
        result.fill(QColor("white"))
        painter = QPainter(result)
        scene.render(painter, QRectF(0, 0, 200, 160), QRectF(0, 0, 200, 160))
        painter.end()
        return result

    normal = render()

    def border_profile(image, x, y):
        return [image.pixelColor(x + dx, y).red() for dx in range(-2, 3)]

    shared = border_profile(normal, 120, 60)
    outside = border_profile(normal, 40, 60)
    internal = border_profile(normal, 80, 60)
    assert sum(value < 180 for value in shared) == sum(value < 180 for value in outside)
    assert min(shared) == min(outside)
    assert min(shared) + 35 < min(internal)
    assert normal.pixelColor(100, 100) == QColor("#f2f5f7")
    # Sample the pool fill away from its centered inventory symbol.
    assert normal.pixelColor(60, 60) == normal.pixelColor(124, 100) == QColor("#e3f3ee")
    # A selected neighbor controls the one shared edge, without another frame.
    scene.select_entity("upper")
    selected = render()
    assert selected.pixelColor(119, 60).red() < normal.pixelColor(119, 60).red()
    assert selected.pixelColor(60, 60) == normal.pixelColor(60, 60)
    scene.select_entity(None)
    assert render() == normal


def test_directionless_agv_cargo_is_transient_and_keeps_selection(app, window):
    images = []
    for heading in ("north", "east", "south", "west"):
        design = FactoryDesign(
            "cargo_preview",
            "Cargo preview",
            GridDesign(3, 3),
            agvs=(
                AGVDesign("vehicle", "Vehicle", Cell(1, 1), initial_heading=heading),
            ),
        )
        original = canonical_json(design)
        document = window.add_design(design)
        app.processEvents()
        item = document.scene.entity_items["vehicle"]
        assert item.loaded is False

        point = window.view.mapFromScene(item.sceneBoundingRect().center())
        QTest.mouseClick(window.view.viewport(), Qt.MouseButton.LeftButton, pos=point)
        app.processEvents()
        assert document.selected_id == "vehicle"
        properties = [(n.text(0), n.text(1)) for n in nodes(window.property_tree)]

        def render():
            result = QImage(80, 80, QImage.Format.Format_ARGB32)
            result.fill(QColor("white"))
            painter = QPainter(result)
            document.scene.render(painter, QRectF(0, 0, 80, 80), QRectF(40, 40, 40, 40))
            painter.end()
            return result

        empty = render()
        assert empty.pixelColor(40, 10) == QColor("#bdefff")
        window.select_entity(None)
        assert render().pixelColor(40, 10) != QColor("#bdefff")
        window.select_entity("vehicle")
        assert render() == empty
        item.set_loaded(True)
        loaded = render()
        assert loaded != empty
        assert loaded.pixelColor(40, 40) == QColor("#B18461")
        # Cargo affects only the center; the vehicle base remains recognizable.
        assert loaded.copy(12, 12, 56, 12) == empty.copy(12, 12, 56, 12)
        assert item.isSelected() and document.selected_id == "vehicle"
        assert [
            (n.text(0), n.text(1)) for n in nodes(window.property_tree)
        ] == properties
        assert canonical_json(document.design) == original
        item.set_loaded(False)
        assert render() == empty
        images.append((empty, loaded))

    # Design heading metadata cannot introduce direction into either glyph.
    assert all(pair == images[0] for pair in images[1:])


def test_binding_highlights_resolve_exact_targets_and_shared_ports(app, window):
    targets = (
        (MachineTarget("machine"), QRectF(0, 0, 80, 80)),
        (BufferSlotTarget("slots", "a"), QRectF(120, 0, 40, 40)),
        (BufferTarget("pool"), QRectF(240, 0, 40, 80)),
        (InspectionSlotTarget("inspection", "a"), QRectF(320, 0, 40, 40)),
        (ScrapBinTarget("scrap"), QRectF(440, 0, 40, 80)),
        (ChargerTarget("charger"), QRectF(520, 0, 40, 40)),
    )
    design = FactoryDesign(
        "binding_preview",
        "Binding preview",
        GridDesign(15, 7),
        machines=(MachineDesign("machine", "Machine", Footprint(0, 0, 2, 2)),),
        buffers=(
            BufferDesign(
                "slots",
                "Slot buffer",
                Footprint(3, 0, 2, 2),
                storage=SlotStorage(
                    (SlotDesign("a", Cell(0, 0)), SlotDesign("b", Cell(1, 1)))
                ),
            ),
            BufferDesign("pool", "Pool", Footprint(6, 0, 1, 2)),
        ),
        inspection_stations=(
            InspectionStationDesign(
                "inspection",
                "Inspection",
                Footprint(8, 0, 2, 2),
                slots=(SlotDesign("a", Cell(0, 0)), SlotDesign("b", Cell(1, 1))),
            ),
        ),
        scrap_bins=(ScrapBinDesign("scrap", "Scrap", Footprint(11, 0, 1, 2)),),
        chargers=(ChargerDesign("charger", "Charger", Footprint(13, 0, 1, 1)),),
        ports=tuple(
            PortDesign(
                f"port_{index}",
                f"Port {index}",
                Cell(index, 4),
                bindings=(
                    PortBinding(
                        target,
                        operations=("charge",)
                        if target.kind == "charger"
                        else ("drop_off",)
                        if target.kind == "scrap_bin"
                        else ("pickup", "drop_off"),
                    ),
                ),
            )
            for index, (target, _) in enumerate(targets)
        )
        + (
            PortDesign(
                "shared",
                "Shared port",
                Cell(7, 4),
                bindings=(PortBinding(targets[1][0]), PortBinding(targets[3][0])),
            ),
        ),
    )
    assert not [i for i in validate_factory_design(design) if i.severity == "error"]
    before = canonical_json(design)
    document = window.add_design(design)
    app.processEvents()
    scene = document.scene

    def visible_targets():
        return {item for _, _, item in scene.binding_items if item.isVisible()}

    def related_ports():
        return {
            p.port_id
            for p in design.ports
            if scene.entity_items[p.port_id].binding_highlighted
        }

    def rects(items):
        return {item.rect().getRect() for item in items}

    # Every supported target type uses its actual area, not a center-to-center line.
    for index, (_, area) in enumerate(targets):
        window.select_entity(f"port_{index}")
        visible = visible_targets()
        assert len(visible) == 1
        assert all(isinstance(item, QGraphicsRectItem) for item in visible)
        assert rects(visible) == {area.adjusted(1, 1, -1, -1).getRect()}
        assert related_ports() == {f"port_{index}"}

    # Two ports share one slot overlay. Reverse lookup does not light the other
    # inspection target merely because the shared port can also access it.
    window.select_entity("slots")
    assert len(visible_targets()) == 1
    assert rects(visible_targets()) == {QRectF(121, 1, 38, 38).getRect()}
    assert related_ports() == {"port_1", "shared"}
    window.select_entity("shared")
    assert rects(visible_targets()) == {
        QRectF(121, 1, 38, 38).getRect(),
        QRectF(321, 1, 38, 38).getRect(),
    }

    # Overlays must not eat clicks on the actual facility underneath.
    point = document.view.mapFromScene(QPointF(140, 20))
    QTest.mouseClick(document.view.viewport(), Qt.MouseButton.LeftButton, pos=point)
    app.processEvents()
    assert document.selected_id == "slots"
    assert related_ports() == {"port_1", "shared"}

    window.binding_combo.setCurrentIndex(window.binding_combo.findData("all"))
    assert len(visible_targets()) == 6
    assert related_ports() == {p.port_id for p in design.ports}
    scene.set_layer("ports", False)
    assert all(not scene.entity_items[p.port_id].isVisible() for p in design.ports)
    assert len(visible_targets()) == 6
    scene.set_layer("ports", True)
    window.binding_combo.setCurrentIndex(window.binding_combo.findData("none"))
    assert not visible_targets() and not related_ports()
    window.binding_combo.setCurrentIndex(window.binding_combo.findData("selected"))
    assert len(visible_targets()) == 1
    window.select_entity(None)
    assert not visible_targets() and not related_ports()
    assert canonical_json(document.design) == before


@pytest.mark.parametrize("scale", [0.7, 1.8])
@pytest.mark.parametrize("loaded", [False, True])
def test_agv_body_and_exposed_port_have_separate_click_targets(
    app, window, scale, loaded
):
    document = window.new_template()
    scene, view = document.scene, document.view
    agv = document.design.agvs[0]
    port = next(p for p in document.design.ports if p.cell == agv.initial_cell)
    before = canonical_json(document.design)
    vehicle = scene.entity_items[agv.agv_id]
    vehicle.set_loaded(loaded)
    view.resetTransform()
    view.scale(scale, scale)
    view.centerOn(vehicle.sceneBoundingRect().center())
    app.processEvents()

    def click_local(x, y):
        point = view.mapFromScene(vehicle.mapToScene(QPointF(x, y)))
        QTest.mouseMove(view.viewport(), point)
        QTest.mouseClick(view.viewport(), Qt.MouseButton.LeftButton, pos=point)
        app.processEvents()

    click_local(20, 20)
    assert document.selected_id == agv.agv_id
    assert scene.hovered_entity_id == agv.agv_id
    for x, y in ((2, 20), (38, 20), (20, 2), (20, 38), (5, 5)):
        click_local(x, y)
        assert document.selected_id == port.port_id
        assert scene.hovered_entity_id == port.port_id
        assert window.property_title.text() == port.name
        assert any(item.isVisible() for _, _, item in scene.binding_items)
        click_local(20, 20)
        assert document.selected_id == agv.agv_id
    scene.set_layer("ports", False)
    click_local(2, 20)
    assert document.selected_id is None
    assert scene.hovered_entity_id is None
    assert canonical_json(document.design) == before
