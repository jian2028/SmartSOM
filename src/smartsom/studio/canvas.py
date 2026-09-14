"""Grid drawing and read-only map navigation."""

import math

from PySide6.QtCore import QEvent, QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QGraphicsScene, QGraphicsView

from smartsom.domain.factory_design import target_cell, target_owner_id
from smartsom.studio.items import (
    BINDING_FILL,
    CELL_SIZE,
    AGVItem,
    BufferBoundaryItem,
    BufferItem,
    ChargerItem,
    EntityItem,
    InspectionStationItem,
    MachineItem,
    PortItem,
    ScrapBinItem,
)


class FactoryScene(QGraphicsScene):
    entity_selected = Signal(str)

    def __init__(self, design, parent=None):
        super().__init__(parent)
        self.design = design
        self.grid_visible = True
        self.names_visible = False
        self.ports_visible = True
        self.binding_mode = "selected"
        self.entity_items = {}
        self.binding_items = []
        self._binding_highlights = {}
        self.hovered_entity_id = None
        self.highlighted_ids = frozenset()
        self.map_rect = QRectF(
            0, 0, design.grid.width * CELL_SIZE, design.grid.height * CELL_SIZE
        )
        self.setSceneRect(
            self.map_rect.adjusted(-CELL_SIZE, -CELL_SIZE, CELL_SIZE, CELL_SIZE)
        )
        collections = [
            (MachineItem, design.machines),
            (BufferItem, design.buffers),
            (InspectionStationItem, design.inspection_stations),
            (ScrapBinItem, design.scrap_bins),
            (ChargerItem, design.chargers),
        ]
        for item_type, resources in collections:
            for resource in resources:
                self._add_entity(item_type(resource))
        for port in design.ports:
            self._add_entity(PortItem(port))
        for agv in design.agvs:
            self._add_entity(AGVItem(agv))
        self.buffer_boundaries = BufferBoundaryItem(
            [
                item
                for item in self.entity_items.values()
                if isinstance(item, BufferItem)
            ]
        )
        self.addItem(self.buffer_boundaries)
        self._hover_groups = self._build_hover_groups()
        for port in design.ports:
            for binding in port.bindings:
                target = binding.target
                cell = target_cell(design, target)
                if cell is None:
                    continue
                owner_id = target_owner_id(target)
                if target not in self._binding_highlights:
                    if target.kind in ("buffer_slot", "inspection_slot"):
                        area = QRectF(
                            cell.x * CELL_SIZE, cell.y * CELL_SIZE, CELL_SIZE, CELL_SIZE
                        )
                    else:
                        owner = self.entity_items[owner_id]
                        area = owner.mapRectToScene(owner.shape().boundingRect())
                    highlight = self.addRect(
                        area.adjusted(1, 1, -1, -1),
                        QPen(Qt.PenStyle.NoPen),
                        BINDING_FILL,
                    )
                    highlight.setZValue(2)
                    highlight.hide()
                    highlight.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
                    self._binding_highlights[target] = highlight
                self.binding_items.append(
                    (port.port_id, owner_id, self._binding_highlights[target])
                )
        self.slot_highlight = self.addRect(
            QRectF(), QPen(Qt.PenStyle.NoPen), BINDING_FILL
        )
        self.slot_highlight.setZValue(6)
        self.slot_highlight.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self.slot_highlight.hide()
        self.selectionChanged.connect(self._selection_changed)

    def _add_entity(self, item):
        item.names_visible = self.names_visible
        self.addItem(item)
        self.entity_items[item.entity_id] = item

    def _build_hover_groups(self):
        owner_machine = {
            machine.machine_id: machine.machine_id for machine in self.design.machines
        }
        for buffer in self.design.buffers:
            if (
                buffer.role in ("machine_pre", "machine_post")
                and buffer.machine_id in owner_machine
            ):
                owner_machine[buffer.buffer_id] = buffer.machine_id
        groups = {
            machine.machine_id: {machine.machine_id} for machine in self.design.machines
        }
        for identifier, machine_id in owner_machine.items():
            groups[machine_id].add(identifier)
        port_machines = {}
        for port in self.design.ports:
            machines = {
                owner_machine[target_owner_id(binding.target)]
                for binding in port.bindings
                if target_owner_id(binding.target) in owner_machine
            }
            port_machines[port.port_id] = machines
            for machine_id in machines:
                groups[machine_id].add(port.port_id)
        result = {
            identifier: frozenset(groups[mid])
            for identifier, mid in owner_machine.items()
        }
        for port_id, machines in port_machines.items():
            result[port_id] = frozenset().union(*(groups[mid] for mid in machines))
        return result

    def hover_entity(self, identifier=None):
        item = self.entity_items.get(identifier)
        if item is None or not item.isVisible():
            identifier = None
        highlighted = self._hover_groups.get(identifier, frozenset())
        changed = self.highlighted_ids ^ highlighted
        self.hovered_entity_id = identifier
        self.highlighted_ids = highlighted
        for entity_id in changed:
            self.entity_items[entity_id].set_group_highlighted(entity_id in highlighted)
        if changed:
            self.buffer_boundaries.update()

    def _selection_changed(self):
        items = [item for item in self.selectedItems() if isinstance(item, EntityItem)]
        selected = items[0].entity_id if items else ""
        self.slot_highlight.hide()
        self.update_bindings(selected)
        self.buffer_boundaries.update()
        self.entity_selected.emit(selected)

    def select_entity(self, selected_id):
        self.select_entities((selected_id,) if selected_id else ())

    def select_entities(self, selected_ids):
        self.blockSignals(True)
        try:
            self.clearSelection()
            for selected_id in selected_ids:
                if selected_id in self.entity_items:
                    self.entity_items[selected_id].setSelected(True)
        finally:
            self.blockSignals(False)
        self.slot_highlight.hide()
        self.update_bindings(selected_ids[0] if selected_ids else "")
        self.buffer_boundaries.update()

    def update_bindings(self, selected_id=None):
        if selected_id is None:
            selected = self.selectedItems()
            selected_id = selected[0].data(0) if selected else None
        visible_targets = set()
        related_ports = set()
        for port_id, owner_id, item in self.binding_items:
            if self.binding_mode == "all" or (
                self.binding_mode == "selected" and selected_id in (port_id, owner_id)
            ):
                visible_targets.add(item)
                related_ports.add(port_id)
        # Shared targets are drawn once; an unrelated binding cannot hide them.
        for item in self._binding_highlights.values():
            item.setVisible(item in visible_targets)
        for port in self.design.ports:
            self.entity_items[port.port_id].set_binding_highlighted(
                port.port_id in related_ports
            )

    def set_layer(self, layer, visible):
        if layer == "grid":
            self.grid_visible = visible
        elif layer == "names":
            self.names_visible = visible
            for item in self.entity_items.values():
                item.names_visible = visible
                item.update()
        elif layer == "ports":
            self.ports_visible = visible
            for item in self.entity_items.values():
                if item.kind == "port":
                    item.setVisible(visible)
            if not visible and self.hovered_entity_id is not None:
                if self.entity_items[self.hovered_entity_id].kind == "port":
                    self.hover_entity()
        self.update()

    def highlight_cell(self, cell):
        self.slot_highlight.setRect(
            QRectF(
                cell.x * CELL_SIZE + 1,
                cell.y * CELL_SIZE + 1,
                CELL_SIZE - 2,
                CELL_SIZE - 2,
            )
        )
        self.slot_highlight.show()

    def drawBackground(self, painter, rect):
        painter.fillRect(rect, QColor("#e9eef2"))
        painter.fillRect(self.map_rect, QColor("#fafcfd"))
        if self.grid_visible:
            visible = rect.intersected(self.map_rect)
            lod = abs(painter.worldTransform().m11())
            step = 5 if lod < 0.18 else 1
            pen = QPen(QColor("#dce4e9"), 0)
            painter.setPen(pen)
            left = max(0, math.floor(visible.left() / CELL_SIZE / step) * step)
            top = max(0, math.floor(visible.top() / CELL_SIZE / step) * step)
            right = min(self.design.grid.width, math.ceil(visible.right() / CELL_SIZE))
            bottom = min(
                self.design.grid.height, math.ceil(visible.bottom() / CELL_SIZE)
            )
            for x in range(left, right + 1, step):
                painter.drawLine(
                    QPointF(x * CELL_SIZE, visible.top()),
                    QPointF(x * CELL_SIZE, visible.bottom()),
                )
            for y in range(top, bottom + 1, step):
                painter.drawLine(
                    QPointF(visible.left(), y * CELL_SIZE),
                    QPointF(visible.right(), y * CELL_SIZE),
                )
        for cell in self.design.grid.blocked_cells:
            blocked = QRectF(
                cell.x * CELL_SIZE, cell.y * CELL_SIZE, CELL_SIZE, CELL_SIZE
            )
            if blocked.intersects(rect):
                painter.fillRect(blocked, QColor("#87959f"))
        painter.setPen(QPen(QColor("#a3b3be"), 0))
        painter.drawRect(self.map_rect)


class FactoryView(QGraphicsView):
    zoom_changed = Signal(float)
    cell_hovered = Signal(int, int)

    def __init__(self, scene, parent=None):
        super().__init__(scene, parent)
        self.setObjectName("factoryView")
        self.setRenderHints(
            QPainter.RenderHint.Antialiasing | QPainter.RenderHint.TextAntialiasing
        )
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setMouseTracking(True)
        self.setFrameShape(QGraphicsView.Shape.NoFrame)
        self._space_pressed = False
        self._pan_position = None
        self._first_show = True
        self._fit_to_map = True
        self._fit_timer = QTimer(self)
        self._fit_timer.setSingleShot(True)
        self._fit_timer.timeout.connect(self._refit_after_resize)
        for scrollbar in (self.horizontalScrollBar(), self.verticalScrollBar()):
            scrollbar.sliderPressed.connect(self._stop_fitting)

    def showEvent(self, event):
        super().showEvent(event)
        if self._first_show:
            self.fit_map()
            self._first_show = False

    def fit_map(self):
        self.clear_hover()
        self._fit_to_map = True
        # Fit mode has no scrollbars; changing their visibility mid-fit otherwise
        # queues a second resize that can clear a freshly established hover.
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.fitInView(
            self.scene().map_rect.adjusted(-20, -20, 20, 20),
            Qt.AspectRatioMode.KeepAspectRatio,
        )
        self.zoom_changed.emit(self.transform().m11())

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.clear_hover()
        if self._fit_to_map:
            self._fit_timer.start(0)

    def _refit_after_resize(self):
        if self._fit_to_map and self.isVisible():
            self.fit_map()

    def _stop_fitting(self):
        self._fit_to_map = False
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

    def zoom_by(self, factor):
        self.clear_hover()
        self._stop_fitting()
        scale = self.transform().m11()
        target = min(8.0, max(0.04, scale * factor))
        self.scale(target / scale, target / scale)
        self.zoom_changed.emit(target)

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        if delta:
            self.zoom_by(1.15 ** (delta / 120))
        event.accept()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self._space_pressed = True
            self.setCursor(Qt.CursorShape.OpenHandCursor)
            event.accept()
        else:
            super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self._space_pressed = False
            if self._pan_position is None:
                self.unsetCursor()
            event.accept()
        else:
            super().keyReleaseEvent(event)

    def focusOutEvent(self, event):
        self.clear_hover()
        self._space_pressed = False
        self._pan_position = None
        self.unsetCursor()
        super().focusOutEvent(event)

    def clear_hover(self):
        scene = self.scene()
        if scene is not None:
            scene.hover_entity()

    def viewportEvent(self, event):
        if event.type() == QEvent.Type.Leave:
            self.clear_hover()
        return super().viewportEvent(event)

    def hideEvent(self, event):
        self.clear_hover()
        super().hideEvent(event)

    def changeEvent(self, event):
        if event.type() == QEvent.Type.ActivationChange and not self.isActiveWindow():
            self.clear_hover()
        super().changeEvent(event)

    def event(self, event):
        if event.type() == QEvent.Type.WindowDeactivate:
            self.clear_hover()
        return super().event(event)

    def scrollContentsBy(self, dx, dy):
        self.clear_hover()
        super().scrollContentsBy(dx, dy)

    def _hover_at(self, position):
        if getattr(self.scene(), "edit_gesture", False):
            return
        item = next(
            (item for item in self.items(position) if isinstance(item, EntityItem)),
            None,
        )
        self.scene().hover_entity(item.entity_id if item is not None else None)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.MiddleButton or (
            self._space_pressed and event.button() == Qt.MouseButton.LeftButton
        ):
            self.clear_hover()
            self._stop_fitting()
            self._pan_position = event.position().toPoint()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._pan_position is not None:
            current = event.position().toPoint()
            delta = current - self._pan_position
            self._pan_position = current
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - delta.x()
            )
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - delta.y()
            )
            event.accept()
        else:
            point = self.mapToScene(event.position().toPoint())
            self.cell_hovered.emit(
                math.floor(point.x() / CELL_SIZE), math.floor(point.y() / CELL_SIZE)
            )
            super().mouseMoveEvent(event)
            self._hover_at(event.position().toPoint())

    def mouseReleaseEvent(self, event):
        if self._pan_position is not None:
            self._pan_position = None
            self.setCursor(
                Qt.CursorShape.OpenHandCursor
                if self._space_pressed
                else Qt.CursorShape.ArrowCursor
            )
            event.accept()
        else:
            super().mouseReleaseEvent(event)
