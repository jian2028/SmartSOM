"""Vector resource symbols with selection and orientation independent of geometry."""

import re

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QFontMetricsF, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QGraphicsItem, QGraphicsObject

from smartsom.domain.factory_design import entity_id
from smartsom.studio.symbols import (
    CELL_SIZE,
    JobVisual,
    draw_job,
    gear_path,
    job_rect,
    magnifier_path,
    validate_job,
)

BINDING_FILL = QColor("#64ffe27b")

COLORS = {
    "machine": ("#eaf0ff", "#526bac"),
    "buffer": ("#e3f3ee", "#267b68"),
    "inspection": ("#efeaf8", "#7b61a7"),
    "scrap": ("#f8ebe6", "#a66950"),
    "charger": ("#fff1d7", "#a8761a"),
    "port": ("#e5f1fc", "#4683b7"),
    "agv": ("#164b5b", "#164b5b"),
}


def text_value(value):
    return str(getattr(value, "value", value))


def _number(identifier):
    suffix = re.search(r"(\d+)$", identifier)
    return str(int(suffix.group(1))) if suffix else ""


def _symbol_pen(color, opacity=0.55):
    tint = QColor(color)
    tint.setAlphaF(opacity)
    pen = QPen(tint, 1.4)
    pen.setCosmetic(True)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    return pen


def _edge_path(edges):
    path = QPainterPath()
    for x1, y1, x2, y2 in sorted(edges):
        path.moveTo(x1 * CELL_SIZE, y1 * CELL_SIZE)
        path.lineTo(x2 * CELL_SIZE, y2 * CELL_SIZE)
    return path


def _rectangle_edges(x, y, width, height):
    edges = set()
    for col in range(x, x + width):
        edges.update(((col, y, col + 1, y), (col, y + height, col + 1, y + height)))
    for row in range(y, y + height):
        edges.update(((x, row, x, row + 1), (x + width, row, x + width, row + 1)))
    return edges


class EntityItem(QGraphicsObject):
    """Common selection, body and labeling for immovable semantic entities."""

    draw_name = True

    def __init__(self, resource, kind):
        super().__init__()
        self.entity_id = entity_id(resource)
        self.name = resource.name or self.entity_id
        self.kind = kind
        self.resource = resource
        self.names_visible = False
        self.state_layer_active = False
        self.group_highlighted = False
        footprint = getattr(resource, "footprint", None)
        if footprint is not None:
            x, y = footprint.x, footprint.y
            width, height = footprint.width, footprint.height
        else:
            cell = resource.cell if kind == "port" else resource.initial_cell
            x, y, width, height = cell.x, cell.y, 1, 1
        self._rect = QRectF(0, 0, width * CELL_SIZE, height * CELL_SIZE)
        self.setPos(x * CELL_SIZE, y * CELL_SIZE)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
        self.setAcceptHoverEvents(True)
        self.setData(0, self.entity_id)
        self.setToolTip(
            f"{self.name}\n{kind.title()} · {self.entity_id}\nCell ({x}, {y})"
        )
        self.setZValue({"port": 3, "agv": 4}.get(kind, 1))

    def boundingRect(self):
        return self._rect.adjusted(-3, -3, 3, 3)

    def shape(self):
        path = QPainterPath()
        path.addRect(self._rect)
        return path

    def paint(self, painter, option, widget=None):
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        fill, stroke = COLORS[self.kind]
        painter.setPen(self.outline_pen())
        painter.setBrush(QColor(fill))
        rect = self._rect.adjusted(3, 3, -3, -3)
        self._draw_geometry(painter, rect, stroke)
        if self.names_visible and self.draw_name:
            self._draw_label(painter, option, rect)

    def outline_pen(self):
        pen = QPen(QColor("#146b91" if self.isSelected() else COLORS[self.kind][1]))
        pen.setCosmetic(True)
        pen.setWidthF(
            2.4 if self.isSelected() else 2.0 if self.group_highlighted else 1.2
        )
        return pen

    def set_group_highlighted(self, highlighted):
        if self.group_highlighted != highlighted:
            self.group_highlighted = highlighted
            self.update()

    def _draw_geometry(self, painter, rect, color):
        painter.drawRoundedRect(rect, 4, 4)
        pen = QPen(QColor(color), 1.4)
        pen.setCosmetic(True)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        self._draw_symbol(painter, self._symbol_area(rect), color)

    def _draw_symbol(self, painter, rect, color):
        raise NotImplementedError("Resource items define their own vector symbol")

    def _label_area(self, rect):
        height = min(26, rect.height() * 0.34)
        return QRectF(rect.left() + 3, rect.bottom() - height, rect.width() - 6, height)

    def _symbol_area(self, rect):
        if not self.draw_name or not self.names_visible:
            return rect
        area = QRectF(rect)
        area.setBottom(self._label_area(rect).top() - 4)
        return area

    def _draw_label(self, painter, option, rect):
        label = _number(self.entity_id)
        if not label:
            return
        area = self._label_area(rect)
        font = QFont()
        font.setPixelSize(max(1, round(min(20, area.height() * 0.8))))
        font.setWeight(QFont.Weight.DemiBold)
        metrics = QFontMetricsF(font)
        if metrics.horizontalAdvance(label) > area.width():
            font.setPixelSize(
                max(
                    1,
                    int(
                        font.pixelSize()
                        * area.width()
                        / metrics.horizontalAdvance(label)
                    ),
                )
            )
        lod = option.levelOfDetailFromTransform(painter.worldTransform())
        if font.pixelSize() * lod < 5:
            return
        painter.setFont(font)
        painter.setPen(QColor("#223c4b"))
        painter.drawText(area, Qt.AlignmentFlag.AlignCenter, label)

    def _slot_label_area(self, slots):
        if slots:
            cell = max((slot.local_cell for slot in slots), key=lambda c: (c.y, c.x))
            return QRectF(cell.x * CELL_SIZE + 5, cell.y * CELL_SIZE + 5, 30, 30)
        return QRectF(self._rect.right() - 35, self._rect.bottom() - 35, 30, 30)

    def _draw_slot_grid(self, painter, slots, *, subtle_grid=False, outline=True):
        """Draw actual slots on cell edges, with each shared edge painted once."""
        painter.fillRect(self._rect, QColor("#f2f5f7"))
        width = int(self._rect.width() / CELL_SIZE)
        height = int(self._rect.height() / CELL_SIZE)
        border = _rectangle_edges(0, 0, width, height)
        edges = set(border)

        def add_cell(x, y):
            edges.update(
                (
                    (x, y, x + 1, y),
                    (x, y + 1, x + 1, y + 1),
                    (x, y, x, y + 1),
                    (x + 1, y, x + 1, y + 1),
                )
            )

        for slot in slots:
            x, y = slot.local_cell.x, slot.local_cell.y
            painter.fillRect(
                QRectF(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE),
                QColor(COLORS[self.kind][0]),
            )
            add_cell(x, y)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        if subtle_grid:
            painter.save()
            pen = _symbol_pen(COLORS[self.kind][1], opacity=0.32)
            pen.setWidthF(0.7)
            painter.setPen(pen)
            painter.drawPath(_edge_path(edges - border))
            painter.restore()
            if outline:
                painter.drawPath(_edge_path(border))
        else:
            painter.drawPath(_edge_path(edges))


class MachineItem(EntityItem):
    def __init__(self, resource):
        super().__init__(resource, "machine")
        self.job = None

    def set_job(self, job: JobVisual | None):
        validate_job(job)
        self.job = job
        self.update()

    def _draw_geometry(self, painter, rect, color):
        if self.state_layer_active:
            painter.drawRect(self._rect)
        else:
            super()._draw_geometry(painter, rect, color)

    def _draw_symbol(self, painter, rect, color):
        if self.state_layer_active:
            return
        if self.job is not None:
            draw_job(painter, rect.center(), self.job, kind="machine")
            return
        painter.save()
        painter.setPen(_symbol_pen(color, opacity=0.65))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        # Keep the idle glyph airy, centered and unfilled at every footprint size.
        inset = min(rect.width(), rect.height()) * 0.19
        painter.drawPath(gear_path(rect.adjusted(inset, inset, -inset, -inset)))
        painter.restore()


class BufferItem(EntityItem):
    def __init__(self, resource):
        super().__init__(resource, "buffer")
        self.outline_in_scene = False
        self.job = None
        self.slot_jobs = {}

    def set_job(self, job: JobVisual | None, *, slot_id=None):
        validate_job(job)
        if self.resource.storage.mode == "slots":
            if slot_id not in {slot.slot_id for slot in self.resource.storage.slots}:
                raise ValueError("job requires an existing buffer slot")
            if job is None:
                self.slot_jobs.pop(slot_id, None)
            else:
                self.slot_jobs[slot_id] = job
        else:
            if slot_id is not None:
                raise ValueError("pool buffers have no slots")
            self.job = job
        self.update()

    def outline_pen(self):
        pen = super().outline_pen()
        if not self.isSelected() and not self.group_highlighted:
            pen.setWidthF(1.4)
        return pen

    def _draw_geometry(self, painter, rect, color):
        if self.resource.storage.mode == "slots":
            self._draw_slot_grid(
                painter,
                self.resource.storage.slots,
                subtle_grid=True,
                outline=not self.outline_in_scene,
            )
            for slot in self.resource.storage.slots:
                if slot.slot_id in self.slot_jobs and not self.state_layer_active:
                    center = (
                        QPointF(slot.local_cell.x + 0.5, slot.local_cell.y + 0.5)
                        * CELL_SIZE
                    )
                    draw_job(painter, center, self.slot_jobs[slot.slot_id])
        else:
            if self.outline_in_scene:
                painter.fillRect(self._rect, QColor(COLORS[self.kind][0]))
            else:
                painter.drawRect(self._rect)
            if self.state_layer_active:
                return
            if self.job is not None:
                draw_job(painter, self._symbol_area(rect).center(), self.job)
                return
            if self.resource.role in ("system_input", "system_output"):
                self._draw_transfer_tray(painter, rect, color)
                return
            area = self._symbol_area(rect)
            center = area.center()
            half_width = min(12, area.width() * 0.35)
            spacing = min(7, area.height() * 0.22)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            for offset in (-spacing, 0, spacing):
                painter.drawLine(
                    center + QPointF(-half_width, offset),
                    center + QPointF(half_width, offset),
                )

    def _draw_transfer_tray(self, painter, rect, color):
        area = self._symbol_area(rect) if self.names_visible else rect
        center = area.center()
        size = min(24, area.width() * 0.72, area.height() * 0.8)

        def point(x, y):
            return center + QPointF(x * size, y * size)

        # Arrows mean receive/dispatch, independent of footprint orientation.
        tray = QPainterPath(point(-0.5, 0.15))
        tray.lineTo(point(-0.5, 0.48))
        tray.lineTo(point(0.5, 0.48))
        tray.lineTo(point(0.5, 0.15))
        receiving = self.resource.role == "system_input"
        tail_y, tip_y = (-0.48, 0.12) if receiving else (0.12, -0.48)
        wing_y = tip_y - 0.2 if receiving else tip_y + 0.2
        tray.moveTo(point(0, tail_y))
        tray.lineTo(point(0, tip_y))
        tray.moveTo(point(-0.18, wing_y))
        tray.lineTo(point(0, tip_y))
        tray.lineTo(point(0.18, wing_y))
        painter.save()
        painter.setPen(_symbol_pen(color, opacity=1))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(tray)
        painter.restore()

    def _label_area(self, rect):
        if self.resource.storage.mode == "pool":
            return super()._label_area(rect)
        return self._slot_label_area(getattr(self.resource.storage, "slots", ()))


class BufferBoundaryItem(QGraphicsItem):
    """One boundary layer above buffer fills, including shared cell edges."""

    def __init__(self, buffers):
        super().__init__()
        self.edges = {}
        self._rect = QRectF()
        for item in buffers:
            item.outline_in_scene = True
            self._rect = self._rect.united(item.sceneBoundingRect())
            box = item.resource.footprint
            for edge in _rectangle_edges(box.x, box.y, box.width, box.height):
                self.edges.setdefault(edge, []).append(item)
        self.setZValue(1.5)
        self.setAcceptedMouseButtons(Qt.MouseButton.NoButton)

    def boundingRect(self):
        return self._rect

    def paint(self, painter, option, widget=None):
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        groups = {}
        for edge, owners in self.edges.items():
            visible = [item for item in owners if item.isVisible()]
            if not visible:
                continue
            owner = max(
                visible, key=lambda item: (item.isSelected(), item.group_highlighted)
            )
            state = owner.isSelected(), owner.group_highlighted
            edges, pen = groups.setdefault(state, (set(), owner.outline_pen()))
            edges.add(edge)
        for edges, pen in groups.values():
            painter.setPen(pen)
            painter.drawPath(_edge_path(edges))


class InspectionStationItem(EntityItem):
    def __init__(self, resource):
        super().__init__(resource, "inspection")
        self.slot_jobs = {}

    def set_job(self, job: JobVisual | None, *, slot_id):
        validate_job(job)
        if slot_id not in {slot.slot_id for slot in self.resource.slots}:
            raise ValueError("job requires an existing inspection slot")
        if job is None:
            self.slot_jobs.pop(slot_id, None)
        else:
            self.slot_jobs[slot_id] = job
        self.update()

    def _draw_geometry(self, painter, rect, color):
        self._draw_slot_grid(painter, self.resource.slots)
        if self.state_layer_active:
            return
        painter.save()
        painter.setPen(_symbol_pen(color, opacity=0.45))
        for slot in self.resource.slots:
            cell = slot.local_cell
            area = QRectF(cell.x * CELL_SIZE + 5, cell.y * CELL_SIZE + 5, 30, 30)
            if self.names_visible and area.intersects(self._label_area(rect)):
                area.setBottom(self._label_area(rect).top() - 2)
            job = self.slot_jobs.get(slot.slot_id)
            if job is None:
                self._draw_symbol(painter, area, color)
            else:
                center = QPointF(cell.x + 0.5, cell.y + 0.5) * CELL_SIZE
                draw_job(painter, center, job, kind="inspection")
        painter.restore()

    def _draw_symbol(self, painter, rect, color):
        painter.drawPath(magnifier_path(rect))

    def _label_area(self, rect):
        area = self._slot_label_area(self.resource.slots)
        area.setTop(area.bottom() - 12)
        return area


class ScrapBinItem(EntityItem):
    def _draw_geometry(self, painter, rect, color):
        if not self.state_layer_active:
            return super()._draw_geometry(painter, rect, color)
        painter.drawRect(self._rect)
        self._draw_symbol(painter, self._rect, color)

    def __init__(self, resource):
        super().__init__(resource, "scrap")

    def _draw_symbol(self, painter, rect, color):
        painter.save()
        painter.setPen(_symbol_pen(color))
        painter.translate(rect.center())
        scale = min(rect.width() / 28, rect.height() / 34, 1)
        painter.scale(scale, scale)
        painter.drawRect(QRectF(-9, -8, 18, 20))
        painter.drawLine(QPointF(-12, -11), QPointF(12, -11))
        painter.drawLine(QPointF(-4, -15), QPointF(4, -15))
        painter.restore()


class ChargerItem(EntityItem):
    draw_name = False

    def __init__(self, resource):
        super().__init__(resource, "charger")

    def _draw_geometry(self, painter, rect, color):
        if not self.state_layer_active:
            return super()._draw_geometry(painter, rect, color)
        painter.drawRect(self._rect)
        self._draw_symbol(painter, self._rect, color)

    def _draw_symbol(self, painter, rect, color):
        # Access ports indicate charging locations; the station needs no arrow.
        painter.save()
        painter.translate(rect.center())
        scale = min(rect.width() / 24, rect.height() / 28, 1)
        if self.state_layer_active:
            scale *= 0.9
        painter.scale(scale, scale)
        painter.setPen(_symbol_pen(color, opacity=0.65))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        bolt = QPainterPath(QPointF(4, -12))
        for point in [(-9, 2), (-1, 2), (-4, 12), (9, -3), (1, -3)]:
            bolt.lineTo(QPointF(*point))
        bolt.closeSubpath()
        painter.drawPath(bolt)
        painter.restore()


class PortItem(EntityItem):
    draw_name = False

    def __init__(self, resource):
        super().__init__(resource, "port")
        self.binding_highlighted = False

    def set_binding_highlighted(self, highlighted):
        if self.binding_highlighted != highlighted:
            self.binding_highlighted = highlighted
            self.update()

    def _draw_geometry(self, painter, rect, color):
        pen = painter.pen()
        pen.setStyle(Qt.PenStyle.DashLine)
        pen.setColor(QColor(color))
        painter.setPen(pen)
        painter.setBrush(
            BINDING_FILL if self.binding_highlighted else Qt.BrushStyle.NoBrush
        )
        painter.drawRect(self._rect.adjusted(1.5, 1.5, -1.5, -1.5))


class AGVItem(EntityItem):
    draw_name = False

    def __init__(self, resource):
        super().__init__(resource, "agv")
        self.loaded = False

    def set_loaded(self, loaded: bool):
        """Set transient cargo visibility without changing the factory design."""
        if self.loaded != loaded:
            self.loaded = loaded
            self.update()

    def shape(self):
        # Only the circular vehicle catches clicks; the exposed port stays reachable.
        path = QPainterPath()
        path.addEllipse(self._rect.center(), 13, 13)
        return path

    def _draw_geometry(self, painter, rect, color):
        painter.save()
        painter.translate(rect.center())
        if self.isSelected():
            # A separate halo remains visible against the dark vehicle body.
            # Keep it inside the cell and leave the circular hit target unchanged.
            painter.save()
            halo = QPen(QColor("#146b91"), 2.4)
            halo.setCosmetic(True)
            painter.setPen(halo)
            painter.setBrush(QColor("#bdefff"))
            painter.drawEllipse(QRectF(-17, -17, 34, 34))
            painter.restore()
        painter.drawEllipse(QRectF(-13, -13, 26, 26))
        if (
            self.state_layer_active
            and self.loaded
            and getattr(self, "state_layer_job", False)
        ):
            painter.restore()
            return
        if self.loaded:
            draw_job(painter, QPointF(0, 0), JobVisual())
        else:
            pen = QPen(QColor("white"), 1.0)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            # Inset the stroke so its outer edge matches the loaded square.
            painter.drawRoundedRect(
                job_rect(QPointF(0, 0)).adjusted(0.5, 0.5, -0.5, -0.5), 0.8, 0.8
            )
        painter.restore()
