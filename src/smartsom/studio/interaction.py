"""Grid gestures preview candidates and submit them through the document editor."""

import math
from dataclasses import replace

from PySide6.QtCore import QEvent, QObject, QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QPen

from smartsom.domain.factory_design import Cell, entity_id, iter_resources
from smartsom.studio import editing
from smartsom.studio.items import CELL_SIZE, EntityItem


class MapInteraction(QObject):
    def __init__(self, editor, document):
        super().__init__(document.view)
        self.editor, self.document = editor, document
        self.tool = "select"
        self.gesture = None
        self.overlays = []
        self.copied = ()
        self.candidate = None
        self.new_selection = None
        self.failure = ""
        document.view.viewport().installEventFilter(self)
        document.view.installEventFilter(self)
        document.view.zoom_changed.connect(lambda _: self.handles())

    @property
    def scene(self):
        return self.document.scene

    @property
    def view(self):
        return self.document.view

    def clear_overlays(self):
        for item in self.overlays:
            try:
                if item.scene() is not None:
                    item.scene().removeItem(item)
            except RuntimeError:
                pass
        self.overlays = []

    def rectangle(self, rect, *, color="#408ab6", fill=None, dashed=True):
        pen = QPen(QColor(color), 1)
        pen.setCosmetic(True)
        if dashed:
            pen.setStyle(Qt.PenStyle.DashLine)
        item = self.scene.addRect(rect, pen, QColor(fill or "#22408ab6"))
        item.setZValue(20)
        item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self.overlays.append(item)
        return item

    def cancel(self):
        self.gesture = self.candidate = self.new_selection = None
        self.failure = ""
        self.tool = "select"
        self.scene.edit_gesture = False
        self.clear_overlays()
        self.view.unsetCursor()
        self.handles()

    def handles(self):
        if self.gesture or self.tool != "select":
            return
        self.clear_overlays()
        if (
            not self.document.edit_mode
            or len(self.document.selected_ids) != 1
            or self.editor.draft_dialog is not None
        ):
            return
        try:
            value = editing.resource(
                self.document.design, self.document.selected_ids[0]
            )
        except StopIteration:
            return
        if not hasattr(value, "footprint"):
            return
        x, y, w, h = editing.rect_of(value)
        size = 6 / max(0.04, self.view.transform().m11())
        for px, py in ((x + w, y + h / 2), (x + w / 2, y + h), (x + w, y + h)):
            self.rectangle(
                QRectF(
                    px * CELL_SIZE - size / 2, py * CELL_SIZE - size / 2, size, size
                ),
                fill="#ffffff",
                dashed=False,
            )

    def handle_at(self, position):
        if len(self.document.selected_ids) != 1:
            return None
        value = editing.resource(self.document.design, self.document.selected_ids[0])
        if not hasattr(value, "footprint"):
            return None
        x, y, w, h = editing.rect_of(value)
        for name, px, py in (
            ("corner", x + w, y + h),
            ("right", x + w, y + h / 2),
            ("bottom", x + w / 2, y + h),
        ):
            point = self.view.mapFromScene(QPointF(px * CELL_SIZE, py * CELL_SIZE))
            if (point - position).manhattanLength() <= 9:
                return name
        return None

    def cell_at(self, position):
        point = self.view.mapToScene(position)
        return Cell(
            math.floor(point.x() / CELL_SIZE), math.floor(point.y() / CELL_SIZE)
        )

    def hit(self, position):
        return next(
            (i for i in self.view.items(position) if isinstance(i, EntityItem)), None
        )

    def eventFilter(self, watched, event):
        if event.type() == QEvent.Type.KeyPress and event.key() == Qt.Key.Key_Escape:
            self.editor.cancel_tool()
            return True
        if (
            not self.document.edit_mode
            or self.document is not self.editor.window.current_document
        ):
            return False
        kind = event.type()
        if kind not in (
            QEvent.Type.MouseButtonPress,
            QEvent.Type.MouseMove,
            QEvent.Type.MouseButtonRelease,
        ):
            return False
        if watched is not self.view.viewport():
            return False
        if self.view._space_pressed or self.view._pan_position is not None:
            return False
        if (
            kind != QEvent.Type.MouseMove
            and event.button() != Qt.MouseButton.LeftButton
        ):
            return False
        position = event.position().toPoint()
        cell = self.cell_at(position)
        if kind == QEvent.Type.MouseButtonPress:
            if self.editor.draft_dialog is not None:
                self.editor.draft_click(self.hit(position), cell)
                return True
            self.view.setFocus()
            if self.tool != "select":
                self.gesture = {
                    "kind": self.tool,
                    "start": cell,
                    "cells": {cell},
                    "last": cell,
                }
            else:
                handle = self.handle_at(position)
                hit = self.hit(position)
                if handle:
                    if not self.editor.resolve_pending():
                        return True
                    self.gesture = {"kind": "resize", "handle": handle, "start": cell}
                elif hit:
                    identifiers = tuple(self.document.selected_ids)
                    if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                        identifiers = (
                            tuple(i for i in identifiers if i != hit.entity_id)
                            if hit.entity_id in identifiers
                            else (*identifiers, hit.entity_id)
                        )
                        self.editor.select_many(identifiers)
                        return True
                    if hit.entity_id not in identifiers:
                        if not self.editor.select_many((hit.entity_id,)):
                            return True
                    self.gesture = {
                        "kind": "move",
                        "start": cell,
                        "press": position,
                        "moved": False,
                    }
                else:
                    if not self.editor.resolve_pending():
                        return True
                    self.gesture = {
                        "kind": "box",
                        "start": cell,
                        "previous": self.document.selected_ids
                        if event.modifiers() & Qt.KeyboardModifier.ShiftModifier
                        else (),
                    }
            self.scene.hover_entity()
            self.scene.edit_gesture = True
            self.preview(cell)
            return True
        if kind == QEvent.Type.MouseMove:
            if self.editor.draft_dialog is not None:
                return True
            if self.gesture:
                if self.gesture["kind"] == "move" and not self.gesture["moved"]:
                    if (position - self.gesture["press"]).manhattanLength() < 5:
                        return True
                    if not self.editor.resolve_pending():
                        self.cancel()
                        return True
                    self.gesture["moved"] = True
                self.preview(cell)
                return True
            if self.tool != "select":
                self.gesture = {
                    "kind": self.tool,
                    "start": cell,
                    "cells": {cell},
                    "last": cell,
                }
                self.preview(cell)
                self.gesture = None
                return True
            return False
        if kind == QEvent.Type.MouseButtonRelease and self.gesture:
            gesture = self.gesture
            self.preview(cell)
            kind = gesture["kind"]
            applied = False
            if kind == "box":
                x, y, w, h = self.bounds(gesture["start"], cell)
                rect = QRectF(
                    x * CELL_SIZE, y * CELL_SIZE, w * CELL_SIZE, h * CELL_SIZE
                )
                identifiers = tuple(
                    entity_id(r)
                    for r in iter_resources(self.document.design)
                    if rect.contains(
                        QRectF(*(n * CELL_SIZE for n in editing.rect_of(r)))
                    )
                )
                self.editor.select_many(
                    tuple(dict.fromkeys((*gesture["previous"], *identifiers)))
                )
            elif self.candidate is not None and (kind != "move" or gesture["moved"]):
                label = (
                    "Move selection"
                    if kind == "move"
                    else "Resize resource"
                    if kind == "resize"
                    else "Paste selection"
                    if kind == "paste"
                    else "Edit obstacles"
                    if kind.startswith("obstacle")
                    else "Add " + kind.replace("_", " ")
                )
                applied = self.editor.commit(
                    self.candidate,
                    label,
                    selection=self.new_selection,
                    confirm=kind == "resize",
                )
            elif self.failure:
                self.editor.status(self.failure)
            self.gesture = None
            self.scene.edit_gesture = False
            self.clear_overlays()
            self.candidate = None
            if (
                applied
                and kind not in ("move", "resize", "box")
                and not self.editor.continuous_action.isChecked()
            ):
                self.tool = "select"
            self.handles()
            self.editor.update_actions()
            return True
        return False

    @staticmethod
    def bounds(start, end):
        return (
            min(start.x, end.x),
            min(start.y, end.y),
            abs(start.x - end.x) + 1,
            abs(start.y - end.y) + 1,
        )

    def preview(self, cell):
        self.clear_overlays()
        self.failure = ""
        self.candidate = None
        self.new_selection = None
        gesture = self.gesture
        kind = gesture["kind"]
        design = self.document.design
        x, y, w, h = self.bounds(gesture["start"], cell)
        boxes = [(x, y, w, h)]
        if kind == "box":
            self.rectangle(
                QRectF(x * CELL_SIZE, y * CELL_SIZE, w * CELL_SIZE, h * CELL_SIZE)
            )
            return
        try:
            if kind == "move":
                dx, dy = cell.x - gesture["start"].x, cell.y - gesture["start"].y
                boxes = [
                    (bx + dx, by + dy, bw, bh)
                    for identifier in self.document.selected_ids
                    for bx, by, bw, bh in [
                        editing.rect_of(editing.resource(design, identifier))
                    ]
                ]
                self.candidate = editing.move(
                    design, self.document.selected_ids, dx, dy
                )
            elif kind == "resize":
                identifier = self.document.selected_ids[0]
                r = editing.resource(design, identifier)
                f = r.footprint
                # A dragged boundary denotes the nearest grid line, not an occupied cell.
                width = (
                    max(1, cell.x - f.x) if gesture["handle"] != "bottom" else f.width
                )
                height = (
                    max(1, cell.y - f.y) if gesture["handle"] != "right" else f.height
                )
                boxes = [(f.x, f.y, width, height)]
                self.candidate = editing.resize(design, identifier, width, height)
            elif kind == "paste":
                left = min(editing.rect_of(r)[0] for r in self.copied)
                top = min(editing.rect_of(r)[1] for r in self.copied)
                boxes = [
                    (bx - left + cell.x, by - top + cell.y, bw, bh)
                    for r in self.copied
                    for bx, by, bw, bh in [editing.rect_of(r)]
                ]
                self.candidate, self.new_selection = editing.paste(
                    design, self.copied, cell.x, cell.y
                )
            elif kind.startswith("obstacle"):
                if kind == "obstacle_rectangle":
                    cells = {
                        Cell(cx, cy) for cy in range(y, y + h) for cx in range(x, x + w)
                    }
                else:
                    last = gesture["last"]
                    steps = max(abs(cell.x - last.x), abs(cell.y - last.y), 1)
                    gesture["cells"].update(
                        Cell(
                            round(last.x + (cell.x - last.x) * t / steps),
                            round(last.y + (cell.y - last.y) * t / steps),
                        )
                        for t in range(steps + 1)
                    )
                    gesture["last"] = cell
                    cells = gesture["cells"]
                boxes = [(c.x, c.y, 1, 1) for c in cells]
                if any(
                    c.x < 0
                    or c.y < 0
                    or c.x >= design.grid.width
                    or c.y >= design.grid.height
                    for c in cells
                ):
                    raise ValueError("Obstacle stroke extends outside the grid")
                blocked = set(design.grid.blocked_cells)
                blocked = (
                    blocked - cells if kind == "obstacle_erase" else blocked | cells
                )
                self.candidate = editing.checked(
                    replace(
                        design,
                        grid=replace(
                            design.grid,
                            blocked_cells=tuple(
                                sorted(blocked, key=lambda c: (c.y, c.x))
                            ),
                        ),
                    )
                )
            else:
                if kind in ("agv", "port"):
                    x, y, w, h = cell.x, cell.y, 1, 1
                    boxes = [(x, y, w, h)]
                self.candidate, identifier = editing.create_resource(
                    design, kind, x, y, w, h
                )
                self.new_selection = (identifier,)
        except (ValueError, TypeError) as exc:
            self.failure = str(exc)
        color = "#b94d3d" if self.failure else "#408ab6"
        fill = "#44d06050" if self.failure else "#30408ab6"
        for bx, by, bw, bh in boxes:
            self.rectangle(
                QRectF(bx * CELL_SIZE, by * CELL_SIZE, bw * CELL_SIZE, bh * CELL_SIZE),
                color=color,
                fill=fill,
            )
        self.editor.status(
            self.failure
            or f"{kind.replace('_', ' ').capitalize()} · {boxes[0][2]} × {boxes[0][3]} cells · release to apply"
        )
