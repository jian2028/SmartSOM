"""Read-only annotations retained from the historical Template 1 viewer.

Uses current scene positions so cargo labels follow interpolated vehicles.
The frozen experiment implementation is neither imported nor changed.
"""

from math import ceil, sqrt

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QGraphicsItem

from smartsom.domain.factory_design import FactoryDesign
from smartsom.studio.symbols import CELL_SIZE, job_rect

QUALITY_COLORS = {
    "UNKNOWN": "#e0ab39",
    "PASS": "#359377",
    "FAIL": "#cf6464",
}


def _short(identifier: str) -> str:
    if "_a" in identifier:
        demand, attempt = identifier.rsplit("_a", 1)
        if attempt.isdigit():
            number = (
                str(int(demand[1:]))
                if demand.startswith("d") and demand[1:].isdigit()
                else demand[-4:]
            )
            return f"{number}:{attempt}"
    suffix = identifier.rsplit("_", 1)[-1]
    return suffix.lstrip("0") or "0"


class RuntimeOverlay(QGraphicsItem):
    """Transient drawing above the unchanged Studio geometry and symbols."""

    def __init__(self, design: FactoryDesign, items):
        super().__init__()
        self.design = design
        self.items = items
        self.frame: dict = {}
        self.setZValue(5)
        self.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self.store_resources = {
            **{item.buffer_id: item for item in design.buffers},
            **{item.inspection_station_id: item for item in design.inspection_stations},
        }

    def boundingRect(self):
        return QRectF(
            0,
            0,
            self.design.grid.width * CELL_SIZE,
            self.design.grid.height * CELL_SIZE,
        )

    def slot_rects(self, owner_id: str, count: int) -> list[QRectF]:
        """Explicit slots retain their design order; pools use a visual inset."""
        resource = self.store_resources[owner_id]
        footprint = resource.footprint
        slots = getattr(resource, "slots", ()) or getattr(
            getattr(resource, "storage", None), "slots", ()
        )
        if slots:
            return [
                QRectF(
                    (footprint.x + slot.local_cell.x) * CELL_SIZE + 8,
                    (footprint.y + slot.local_cell.y) * CELL_SIZE + 8,
                    CELL_SIZE - 16,
                    CELL_SIZE - 16,
                )
                for slot in slots[:count]
            ]
        # Eight logical INPUT candidates can live in a two-cell pool footprint.
        # This is a display arrangement, never an inferred physical slot binding.
        columns = max(1, ceil(sqrt(count * footprint.width / footprint.height)))
        rows = max(1, ceil(count / columns))
        width = footprint.width * CELL_SIZE / columns
        height = footprint.height * CELL_SIZE / rows
        return [
            QRectF(
                footprint.x * CELL_SIZE + (i % columns) * width + 3,
                footprint.y * CELL_SIZE + (i // columns) * height + 3,
                width - 6,
                height - 6,
            )
            for i in range(count)
        ]

    def job_locations(self):
        """One display location per held job, including processing and cargo."""
        locations = {}
        for owner, store in self.frame.get("stores", {}).items():
            if owner not in self.store_resources:
                continue
            slots = store.get("slots", [])
            resource = self.store_resources[owner]
            is_pool = (
                getattr(getattr(resource, "storage", None), "mode", None) == "pool"
            )
            for rect, jid in zip(self.slot_rects(owner, len(slots)), slots):
                if jid is not None:
                    f = resource.footprint
                    center = (
                        QPointF(f.x + f.width / 2, f.y + f.height / 2) * CELL_SIZE
                        if is_pool
                        else rect.center()
                    )
                    locations[jid] = (center, owner, store.get("beacon") == jid)
        for resource in self.design.machines:
            record = self.frame.get("machines", {}).get(resource.machine_id, {})
            if record.get("job_id"):
                f = resource.footprint
                locations[record["job_id"]] = (
                    QPointF(f.x + f.width / 2, f.y + f.height / 2) * CELL_SIZE,
                    resource.machine_id,
                    False,
                )
        for owner, record in self.frame.get("agvs", {}).items():
            if record.get("job_id"):
                locations[record["job_id"]] = (
                    self.items[owner].pos() + QPointF(0.5, 0.5) * CELL_SIZE,
                    owner,
                    False,
                )
        return locations

    def paint(self, painter, option, widget=None):
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        jobs = self.frame.get("jobs", {})
        drawn_pools = set()
        for jid, (center, owner, selected) in self.job_locations().items():
            resource = self.store_resources.get(owner)
            # Explicit slots, machines and AGVs use Studio's own transient job API.
            # A pool uses one Studio stack symbol with an exact count; its full
            # resident job list is shown in the table, not hidden or truncated.
            if (
                resource is not None
                and getattr(getattr(resource, "storage", None), "mode", None) == "pool"
            ):
                if owner in drawn_pools:
                    continue
                drawn_pools.add(owner)
                store = self.frame["stores"][owner]
                selected = store.get("beacon") is not None
                jid = store.get("beacon") or jid
                font = QFont()
                font.setPixelSize(8)
                painter.setFont(font)
                painter.setPen(QColor("#254455"))
                count = sum(j is not None for j in store["slots"])
                painter.drawText(
                    QRectF(center.x() - 19, center.y() + 12, 38, 12),
                    Qt.AlignmentFlag.AlignCenter,
                    f"×{count}",
                )
            rect = job_rect(center)
            if selected:
                painter.setPen(QPen(QColor("#784500"), 1.2))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawRoundedRect(rect.adjusted(-2, -2, 2, 2), 1, 1)
            quality = str(jobs.get(jid, {}).get("quality", "UNKNOWN")).upper()
            painter.setPen(QPen(QColor("white"), 0.5))
            painter.setBrush(
                QColor(QUALITY_COLORS.get(quality, QUALITY_COLORS["UNKNOWN"]))
            )
            painter.drawEllipse(QPointF(rect.right() - 1, rect.top() + 1), 2.5, 2.5)
            if (
                resource is None
                or getattr(getattr(resource, "storage", None), "mode", None) != "pool"
            ):
                font = QFont()
                font.setPixelSize(8)
                painter.setFont(font)
                painter.setPen(QColor("#254455"))
                painter.drawText(
                    QRectF(center.x() - 20, center.y() + 9, 40, 10),
                    Qt.AlignmentFlag.AlignCenter,
                    _short(jid),
                )
        font = QFont()
        font.setPixelSize(8)
        painter.setFont(font)
        for resource in self.design.machines:
            record = self.frame.get("machines", {}).get(resource.machine_id, {})
            f = resource.footprint
            status = str(record.get("status", "IDLE")).upper()
            label = f"M{_short(resource.machine_id)} · {status}"
            painter.setPen(QColor("#bf723d" if status == "BLOCKED" else "#366d96"))
            painter.drawText(
                QRectF(
                    f.x * CELL_SIZE,
                    (f.y + f.height) * CELL_SIZE - 12,
                    f.width * CELL_SIZE,
                    10,
                ),
                Qt.AlignmentFlag.AlignCenter,
                label,
            )
        for identifier, record in self.frame.get("agvs", {}).items():
            x, y = self.items[identifier].x(), self.items[identifier].y()
            painter.setPen(QColor("#234b74"))
            painter.drawText(
                QRectF(x, y, CELL_SIZE, 9),
                Qt.AlignmentFlag.AlignCenter,
                f"A{_short(identifier)}",
            )
