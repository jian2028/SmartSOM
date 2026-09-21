"""Factory integer-state presentation, with an optional replay transition.

The layer consumes display state only. It neither edits a design nor advances
simulation. Studio can reuse integer presentation without replay interpolation.
"""

import math

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QFontMetricsF, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QGraphicsItem

from smartsom.studio.replay_evidence import (
    displayed_quality,
    job_marks,
    movement_conflicts,
    resource_conflicts,
)
from smartsom.studio.symbols import CELL_SIZE, gear_path, magnifier_path


class FactoryStateLayer(QGraphicsItem):
    def __init__(self, design, items):
        super().__init__()
        self.design, self.items = design, items
        self.setZValue(5)
        self.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self.store_resources = {
            **{item.buffer_id: item for item in design.buffers},
            **{item.inspection_station_id: item for item in design.inspection_stations},
        }
        self.state = {}
        self.row = {}
        self.following = None
        self.alpha = 0.0
        self.evidence = None
        self.output_window = 100
        self.charging_preview = set()
        for item in items.values():
            if item.kind in (
                "machine",
                "buffer",
                "inspection",
                "agv",
                "charger",
                "scrap",
            ):
                item.state_layer_active = True

    def boundingRect(self):
        return QRectF(
            -35,
            -35,
            self.design.grid.width * CELL_SIZE + 70,
            self.design.grid.height * CELL_SIZE + 70,
        )

    def set_row(self, row):
        self.row = row
        self.state = row["state"]
        for key, data in self.state.get("agvs", {}).items():
            if key in self.items:
                self.items[key].state_layer_job = bool(data.get("job"))
        self.frame = row.get("historical_frame", {})
        self.following = None
        self.alpha = 0.0
        self.update()

    def transition(self, following, alpha):
        self.following, self.alpha = following, alpha
        self.update()

    def owner_center(self, owner, slot=None):
        if owner not in self.items:
            return None
        item = self.items[owner]
        resource = self.store_resources.get(owner)
        if slot and resource:
            slots = getattr(resource, "slots", ()) or getattr(
                getattr(resource, "storage", None), "slots", ()
            )
            for entry in slots:
                if entry.slot_id == slot:
                    return (
                        item.pos()
                        + QPointF(entry.local_cell.x + 0.5, entry.local_cell.y + 0.5)
                        * CELL_SIZE
                    )
        return item.pos() + item._rect.center()

    def locations(self, state):
        locations = {}
        for owner, slots in state.get("storage", {}).items():
            for slot, jobs in slots.items():
                for jid in jobs:
                    center = self.owner_center(owner, slot)
                    if center is not None:
                        locations[jid] = (center, owner, False)
        for owner, data in state.get("machines", {}).items():
            if data.get("job"):
                locations[data["job"]] = (self.owner_center(owner), owner, False)
        for owner, data in state.get("agvs", {}).items():
            if data.get("job"):
                locations[data["job"]] = (self.owner_center(owner), owner, False)
        return locations

    def job_locations(self):
        return self.locations(self.state)

    def paths(self):
        """Events retain intermediate hops even when a job exits in one tick."""
        if not self.following or not self.alpha:
            return {}
        before, after = self.job_locations(), self.locations(self.following["state"])
        paths = {}
        for event in self.following.get("events", ()):
            kind = event["kind"]
            jid = event.get("job", event.get("job_id"))
            if not jid:
                continue
            if kind == "pickup":
                target = self.owner_center(event.get("agv", event.get("agent_id")))
            elif kind in ("drop", "machine_released", "input_admitted"):
                target = self.owner_center(event.get("owner"), event.get("slot"))
            elif kind in ("processing_started", "process_start"):
                target = self.owner_center(
                    event.get("machine", event.get("machine_id"))
                )
            else:
                continue
            if target is not None:
                origin = before.get(jid, (target,))[0]
                points = paths.setdefault(jid, [origin])
                if target != points[-1]:
                    points.append(target)
        for jid, (end, owner, _) in after.items():
            if jid in before and before[jid][1] != owner:
                points = paths.setdefault(jid, [before[jid][0]])
                if points[-1] != end:
                    points.append(end)
        return {jid: points for jid, points in paths.items() if len(points) > 1}

    @staticmethod
    def label(painter, rect, text, color="#294550", size=8):
        font = QFont()
        font.setPixelSize(size)
        painter.setFont(font)
        painter.setPen(QColor(color))
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, str(text))

    @staticmethod
    def segments(painter, rect, total, remaining, color="#526bac"):
        if total <= 0:
            painter.fillRect(rect, QColor("#dbe2ea"))
            return
        width = rect.width() / total
        elapsed = total - remaining
        painter.setPen(Qt.PenStyle.NoPen)
        for index in range(total):
            box = QRectF(
                rect.x() + index * width + 0.4,
                rect.y(),
                max(0.15, width - 0.8),
                rect.height(),
            )
            painter.setBrush(QColor("#dbe2ea"))
            painter.drawRect(box)
            fraction = max(0, min(1, index + 1 - elapsed))
            if fraction:
                box.setLeft(box.right() - box.width() * fraction)
                painter.fillRect(box, QColor(color))

    def machine_geometry(self, key):
        item = self.items[key]
        rect = item._rect.translated(item.pos())
        band = max(6, min(12, rect.height() * 0.125))
        side = min(rect.width() - 8, rect.height() - 2 * band - 4)
        center = rect.center()
        gear = QRectF(center.x() - side / 2, center.y() - side / 2, side, side)
        # gear_path's inner radius is outer radius * .34; cover its full diameter.
        job_size = max(12, side * 0.43 * 0.34 * 2 + 3)
        return rect, band, gear, job_size

    def machine(self, painter, key, data):
        item = self.items[key]
        rect, band_height, gear, _ = self.machine_geometry(key)
        total = data.get("elapsed", 0) + data.get("remaining", 0)
        remaining = data.get("remaining", 0)
        active = data.get("status") == "PROCESSING" and not data.get("down")
        after = (
            self.following["state"]["machines"].get(key, {}) if self.following else {}
        )
        advances = active and (
            after.get("remaining") == remaining - 1
            or remaining == 1
            and after.get("job") != data.get("job")
        )
        self.segments(
            painter,
            QRectF(rect.x(), rect.y(), rect.width(), band_height),
            total,
            remaining - (self.alpha if advances else 0),
        )
        painter.save()
        center = gear.center()
        painter.translate(center)
        painter.rotate(
            (data.get("elapsed", 0) + (self.alpha if advances else 0)) * 25
            if data.get("job")
            else 0
        )
        painter.setPen(QPen(QColor("#526bac"), 1.3))
        painter.setBrush(QColor("#eef2ff"))
        painter.drawPath(gear_path(gear.translated(-center)))
        painter.restore()
        modes = data.get("available_modes")
        if modes is None:
            modes = [mode.quality_mode_id for mode in item.resource.quality_modes]
        labels = list(dict.fromkeys(["slow", "normal", "fast", *modes]))
        width = rect.width() / len(labels)
        font = QFont()
        font.setPixelSize(max(5, int(band_height * 0.75)))
        while (
            font.pixelSize() > 3
            and max(
                QFontMetricsF(font).horizontalAdvance(label.upper()) for label in labels
            )
            > width - 2
        ):
            font.setPixelSize(font.pixelSize() - 1)
        for i, mode in enumerate(labels):
            band = QRectF(
                rect.x() + i * width, rect.bottom() - band_height, width, band_height
            )
            on = mode == data.get("mode")
            painter.fillRect(band, QColor("#526bac" if on else "#e4e8ef"))
            self.label(
                painter,
                band,
                mode.upper(),
                "white" if on else "#526bac" if mode in modes else "#a6aebb",
                font.pixelSize(),
            )
            painter.setPen(QPen(QColor("#526bac"), 0.7))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(band)
        painter.setPen(QPen(QColor("#526bac"), 0.7))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(QRectF(rect.x(), rect.y(), rect.width(), band_height))
        if data.get("down") or data.get("status") == "DOWN":
            # A frozen gear and pulsing warning icon distinguish a real outage.
            warning = QRectF(rect.right() - 17, rect.y() + band_height + 3, 14, 14)
            painter.setBrush(QColor("#ffe5cc"))
            painter.setPen(
                QPen(QColor("#c55a37"), 1 + 0.6 * math.sin(self.alpha * math.pi))
            )
            painter.drawEllipse(warning)
            self.label(painter, warning, "!", "#b64e30", 12)
        elif data.get("status") == "BLOCKED":
            self.label(
                painter,
                QRectF(rect.right() - 17, rect.y() + band_height + 3, 14, 14),
                "Ⅱ",
                "#bc8135",
                12,
            )

    def badge(self, painter, center, jid, quality, inspecting=False, size=18):
        rect = QRectF(center.x() - size / 2, center.y() - size / 2, size, size)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#b18461"))
        painter.drawRoundedRect(rect, 1, 1)
        number, attempt = job_marks(jid)
        self.label(
            painter,
            rect,
            number,
            "white",
            max(5, min(13, int(size / max(2, len(number)) * 1.6))),
        )
        if attempt > 1:
            retry = QRectF(rect.right() - 8, rect.bottom() - 4, 15, 9)
            painter.fillRect(retry, QColor("#f7ead7"))
            self.label(painter, retry, f"↻{attempt}", "#795335", 7)
        badge = QRectF(rect.right() - 5, rect.top() - 4, 10, 10)
        color = (
            "#7b61a7"
            if inspecting
            else {"PASS": "#22835c", "FAIL": "#cc4f45"}.get(quality, "#344b58")
        )
        painter.setBrush(QColor("#fbfaf5"))
        painter.setPen(QPen(QColor(color), 0.7))
        painter.drawEllipse(badge)
        if inspecting:
            painter.drawPath(magnifier_path(badge))
        else:
            self.label(
                painter, badge, {"PASS": "✓", "FAIL": "×"}.get(quality, "?"), color, 10
            )

    def pool_stack(self, painter, rect, count):
        center = rect.center() - QPointF(2, 6)
        size = min(19, rect.width() * 0.5, rect.height() * 0.4)
        box = QRectF(center.x() - size / 2, center.y() - size / 2, size, size)
        for offset in (QPointF(4, -4), QPointF(2, -2), QPointF(0, 0)):
            painter.setPen(QPen(QColor("#f7eee4"), 0.8))
            painter.setBrush(QColor("#b18461") if count else QColor("#efe5dc"))
            painter.drawRect(box.translated(offset))
        self.label(
            painter,
            QRectF(rect.x(), box.bottom() + 3, rect.width(), 13),
            f"× {count if count is not None else '—'}",
            "#267b68",
            11,
        )

    @staticmethod
    def pool_geometry(rect):
        band = min(12, rect.height() * 0.15)
        return (
            rect.adjusted(0, band, 0, -band),
            QRectF(rect.left(), rect.top(), rect.width(), band),
            QRectF(rect.left(), rect.bottom() - band, rect.width(), band),
        )

    @staticmethod
    def pool_metric_icon(painter, rect, kind):
        painter.save()
        painter.translate(rect.center())
        painter.scale(rect.width() / 12, rect.height() / 12)
        painter.setPen(QPen(QColor("#267b68"), 0.9))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        path = QPainterPath()
        if kind == "waiting":
            path.moveTo(-3.5, -5)
            for x, y in [
                (3.5, -5),
                (3.5, -3),
                (-3.5, 3),
                (-3.5, 5),
                (3.5, 5),
                (3.5, 3),
                (-3.5, -3),
                (-3.5, -5),
            ]:
                path.lineTo(x, y)
        else:
            # Scalloped quality seal with a check, independent of font glyphs.
            for index in range(24):
                angle = index * math.pi / 12 - math.pi / 2
                radius = 5.4 if index % 2 == 0 else 4.5
                point = QPointF(math.cos(angle) * radius, math.sin(angle) * radius)
                if index == 0:
                    path.moveTo(point)
                else:
                    path.lineTo(point)
            path.closeSubpath()
            path.moveTo(-2.5, 0)
            path.lineTo(-0.6, 2)
            path.lineTo(2.8, -2)
        painter.drawPath(path)
        painter.restore()

    def pool_header(self, painter, rect, count, text, kind):
        content, header, footer = self.pool_geometry(rect)
        self.pool_stack(painter, content, count)
        painter.fillRect(header, QColor("#d6ebe4"))
        painter.setPen(QPen(QColor("#267b68"), 0.6))
        painter.drawLine(header.bottomLeft(), header.bottomRight())
        font = QFont()
        font.setPixelSize(7)
        icon_size = min(9, header.height() - 2)
        gap = 2
        while (
            font.pixelSize() > 4
            and QFontMetricsF(font).horizontalAdvance(text) + icon_size + gap
            > rect.width() - 3
        ):
            font.setPixelSize(font.pixelSize() - 1)
        width = QFontMetricsF(font).horizontalAdvance(text)
        left = header.center().x() - (icon_size + gap + width) / 2
        self.pool_metric_icon(
            painter,
            QRectF(left, header.center().y() - icon_size / 2, icon_size, icon_size),
            kind,
        )
        self.label(
            painter,
            QRectF(left + icon_size + gap, header.top(), width, header.height()),
            text,
            "#267b68",
            font.pixelSize(),
        )
        return footer

    def input_buffer(self, painter, rect, owner, count):
        waiting = self.evidence.input_waiting(owner, self.state)
        footer = self.pool_header(
            painter,
            rect,
            count,
            str(waiting) if waiting is not None else "—",
            "waiting",
        )
        progress = self.evidence.input_progress(owner, self.row.get("tick", 0))
        total, remaining = progress[:2] if progress else (0, 0)
        self.segments(painter, footer, total, remaining, "#267b68")
        painter.setPen(QPen(QColor("#267b68"), 0.6))
        painter.drawLine(footer.topLeft(), footer.topRight())

    def output_buffer(self, painter, rect, owner):
        tick = self.row.get("tick", 0)
        count, rate, throughput = self.evidence.output_metrics(
            owner, tick, self.output_window
        )
        footer = self.pool_header(
            painter, rect, count, f"{rate:.0%}" if rate is not None else "—", "quality"
        )
        painter.setPen(QPen(QColor("#267b68"), 0.6))
        painter.drawLine(footer.topLeft(), footer.topRight())
        self.label(
            painter,
            footer,
            f"{throughput:.2f}/tick" if throughput is not None else "—/tick",
            "#267b68",
            7,
        )

    def transfer_feedback(self, jid):
        events = self.following.get("events", ()) if self.following else ()
        for event in events:
            if event.get("job", event.get("job_id")) != jid:
                continue
            if event["kind"] == "output":
                return (
                    ("✓ +1", "#267b68")
                    if event.get("quality") == "PASS"
                    else ("× rejected", "#c84b43")
                )
            if event["kind"] == "drop":
                owner = event.get("owner")
                resource = self.store_resources.get(owner)
                if getattr(resource, "role", None) == "system_output":
                    quality = (
                        self.following["state"]["jobs"].get(jid, {}).get("quality")
                    )
                    return (
                        ("✓ +1", "#267b68")
                        if quality == "PASS"
                        else ("× rejected", "#c84b43")
                    )
        return "+1", "#267b68"

    def paint(self, painter, option, widget=None):
        if not self.state:
            return
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        for key, data in self.state["machines"].items():
            display = data
            if not data.get("job") and self.following and self.alpha:
                for event in self.following.get("events", ()):
                    if (
                        event["kind"] == "processing_started"
                        and event.get("machine") == key
                    ):
                        display = {
                            **data,
                            "job": event["job"],
                            "status": "PROCESSING",
                            "elapsed": 0,
                            "remaining": event["actual_ticks"],
                            "mode": event["mode"],
                        }
                        break
            self.machine(painter, key, display)
        paths = self.paths()
        for jid, (center, owner, _) in self.job_locations().items():
            if jid in paths:
                continue
            item = self.items[owner]
            if getattr(getattr(item.resource, "storage", None), "mode", None) == "pool":
                continue
            quality = displayed_quality(self.state, jid, owner)
            inspecting = jid in self.state["stations"].get(owner, {}).get("batch", ())
            size = (
                self.machine_geometry(owner)[3]
                if owner in self.state["machines"]
                else 18
            )
            self.badge(painter, center, jid, quality, inspecting, size)
        for jid, points in paths.items():
            position = self.alpha * (len(points) - 1)
            index = min(len(points) - 2, int(position))
            fraction = position - index
            start, end = points[index : index + 2]
            center = (
                start * (1 - fraction)
                + end * fraction
                - QPointF(0, math.sin(fraction * math.pi) * 8)
            )
            quality = self.state["jobs"].get(jid, {}).get("quality", "UNKNOWN")
            if quality != "FAIL" and any(
                e["kind"] in ("processing_started", "process_start")
                and e.get("job", e.get("job_id")) == jid
                for e in self.following.get("events", ())
            ):
                quality = "UNKNOWN"
            self.badge(painter, center, jid, quality)
            if fraction > 0.7:
                self.label(
                    painter,
                    QRectF(end.x() - 36, end.y() - 25, 72, 12),
                    self.transfer_feedback(jid)[0],
                    self.transfer_feedback(jid)[1],
                    10,
                )
        tick = self.row.get("tick", 0)
        for owner, slots in self.state["storage"].items():
            item = self.items.get(owner)
            if not item:
                continue
            count = sum(len(jobs) for jobs in slots.values())
            rect = item._rect.translated(item.pos())
            if getattr(getattr(item.resource, "storage", None), "mode", None) == "pool":
                if (
                    getattr(item.resource, "role", None) == "system_output"
                    and self.evidence
                ):
                    self.output_buffer(painter, rect, owner)
                elif (
                    getattr(item.resource, "role", None) == "system_input"
                    and self.evidence
                ):
                    self.input_buffer(painter, rect, owner, count)
                else:
                    self.pool_stack(painter, rect, count)
            station = self.state["stations"].get(owner, {})
            if station.get("batch"):
                total = station.get(
                    "total", getattr(item.resource, "inspection_ticks", 1)
                )
                after = (
                    self.following["state"]["stations"].get(owner, {})
                    if self.following
                    else {}
                )
                rem = station["remaining"] - (
                    self.alpha
                    if after.get("remaining") == station["remaining"] - 1
                    else 0
                )
                self.segments(
                    painter,
                    QRectF(rect.x() + 1, rect.y() + 1, rect.width() - 2, 4),
                    total,
                    rem,
                    "#7b61a7",
                )
                painter.setPen(QPen(QColor("#987ac0"), 0.8))
                y = rect.y() + 6 + self.alpha * (rect.height() - 16)
                painter.drawLine(QPointF(rect.left(), y), QPointF(rect.right(), y))
        for resource in self.design.scrap_bins:
            owner = resource.scrap_bin_id
            count = (
                self.state["metrics"].get(f"scrap:{owner}", 0)
                if "completed" in self.state
                else self.evidence.disposals[tick].get(owner, 0)
                if self.evidence and getattr(self, "has_events", False)
                else None
            )
            rect = self.items[owner]._rect.translated(self.items[owner].pos())
            self.label(
                painter,
                QRectF(rect.center().x() - 8, rect.center().y() - 7, 16, 18),
                count if count is not None else "—",
                "#a66950",
                12,
            )
        for owner in movement_conflicts(self.row):
            center = self.owner_center(owner)
            if center is not None:
                self.label(
                    painter,
                    QRectF(
                        center.x() - 22 + 2 * math.sin(self.alpha * math.pi * 6),
                        center.y() - 25,
                        44,
                        12,
                    ),
                    "✦  !  ✦",
                    "#c26336",
                    10,
                )
        for owner in resource_conflicts(self.row):
            center = self.owner_center(owner)
            if center is not None:
                self.label(
                    painter,
                    QRectF(center.x() - 28, center.y() - 25, 56, 12),
                    "RESOURCE !",
                    "#a97d35",
                    7,
                )
        # Explicit presentation fixture only; never infer charge from position.
        for owner in self.charging_preview:
            center = self.owner_center(owner)
            if center is not None:
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.setPen(QPen(QColor("#dea324"), 2))
                painter.drawEllipse(center, 15 + 3 * self.alpha, 15 + 3 * self.alpha)
                self.label(
                    painter,
                    QRectF(center.x() - 35, center.y() - 29, 70, 12),
                    getattr(self, "charging_label", "⚡ CHARGE PREVIEW"),
                    "#9b6c12",
                    6,
                )
