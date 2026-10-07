"""Six evidence tracks sharing the existing replay clock, with pixel aggregation."""

from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QToolTip, QWidget

from smartsom.studio.replay_model import LANES


class EventTimeline(QWidget):
    seek_requested = Signal(int)
    events_selected = Signal(object)

    def __init__(self, index):
        super().__init__()
        self.index = index
        self.tick, self.owner = 0, None
        self.start, self.end = 0, max(1, index.last_tick)
        self.setMinimumHeight(110)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setAccessibleName(
            "Recorded event timeline; arrows step, plus/minus zoom, Home fits"
        )
        self.dark = False

    def set_frame(self, tick, owner=None):
        self.tick, self.owner = tick, owner
        self.update()

    def plot(self):
        return QRectF(170, 26, max(20, self.width() - 184), max(30, self.height() - 54))

    def x(self, tick):
        rect = self.plot()
        return (
            rect.left()
            + (tick - self.start) / max(1, self.end - self.start) * rect.width()
        )

    def time_at(self, x):
        rect = self.plot()
        return max(
            0,
            min(
                self.index.last_tick,
                round(
                    self.start
                    + (x - rect.left()) / rect.width() * (self.end - self.start)
                ),
            ),
        )

    def visible_events(self):
        return [
            e
            for e in self.index.filtered(self.owner)
            if e.lane
            and e.tick <= self.end
            and (e.end if e.end is not None else e.tick) >= self.start
        ]

    def buckets(self):
        groups = {}
        for event in self.visible_events():
            key = (event.lane, int(self.x(max(self.start, event.tick))))
            groups.setdefault(key, []).append(event)
        return groups

    def hit(self, position):
        rect = self.plot()
        lane_index = int((position.y() - rect.top()) / (rect.height() / len(LANES)))
        if not 0 <= lane_index < len(LANES) or position.x() < rect.left():
            return []
        lane = LANES[lane_index][0]
        return [
            e
            for e in self.visible_events()
            if e.lane == lane
            and (
                abs(self.x(e.tick) - position.x()) <= 5
                or (
                    e.end is not None
                    and self.x(e.tick) <= position.x() <= self.x(e.end)
                )
            )
        ]

    def paintEvent(self, event):
        painter = QPainter(self)
        rect = self.plot()
        foreground = QColor("#cbd7e0" if self.dark else "#405b6c")
        divider = QColor("#354856" if self.dark else "#e1e8ed")
        painter.setPen(foreground)
        painter.drawText(
            QRectF(8, 0, self.width() - 16, 22),
            Qt.AlignmentFlag.AlignLeft,
            "Event counts: entire recording"
            + (" · selected object" if self.owner else " · all objects"),
        )
        groups = self.buckets()
        row_height = rect.height() / len(LANES)
        for i, (key, label, color) in enumerate(LANES):
            y = rect.top() + i * row_height
            count = len([e for e in self.index.filtered(self.owner, key)])
            painter.setPen(foreground)
            painter.drawText(
                QRectF(8, y, 158, row_height),
                Qt.AlignmentFlag.AlignVCenter,
                f"{label}  {count}",
            )
            painter.setPen(divider)
            painter.drawLine(rect.left(), y + row_height, rect.right(), y + row_height)
            for (lane, x), events in groups.items():
                if lane != key:
                    continue
                painter.fillRect(
                    QRectF(x, y + 5, min(6, 2 + len(events)), max(3, row_height - 10)),
                    QColor(color),
                )
                for recorded in events:
                    if recorded.end is not None:
                        end = min(rect.right(), self.x(recorded.end))
                        painter.fillRect(
                            QRectF(x, y + row_height / 2 - 3, max(2, end - x), 6),
                            QColor(color),
                        )
        painter.setPen(foreground)
        for i in range(6):
            t = round(self.start + (self.end - self.start) * i / 5)
            painter.drawText(
                QRectF(
                    max(0, min(self.width() - 56, self.x(t) - 22)),
                    rect.bottom() + 3,
                    55,
                    22,
                ),
                Qt.AlignmentFlag.AlignCenter,
                str(t),
            )
        if self.start <= self.tick <= self.end:
            painter.setPen(QPen(QColor("#4b99ca" if self.dark else "#235f88"), 2))
            painter.drawLine(
                self.x(self.tick), rect.top(), self.x(self.tick), rect.bottom()
            )

    def mouseMoveEvent(self, event):
        hits = self.hit(event.position())
        if hits:
            QToolTip.showText(
                event.globalPosition().toPoint(),
                "\n".join(f"Tick {e.tick} · {e.detail}" for e in hits[:16])
                + (
                    f"\n+ {len(hits) - 16} more; click to list"
                    if len(hits) > 16
                    else ""
                ),
                self,
            )
        else:
            QToolTip.hideText()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            hits = self.hit(event.position())
            tick = (
                min(hits, key=lambda e: abs(self.x(e.tick) - event.position().x())).tick
                if hits
                else self.time_at(event.position().x())
            )
            self.seek_requested.emit(tick)
            self.events_selected.emit(hits)
            self.setFocus()

    def fit(self):
        self.start, self.end = 0, max(1, self.index.last_tick)
        self.update()

    def zoom(self, factor, anchor=None):
        center = self.tick if anchor is None else anchor
        span = min(
            max(1, self.index.last_tick), max(10, (self.end - self.start) * factor)
        )
        self.start = max(0, min(center - span / 2, self.index.last_tick - span))
        self.end = self.start + span
        self.update()

    def wheelEvent(self, event):
        self.zoom(
            0.5 if event.angleDelta().y() > 0 else 2, self.time_at(event.position().x())
        )
        event.accept()

    def keyPressEvent(self, event):
        key = event.key()
        if key in (Qt.Key.Key_Left, Qt.Key.Key_Right):
            self.seek_requested.emit(
                max(
                    0,
                    min(
                        self.index.last_tick,
                        self.tick + (-1 if key == Qt.Key.Key_Left else 1),
                    ),
                )
            )
        elif key in (Qt.Key.Key_Plus, Qt.Key.Key_Equal):
            self.zoom(0.5)
        elif key == Qt.Key.Key_Minus:
            self.zoom(2)
        elif key == Qt.Key.Key_Home:
            self.fit()
        else:
            super().keyPressEvent(event)
