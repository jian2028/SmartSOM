"""Shared vector job symbols; display values never advance a simulation clock."""

import math
from dataclasses import dataclass

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QPainterPath, QPen

CELL_SIZE = 40.0
JOB_SIZE = CELL_SIZE * 0.4
JOB_COLOR = "#B18461"
RING_DIAMETER = CELL_SIZE * 0.84
RING_WIDTH = 1.4
PROCESS_COLORS = {"machine": "#507fa8", "inspection": "#8064a5"}


@dataclass(frozen=True)
class TickProgress:
    total: int
    remaining: int

    def __post_init__(self):
        if type(self.total) is not int or self.total < 1:
            raise ValueError("total ticks must be a positive integer")
        if type(self.remaining) is not int or not 0 <= self.remaining <= self.total:
            raise ValueError("remaining ticks must be an integer between 0 and total")


@dataclass(frozen=True)
class JobVisual:
    """One visible job, optionally processing; no identity or workload is invented."""

    progress: TickProgress | None = None

    def __post_init__(self):
        if self.progress is not None and not isinstance(self.progress, TickProgress):
            raise TypeError("progress must be TickProgress or None")


def validate_job(job):
    if job is not None and not isinstance(job, JobVisual):
        raise TypeError("job must be JobVisual or None")


def job_rect(center):
    return QRectF(
        center.x() - JOB_SIZE / 2, center.y() - JOB_SIZE / 2, JOB_SIZE, JOB_SIZE
    )


def gear_path(rect):
    center = rect.center()
    radius = min(rect.width(), rect.height()) * 0.43
    path = QPainterPath()
    for tooth in range(8):
        for offset, scale in (
            (-22.5, 0.76),
            (-14, 0.76),
            (-10, 1),
            (10, 1),
            (14, 0.76),
            (22.5, 0.76),
        ):
            angle = math.radians(tooth * 45 + offset)
            point = center + QPointF(math.cos(angle), math.sin(angle)) * radius * scale
            if tooth == 0 and offset == -22.5:
                path.moveTo(point)
            else:
                path.lineTo(point)
    path.closeSubpath()
    path.addEllipse(center, radius * 0.34, radius * 0.34)
    return path


def magnifier_path(rect):
    radius = min(rect.width(), rect.height()) * 0.23
    center = rect.center() - QPointF(radius * 0.28, radius * 0.28)
    path = QPainterPath()
    path.addEllipse(center, radius, radius)
    diagonal = radius / math.sqrt(2)
    path.moveTo(center + QPointF(diagonal, diagonal))
    path.lineTo(center + QPointF(radius * 1.35, radius * 1.35))
    return path


def progress_arcs(progress):
    """Qt angles: remove fixed segments clockwise, beginning at twelve o'clock."""
    step = 360 / progress.total
    gap = min(10, step * 0.15) if progress.total > 1 else 0
    return tuple(
        (90 - index * step - gap / 2, -(step - gap))
        for index in range(progress.total - progress.remaining, progress.total)
    )


def draw_job(painter, center, job, *, kind=None):
    painter.save()
    rect = job_rect(center)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor(JOB_COLOR))
    painter.drawRoundedRect(rect, 0.8, 0.8)
    if job.progress is not None and kind in PROCESS_COLORS:
        painter.setBrush(Qt.BrushStyle.NoBrush)
        pen = QPen(QColor("white"), 0.9)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        glyph = gear_path if kind == "machine" else magnifier_path
        painter.drawPath(glyph(rect.adjusted(1, 1, -1, -1)))
        painter.setPen(QPen(QColor(PROCESS_COLORS[kind]), RING_WIDTH))
        radius = (RING_DIAMETER - RING_WIDTH) / 2
        ring = QRectF(center.x() - radius, center.y() - radius, 2 * radius, 2 * radius)
        for start, span in progress_arcs(job.progress):
            painter.drawArc(ring, round(start * 16), round(span * 16))
    painter.restore()
