"""Rendering and isolation of transient appearance samples."""

import importlib.util
import os

import pytest

if importlib.util.find_spec("PySide6") is None:
    if os.environ.get("SMARTSOM_REQUIRE_STUDIO") == "1":
        raise ImportError("Studio acceptance requires the studio extra")
    pytest.skip("optional studio extra is not installed", allow_module_level=True)

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPointF, QRectF
from PySide6.QtGui import QColor, QImage, QPainter
from PySide6.QtWidgets import QApplication, QGraphicsScene

from smartsom.config.codec import canonical_json
from smartsom.domain.factory_design import Footprint, MachineDesign
from smartsom.studio.appearance import AppearanceDialog
from smartsom.studio.export import export_scene, render_image
from smartsom.studio.items import COLORS, MachineItem
from smartsom.studio.symbols import (
    JOB_COLOR,
    JobVisual,
    TickProgress,
    draw_job,
    progress_arcs,
)
from smartsom.studio.templates import load_template_1

pytestmark = pytest.mark.studio


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.mark.parametrize("total", [1, 2, 3, 4, 10])
def test_segments_disappear_without_redistribution(app, total):
    full = progress_arcs(TickProgress(total, total))
    previous = None
    for remaining in range(total, -1, -1):
        arcs = progress_arcs(TickProgress(total, remaining))
        assert len(arcs) == remaining
        assert arcs == full[total - remaining :]
        image = QImage(80, 80, QImage.Format.Format_ARGB32)
        image.fill(QColor("white"))
        painter = QPainter(image)
        painter.scale(2, 2)
        draw_job(
            painter,
            QPointF(20, 20),
            JobVisual(TickProgress(total, remaining)),
            kind="inspection",
        )
        painter.end()
        assert image.pixelColor(28, 28) == QColor(JOB_COLOR)
        if previous is not None:
            assert image != previous
        previous = image


@pytest.mark.parametrize(
    "total,remaining", [(0, 0), (4, -1), (4, 5), (True, 1), (4, 1.5)]
)
def test_invalid_tick_values(total, remaining):
    with pytest.raises(ValueError):
        TickProgress(total, remaining)


def test_idle_gear_is_small_centered_and_unfilled(app):
    scene = QGraphicsScene()
    item = MachineItem(MachineDesign("machine", "Machine", Footprint(0, 0, 2, 2)))
    scene.addItem(item)
    image = QImage(80, 80, QImage.Format.Format_ARGB32)
    image.fill(QColor("white"))
    painter = QPainter(image)
    scene.render(painter, QRectF(0, 0, 80, 80), QRectF(0, 0, 80, 80))
    painter.end()
    fill = QColor(COLORS["machine"][0])
    # Both the hub and space inside the tooth outline retain the machine fill.
    assert image.pixelColor(40, 40) == fill
    assert image.pixelColor(50, 40) == fill
    points = [
        (x, y)
        for x in range(10, 70)
        for y in range(10, 70)
        if image.pixelColor(x, y) != fill
    ]
    assert points
    assert 19 <= min(x for x, _ in points) <= 22
    assert 57 <= max(x for x, _ in points) <= 60


def test_preview_controls_and_export_remain_isolated(app):
    design = load_template_1()
    original = canonical_json(design)
    source = export_scene(design)
    before = render_image(source)
    dialog = AppearanceDialog(design)
    dialog.show()
    app.processEvents()
    assert render_image(dialog.factory_scene) != before
    for _ in range(3):
        dialog.next_tick.click()
    assert dialog.remaining.value() == 0
    assert not dialog.next_tick.isEnabled()
    for item, slot_id in dialog.processing_items:
        job = item.job if slot_id is None else item.slot_jobs[slot_id]
        assert job.progress == TickProgress(4, 0)
    dialog.total.setValue(2)
    assert dialog.remaining.value() == 2
    assert dialog.next_tick.isEnabled()
    assert canonical_json(design) == original
    assert render_image(source) == before
    assert render_image(export_scene(design)) == before
    dialog.close()
