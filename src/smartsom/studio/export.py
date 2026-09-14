"""Render an isolated scene so export never mutates editor selection or layers."""

import os
import tempfile
from pathlib import Path

from PySide6.QtCore import QRect, QRectF, QSize, Qt
from PySide6.QtGui import QImage, QPainter
from PySide6.QtSvg import QSvgGenerator

from smartsom.studio.canvas import FactoryScene


def export_scene(
    design, *, grid=True, numbers=False, ports=True, bindings="none", selected=None
):
    scene = FactoryScene(design)
    scene.set_layer("grid", grid)
    scene.set_layer("names", numbers)
    scene.set_layer("ports", ports)
    scene.binding_mode = bindings
    scene.update_bindings(selected)
    return scene


def render_image(scene, cell_pixels=40):
    width = scene.design.grid.width * cell_pixels
    height = scene.design.grid.height * cell_pixels
    if width * height > 64_000_000:
        raise ValueError(
            "PNG exceeds 64 million pixels. Reduce pixels per cell or use SVG."
        )
    image = QImage(width, height, QImage.Format.Format_ARGB32)
    if image.isNull():
        raise ValueError("Cannot allocate image; reduce pixels per cell.")
    image.fill(Qt.GlobalColor.white)
    painter = QPainter(image)
    try:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        scene.render(painter, QRectF(0, 0, width, height), scene.map_rect)
    finally:
        painter.end()
    return image


def export_map(path, design, *, cell_pixels=40, **options):
    path = Path(path)
    if path.suffix.lower() not in (".png", ".svg"):
        raise ValueError("Choose a PNG or SVG file")
    if type(cell_pixels) is not int or cell_pixels < 1:
        raise ValueError("Pixels per cell must be a positive integer")
    destination = path
    with tempfile.NamedTemporaryFile(
        dir=path.parent, suffix=path.suffix, delete=False
    ) as stream:
        path = Path(stream.name)
    scene = export_scene(design, **options)
    try:
        if path.suffix.lower() == ".png":
            if not render_image(scene, cell_pixels).save(str(path), "PNG"):
                raise OSError(f"Could not write {path}")
        elif path.suffix.lower() == ".svg":
            generator = QSvgGenerator()
            generator.setFileName(str(path))
            size = QSize(
                design.grid.width * cell_pixels, design.grid.height * cell_pixels
            )
            generator.setSize(size)
            generator.setViewBox(QRect(0, 0, size.width(), size.height()))
            generator.setTitle(design.name)
            painter = QPainter()
            if not painter.begin(generator):
                raise OSError(f"Could not write {path}")
            try:
                scene.render(
                    painter, QRectF(0, 0, size.width(), size.height()), scene.map_rect
                )
            finally:
                painter.end()
            if not path.is_file() or path.stat().st_size == 0:
                raise OSError(f"Could not write {path}")
        os.replace(path, destination)
    finally:
        path.unlink(missing_ok=True)
        scene.deleteLater()
