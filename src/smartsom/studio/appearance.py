"""Isolated appearance samples built with the editor's actual vector items."""

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QPainter
from PySide6.QtWidgets import (
    QDialog,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
)

from smartsom.domain.factory_design import (
    AGVDesign,
    BufferDesign,
    Cell,
    Footprint,
    InspectionStationDesign,
    MachineDesign,
    SlotDesign,
    SlotStorage,
)
from smartsom.studio.canvas import FactoryScene
from smartsom.studio.items import (
    AGVItem,
    BufferItem,
    InspectionStationItem,
    MachineItem,
)
from smartsom.studio.symbols import JobVisual, TickProgress


class SampleView(QGraphicsView):
    def __init__(self, scene, parent=None):
        super().__init__(scene, parent)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setBackgroundBrush(QColor("white"))
        self.setInteractive(False)
        self.setFrameShape(QGraphicsView.Shape.NoFrame)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def showEvent(self, event):
        super().showEvent(event)
        self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)


class AppearanceDialog(QDialog):
    """Manual tick examples, never a source of simulator or factory state."""

    def __init__(self, design, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Appearance preview · Warm clay")
        self.resize(940, 760)
        layout = QVBoxLayout(self)
        layout.addWidget(
            QLabel("Display samples only · Changes here do not edit the factory.")
        )
        controls = QHBoxLayout()
        self.total = QSpinBox()
        self.total.setRange(1, 10)
        self.total.setValue(4)
        self.remaining = QSpinBox()
        self.remaining.setRange(0, 4)
        self.remaining.setValue(3)
        self.next_tick = QPushButton("Next tick")
        controls.addWidget(QLabel("Total ticks"))
        controls.addWidget(self.total)
        controls.addWidget(QLabel("Remaining ticks"))
        controls.addWidget(self.remaining)
        controls.addWidget(self.next_tick)
        controls.addStretch()
        layout.addLayout(controls)
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)
        self.factory_scene = FactoryScene(design, self)
        self.factory_scene.setSceneRect(
            self.factory_scene.map_rect.adjusted(-10, -10, 10, 10)
        )
        self.factory_view = SampleView(self.factory_scene)
        self.tabs.addTab(self.factory_view, "Factory sample")
        self.symbol_scene = QGraphicsScene(self)
        self.symbol_scene.setSceneRect(0, 0, 440, 440)
        self.symbol_view = SampleView(self.symbol_scene)
        self.tabs.addTab(self.symbol_view, "Symbols")
        self.processing_items = []
        self._populate_factory()
        self._populate_symbols()
        self.total.valueChanged.connect(self._total_changed)
        self.remaining.valueChanged.connect(self._update_progress)
        self.next_tick.clicked.connect(
            lambda: self.remaining.setValue(self.remaining.value() - 1)
        )
        self._update_progress()

    def _populate_factory(self):
        seen = set()
        for item in self.factory_scene.entity_items.values():
            if isinstance(item, MachineItem) and "machine" not in seen:
                self.processing_items.append((item, None))
                seen.add("machine")
            elif (
                isinstance(item, InspectionStationItem)
                and item.resource.slots
                and "inspection" not in seen
            ):
                self.processing_items.append((item, item.resource.slots[0].slot_id))
                seen.add("inspection")
            elif (
                isinstance(item, BufferItem)
                and item.resource.storage.mode == "slots"
                and item.resource.storage.slots
                and "buffer" not in seen
            ):
                item.set_job(
                    JobVisual(), slot_id=item.resource.storage.slots[0].slot_id
                )
                seen.add("buffer")
            elif isinstance(item, AGVItem) and "agv" not in seen:
                item.set_loaded(True)
                seen.add("agv")

    def _text(self, text, x, y, size=10):
        font = QFont()
        font.setPixelSize(size)
        item = self.symbol_scene.addSimpleText(text, font)
        item.setBrush(QColor("#294758"))
        item.setPos(x, y)

    def _add(self, item, x, y):
        self.symbol_scene.addItem(item)
        item.setPos(x, y)
        return item

    def _inspection(self, identifier):
        return InspectionStationItem(
            InspectionStationDesign(
                identifier,
                "Inspection",
                Footprint(0, 0, 1, 1),
                slots=(SlotDesign("slot", Cell(0, 0)),),
            )
        )

    def _populate_symbols(self):
        self._text("UNIFORM JOB SIZE · ACTUAL VECTOR SYMBOLS", 20, 10, 13)
        for col, loaded in enumerate((False, True)):
            x = 125 + col * 170
            agv = self._add(
                AGVItem(AGVDesign(f"agv{col}", "AGV", Cell(0, 0))), x - 20, 45
            )
            agv.set_loaded(loaded)
            self._text("Loaded" if loaded else "Empty", x - 17, 87)
            buffer = self._add(
                BufferItem(
                    BufferDesign(
                        f"buffer{col}",
                        "Buffer",
                        Footprint(0, 0, 1, 1),
                        storage=SlotStorage((SlotDesign("slot", Cell(0, 0)),)),
                    )
                ),
                x - 20,
                119,
            )
            inspection = self._add(self._inspection(f"inspection{col}"), x - 20, 192)
            machine = self._add(
                MachineItem(
                    MachineDesign(
                        f"machine{col}",
                        "Machine",
                        Footprint(0, 0, 2, 2),
                    )
                ),
                x - 40,
                262,
            )
            if loaded:
                buffer.set_job(JobVisual(), slot_id="slot")
                self.processing_items.extend(((inspection, "slot"), (machine, None)))
        for label, y in (
            ("AGV", 58),
            ("Buffer", 131),
            ("Inspection", 204),
            ("Machine", 295),
        ):
            self._text(label, 20, y)
        self._text("Discrete countdown · 4 total ticks", 20, 355)
        for index, remaining in enumerate(range(4, -1, -1)):
            item = self._add(self._inspection(f"tick{index}"), 55 + index * 75, 382)
            item.set_job(JobVisual(TickProgress(4, remaining)), slot_id="slot")
            self._text(str(remaining), 70 + index * 75, 424)

    def _total_changed(self, total):
        self.remaining.setMaximum(total)
        self.remaining.setValue(total)
        self._update_progress()

    def _update_progress(self):
        job = JobVisual(TickProgress(self.total.value(), self.remaining.value()))
        for item, slot_id in self.processing_items:
            if slot_id is None:
                item.set_job(job)
            else:
                item.set_job(job, slot_id=slot_id)
        self.next_tick.setEnabled(self.remaining.value() > 0)
