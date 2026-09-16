"""Read-only live/replay window sharing the Studio scene and job symbols."""

import json

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSlider,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from smartsom.domain.factory_design import PoolStorage
from smartsom.studio.canvas import FactoryScene, FactoryView
from smartsom.studio.symbols import CELL_SIZE, JobVisual, TickProgress


class PlaybackWindow(QMainWindow):
    def __init__(self, factory, *, controls=None, playback=None):
        super().__init__()
        self.factory, self.controls, self.playback = factory, controls, playback
        self.current_tick = -1
        self.playing = False
        self.scene = FactoryScene(factory)
        self.view = FactoryView(self.scene)
        self.setWindowTitle("SmartSOM — Live" if controls else "SmartSOM — Replay")
        context = (
            getattr(controls, "context", None)
            if controls
            else playback.manifest.get("context")
        )
        if context:
            self.setWindowTitle(
                self.windowTitle()
                + " | "
                + " · ".join(
                    f"{key}: {context[key]}"
                    for key in ("case", "seed", "replication")
                    if key in context
                )
            )
        self.resize(1100, 720)
        self.tick_label = QLabel("Tick 0")
        self.tick_label.setObjectName("simulationTick")
        self.details = QTextEdit()
        self.details.setReadOnly(True)
        self.details.setObjectName("simulationState")
        self.pause_button = QPushButton("Pause" if controls else "Play")
        self.pause_button.clicked.connect(self.toggle)
        self.step_button = QPushButton("Step +1")
        self.step_button.clicked.connect(self.forward)
        self.back_button = QPushButton("Step −1")
        self.back_button.setEnabled(playback is not None)
        self.back_button.clicked.connect(
            lambda: self.seek(max(0, self.current_tick - 1))
        )
        self.stop_button = QPushButton("Stop run")
        self.stop_button.setVisible(controls is not None)
        self.stop_button.clicked.connect(lambda: controls.stop())
        # A plain button avoids the native macOS combo popup's accessibility
        # lifetime crash while keeping every playback speed available.
        self.speed_labels = ("0.25×", "0.5×", "1×", "2×", "4×", "10×", "Maximum")
        self.speed_index = 2
        self.speed = QPushButton("Speed: 1×")
        self.speed.setObjectName("playbackSpeed")
        self.speed.setToolTip("Click to cycle playback speed; Maximum wraps to 0.25×")
        self.speed.clicked.connect(self.cycle_speed)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setObjectName("replayTimeline")
        self.slider.setEnabled(playback is not None)
        self.slider.setRange(0, playback.last_tick if playback else 0)
        self.slider.valueChanged.connect(self.seek)
        controls_row = QHBoxLayout()
        for item in (
            self.tick_label,
            self.back_button,
            self.pause_button,
            self.step_button,
            self.speed,
            self.stop_button,
        ):
            controls_row.addWidget(item)
        splitter = QSplitter()
        splitter.addWidget(self.view)
        splitter.addWidget(self.details)
        splitter.setSizes([780, 320])
        layout = QVBoxLayout()
        layout.addLayout(controls_row)
        layout.addWidget(splitter)
        layout.addWidget(self.slider)
        central = QWidget()
        central.setLayout(layout)
        self.setCentralWidget(central)
        self.timer = QTimer(self)
        self.timer.setInterval(100)
        self.timer.timeout.connect(self.poll)
        self.timer.start()
        if playback:
            self.seek(0)

    def cycle_speed(self):
        self.speed_index = (self.speed_index + 1) % len(self.speed_labels)
        self.speed.setText(f"Speed: {self.speed_labels[self.speed_index]}")
        self.change_speed(self.speed_index)

    def change_speed(self, index):
        delays = [400, 200, 100, 50, 25, 10, 1]
        if self.controls:
            self.controls.delay = 0 if index == 6 else delays[index] / 1000
        else:
            self.timer.setInterval(delays[index])

    def toggle(self):
        if self.controls:
            self.controls.pause(not self.controls.paused)
            self.pause_button.setText("Play" if self.controls.paused else "Pause")
        else:
            self.playing = not self.playing
            self.pause_button.setText("Pause" if self.playing else "Play")

    def forward(self):
        if self.controls:
            self.controls.step()
            self.pause_button.setText("Play")
        else:
            self.playing = False
            self.pause_button.setText("Play")
            self.seek(min(self.playback.last_tick, self.current_tick + 1))

    def seek(self, tick):
        if self.playback:
            self.show_row(self.playback.row(tick))
            self.slider.blockSignals(True)
            self.slider.setValue(tick)
            self.slider.blockSignals(False)

    def poll(self):
        if self.controls:
            row = self.controls.latest
            if row is not None and row["tick"] != self.current_tick:
                self.show_row(row)
            if self.controls.finished:
                self.pause_button.setEnabled(False)
                self.step_button.setEnabled(False)
                self.stop_button.setEnabled(False)
                self.statusBar().showMessage(
                    str(self.controls.error)
                    if self.controls.error
                    else f"Run finished: {getattr(self.controls, 'outcome', None) or 'stopped'}"
                )
        elif self.playing:
            if self.current_tick < self.playback.last_tick:
                self.seek(self.current_tick + 1)
            else:
                self.playing = False
                self.pause_button.setText("Play")

    def show_row(self, row):
        state = row["state"]
        self.current_tick = row["tick"]
        self.tick_label.setText(
            f"Tick {self.current_tick} · delivered {len(state['completed'])}"
        )
        items = self.scene.entity_items
        for key, data in state["agvs"].items():
            items[key].setPos(data["cell"][0] * CELL_SIZE, data["cell"][1] * CELL_SIZE)
            items[key].set_loaded(data["job"] is not None)
        for key, data in state["machines"].items():
            progress = None
            if data["job"] and data["status"] == "PROCESSING":
                progress = TickProgress(
                    data["elapsed"] + data["remaining"], data["remaining"]
                )
            items[key].set_job(JobVisual(progress) if data["job"] else None)
            items[key].setToolTip(
                f"{key}: {data['status']} · {'DOWN' if data['down'] else 'UP'}"
            )
        for buffer in self.factory.buffers:
            slots = state["storage"][buffer.buffer_id]
            if isinstance(buffer.storage, PoolStorage):
                items[buffer.buffer_id].set_job(JobVisual() if slots["pool"] else None)
            else:
                for slot, jobs in slots.items():
                    items[buffer.buffer_id].set_job(
                        JobVisual() if jobs else None, slot_id=slot
                    )
        for station in self.factory.inspection_stations:
            key = station.inspection_station_id
            data = state["stations"][key]
            for slot, jobs in state["storage"][key].items():
                progress = None
                if jobs and jobs[0] in data["batch"]:
                    progress = TickProgress(station.inspection_ticks, data["remaining"])
                items[key].set_job(JobVisual(progress) if jobs else None, slot_id=slot)
        # Exact read-only values make visual spot checks possible without guessing
        # how many jobs a single pool symbol represents.
        public_jobs = {
            k: {f: v for f, v in job.items() if f != "defective"}
            for k, job in state["jobs"].items()
            if job["location"] != "queue"
        }
        self.details.setPlainText(
            json.dumps(
                {
                    "tick": self.current_tick,
                    "status": state["status"],
                    "agvs": state["agvs"],
                    "machines": state["machines"],
                    "stations": state["stations"],
                    "storage": state["storage"],
                    "rankings": state["rankings"],
                    "jobs": public_jobs,
                    "metrics": state["metrics"],
                },
                indent=2,
            )
        )

    def closeEvent(self, event):
        if self.controls:
            self.controls.detach()
        self.timer.stop()
        super().closeEvent(event)


def live_window(factory, controls, thread):
    app = QApplication.instance() or QApplication([])
    window = PlaybackWindow(factory, controls=controls)
    window.show()
    thread.start()
    app.exec()
    controls.detach()


def playback_window(source):
    from smartsom.config.production import scenario_from_snapshot
    from smartsom.trace.production import Playback

    recording = Playback(source)
    factory = scenario_from_snapshot(recording.manifest["inputs"]["scenario"]).factory
    app = QApplication.instance() or QApplication([])
    window = PlaybackWindow(factory, playback=recording)
    window.show()
    app.exec()
