"""Read historical public frames through the current presentation-only player.

The projection below is a viewer contract, not a converted simulator trace.
Original historical frames remain the authority for details and metrics.
"""

import hashlib
import json
from pathlib import Path

from PySide6.QtWidgets import QApplication

from smartsom.config.factory_design import load_factory_design
from smartsom.domain.factory_design import PoolStorage
from smartsom.studio.playback import PlaybackWindow


class HistoricalRecording:
    def __init__(self, root, seed=24004):
        self.root = Path(root)
        self.factory, _ = load_factory_design(self.root / "factory.yaml")
        self.path = self.root / f"frames-{seed}.jsonl"
        self.manifest = {
            "context": {"case": f"Historical {self.root.name}", "seed": seed}
        }
        run = json.loads((self.root / "run.json").read_text())
        if run.get("schema") != "smartsom.spatial-run/v1":
            raise ValueError("Unsupported historical recording format")
        factory_hash = hashlib.sha256(
            (self.root / "factory.yaml").read_bytes()
        ).hexdigest()
        # The old manifest hashes its old domain object, not YAML bytes.
        # Compare byte identity to the preserved original and compare the two
        # historical manifests without importing the obsolete domain/runtime.
        original = self.root.parent / "original" / self.root.name
        original_run = json.loads((original / "run.json").read_text())
        original_hash = hashlib.sha256(
            (original / "factory.yaml").read_bytes()
        ).hexdigest()
        if factory_hash != original_hash or run.get(
            "factory_sha256"
        ) != original_run.get("factory_sha256"):
            raise ValueError("Historical factory differs from the original evidence")
        episode = json.loads((self.root / f"episode-{seed}.json").read_text())
        config_path = self.root / "config.json"
        scales = (
            json.loads(config_path.read_text())
            .get("environment", {})
            .get("mode_scales", [])
            if config_path.exists()
            else []
        )
        self.mode_names = [
            "slow" if scale > 1 else "fast" if scale < 1 else "normal"
            for scale in scales
        ]
        self.events = {}
        trace_path = self.root / f"trace-{seed}.jsonl"
        if trace_path.exists():
            with trace_path.open() as stream:
                for line in stream:
                    trace = json.loads(line)
                    if (
                        not line.endswith("\n")
                        or trace.get("tick") != len(self.events) + 1
                    ):
                        raise ValueError(
                            "Historical events require complete contiguous ticks starting at one"
                        )
                    self.events[trace["tick"]] = trace.get("events", [])
        self.offsets = []
        with self.path.open("rb") as stream:
            while True:
                offset = stream.tell()
                line = stream.readline()
                if not line:
                    break
                try:
                    frame = json.loads(line)
                    if not line.endswith(b"\n") or frame["tick"] != len(self.offsets):
                        raise ValueError("missing, duplicate or incomplete tick")
                    self.project(frame)
                except (KeyError, TypeError, ValueError) as error:
                    raise ValueError(
                        f"Invalid historical frame {len(self.offsets)}: {error}"
                    ) from error
                self.offsets.append(offset)
        if trace_path.exists() and len(self.events) != len(self.offsets) - 1:
            raise ValueError("Historical events do not match the frame endpoint")
        if not self.offsets:
            raise ValueError("Historical recording is empty")
        if (
            self.last_tick != episode["tick"]
            or frame["metrics"]["fulfilled"] != episode["fulfilled"]
        ):
            raise ValueError("Historical recording does not match the episode endpoint")

    @property
    def last_tick(self):
        return len(self.offsets) - 1

    def row(self, tick):
        if type(tick) is not int or not 0 <= tick <= self.last_tick:
            raise ValueError("tick is outside recorded trajectory")
        with self.path.open("rb") as stream:
            stream.seek(self.offsets[tick])
            frame = json.loads(stream.readline())
        if frame["tick"] != tick:
            raise ValueError("Historical recording changed after opening")
        return {
            "tick": tick,
            "state": self.project(frame),
            "historical_frame": frame,
            "events": self.events.get(tick, []),
        }

    def project(self, frame):
        storage = {}
        for resource in (*self.factory.buffers, *self.factory.inspection_stations):
            key = getattr(resource, "buffer_id", None) or resource.inspection_station_id
            values = frame["stores"].get(key, {}).get("slots", [])
            if isinstance(getattr(resource, "storage", None), PoolStorage):
                storage[key] = {"pool": [job for job in values if job is not None]}
            else:
                slots = getattr(resource, "slots", None) or resource.storage.slots
                if len(values) > len(slots):
                    raise ValueError(f"too many recorded slots for {key}")
                storage[key] = {
                    slot.slot_id: [values[index]]
                    if index < len(values) and values[index] is not None
                    else []
                    for index, slot in enumerate(slots)
                }
        machines = {}
        for key, data in frame["machines"].items():
            machines[key] = {
                "job": data["job_id"],
                "status": data["status"],
                "down": data["status"] == "DOWN",
                "mode": self.mode_names[data["mode"]]
                if isinstance(data.get("mode"), int)
                and 0 <= data["mode"] < len(self.mode_names)
                else None,
                "available_modes": self.mode_names,
                "elapsed": data["total"] - data["remaining"],
                "remaining": data["remaining"],
            }
        return {
            "agvs": {
                key: {**data, "cell": [data["x"], data["y"]], "job": data["job_id"]}
                for key, data in frame["agvs"].items()
            },
            "machines": machines,
            "stations": frame["inspections"],
            "storage": storage,
            "jobs": {
                key: {**data, "location": data["holder"]}
                for key, data in frame["jobs"].items()
            },
            "metrics": frame["metrics"],
            "rankings": {},
            "status": "historical recording",
        }


class HistoricalPlaybackWindow(PlaybackWindow):
    def __init__(self, recording):
        self.overlay = None
        super().__init__(recording.factory, playback=recording)
        self.overlay = self.state_layer
        self.show_row(self.cached_row(0))

    @staticmethod
    def delivered(state):
        return int(state["metrics"]["fulfilled"])

    def show_row(self, row):
        super().show_row(row)
        frame = row["historical_frame"]
        # Preserve the historical vocabulary and values rather than displaying
        # projection fields as though these were current simulator records.
        self.details.setPlainText(json.dumps(frame, indent=2))
        if self.overlay is not None:
            self.overlay.frame = frame
            self.overlay.update()

    def interpolate(self, state, following, alpha):
        super().interpolate(state, following, alpha)
        if self.overlay is not None:
            self.overlay.update()


def historical_playback_window(root, seed=24004):
    recording = HistoricalRecording(root, seed)
    app = QApplication.instance() or QApplication([])
    window = HistoricalPlaybackWindow(recording)
    window.show()
    app.exec()
