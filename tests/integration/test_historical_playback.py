"""Historical frames use the shared player without importing frozen code."""

import copy
import hashlib
import json
import os
import shutil

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtWidgets import QApplication

from smartsom.config.factory_design import load_factory_design
from smartsom.domain.factory_design import PoolStorage
from smartsom.studio.historical_playback import (
    HistoricalPlaybackWindow,
    HistoricalRecording,
)

pytestmark = pytest.mark.studio


@pytest.fixture
def historical(tmp_path):
    tmp_path = tmp_path / "rule"
    tmp_path.mkdir()
    shutil.copyfile("configs/factories/template1_demo.yaml", tmp_path / "factory.yaml")
    factory, _ = load_factory_design(tmp_path / "factory.yaml")
    stores = {}
    for item in (*factory.buffers, *factory.inspection_stations):
        key = getattr(item, "buffer_id", None) or item.inspection_station_id
        slots = getattr(item, "slots", ()) or getattr(
            getattr(item, "storage", None), "slots", ()
        )
        stores[key] = {"slots": [None] * len(slots), "beacon": None}
        if isinstance(getattr(item, "storage", None), PoolStorage):
            stores[key]["slots"] = [None] * 10
    frame = {
        "tick": 0,
        "agvs": {
            item.agv_id: {
                "x": item.initial_cell.x,
                "y": item.initial_cell.y,
                "job_id": None,
            }
            for item in factory.agvs
        },
        "stores": stores,
        "machines": {
            item.machine_id: {
                "job_id": None,
                "remaining": 0,
                "total": 0,
                "status": "IDLE",
            }
            for item in factory.machines
        },
        "inspections": {
            item.inspection_station_id: {
                "remaining": 0,
                "total": 2,
                "batch": [],
                "status": "IDLE",
            }
            for item in factory.inspection_stations
        },
        "jobs": {},
        "metrics": dict(
            released=20,
            fulfilled=0,
            submitted=0,
            output_failed=0,
            external_backlog=10,
            wip=10,
            conflicts=0,
        ),
    }
    frame["agvs"]["agv_001"]["job_id"] = "d0_a0"
    frame["jobs"]["d0_a0"] = dict(
        holder="agv_001", slot=None, quality="UNKNOWN", next_operation=0
    )
    second = copy.deepcopy(frame)
    second["tick"] = 1
    second["agvs"]["agv_001"]["x"] += 1
    second["jobs"]["d0_a0"]["quality"] = "PASS"
    frames = [frame, second]
    path = tmp_path / "frames-24004.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in frames))
    (tmp_path / "run.json").write_text(
        json.dumps(
            {
                "schema": "smartsom.spatial-run/v1",
                "factory_sha256": hashlib.sha256(
                    (tmp_path / "factory.yaml").read_bytes()
                ).hexdigest(),
            }
        )
    )
    (tmp_path / "episode-24004.json").write_text(
        json.dumps({"tick": 1, "fulfilled": 0})
    )
    original = tmp_path.parent / "original" / tmp_path.name
    original.mkdir(parents=True)
    shutil.copyfile(tmp_path / "factory.yaml", original / "factory.yaml")
    shutil.copyfile(tmp_path / "run.json", original / "run.json")
    return tmp_path, path, frames


def test_native_projection_keeps_original_details_and_moving_annotations(historical):
    root, path, frames = historical
    app = QApplication.instance() or QApplication([])
    original = path.read_bytes()
    recording = HistoricalRecording(root)
    window = HistoricalPlaybackWindow(recording)
    window.timer.stop()
    window.clock.position = 0.5
    window.render_position()
    item = window.scene.entity_items["agv_001"]
    assert item.x() == (frames[0]["agvs"]["agv_001"]["x"] + 0.5) * 40
    center = window.overlay.job_locations()["d0_a0"][0]
    assert center.x() == item.x() + 20
    assert json.loads(window.details.toPlainText()) == frames[0]
    assert "Outside queue: 10" in window.statusBar().currentMessage()
    assert window.overlay.frame["jobs"]["d0_a0"]["quality"] == "UNKNOWN"
    window.forward()
    assert json.loads(window.details.toPlainText()) == frames[1]
    assert path.read_bytes() == original
    app.processEvents()
    window.close()


@pytest.mark.parametrize("damage", ["empty", "gap", "partial", "json", "missing"])
def test_invalid_recording_is_rejected(historical, damage):
    root, path, frames = historical
    if damage == "empty":
        path.write_text("")
    elif damage == "gap":
        frames[1]["tick"] = 2
        path.write_text("".join(json.dumps(row) + "\n" for row in frames))
    elif damage == "partial":
        path.write_text(json.dumps(frames[0]))
    elif damage == "json":
        path.write_text("broken\n")
    else:
        del frames[0]["agvs"]
        path.write_text(json.dumps(frames[0]) + "\n")
    with pytest.raises(
        ValueError, match="Historical recording is empty|Invalid historical frame"
    ):
        HistoricalRecording(root)


def test_complete_lines_still_require_episode_endpoint(historical):
    root, path, frames = historical
    path.write_text(json.dumps(frames[0]) + "\n")
    with pytest.raises(ValueError, match="episode endpoint"):
        HistoricalRecording(root)


def test_wrong_factory_is_rejected(historical):
    root, _, _ = historical
    with (root / "factory.yaml").open("a") as stream:
        stream.write("\n# changed\n")
    with pytest.raises(ValueError, match="factory differs"):
        HistoricalRecording(root)


def test_inspection_progress_and_quality_are_separate(historical):
    root, _, _ = historical
    app = QApplication.instance() or QApplication([])
    recording = HistoricalRecording(root)
    window = HistoricalPlaybackWindow(recording)
    state = copy.deepcopy(recording.row(0)["state"])
    station = "inspection_station_001"
    state["storage"][station]["slot_001"] = ["d0_a0"]
    state["stations"][station].update(batch=["d0_a0"], remaining=2, total=2)
    following = copy.deepcopy(state)
    following["stations"][station]["remaining"] = 1
    window.interpolate(state, following, 0.5)
    job = window.scene.entity_items[station].slot_jobs["slot_001"]
    assert job.display_remaining == 1.5 and job.progress.remaining == 2
    assert window.overlay.frame["jobs"]["d0_a0"]["quality"] == "UNKNOWN"
    following["stations"][station]["batch"] = ["other"]
    window.interpolate(state, following, 0.5)
    assert (
        window.scene.entity_items[station].slot_jobs["slot_001"].display_remaining == 2
    )
    app.processEvents()
    window.close()
