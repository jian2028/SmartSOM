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
    assert window.statusBar().currentMessage() == "Paused · Read-only"
    assert "Outside 10" in window.workspace.dashboard.inventory.text()
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


def test_historical_inspector_preserves_ids_quality_and_original_frame(historical):
    root, _, frames = historical
    app = QApplication.instance() or QApplication([])
    window = HistoricalPlaybackWindow(HistoricalRecording(root))
    window.workspace.select_resource("agv_001")
    values = window.workspace._state_values()
    assert values["jobs"]["d0_a0"]["quality"] == "UNKNOWN"
    assert values["agvs"]["job_id"] == "d0_a0"
    window.workspace.set_layout("focus")
    window.forward()
    assert window.workspace._state_values()["jobs"]["d0_a0"]["quality"] == "PASS"
    assert json.loads(window.details.toPlainText()) == frames[1]
    assert window.overlay.frame == frames[1]
    app.processEvents()
    window.close()


def test_historical_modes_use_config_and_incomplete_event_stream_is_rejected(
    historical,
):
    root, _, frames = historical
    (root / "config.json").write_text(
        json.dumps({"environment": {"mode_scales": [1.25, 1.0, 0.8]}})
    )
    recording = HistoricalRecording(root)
    frame = copy.deepcopy(frames[0])
    frame["machines"]["machine_001"]["mode"] = 2
    assert recording.project(frame)["machines"]["machine_001"]["mode"] == "fast"
    trace = root / "trace-24004.jsonl"
    trace.write_text("")
    with pytest.raises(ValueError, match="events do not match"):
        HistoricalRecording(root)
    trace.write_text(json.dumps({"tick": 2, "events": []}) + "\n")
    with pytest.raises(ValueError, match="contiguous"):
        HistoricalRecording(root)
    trace.write_text(json.dumps({"tick": 1, "events": []}) + "\n")
    assert HistoricalRecording(root).row(1)["events"] == []


def test_machine_modes_and_inventory_have_separate_views(historical):
    root, _, _ = historical
    app = QApplication.instance() or QApplication([])
    window = HistoricalPlaybackWindow(HistoricalRecording(root))
    window.show()
    app.processEvents()
    dashboard = window.workspace.dashboard
    dashboard.tabs.setCurrentIndex(2)
    app.processEvents()
    assert dashboard.mode_chart.isVisible()
    assert not dashboard.jobs.isVisible()
    assert len(dashboard.mode_chart.names) == 4
    dashboard.tabs.setCurrentIndex(3)
    app.processEvents()
    assert dashboard.jobs.isVisible()
    assert not dashboard.mode_chart.isVisible()
    window.close()


def test_historical_inspector_decodes_actions_and_keeps_unknowns(historical):
    root, path, frames = historical
    frames[0]["agvs"]["agv_001"].update(
        previous_action=0, previous_outcome="SUCCESS", battery=100.0
    )
    frames[1]["agvs"]["agv_001"].update(previous_action=99)
    path.write_text("".join(json.dumps(frame) + "\n" for frame in frames))
    original = path.read_bytes()
    app = QApplication.instance() or QApplication([])
    window = HistoricalPlaybackWindow(HistoricalRecording(root))
    window.timer.stop()
    window.workspace.select_resource("agv_001")
    inspector = window.workspace.runtime_inspector
    assert ">Up<" in inspector.feedback.text()
    assert "100%" in inspector.summary.text()
    assert "Attempt 1" in inspector.summary.text()
    window.seek(1)
    assert "Unknown (99)" in inspector.feedback.text()
    assert "Not recorded" in inspector.summary.text()
    assert path.read_bytes() == original
    app.processEvents()
    window.close()
