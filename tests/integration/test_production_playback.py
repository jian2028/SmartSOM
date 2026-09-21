"""The native viewer projects recorded states; controls never own physics."""

import os
import threading

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtWidgets import QApplication

from smartsom.api import load_config, prepare
from smartsom.experiments.production import RunControls, execute
from smartsom.studio.playback import PlaybackWindow
from smartsom.studio.symbols import CELL_SIZE
from smartsom.trace.production import Playback

pytestmark = pytest.mark.studio


def test_seek_step_and_reverse_project_exact_states(tmp_path):
    app = QApplication.instance() or QApplication([])
    recipe = prepare(
        load_config("configs/runs/production_hand.yaml"), training=False
    ).resolved
    scenario, algorithm = recipe.scenario, recipe.algorithm
    root = execute(scenario, algorithm, output_root=tmp_path, verbose=False)
    playback = Playback(root)
    window = PlaybackWindow(scenario.factory, playback=playback)
    for tick in (0, 3, 4, 6, 8, 4, 1, 0):
        window.slider.setValue(tick)
        app.processEvents()
        state = playback.row(tick)["state"]
        assert window.current_tick == tick
        agv = window.scene.entity_items["agv"]
        assert [agv.x() / CELL_SIZE, agv.y() / CELL_SIZE] == state["agvs"]["agv"][
            "cell"
        ]
        assert agv.loaded == (state["agvs"]["agv"]["job"] is not None)
        machine = window.scene.entity_items["machine"]
        assert (machine.job is not None) == (
            state["machines"]["machine"]["job"] is not None
        )
    window.forward()
    assert window.current_tick == 1
    window.close()


def test_pause_single_step_and_detach_preserve_physical_result(tmp_path):
    recipe = prepare(
        load_config("configs/runs/production_hand.yaml"), training=False
    ).resolved
    scenario, algorithm = recipe.scenario, recipe.algorithm
    controls = RunControls()
    controls.pause(True)
    controls.delay = 0
    result = []
    thread = threading.Thread(
        target=lambda: result.append(
            execute(
                scenario,
                algorithm,
                output_root=tmp_path,
                verbose=False,
                controls=controls,
            )
        )
    )
    thread.start()
    import time

    deadline = time.monotonic() + 5
    while controls.latest is None and time.monotonic() < deadline:
        time.sleep(0.005)
    try:
        assert controls.latest["tick"] == 0
        controls.step()
        while controls.latest["tick"] == 0 and time.monotonic() < deadline:
            time.sleep(0.005)
        assert controls.latest["tick"] == 1 and controls.paused
        controls.detach()
        thread.join(5)
        assert not thread.is_alive() and controls.finished
        assert Playback(result[0]).last_tick == 8
    finally:
        controls.stop()
        thread.join(5)


def test_speed_selector_chooses_directly_without_mutating_recording(tmp_path):
    app = QApplication.instance() or QApplication([])
    recipe = prepare(
        load_config("configs/runs/production_hand.yaml"), training=False
    ).resolved
    root = execute(
        recipe.scenario, recipe.algorithm, output_root=tmp_path, verbose=False
    )
    recording = Playback(root)
    window = PlaybackWindow(recipe.scenario.factory, playback=recording)
    original = (root / "trace.jsonl").read_bytes()
    assert window.speed.currentText() == "Speed: 0.25×"
    assert not window.clock.playing
    for label, interval in (
        ("0.5×", 200),
        ("1×", 100),
        ("2×", 50),
        ("4×", 25),
        ("10×", 10),
        ("Maximum", 1),
        ("0.25×", 400),
    ):
        window.speed.setCurrentIndex(window.speed_labels.index(label))
        app.processEvents()
        assert window.speed.currentText() == f"Speed: {label}"
        assert window.clock.seconds_per_tick == interval / 1000
        assert window.timer.interval() == 16
        assert window.current_tick == 0
    assert (root / "trace.jsonl").read_bytes() == original
    window.close()


@pytest.fixture
def recorded_window(tmp_path):
    app = QApplication.instance() or QApplication([])
    recipe = prepare(
        load_config("configs/runs/production_hand.yaml"), training=False
    ).resolved
    root = execute(
        recipe.scenario, recipe.algorithm, output_root=tmp_path, verbose=False
    )
    window = PlaybackWindow(recipe.scenario.factory, playback=Playback(root))
    window.timer.stop()
    yield app, window
    window.close()


def test_fractional_motion_steps_and_discrete_details(recorded_window):
    _, window = recorded_window
    for tick in range(window.playback.last_tick):
        before = window.playback.row(tick)["state"]
        after = window.playback.row(tick + 1)["state"]
        window.clock.position = tick + 0.4
        window.render_position()
        assert window.current_tick == tick
        for key, data in before["agvs"].items():
            item = window.scene.entity_items[key]
            end = after["agvs"][key]["cell"]
            assert item.x() / CELL_SIZE == pytest.approx(
                data["cell"][0] * 0.6 + end[0] * 0.4
            )
            assert item.y() / CELL_SIZE == pytest.approx(
                data["cell"][1] * 0.6 + end[1] * 0.4
            )
            assert item.loaded == (data["job"] is not None)
        import json

        assert json.loads(window.details.toPlainText())["tick"] == tick
        window.backward()
        assert window.clock.position == tick
        window.clock.position = tick + 0.4
        window.render_position()
        window.forward()
        assert window.clock.position == tick + 1
        assert not window.playing


def test_scrub_cancels_playback_and_cache_avoids_frame_reads(recorded_window):
    _, window = recorded_window
    from unittest.mock import patch

    with patch.object(window.playback, "row", wraps=window.playback.row) as read:
        window.seek(3)
        reads = read.call_count
        for position in (3.1, 3.2, 3.4, 3.9):
            window.clock.position = position
            window.render_position()
        assert read.call_count == reads
        window.toggle()
        window.slider.setValue(2)
        assert not window.playing and window.clock.position == 2


def test_processing_interpolation_freezes_during_outage(recorded_window):
    from copy import deepcopy

    _, window = recorded_window
    state = window.playback.row(0)["state"]
    state = deepcopy(state)
    data = state["machines"]["machine"]
    data.update(job="example", elapsed=2, remaining=4, status="PROCESSING", down=False)
    following = deepcopy(state)
    following["machines"]["machine"].update(elapsed=3, remaining=3)
    window.interpolate(state, following, 0.5)
    job = window.scene.entity_items["machine"].job
    assert job.progress.remaining == 4 and job.display_remaining == 3.5
    data["down"] = True
    window.interpolate(state, following, 0.5)
    assert window.scene.entity_items["machine"].job.display_remaining == 4
    data["down"] = False
    following["machines"]["machine"]["job"] = "different"
    window.interpolate(state, following, 0.5)
    assert window.scene.entity_items["machine"].job.display_remaining == 4


def test_invalid_long_move_is_not_animated(recorded_window):
    from copy import deepcopy

    _, window = recorded_window
    state = window.playback.row(0)["state"]
    following = deepcopy(state)
    following["agvs"]["agv"]["cell"][0] += 2
    with pytest.raises(ValueError, match="Non-adjacent"):
        window.interpolate(state, following, 0.5)


def test_final_processing_tick_smoothly_finishes_after_release(recorded_window):
    from copy import deepcopy

    _, window = recorded_window
    state = deepcopy(window.playback.row(0)["state"])
    state["machines"]["machine"].update(
        job="example", status="PROCESSING", elapsed=3, remaining=1, down=False
    )
    following = deepcopy(state)
    following["machines"]["machine"].update(
        job=None, status="IDLE", elapsed=0, remaining=0
    )
    following["jobs"]["example"] = {"location": "postbuffer"}
    window.interpolate(state, following, 0.75)
    assert window.scene.entity_items["machine"].job.display_remaining == 0.25


def test_workspace_selection_filter_layout_and_raw_frame(recorded_window):
    _, window = recorded_window
    workspace = window.workspace
    workspace.resource_tree.setCurrentItem(workspace.tree_items["machine"])
    assert workspace.selected_id == "machine"
    assert window.scene.entity_items["machine"].isSelected()
    assert workspace.title.text() == "Machine"
    window.scene.entity_items["agv"].setSelected(True)
    workspace.select_resource("input")
    assert workspace.selected_id == "input"
    assert "defective" not in repr(workspace._state_values())
    workspace.filter.setText("machine")
    assert not workspace.tree_items["machine"].isHidden()
    assert workspace.tree_items["agv"].isHidden()
    workspace.filter.clear()
    workspace.set_layout("focus")
    assert workspace.resource_panel.isHidden()
    assert workspace.property_panel.isHidden()
    window.seek(3)
    assert workspace.selected_id == "input"
    assert workspace._state_values()["recorded_tick"] == 3
    workspace.set_layout("standard")
    assert workspace.resource_panel.isHidden()
    assert not workspace.property_panel.isHidden()
    assert workspace.splitter.indexOf(workspace.property_panel) == 2
    assert workspace.inspector.widget(2) is window.details
    assert window.details.isReadOnly()
    assert window.current_tick == 3


def test_live_window_keeps_existing_controls_and_layout(recorded_window):
    _, recording_window = recorded_window
    controls = RunControls()
    live = PlaybackWindow(recording_window.factory, controls=controls)
    assert live.workspace is None and live.clock is None
    assert live.timer.interval() == 100
    controls.latest = recording_window.playback.row(0)
    live.poll()
    assert live.current_tick == 0
    live.toggle()
    assert controls.paused
    live.close()


def test_job_transfers_use_one_path_and_seek_discards_transients(recorded_window):
    from copy import deepcopy

    _, window = recorded_window
    found = False
    for tick in range(window.playback.last_tick):
        row, following = window.playback.row(tick), window.playback.row(tick + 1)
        if not any(
            e["kind"] in ("pickup", "drop", "machine_released")
            for e in following["events"]
        ):
            continue
        original = deepcopy(row)
        window.clock.position = tick + 0.5
        window.render_position()
        paths = window.state_layer.paths()
        assert paths
        for jid, points in paths.items():
            assert len(points) >= 2
            assert jid in row["state"]["jobs"]
        assert row == original
        position = window.clock.position
        window.poll()
        assert window.clock.position == position  # starts paused
        window.seek(tick + 1)
        assert window.state_layer.alpha == 0
        assert window.state_layer.paths() == {}
        found = True
        break
    assert found


def test_dashboard_has_zero_denominator_and_locates_real_job(recorded_window):
    app, window = recorded_window
    window.show()
    app.processEvents()
    assert window.workspace.resource_panel.isHidden()
    assert "Output passing rate   —" in window.workspace.dashboard.summary.text()
    assert "From recording" in window.workspace.outside.text()
    jobs = window.workspace.dashboard.jobs
    assert jobs.count()
    item = jobs.item(0)
    window.workspace.dashboard.locate(item)
    assert window.workspace.selected_id is not None
    window.workspace._charging_preview(True)
    assert window.state_layer.charging_preview
    window.workspace._charging_preview(False)
    assert not window.state_layer.charging_preview
    window.grab()  # exercise the painter, including integer machine bars


def test_analysis_navigation_keeps_recorded_time_and_selection(recorded_window):
    from PySide6.QtCore import QPoint, Qt

    app, window = recorded_window
    window.show()
    workspace = window.workspace
    dashboard = workspace.dashboard
    window.seek(3)
    workspace.select_resource("machine")
    for index in range(5):
        dashboard.tabs.setCurrentIndex(index)
        app.processEvents()
        assert window.current_tick == 3
        assert workspace.selected_id == "machine"
    assert (
        window.slider.mapToGlobal(QPoint(0, 0)).y()
        < window.view.mapToGlobal(QPoint(0, 0)).y()
    )
    assert (
        dashboard.mapToGlobal(QPoint(0, 0)).y()
        >= window.view.mapToGlobal(QPoint(0, window.view.height())).y()
    )
    for index in range(dashboard.events.count()):
        event_tick = dashboard.events.item(index).data(Qt.ItemDataRole.UserRole)
        assert event_tick is None or event_tick <= 3
    dashboard.collapse.click()
    app.processEvents()
    assert dashboard.body.isHidden()
    assert dashboard.header_widget.isHidden()
    assert dashboard.collapse.isVisible()
    assert dashboard.height() <= 28
    dashboard.collapse.click()
    app.processEvents()
    assert not dashboard.body.isHidden()
    assert dashboard.tabs.currentIndex() == 4
    window.seek(0)
    assert dashboard.quality_chart.tick == 0
    assert workspace.selected_id == "machine"


@pytest.mark.parametrize("layout_index", [0, 1])
@pytest.mark.parametrize("size", [(1000, 620), (1420, 860)])
def test_analysis_collapse_releases_space_to_canvas(
    recorded_window, layout_index, size
):
    app, window = recorded_window
    window.timer.stop()
    window.resize(*size)
    window.show()
    workspace = window.workspace
    dashboard = workspace.dashboard
    splitter = workspace.analysis_splitter
    dashboard.tabs.setCurrentIndex(layout_index)
    app.processEvents()
    dashboard.collapse.setChecked(False)
    app.processEvents()
    expanded_sizes = splitter.sizes()
    expanded_canvas_height = window.view.height()

    for _ in range(2):
        dashboard.collapse.click()
        app.processEvents()
        assert splitter.sizes()[1] == 28
        assert window.view.height() == (expanded_canvas_height + expanded_sizes[1] - 28)
        assert dashboard.geometry().bottom() == splitter.contentsRect().bottom()
        assert dashboard.collapse.isVisible()
        dashboard.collapse.click()
        app.processEvents()
        assert splitter.sizes() == expanded_sizes
        assert window.view.height() == expanded_canvas_height
        assert dashboard.tabs.currentIndex() == layout_index


def test_analysis_switch_preserves_playback_zoom_and_chart_settings(recorded_window):
    app, window = recorded_window
    window.show()
    window.timer.stop()
    dashboard = window.workspace.dashboard
    assert dashboard.tabs.currentIndex() == 0
    window.seek(3)
    window.workspace.select_resource("machine")
    app.processEvents()
    window.view.zoom_by(1.2)
    transform = window.view.transform()
    window.clock.resume()
    dashboard.window_selector.setCurrentIndex(2)
    selected_window = dashboard.chart.window
    for index in (1, 2, 3, 4, 0):
        dashboard.tabs.setCurrentIndex(index)
        app.processEvents()
        assert window.clock.playing
        assert window.current_tick == 3
        assert window.workspace.selected_id == "machine"
        assert window.view.transform() == transform
        assert dashboard.chart.window == selected_window
    assert dashboard.chart.isVisible()
    assert dashboard.quality_chart.isVisible()
    assert dashboard.mode_chart.isVisible()
    dashboard.collapse.click()
    dashboard.tabs.setCurrentIndex(1)
    assert dashboard.body.isHidden()
    dashboard.collapse.click()
    dashboard.tabs.setCurrentIndex(0)
    dashboard.tabs.setCurrentIndex(4)
    assert dashboard.tabs.currentIndex() == 4
    window.clock.pause()


def test_compact_inspector_updates_and_does_not_expose_hidden_quality(recorded_window):
    app, window = recorded_window
    window.show()
    workspace = window.workspace
    workspace.select_resource("machine")
    inspector = workspace.runtime_inspector
    for tick in (0, 3, 0):
        window.seek(tick)
        app.processEvents()
        machine = window.playback.row(tick)["state"]["machines"]["machine"]
        assert f"Tick {tick}" in window.tick_label.text()
        assert machine["status"].title() in inspector.status.text()
        assert "defective" not in inspector.jobs.text()
        if machine.get("job"):
            assert inspector.progress.value() == machine["remaining"]
        else:
            assert inspector.progress.isHidden()
    workspace.select_resource(None)
    assert inspector.progress.isHidden()
    assert inspector.job_toggle.isHidden()
    assert "Click a machine" in inspector.summary.text()


def test_compact_drawer_preserves_selection_and_playback(recorded_window):
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest

    app, window = recorded_window
    window.timer.stop()
    window.show()
    window.seek(3)
    app.processEvents()
    workspace = window.workspace
    window.view.zoom_by(1.2)
    transform = window.view.transform()
    window.resize(1000, 620)
    app.processEvents()
    assert workspace.narrow
    assert workspace.property_panel.isHidden()
    assert workspace.dashboard.collapse.isChecked()
    workspace.select_resource("agv")
    app.processEvents()
    assert workspace.drawer_open and workspace.property_panel.isVisible()
    assert workspace.splitter.indexOf(workspace.property_panel) == -1
    assert workspace.property_panel.geometry().right() < workspace.width()
    assert workspace.property_panel.geometry().bottom() < workspace.height()
    window.activateWindow()
    workspace.drawer_close.setFocus()
    QTest.keyClick(workspace.drawer_close, Qt.Key.Key_Escape)
    app.processEvents()
    assert not workspace.drawer_open
    assert workspace.selected_id == "agv"
    workspace.inspector_action.trigger()
    assert workspace.drawer_open
    workspace.set_layout("focus")
    assert workspace.property_panel.isHidden()
    workspace.set_layout("standard")
    assert workspace.property_panel.isVisible()
    window.resize(1420, 860)
    app.processEvents()
    assert not workspace.narrow
    assert workspace.splitter.indexOf(workspace.property_panel) == 2
    assert workspace.property_panel.isVisible()
    assert window.current_tick == 3
    assert window.view.transform() == transform


def test_chart_ranges_values_and_pinned_readout_are_tick_bounded(recorded_window):
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtTest import QTest

    app, window = recorded_window
    window.timer.stop()
    window.show()
    dashboard = window.workspace.dashboard
    window.seek(window.playback.last_tick)
    app.processEvents()
    chart = dashboard.chart
    chart.visible_window = 3
    assert chart.start_tick == window.current_tick - 3
    assert chart.tick_at(-100) == chart.start_tick
    assert chart.tick_at(chart.width() + 100) == chart.tick
    assert (
        chart.formatted_value(chart.tick)
        == f"{window.evidence.throughput(chart.tick, 100):.3f} jobs/tick"
    )
    dashboard.window_selector.setCurrentIndex(0)
    dashboard.range_selector.setCurrentIndex(2)
    assert chart.window == 20
    assert chart.start_tick == 0
    assert dashboard.quality_chart.start_tick == 0
    assert window.state_layer.output_window == 20
    QTest.mouseClick(
        chart, Qt.MouseButton.LeftButton, pos=QPoint(chart.width() // 2, 55)
    )
    assert chart.pinned
    assert 0 <= chart.hover_tick <= chart.tick
    pinned = chart.hover_tick
    dashboard.tabs.setCurrentIndex(1)
    assert chart.hover_tick == pinned
    window.seek(0)
    assert chart.hover_tick is None and not chart.pinned
    assert chart.formatted_value(0) == "—"
    assert dashboard.quality_chart.formatted_value(0) == "—"


def test_analysis_arrow_strip_clicks_away_from_center(recorded_window):
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtTest import QTest

    app, window = recorded_window
    window.show()
    app.processEvents()
    dashboard = window.workspace.dashboard
    assert dashboard.collapse.width() >= dashboard.width() - 4
    QTest.mouseClick(dashboard.collapse, Qt.MouseButton.LeftButton, pos=QPoint(10, 10))
    assert dashboard.body.isHidden()
    QTest.mouseClick(dashboard.collapse, Qt.MouseButton.LeftButton, pos=QPoint(10, 10))
    assert not dashboard.body.isHidden()
