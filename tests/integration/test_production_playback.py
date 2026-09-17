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


def test_speed_button_cycles_without_mutating_recording(tmp_path):
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
    for label, interval in (
        ("2×", 50),
        ("4×", 25),
        ("10×", 10),
        ("Maximum", 1),
        ("0.25×", 400),
        ("0.5×", 200),
        ("1×", 100),
    ):
        window.speed.click()
        app.processEvents()
        assert window.speed.text() == f"Speed: {label}"
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
