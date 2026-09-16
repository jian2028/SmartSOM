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
