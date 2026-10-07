"""Replay shows task performance beside the map and seeks recorded markers."""

import importlib.util
import os

import pytest

if importlib.util.find_spec("PySide6") is None:
    if os.environ.get("SMARTSOM_REQUIRE_STUDIO") == "1":
        raise ImportError("Studio acceptance requires uv sync --locked --extra studio")
    pytest.skip("optional studio extra is not installed", allow_module_level=True)

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from smartsom.api import load_config, prepare  # noqa: E402
from smartsom.experiments.production import execute  # noqa: E402
from smartsom.studio.playback import PlaybackWindow  # noqa: E402
from smartsom.trace.performance import TaskPerformance  # noqa: E402
from smartsom.trace.production import Playback  # noqa: E402

pytestmark = pytest.mark.studio


@pytest.fixture(scope="module")
def application():
    return QApplication.instance() or QApplication([])


@pytest.fixture(scope="module")
def recorded(tmp_path_factory):
    """One real recorded run, so the panel is checked against actual evidence."""
    recipe = prepare(
        load_config("configs/test/runs/template1_static.yaml"), training=False
    ).resolved
    directory = execute(
        recipe.scenario,
        recipe.algorithm,
        output_root=tmp_path_factory.mktemp("runs"),
        verbose=False,
    )
    return Playback(directory)


def window(application, recording):
    from smartsom.config.production import scenario_from_snapshot

    factory = scenario_from_snapshot(recording.manifest["inputs"]["scenario"]).factory
    return PlaybackWindow(factory, playback=recording)


def cell(panel, label, column):
    return panel.cells[label, column].text()


def test_panel_sits_beside_the_map_and_matches_the_shared_definitions(
    application, recorded
):
    player = window(application, recorded)
    workspace = player.workspace
    panel = workspace.performance_panel
    assert panel.parent() is workspace.property_panel

    player.seek(recorded.last_tick)
    expected = TaskPerformance(recorded).cumulative(recorded.last_tick)
    assert cell(panel, "Qualified jobs", "total") == str(expected["qualified"])
    assert cell(panel, "Throughput", "total") == f"{expected['throughput']:.3f}"
    assert cell(panel, "Total tardiness", "total") == (
        f"{expected['total_tardiness']} ticks"
    )
    player.close()


def test_window_column_follows_the_selected_calculation_window(application, recorded):
    player = window(application, recorded)
    workspace = player.workspace
    player.seek(recorded.last_tick)
    assert workspace.performance_panel.window_heading.text() == "Last 100"

    workspace.dashboard.window_selector.setCurrentIndex(0)
    workspace.update_row(player.cached_row(recorded.last_tick))
    assert workspace.performance_panel.window_heading.text() == "Last 20"
    player.close()


def test_marker_arrows_seek_recorded_ticks_without_wrapping(application, recorded):
    player = window(application, recorded)
    workspace = player.workspace
    workspace.marker_selector.setCurrentIndex(1)
    ticks = player.evidence.marker_ticks("delivery")
    assert ticks, "the reference run delivers at least one qualified job"

    player.seek(0)
    workspace.sync_markers()
    assert workspace.jump_back.isEnabled() is False
    assert workspace.marker_label.text() == f"0/{len(ticks)}"

    workspace.jump(1)
    assert player.current_tick == ticks[0]
    workspace.jump(-1)
    assert player.current_tick == ticks[0], "no wrap before the first marker"

    player.seek(recorded.last_tick)
    workspace.sync_markers()
    assert workspace.jump_forward.isEnabled() is False
    workspace.jump(1)
    assert player.current_tick == recorded.last_tick
    player.close()


def test_missing_reference_leaves_the_bound_column_empty(application, recorded):
    player = window(application, recorded)
    panel = player.workspace.performance_panel
    panel.reference = None
    player.seek(recorded.last_tick)
    assert cell(panel, "Throughput", "bound") == "Not recorded"
    assert "No declared reference" in panel.bound_tooltip()
    player.close()


def test_zero_denominators_are_not_missing_evidence(application, recorded):
    player = window(application, recorded)
    panel = player.workspace.performance_panel
    player.seek(0)
    assert cell(panel, "Throughput", "total") == "No elapsed ticks"
    assert cell(panel, "Passing rate", "total") == "No submissions"
    assert cell(panel, "Passing rate", "bound") == "Not applicable"
    assert panel.reference_note.isHidden()
    panel.reference_toggle.click()
    assert not panel.reference_note.isHidden()
    player.close()


def test_compact_timeline_leaves_room_for_tick_labels(application, recorded):
    from smartsom.studio.replay_model import ReplayIndex
    from smartsom.studio.replay_timeline import EventTimeline

    player = window(application, recorded)
    timeline = EventTimeline(ReplayIndex(recorded, player.factory))
    timeline.setMinimumHeight(0)
    timeline.resize(800, 125)
    assert timeline.plot().bottom() + 25 <= timeline.height()
    player.close()
