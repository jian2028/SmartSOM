"""Grid reports derive intervals from records without a GUI dependency."""

from smartsom.api import load_config, prepare
from smartsom.experiments.production import execute
from smartsom.trace.production import Playback


def test_report_timeline_matches_hand_calculated_ticks_and_marks_partial_work(tmp_path):
    from smartsom.experiments.report import grid_timeline

    recipe = prepare(
        load_config("configs/runs/production_hand.yaml"), training=False
    ).resolved
    root = execute(
        recipe.scenario, recipe.algorithm, output_root=tmp_path, verbose=False
    )
    playback = Playback(root)
    trace = [playback.row(tick) for tick in range(1, 9)]
    timeline = grid_timeline(trace, initial_state=playback.manifest["initial_state"])
    processing = [row for row in timeline["intervals"] if row["kind"] == "processing"]
    assert [(row["start"], row["end"], row["complete"]) for row in processing] == [
        (3, 5, True)
    ]
    travel = [row for row in timeline["intervals"] if row["kind"] == "loaded travel"]
    assert [(row["start"], row["end"]) for row in travel] == [(1, 2), (6, 7)]
    for interval in timeline["intervals"]:
        assert (
            timeline["events"][interval["start_sequence"]]["time"] == interval["start"]
        )
        assert timeline["events"][interval["end_sequence"]]["time"] == interval["end"]
    partial = grid_timeline(trace[:4])
    assert partial["intervals"][-1]["kind"] == "processing"
    assert partial["intervals"][-1]["complete"] is False
