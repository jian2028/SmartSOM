"""Recorded grid commands preserve idle time; audit is separate from visual replay."""

import json

import pytest
from test_production_runtime import small_scenario
from test_static_engine import hand_commands

from smartsom.config.production import AlgorithmConfig, frozen_inputs
from smartsom.domain.production import JointCommand
from smartsom.engine.production import ProductionSimulator
from smartsom.trace.production import Playback, Recorder, audit, canonical, state_hash


def recorded_hand(root, idle=0):
    case = small_scenario()
    sim = ProductionSimulator(case)
    writer = Recorder(root, frozen_inputs(case, AlgorithmConfig()), sim.snapshot())
    for command in (*[JointCommand() for _ in range(idle)], *hand_commands()):
        writer.append(sim.step(command))
    writer.finish(sim.status)
    return sim


@pytest.mark.parametrize("idle", [0, 1, 5])
def test_recording_replays_explicit_idle_with_identical_terminal_state(tmp_path, idle):
    root = tmp_path / "run"
    sim = recorded_hand(root, idle)
    playback = Playback(root)
    assert playback.last_tick == sim.tick == 8 + idle
    assert playback.row(playback.last_tick)["state"] == sim.snapshot()
    assert audit(root)["status"] == "passed"
    starts = [
        e["tick"]
        for t in range(1, playback.last_tick + 1)
        for e in playback.row(t)["events"]
        if e["kind"] == "processing_started"
    ]
    assert starts == [3 + idle]


@pytest.mark.parametrize("field", ["command", "event", "state", "missing_tick"])
def test_semantic_audit_rejects_tampered_records_even_when_view_hash_is_resealed(
    tmp_path, field
):
    root = tmp_path / "run"
    recorded_hand(root)
    path = root / "trace.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if field == "command":
        rows[0]["actions"]["agvs"] = [["agv", "WAIT"]]
    elif field == "event":
        rows[0]["events"][0]["tick"] = 99
    elif field == "state":
        rows[0]["state"]["agvs"]["agv"]["cell"] = [2, 0]
        rows[0]["state_hash"] = state_hash(rows[0]["state"])
    else:
        del rows[2]
    path.write_text("".join(canonical(row) + "\n" for row in rows))
    with pytest.raises(ValueError):
        audit(root)
