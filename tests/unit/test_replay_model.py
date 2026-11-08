"""Replay adapters preserve evidence semantics without running policies."""

from types import SimpleNamespace

from smartsom.studio.replay_model import ReplayIndex, decision_rows


def decision(action=("b",)):
    return {
        "tick": 4,
        "decisions": [
            {
                "decision": {
                    "tick": 3,
                    "owner": "a",
                    "group": "dispatcher",
                    "stage": "pickup",
                    "candidates": [
                        {"identity": "first", "action": ("a",)},
                        {"identity": "second", "action": ("b",)},
                    ],
                },
                "action": action,
            }
        ],
    }


def test_current_decision_keeps_identity_and_does_not_invent_scores():
    values = decision_rows(decision(), "a", "ppo")
    assert values[0]["tick"] == 3 and values[0]["result_tick"] == 4
    assert [c["selected"] for c in values[0]["choices"]] == [False, True]
    assert all(c["legal"] and c["score"] is None for c in values[0]["choices"])
    assert decision_rows(decision(), "other") == []
    assert decision_rows(decision(), job="b") == values
    assert decision_rows(decision(), job="bb") == []
    assert decision_rows(decision("first"))[0]["choices"][0]["selected"]


def index():
    rows = []
    for tick in range(4):
        rows.append(
            {
                "tick": tick,
                "state": {
                    "agvs": {
                        "a": {"cell": [tick, 0], "target": {"port": "p"}},
                        "b": {"cell": [0, 1], "job": "j", "target": {"port": "p"}},
                    },
                    "machines": {"m": {"down": tick in (1, 2)}},
                    "completed": ["order"] if tick > 1 else [],
                    "jobs": {},
                },
                "actions": {"movers": [("a", "UP"), ("b", "LEFT")]},
                "rejections": {"agv:a": "conflict", "agv:b": "obstacle"}
                if tick == 1
                else {},
                "events": [{"kind": "quality_revealed", "machine": "m", "job": "j"}]
                if tick == 2
                else [],
            }
        )
    recording = SimpleNamespace(last_tick=3, manifest={}, row=lambda t: rows[t])
    factory = SimpleNamespace(
        ports=[SimpleNamespace(port_id="p", cell=SimpleNamespace(x=5, y=2))]
    )
    return ReplayIndex(recording, factory), rows


def test_conflicts_outages_same_tick_filter_targets_and_backward_history():
    model, rows = index()
    assert len(model.filtered(lane="movement")) == 1
    assert model.filtered("b", "movement") == []  # obstacle is not a collision
    outage = model.filtered("m", "outage")[0]
    assert (outage.tick, outage.end) == (1, 3)
    assert {e.lane for e in model.events if e.tick == 2} == {"inspection", "delivery"}
    assert model.targets(rows[1])[(5, 2)] == [
        ("a", "pickup", "p"),
        ("b", "drop-off", "p"),
    ]
    assert model.history("a", 1, 12) == [(0, 0), (1, 0)]
    assert model.history("a", 3, 0) == []


def test_central_recording_filters_exact_entity_ids():
    from smartsom.engine.central_control import CentralDecision, task
    from smartsom.trace.current import CurrentRecording

    candidate = task(
        "transport", job="xjob/attempt/1", vehicle="vehicle-1", destination="buffer-1"
    )
    recorder = CurrentRecording("unused")
    recorder.decision(CentralDecision(0, (candidate,), None), candidate.identity)
    row = {"tick": 1, "decisions": recorder.pending_decisions}
    assert decision_rows(row, owner="vehicle-1")
    assert decision_rows(row, owner="buffer-1")
    assert not decision_rows(row, owner="vehicle-2")
    assert decision_rows(row, job="xjob/attempt/1")
    assert not decision_rows(row, job="job/attempt/1")
