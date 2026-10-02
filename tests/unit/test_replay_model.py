"""Replay adapters preserve evidence semantics without running policies."""

from types import SimpleNamespace

from smartsom.studio.replay_model import ReplayIndex, decision_rows


def decision(scores=None, mask=(True, True), **extra):
    return {
        "tick": 4,
        "decisions": [
            {
                "owner": "a",
                "role": "mover",
                "candidates": ["UP", "DOWN"],
                "mask": mask,
                "scores": scores,
                "selected_index": 1,
                **extra,
            }
        ],
    }


def test_probability_requires_complete_known_policy_scores():
    assert (
        decision_rows(decision(log_probability=-0.3), "a", "ppo")[0]["score_kind"]
        == "Action probabilities not recorded"
    )
    ppo = decision_rows(decision([0, 0]), "a", "ppo")[0]
    assert [c["score"] for c in ppo["choices"]] == [0.5, 0.5]
    assert ppo["choices"][1]["selected"]
    for scores, mask in (([0, None], [True, True]), ([0, 0], [])):
        incomplete = decision_rows(decision(scores, mask), "a", "ppo")[0]
        assert not incomplete["score_kind"].startswith("Policy probability")
    q = decision_rows(decision([2, 4]), "a", "dqn")[0]
    assert q["score_kind"] == "Q value"
    assert [c["score"] for c in q["choices"]] == [2, 4]


def test_stages_keep_order_and_do_not_convert_single_log_probability():
    row = {
        "tick": 8,
        "decisions": [
            dict(
                owner="a",
                role=role,
                tick=7,
                stage=i,
                candidate="go",
                candidates=[{"identity": "go", "legal": True}],
                log_probability=0,
            )
            for i, role in enumerate(("dispatcher", "mover"))
        ],
    }
    decisions = decision_rows(row, "a", "composable")
    assert [d["role"] for d in decisions] == ["dispatcher", "mover"]
    assert all(d["tick"] == 7 and d["result_tick"] == 8 for d in decisions)
    assert all(d["choices"][0]["score"] is None for d in decisions)


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
                "events": [{"kind": "inspection_result", "machine": "m", "job": "j"}]
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
    rows[1]["state"]["agvs"]["a"]["travel"] = {}
    assert model.history("a", 1, 12) == []
    assert model.history("a", 3, 0) == []


def test_legacy_actor_and_padded_logits_keep_entity_identity():
    row = {
        "tick": 2,
        "decisions": [
            {
                "actor": "agv:a",
                "candidates": ["UP", "WAIT"],
                "mask": [True, True, False],
                "scores": [0, 0, None],
                "selected_index": 0,
            }
        ],
    }
    values = decision_rows(row, "a", "maskable_ppo")
    assert len(values) == 1 and values[0]["role"] == "mover"
    assert values[0]["choices"][0]["score"] == 0.5


def test_order_decisions_follow_recorded_candidate_identity():
    row = {
        "tick": 3,
        "decisions": [
            {
                "owner": "buffer1",
                "role": "buffer",
                "candidates": [{"identity": "job1", "action": "job1", "legal": True}],
            },
            {
                "owner": "machine1",
                "role": "machine",
                "candidates": [
                    {
                        "identity": "START:job1:normal",
                        "action": ["job1", "normal"],
                        "legal": True,
                    }
                ],
            },
        ],
    }
    assert [d["owner"] for d in decision_rows(row, "old_holder", job="job1")] == [
        "buffer1",
        "machine1",
    ]
    assert decision_rows(row, job="job10") == []


def test_missing_scores_preserve_provider_semantics():
    assert (
        decision_rows(decision(), "a", "dqn")[0]["score_kind"]
        == "Action Q values not recorded"
    )
    assert (
        decision_rows(decision(), "a", None)[0]["score_kind"]
        == "Action scores not recorded"
    )
