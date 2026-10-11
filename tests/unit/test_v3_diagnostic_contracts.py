"""Diagnostics describe evidence without modifying actions, rewards or learning."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from smartsom.experiments.pickup_diagnostics import PickupDiagnostics
from smartsom.learning.production_collection import PhysicalCollector
from smartsom.learning.production_diagnostics import learning_coverage
from smartsom.learning.production_provenance import execution_provenance


def test_pickup_choice_is_not_a_service_failure_and_prefix_is_counted_once():
    decision = {
        "role": "dispatcher",
        "owner": "a",
        "candidate": "TARGET:empty",
        "observation": {
            "agvs": {"a": {"job": None}},
            "sources": {"ready": {"ready": ["j"]}},
        },
        "candidates": [
            {"identity": "TARGET:ready", "action": {"owner": "ready"}, "legal": True},
            {"identity": "TARGET:empty", "action": {"owner": "empty"}, "legal": True},
        ],
    }
    coordinator = SimpleNamespace(records=[decision], sim=SimpleNamespace(agvs={}))
    observer = PickupDiagnostics()
    observer.observe(coordinator, {"events": []})
    row = observer.summary()
    assert row["ready_source_choice_opportunities"] == 1
    assert row["nonready_source_choices_when_ready_available"] == 1
    assert row["admitted_service_slots"] == row["pickup_started"] == 0
    assert row["missed_pickup_boundaries"] is None
    assert row["first_reservation_tick"] is None
    assert row["arrival_to_pickup_ticks_mean"] is None
    coordinator.records = [{"role": "buffer", "owner": "ready", "count": 2}] * 2
    coordinator.sim.agvs = {"a": {"arrived_at": 4}, "b": {"arrived_at": 6}}
    observer.observe(
        coordinator,
        {
            "events": [
                {"kind": "pickup_started", "agv": "a", "tick": 8},
                {"kind": "pickup_started", "agv": "b", "tick": 8},
            ]
        },
    )
    row = observer.summary()
    assert row["admitted_service_slots"] == 2
    assert row["pickup_started"] == 2
    assert row["arrival_to_pickup_ticks_mean"] == 3
    assert row["arrival_to_pickup_ticks_max"] == 4
    size = len(json.dumps(row))
    for _ in range(1000):
        observer.observe(SimpleNamespace(records=[]), {"events": []})
    assert len(json.dumps(observer.summary())) == size


def test_critic_only_and_historical_q_coverage_are_not_actor_learning():
    state = {"metrics": {"actor_loss": {"weight": 0}, "value_loss": {"weight": 20}}}
    row = learning_coverage(state, "ppo", {"actor_packets": 0}, 5)
    assert row["optimizer_observed"]
    assert row["actor_training_exposures"] == 0
    assert row["critic_training_exposures"] == 20
    row = learning_coverage({"metrics": {"td_loss": {"weight": 8}}}, "dqn", {}, 1)
    assert row["q_training_exposures"] == 8
    assert row["q_choice_training_exposures"] is None
    assert not row["q_attribution_complete"]
    collector = PhysicalCollector(["g"], 1)
    old = collector.state_dict()
    old.pop("decision_coverage")
    old.pop("decision_coverage_complete")
    collector.load_state_dict(old)
    assert not collector.decision_coverage_complete


def test_q_coverage_reports_forced_choice_and_unknown_replay_rows():
    torch = pytest.importorskip("torch")
    from smartsom.learning.production_diagnostics import record_dqn

    learner = SimpleNamespace()
    record_dqn(learner, torch.tensor(0.1), 3, torch.tensor([1, 0, -1]))
    row = learning_coverage(learner.training_diagnostics, "dqn", {}, 1)
    assert row["q_choice_training_exposures"] == 1
    assert row["q_forced_training_exposures"] == 1
    assert row["q_unknown_training_exposures"] == 1
    assert not row["q_attribution_complete"]


def test_native_evaluation_diagnostics_preserve_exact_results(tmp_path, monkeypatch):
    from smartsom import api
    from smartsom.experiments import composable

    root = Path(__file__).resolve().parents[2]
    cfg = api.load_config(root / "configs/test/runs/evaluate_all_rules.yaml")
    cfg.scenario_overrides["tick_limit"] = 32
    cfg.evaluation.replications = 1
    cfg.evaluation.record = True
    cfg.evaluation.full_replay = True
    cfg.output.root = str(tmp_path)
    first = api.run(cfg)
    # Disable only the observer, leaving all native evaluation/recording intact.
    monkeypatch.setattr(PickupDiagnostics, "observe", lambda *args: None)
    second = api.run(cfg)

    def traces(run):
        return [
            json.loads(line)
            for f in sorted(run.rglob("trace.jsonl"))
            for line in f.read_text().splitlines()
        ]

    assert traces(first.run_dir) == traces(second.run_dir)

    def results(run):
        rows = []
        for f in sorted(run.rglob("result.json")):
            row = json.loads(f.read_text())
            row.pop("dispatcher_diagnostics", None)
            rows.append(row)
        return rows

    assert results(first.run_dir) == results(second.run_dir)
    assert results(first.run_dir)
    assert composable.PickupDiagnostics is PickupDiagnostics


@pytest.mark.parametrize("external_record", [False, True])
def test_native_model_provenance_and_resume_preflight(tmp_path, external_record):
    pytest.importorskip("ray")
    pytest.importorskip("torch")
    from test_composable_learning import tiny

    from smartsom.experiments.composable import TrainingSession, allocate

    with execution_provenance("test-harness", {"test.py": __file__}):
        prepared = tiny("train_all_ppo", tmp_path, ticks=8)
        root, record, prepared = allocate(prepared, "training")
    if external_record:
        # Tune supplies its own run record instead of the native allocator record.
        record.pop("execution_provenance")
    session = TrainingSession(prepared, root, record)
    try:
        assert record["execution_provenance"]["caller"]["files"]["test.py"]
        assert record["execution_provenance"]["effective_matching"] == "first_arrival"
        state = copy.deepcopy(session.state_dict())
        contracts = state["effective_model_contracts"]
        assert contracts
        assert all(c["prefix"]["width"] == 32 for c in contracts.values())
        broken = copy.deepcopy(state)
        next(iter(broken["effective_model_contracts"].values()))["prefix"]["width"] = 33
        with pytest.raises(ValueError, match="effective model"):
            session.restore(broken)
        assert session.ticks == 0
        old = copy.deepcopy(state)
        old.pop("effective_model_contracts")
        session.restore(old)
        assert session.ticks == 0
    finally:
        session.close()


def test_harness_scope_is_portable_and_does_not_leak():
    from smartsom.learning.production_provenance import execution_identity

    prepared = SimpleNamespace(
        composition_json='{"matching":{"name":"global_optimal"}}'
    )
    assert execution_identity(prepared)["caller_status"] == "not_declared"
    with execution_provenance("batch04", {"scripts/driver.py": __file__}):
        assert execution_identity(prepared)["caller_status"] == "declared"
    assert execution_identity(prepared)["caller_status"] == "not_declared"
    with pytest.raises(ValueError):
        with execution_provenance("bad", {"../driver.py": __file__}):
            pass


def test_prepared_caller_survives_serialization_without_changing_science():
    from dataclasses import asdict

    from smartsom import api
    from smartsom.config.experiment_v3 import PreparedComposition
    from smartsom.learning.production_provenance import execution_identity

    root = Path(__file__).resolve().parents[2]
    cfg = api.load_config(root / "configs/test/runs/evaluate_all_rules.yaml")
    plain = api.prepare(cfg, training=False)
    with execution_provenance("batch04", {"driver.py": __file__}):
        declared = api.prepare(cfg, training=False)
    restored = PreparedComposition(**json.loads(json.dumps(asdict(declared))))
    assert plain.scientific_sha256 == restored.scientific_sha256
    assert execution_identity(restored)["caller"]["name"] == "batch04"
    legacy = asdict(plain)
    legacy.pop("execution_provenance_json")
    assert execution_identity(PreparedComposition(**legacy))["caller"] is None
