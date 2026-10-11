"""Frozen observational settings, bounded capture and real learning invariance."""

import copy
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from smartsom import api
from smartsom.config.codec import primitive
from smartsom.config.diagnostics import DiagnosticsOptions, capture_identity
from smartsom.experiments import diagnostic_report as report
from smartsom.experiments import progress_diagnostics as progress
from smartsom.learning.production_provenance import execution_identity

ROOT = Path(__file__).resolve().parents[2]
CUSTOM = {
    "event_context": {
        "max_snippets": 2,
        "max_payload_bytes": 8192,
        "lookback_boundaries": 1,
        "stagnation_ticks": 2,
    },
    "reports": {"interval_updates": 2},
}


@pytest.mark.parametrize(
    "value",
    [
        {"enabled": False},
        {"profile": "off"},
        {"event_context": {"max_snippets": 0}},
        {"event_context": {"max_snippets": True}},
        {"event_context": {"max_snippets": 65}},
        {"event_context": {"max_payload_bytes": 4095}},
        {"event_context": {"max_payload_bytes": 1048577}},
        {"event_context": {"lookback_boundaries": 33}},
        {"event_context": {"stagnation_ticks": 0}},
        {"event_context": {"stagnation_ticks": 2.5}},
        {"reports": {"interval_updates": "2"}},
        {"reports": {"interval_updates": 0}},
    ],
)
def test_capture_options_reject_invalid_or_semantic_knobs(value):
    with pytest.raises(ValidationError):
        DiagnosticsOptions.model_validate(value)


def test_v3_capture_is_frozen_but_not_scientific_identity():
    config = api.load_config(ROOT / "configs/test/runs/evaluate_all_rules.yaml")
    plain = api.prepare(config, training=False)
    config.diagnostics = DiagnosticsOptions.model_validate(CUSTOM)
    changed = api.prepare(config, training=False)
    assert plain.scientific_sha256 == changed.scientific_sha256
    assert plain.policies_json == changed.policies_json
    assert (
        execution_identity(plain)["diagnostics"]
        != execution_identity(changed)["diagnostics"]
    )
    assert json.loads(changed.config_json)["diagnostics"] == CUSTOM
    legacy = replace(
        plain,
        config_json=json.dumps(
            {
                k: v
                for k, v in json.loads(plain.config_json).items()
                if k != "diagnostics"
            }
        ),
    )
    assert primitive(legacy.config.diagnostics) == primitive(DiagnosticsOptions())
    assert execution_identity(legacy)["diagnostics"] == capture_identity()


def test_v4_capture_propagates_without_changing_science(tmp_path):
    import runpy

    import yaml

    from smartsom.config.experiment_v4 import compile_experiment

    helpers = runpy.run_path(str(ROOT / "tests/integration/test_author_learning.py"))
    path = helpers["_inputs"](tmp_path / "inputs", "rllib")
    before = compile_experiment(path).entries[0].prepared
    data = yaml.safe_load(path.read_text())
    data["diagnostics"] = CUSTOM
    path.write_text(yaml.safe_dump(data))
    after = compile_experiment(path).entries[0].prepared
    assert before.scientific_sha256 == after.scientific_sha256
    assert primitive(after.config.diagnostics) == CUSTOM
    assert execution_identity(after)["diagnostics"] == capture_identity(CUSTOM)


def test_capture_trigger_budgets_and_changed_resume_segment():
    state = progress.initial_state(
        DiagnosticsOptions.model_validate(CUSTOM).event_context
    )
    sim = SimpleNamespace(tick=0, agvs={}, machine_state={}, shipped=set())
    coordinator = SimpleNamespace(sim=sim, records=[])
    for tick in range(1, 101):
        sim.tick = tick
        progress.observe(
            state, coordinator, {"events": [], "rejections": {"x": "test"}}
        )
    assert len(state["snippets"]) == 2 and state["snippet_bytes"] <= 8192
    assert len(state["recent"]) == 1 and state["suppressed_triggers"] == 98
    assert "pickup_no_progress_2_ticks" in state["snippets"][1]["triggers"]
    counts = copy.deepcopy(state["progress"])
    progress.configure_capture(state, DiagnosticsOptions().event_context, sim.tick)
    assert state["progress"] == counts and not state["snippets"] and not state["recent"]
    assert state["capture_changes"][-1]["discarded_snippets"] == 2
    for tick in range(101, 228):
        sim.tick = tick
        progress.observe(state, coordinator, {"events": []})
    assert not state["snippets"]
    sim.tick = 228
    progress.observe(state, coordinator, {"events": []})
    assert "pickup_no_progress_128_ticks" in state["snippets"][0]["triggers"]


def test_report_granularity_is_weighted_and_splits_capture_segments():
    history = [
        {
            "update": i,
            "diagnostic_capture": capture_identity(),
            "learner_diagnostics": {
                "groups": {
                    "g": {"metrics": {"loss": {"weight": w, "weighted_sum": total}}}
                }
            },
        }
        for i, w, total in [(1, 2, 4), (2, 6, 24), (3, 9, 36)]
    ]
    merged = report.interval_diagnostics(history, 2)
    assert len(merged) == 2
    assert merged[0]["update_start"] == 1 and merged[0]["update"] == 2
    assert merged[0]["groups"]["g"]["metrics"]["loss"] == {
        "mean": 4,
        "weight": 6,
        "status": "available",
    }
    history[1]["diagnostic_capture"] = capture_identity(CUSTOM)
    assert len(report.interval_diagnostics(history, 3)) == 3


@pytest.mark.parametrize("name", ["train_all_ppo", "train_all_dqn"])
def test_capture_changes_preserve_real_learning_and_resume(tmp_path, name):
    pytest.importorskip("ray")
    torch = pytest.importorskip("torch")
    from test_composable_learning import tiny

    from smartsom.experiments.composable import TrainingSession, allocate

    root, record, prepared = allocate(tiny(name, tmp_path, ticks=16), "training")
    first = TrainingSession(prepared, root, record)
    try:
        first.step_update()
        state = copy.deepcopy(first.state_dict())
        first.execute()
        actions = copy.deepcopy(first.actions)
        weights = {
            g: copy.deepcopy(p.network.state_dict())
            for g, p in first.policies.items()
            if hasattr(p, "network")
        }
        rng = copy.deepcopy(first.random.getstate())
    finally:
        first.close()
    data = json.loads(prepared.config_json)
    data["diagnostics"] = CUSTOM
    root, record, changed = allocate(
        replace(prepared, config_json=json.dumps(data)), "training"
    )
    second = TrainingSession(changed, root, record)
    try:
        second.restore(state)
        assert second.pickup_diagnostics[0].values["progress_state"]["capture_changes"]
        second.execute()
        assert second.actions == actions and second.random.getstate() == rng
        for g, expected in weights.items():
            assert all(
                torch.equal(v, second.policies[g].network.state_dict()[k])
                for k, v in expected.items()
            )
        assert record["execution_provenance"]["diagnostics"] == capture_identity(CUSTOM)
        assert (
            len(json.loads((root / "reports/learner-intervals.json").read_text())) == 2
        )
    finally:
        second.close()
