"""Analytical definitions, bounded event state and read-only report entrypoint."""

import copy
import json
from types import SimpleNamespace

import pytest
import torch

from smartsom.experiments import diagnostic_report as reports
from smartsom.experiments import progress_diagnostics as progress
from smartsom.learning import production_diagnostics as learner_metrics
from smartsom.learning.decision_diagnostics import record_choice, summary


def test_decision_scores_distinguish_q_from_behavior_and_single_choice():
    scores = torch.tensor([10.0, 9.0, -1e30])
    dist = torch.distributions.Categorical(logits=scores)
    request = SimpleNamespace(
        role="dispatcher",
        owner="agv",
        stage="proposals",
        tick=4,
        observation={
            "agvs": {
                "agv": {
                    "job": None,
                    "travel": None,
                    "cell": [1, 2],
                    "target": {"port": "p"},
                }
            },
            "topology": {"ports": {"p": [1, 2]}},
        },
        candidates=[
            SimpleNamespace(legal=True, identity="TARGET:one"),
            SimpleNamespace(legal=True, identity="TARGET:two"),
            SimpleNamespace(legal=False, identity="TARGET:illegal"),
        ],
    )
    policy = SimpleNamespace(
        decision_diagnostics={},
        training=True,
        deterministic=True,
        metadata={"algorithm": "dqn"},
        epsilon=0.5,
    )
    for _ in range(1000):
        record_choice(policy, request, scores, dist, 1, True)
    result = summary(policy)
    metrics = result["strata"]["dispatcher:empty:arrived"]
    assert metrics["behavior_probability"]["mean"] == 0.25
    assert metrics["dqn_exploratory_branch"]["mean"] == 1
    assert metrics["ppo_probability"]["mean"] is None
    assert metrics["top1_top2_gap"]["mean"] == 1
    assert len(result["examples"]) == 8
    assert len(json.dumps(result)) < 16384
    request.candidates = request.candidates[:1]
    record_choice(
        policy,
        request,
        scores[:1],
        torch.distributions.Categorical(logits=scores[:1]),
        0,
        False,
    )
    assert (
        policy.decision_diagnostics["strata"]["dispatcher:empty:arrived"][
            "top1_top2_gap"
        ]["unknown"]
        == 1
    )


def test_minibatch_scales_clip_fraction_and_pooled_constant_target_ev():
    scalar = torch.tensor(0.0, requires_grad=True)
    obj = SimpleNamespace()
    learner_metrics.record_ppo(
        obj,
        scalar,
        scalar,
        scalar,
        scalar,
        torch.tensor([0.0, 0.5]),
        torch.ones(2),
        torch.ones(2),
        clip_range=0.2,
        raw_advantage=torch.tensor([-2.0, 6.0]),
        values=torch.tensor([0.0, 1.0]),
        returns=torch.tensor([2.0, 2.0]),
    )
    row = learner_metrics.summarize(obj.training_diagnostics)
    assert row["metrics"]["clip_fraction"]["mean"] == 0.5
    assert row["metrics"]["raw_advantage_mean"]["mean"] == 2
    assert row["value_explained_variance"] is None
    assert scalar.grad is None
    learner_metrics.record_dqn(
        obj,
        scalar,
        2,
        q=torch.tensor([1.0, 3.0]),
        target=torch.tensor([2.0, 1.0]),
        dt=torch.tensor([1, 3]),
        terminated=torch.tensor([True, False]),
        replay_age=torch.tensor([-1, 4]),
    )
    metrics = learner_metrics.summarize(obj.training_diagnostics)["metrics"]
    assert metrics["td_error_abs_mean"]["mean"] == 1.5
    assert metrics["terminal_fraction"]["mean"] == 0.5
    assert metrics["replay_age_insertions_mean"]["weight"] == 1
    assert metrics["replay_age_insertions_mean"]["mean"] == 4


def test_progress_snippets_resume_and_storage_are_bounded():
    state = progress.initial_state()
    sim = SimpleNamespace(tick=0, agvs={}, machine_state={}, shipped=set())
    coordinator = SimpleNamespace(sim=sim, records=[])
    for tick in range(1, 1000):
        sim.tick = tick
        progress.observe(state, coordinator, {"events": [], "rejections": {"a": "x"}})
        if tick == 128:
            resumed = copy.deepcopy(state)
        if tick > 128:
            progress.observe(
                resumed, coordinator, {"events": [], "rejections": {"a": "x"}}
            )
    assert state == resumed
    assert state["progress"]["pickup"]["open_gap_ticks"] == 999
    assert state["progress"]["pickup"]["right_censored"]
    assert len(state["snippets"]) == 8
    assert len(state["recent"]) == 4
    assert state["suppressed_triggers"] > 0
    assert len(json.dumps(state).encode()) < progress.EVENT_CAP


def test_report_pairs_components_cohorts_and_weighted_intervals():
    contract = {
        "shipment_weight": 2,
        "passing_weight": 1,
        "tardiness_weight": 3,
        "reference_jobs": 10,
        "reference_ticks": 100,
    }
    row = {
        "case_id": "0",
        "replication": 0,
        "seed": 3,
        "world_sha256": "same",
        "return": 1.5,
        "delivered": 5,
        "task_reward_contract": contract,
        "privileged_output_quality": {"shipped": 5, "passing_rate": 0.8},
        "accumulated_overdue_time": 100,
    }
    assert reports.reward_components(row)["sum"] == pytest.approx(1.5)
    other = dict(row, delivered=3)
    paired = reports.paired_differences([row], [other])
    assert paired["verified"] == 1
    assert paired["pairs"][0]["candidate_minus_baseline"]["delivered"] == 2
    assert (
        reports.paired_differences([row], [dict(other, world_sha256="wrong")])["pairs"][
            0
        ]["candidate_minus_baseline"]["return"]
        is None
    )
    with pytest.raises(ValueError, match="duplicate"):
        reports.paired_differences([row], [other, other])
    sim = SimpleNamespace(
        demands={"a": None, "b": None},
        shipped={"a"},
        released={"a"},
        _qualified_shipments={"a"},
    )
    cohort = reports.cohort_metrics(
        sim, {"pool": [{"id": "a", "novel": False}, {"id": "b", "novel": True}]}
    )
    assert cohort["novel"]["declared_demands"] == 1
    assert cohort["novel"]["released_demands"] == 0
    assert cohort["novel"]["passing_fraction"] is None
    assert reports.cohort_metrics(sim)["common"]["shipped"] is None
    history = [
        {
            "update": i,
            "learner_diagnostics": {
                "groups": {
                    "g": {"metrics": {"loss": {"mean": value, "weight": weight}}}
                }
            },
        }
        for i, value, weight in [(1, 2, 2), (2, 4, 6)]
    ]
    assert (
        reports.interval_diagnostics(history)[1]["groups"]["g"]["metrics"]["loss"][
            "mean"
        ]
        == 5
    )


def test_report_cli_reads_saved_data_and_never_overwrites(tmp_path, monkeypatch):
    case = tmp_path / "evaluation" / "case-0000"
    case.mkdir(parents=True)
    raw = '{"case_id":"0","replication":0,"seed":4,"return":2}'
    (case / "result.json").write_text(raw)
    out = tmp_path / "analysis.json"
    monkeypatch.setattr(
        "sys.argv", ["diagnostic_report", "--run", str(tmp_path), "--output", str(out)]
    )
    reports.main()
    report = json.loads(out.read_text())
    assert report["evaluation"]["cases"] == 1
    assert report["evaluation"]["rows"][0]["reward"]["status"] == "unavailable"
    assert (case / "result.json").read_text() == raw
    with pytest.raises(FileExistsError):
        reports.main()
    assert reports.build_report()["evaluation"]["zero_cases_failure"]


def test_batch04_adapter_verifies_commits_and_keeps_training_separate(tmp_path):
    import hashlib

    cell = tmp_path / "cell"
    suite = cell / "final-primary" / "world"
    suite.mkdir(parents=True)
    raw = b'{"case_id":"world","replication":0,"seed":2,"return":1}'
    (suite / "result.json").write_bytes(raw)
    row = json.loads(raw)
    committed = {
        "identity": {"schema": "batch04.evaluation-case/v2", "world_sha256": "world"},
        "row": row,
        "files": {"result.json": hashlib.sha256(raw).hexdigest()},
    }
    (suite / "committed.json").write_text(json.dumps(committed))
    (cell / "training-episodes.json").write_text(
        json.dumps(
            {"completed_objective_windows": [row], "partial_budget_episodes": []}
        )
    )
    report, sources = reports.batch04_report(cell)
    assert report["evaluation"]["cases"] == 1
    assert len(report["training_whole_episodes"]["rows"]) == 1
    assert report["development"] == []
    assert len(sources) == 3
    (suite / "result.json").write_text("changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        reports.batch04_report(cell)


def test_report_budget_rejects_oversized_content_before_write(tmp_path, monkeypatch):
    monkeypatch.setattr(reports, "REPORT_CAP", 400)
    path = tmp_path / "report.json"
    reports.write_report(path, {"examples": ["x" * 1000]})
    result = json.loads(path.read_text())
    assert "omissions" in result and "examples" not in result
    rejected = tmp_path / "rejected.json"
    with pytest.raises(ValueError, match="exceeds"):
        reports.write_report(rejected, {"metrics": "x" * 1000})
    assert not rejected.exists()


def test_parallel_score_reporting_keeps_sampler_scopes_separate():
    from smartsom.experiments.composable import decision_summaries

    policy = SimpleNamespace(decision_diagnostics={"decisions": 99})
    result = decision_summaries(
        {"g": policy},
        [
            {
                "g": {
                    "decision_diagnostics": {"decisions": 2},
                    "decision_history_complete": True,
                }
            },
            {
                "g": {
                    "decision_diagnostics": {"decisions": 3},
                    "decision_history_complete": True,
                }
            },
        ],
    )
    assert [s["groups"]["g"]["decisions"] for s in result["samplers"]] == [2, 3]


def test_legacy_inference_without_algorithm_keeps_unknown_metrics():
    request = SimpleNamespace(
        role="machine",
        owner="m",
        stage="proposals",
        tick=0,
        observation={},
        candidates=[SimpleNamespace(legal=True, identity="START:j")],
    )
    scores = torch.tensor([0.0])
    policy = SimpleNamespace(
        metadata={},
        training=False,
        deterministic=True,
        epsilon=0.0,
        decision_diagnostics={},
    )
    record_choice(
        policy,
        request,
        scores,
        torch.distributions.Categorical(logits=scores),
        0,
        False,
    )
    metrics = summary(policy)["strata"]["machine"]
    assert metrics["ppo_probability"]["mean"] is None
    assert metrics["dqn_epsilon"]["mean"] is None
    assert metrics["behavior_probability"]["mean"] == 1


def test_phase_denominators_and_legacy_progress_are_not_historical_zeroes():
    state = progress.initial_state()
    state["history_complete"] = False
    port = SimpleNamespace(cell=SimpleNamespace(x=1, y=2))
    vehicle = {
        "job": None,
        "target": {"owner": "source", "port": "p"},
        "cell": [1, 2],
        "travel": None,
        "service": None,
    }
    sim = SimpleNamespace(
        tick=201,
        agvs={"a": vehicle, "b": dict(vehicle)},
        machine_state={"m": {"status": "PROCESSING"}},
        shipped={"old"},
        protocol=SimpleNamespace(ports={"p": port}),
    )
    record = {
        "role": "dispatcher",
        "owner": "a",
        "proposal": {"owner": "source", "port": "q"},
        "candidate": "TARGET:source:q",
        "candidates": [{"identity": "TARGET:source:q", "features": [0] * 12}],
        "observation": {"agvs": sim.agvs, "sources": {"source": {"ready": []}}},
    }
    coordinator = SimpleNamespace(sim=sim, records=[record])
    progress.observe(state, coordinator, {"events": []})
    assert (
        state["counts"]["dispatch_empty_selected_source_no_ready_stock"]["numerator"]
        == 1
    )
    assert (
        state["counts"]["postcommit_arrived_idle_agv_shared_target_port"]["denominator"]
        == 2
    )
    assert state["counts"]["dispatch_retarget_existing_intention"]["numerator"] == 1
    assert state["progress"]["shipment"]["events"] == 0
    assert state["progress"]["shipment"]["open_gap_ticks"] == 1
    assert not state["history_complete"]


def test_destination_capacity_uses_selected_port_and_direct_machine():
    port = SimpleNamespace(
        bindings=[SimpleNamespace(operations=("drop_off",), target=("buffer", "a"))]
    )
    sim = SimpleNamespace(
        protocol=SimpleNamespace(ports={"p": port}),
        scrap={},
        machines={"m": object()},
        capacity={"buffer": {"a": 1, "b": 1}},
        _target=lambda target: target,
    )
    view = {
        "storage": {"buffer": {"a": ["occupied"], "b": []}},
        "machines": {"m": {"job": "busy"}},
    }
    assert (
        progress.destination_full(sim, view, {"owner": "buffer", "port": "p"}) is True
    )
    assert progress.destination_full(sim, view, {"owner": "m", "port": "p"}) is True
    view["machines"]["m"]["job"] = None
    assert progress.destination_full(sim, view, {"owner": "m", "port": "p"}) is False
    assert (
        progress.destination_full(sim, view, {"owner": "unknown", "port": "p"}) is None
    )


def test_first_resumed_tick_shipment_counts_after_pretransition_seed():
    state = progress.initial_state()
    state["history_complete"] = False
    sim = SimpleNamespace(tick=200, shipped={"old"}, agvs={}, machine_state={})
    progress.seed_restored_progress(state, sim)
    sim.tick += 1
    sim.shipped.add("new")
    progress.observe(state, SimpleNamespace(sim=sim, records=[]), {"events": []})
    assert state["progress"]["shipment"]["events"] == 1
    assert state["progress"]["shipment"]["last_tick"] == 201
    assert state["progress"]["shipment"]["open_gap_ticks"] == 0
    assert not state["history_complete"]


def test_report_cli_reads_and_writes_long_native_paths(tmp_path, monkeypatch):
    from pathlib import Path

    from smartsom._filesystem import native_path

    # Use an ordinary caller path, including when pytest has an extended base.
    root = Path(str(tmp_path).removeprefix("\\\\?\\")) / ("a" * 100) / ("b" * 100)
    disk = native_path(root)
    case = disk / "case-0000"
    case.mkdir(parents=True)
    (case / "result.json").write_text('{"case_id":"long-path"}')
    (disk / "config").mkdir()
    (disk / "config/experiment.json").write_text(
        '{"diagnostics":{"reports":{"interval_updates":2}}}'
    )
    (disk / "reports").mkdir()
    (disk / "reports/training.json").write_text('[{"update":1},{"update":2}]')
    monkeypatch.setattr(
        "sys.argv",
        ["report", "--run", str(root), "--output", str(root / "derived.json")],
    )
    reports.main()
    saved = json.loads((disk / "derived.json").read_text())
    assert saved["evaluation"]["cases"] == 1
    assert saved["interval_learner_diagnostics"][0]["updates_observed"] == 2
    with pytest.raises(FileExistsError):
        reports.main()
