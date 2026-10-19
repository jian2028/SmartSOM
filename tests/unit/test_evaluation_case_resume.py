"""Small real rule workloads exercise durable cases independently of learners."""

import hashlib
import json
from contextlib import contextmanager
from dataclasses import replace
from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console

from smartsom import api
from smartsom.config.codec import canonical_json
from smartsom.config.experiment_v3 import prepare_v3
from smartsom.experiments import composable, evaluation_resume
from smartsom.telemetry.runtime import CURRENT, RuntimeDisplay

ROOT = Path(__file__).resolve().parents[2]


def prepared(tmp_path, count=5, initial=None):
    config = api.load_config(ROOT / "configs/test/runs/evaluate_all_rules.yaml")
    config.output.root = str(tmp_path)
    config.evaluation.replications = count
    if initial is not None:
        config.evaluation.initial_control = True
        config.evaluation.initial_control_replications = initial
    config.scenario_overrides["tick_limit"] = 12
    config.evaluation.record = False
    return prepare_v3(config, training=False)


def normalized(rows):
    return [{k: v for k, v in row.items() if k != "run_dir"} for row in rows]


def stop_after(count):
    def progress(event):
        if event["evaluation_finished"] == count:
            raise KeyboardInterrupt("injected after durable case commit")

    return progress


def watch_calls(monkeypatch):
    calls = []
    original = composable.policies_for

    def counted(*args, **kwargs):
        calls.append(kwargs.get("policy_seed"))
        return original(*args, **kwargs)

    monkeypatch.setattr(composable, "policies_for", counted)
    return calls


def test_three_of_five_reuses_only_verified_cases_and_matches_uninterrupted(
    tmp_path, monkeypatch
):
    frozen = prepared(tmp_path)
    target = tmp_path / "partial"
    with pytest.raises(KeyboardInterrupt):
        composable.evaluate(frozen, resume_dir=target, on_progress=stop_after(3))
    reference = composable.evaluate(frozen)
    calls = watch_calls(monkeypatch)
    stream = StringIO()
    view = RuntimeDisplay(
        kind="evaluation", quiet=True, console=Console(file=stream, width=80)
    )
    token = CURRENT.set(view)
    events = []
    original_update = view.update

    def capture(task, event, **kwargs):
        if "evaluation_historical_completed" in event:
            events.append(
                tuple(
                    event[key]
                    for key in (
                        "evaluation_historical_completed",
                        "evaluation_current_completed",
                        "evaluation_remaining",
                    )
                )
            )
        return original_update(task, event, **kwargs)

    monkeypatch.setattr(view, "update", capture)
    try:
        result = composable.resume(target)
        assert normalized(result.results) == normalized(reference.results)
        assert len(calls) == 2
        assert (3, 0, 2) in events and (3, 1, 1) in events and (3, 2, 0) in events
        assert json.loads((target / "summary.json").read_text()) == json.loads(
            (reference.run_dir / "summary.json").read_text()
        )
        view.console.print(view.render())
        assert "历史完成 3" in stream.getvalue() and "本次完成 2" in stream.getvalue()
        assert "总目标 5" in stream.getvalue()
        calls.clear()
        measured = view.execution_telemetry.native.snapshot()["execution_ticks"]
        repeated = composable.resume(target)
        assert not calls and normalized(repeated.results) == normalized(
            reference.results
        )
        assert (
            view.tasks["evaluation"]["values"]["evaluation_historical_completed"] == 5
        )
        assert view.tasks["evaluation"]["values"]["evaluation_current_completed"] == 0
        observation = view.execution_telemetry.native.snapshot()
        assert observation["reason"] is None
        assert observation["execution_ticks"] == measured
    finally:
        CURRENT.reset(token)


def test_interrupted_active_case_is_rerun_without_repeating_committed_prefix(
    tmp_path, monkeypatch
):
    frozen = prepared(tmp_path, 3)
    original = composable.BoundaryCoordinator.tick
    calls = 0

    def interrupted(coordinator):
        nonlocal calls
        calls += 1
        if calls == 15:  # first 12-tick case committed; second is in progress
            raise KeyboardInterrupt("mid-case")
        return original(coordinator)

    target = tmp_path / "partial"
    with monkeypatch.context() as patch:
        patch.setattr(composable.BoundaryCoordinator, "tick", interrupted)
        with pytest.raises(KeyboardInterrupt):
            composable.evaluate(frozen, resume_dir=target)
    executions = watch_calls(monkeypatch)
    result = api.resume(target)
    assert result.engineering_failures == 0 and len(executions) == 2
    assert len(list((target / "evidence").glob(".interrupted-case-0001-*"))) == 1
    assert len(list((target / "evidence/.case-resume").glob("case-*.json"))) == 3


@pytest.mark.parametrize("change", ["seed", "input", "target", "phase", "protocol"])
def test_mismatched_identity_fails_before_executing_or_reusing(
    tmp_path, monkeypatch, change
):
    frozen = prepared(tmp_path)
    target = tmp_path / "partial"
    with pytest.raises(KeyboardInterrupt):
        composable.evaluate(frozen, resume_dir=target, on_progress=stop_after(3))
    data = json.loads(frozen.config_json)
    cases = json.loads(frozen.evaluation_json)
    kwargs = {}
    if change == "seed":
        data["seed"] += 1
    elif change == "input":
        cases[0]["scenario"]["tick_limit"] += 1
    elif change == "target":
        data["evaluation"]["replications"] = 4
    elif change == "phase":
        kwargs["purpose"] = "initial_control"
    else:
        data["evaluation"]["full_replay"] = False
    changed = replace(
        frozen, config_json=canonical_json(data), evaluation_json=canonical_json(cases)
    )
    executions = watch_calls(monkeypatch)
    with pytest.raises(ValueError, match="identity|changed"):
        composable.evaluate(changed, resume_dir=target, **kwargs)
    assert not executions


def test_model_payload_is_hashed_from_actual_bytes_and_changed_model_is_rejected(
    tmp_path,
):
    from smartsom.domain.production_decisions import (
        ACTION_CONTRACT,
        OBSERVATION_CONTRACT,
    )
    from smartsom.domain.travel_time import physical_contract

    frozen = prepared(tmp_path)
    model = tmp_path / "model"
    model.mkdir()
    (model / "weights.pt").write_bytes(b"tiny metadata-only inference identity fixture")
    (model / "encoder.json").write_text("{}")
    metadata = dict(
        action_contract=ACTION_CONTRACT,
        observation_contract=OBSERVATION_CONTRACT,
        physical_contract=physical_contract(frozen.scenario),
        weights_file="weights.pt",
        weights_sha256=hashlib.sha256((model / "weights.pt").read_bytes()).hexdigest(),
        encoder_sha256=hashlib.sha256(
            (model / "encoder.json").read_bytes()
        ).hexdigest(),
    )
    (model / "model.json").write_text(json.dumps(metadata))
    policies = json.loads(frozen.policies_json)
    group = next(iter(policies))
    policies[group]["implementation"] = {
        "kind": "model",
        "model": {"source": str(model)},
    }
    policies[group]["resolved_model"] = {"source": str(model), "metadata": metadata}
    frozen = replace(frozen, policies_json=canonical_json(policies))
    cases = json.loads(frozen.evaluation_json)
    evaluation_resume.CaseStore(tmp_path / "journal", frozen, cases, "evaluation")
    (model / "weights.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="weights hash"):
        evaluation_resume.CaseStore(tmp_path / "journal", frozen, cases, "evaluation")
    metadata["weights_sha256"] = hashlib.sha256(b"changed").hexdigest()
    (model / "model.json").write_text(json.dumps(metadata))
    policies[group]["resolved_model"]["metadata"] = metadata
    changed = replace(frozen, policies_json=canonical_json(policies))
    with pytest.raises(ValueError, match="plan/identity changed"):
        evaluation_resume.CaseStore(tmp_path / "journal", changed, cases, "evaluation")


def test_final_and_initial_have_independent_targets_labels_and_receipts(
    tmp_path, monkeypatch
):
    frozen = prepared(tmp_path, 5, initial=2)
    final = composable.evaluate(frozen, resume_dir=tmp_path / "final")
    with pytest.raises(KeyboardInterrupt):
        composable.evaluate(
            frozen,
            purpose="initial_control",
            resume_dir=tmp_path / "initial",
            on_progress=stop_after(1),
        )
    calls = watch_calls(monkeypatch)
    initial = api.resume(tmp_path / "initial")
    assert len(calls) == 1 and len(initial.results) == 2 and len(final.results) == 5
    assert all(
        row["evaluation_label"] == "INITIAL-CONTROL"
        and row["ranking_eligible"] is False
        for row in initial.results
    )
    assert all("evaluation_label" not in row for row in final.results)
    for path in (tmp_path / "initial/evidence").glob("case-*/result.json"):
        assert json.loads(path.read_text())["ranking_eligible"] is False
    with pytest.raises(ValueError, match="requires"):
        prepared_options = frozen.config.evaluation.model_dump()
        prepared_options["initial_control"] = False
        type(frozen.config.evaluation).model_validate(prepared_options)


@pytest.mark.parametrize("count", [0, 5])
def test_all_committed_or_zero_target_rebuilds_missing_summary_without_simulation(
    tmp_path, monkeypatch, count
):
    frozen = prepared(tmp_path, count)
    target = tmp_path / "evaluation"
    if count:
        with pytest.raises(KeyboardInterrupt):
            composable.evaluate(
                frozen, resume_dir=target, on_progress=stop_after(count)
            )
        assert not (target / "summary.json").exists()
    else:
        composable.evaluate(frozen, resume_dir=target)
        (target / "summary.json").unlink()
    calls = watch_calls(monkeypatch)
    result = api.resume(target)
    assert (
        not calls
        and len(result.results) == count
        and (target / "summary.json").is_file()
    )


def test_failed_commit_and_half_written_temp_never_count_as_complete(
    tmp_path, monkeypatch
):
    frozen = prepared(tmp_path)
    target = tmp_path / "evaluation"
    original = evaluation_resume.durable_json
    commits = 0

    def fail(path, value):
        nonlocal commits
        if Path(path).name.startswith("case-"):
            commits += 1
            if commits == 4:
                raise OSError("injected commit failure")
        return original(path, value)

    with monkeypatch.context() as patch:
        patch.setattr(evaluation_resume, "durable_json", fail)
        with pytest.raises(OSError, match="commit failure"):
            composable.evaluate(frozen, resume_dir=target)
    (target / "evidence/.case-resume/.pending-crashed").write_text('{"result":')
    calls = watch_calls(monkeypatch)
    result = api.resume(target)
    assert len(result.results) == 5 and len(calls) == 2


def test_corrupt_receipt_and_duplicate_case_identity_fail_closed(tmp_path, monkeypatch):
    frozen = prepared(tmp_path)
    target = tmp_path / "evaluation"
    with pytest.raises(KeyboardInterrupt):
        composable.evaluate(frozen, resume_dir=target, on_progress=stop_after(3))
    receipt = next((target / "evidence/.case-resume").glob("case-*.json"))
    receipt.write_text('{"result":')
    calls = watch_calls(monkeypatch)
    with pytest.raises(ValueError, match="corrupt"):
        api.resume(target)
    assert not calls
    cases = json.loads(frozen.evaluation_json)
    with pytest.raises(ValueError, match="duplicate"):
        evaluation_resume.CaseStore(
            tmp_path / "duplicate", frozen, [cases[0], cases[0]], "evaluation"
        )


def test_exclusive_lock_rejects_thread_and_process_writers(tmp_path):
    import subprocess
    import sys

    with evaluation_resume.stage_lock(tmp_path):
        with pytest.raises(RuntimeError, match="already running"):
            with evaluation_resume.stage_lock(tmp_path):
                pass
        code = (
            "from smartsom.experiments.evaluation_resume import stage_lock; from pathlib import Path\nwith stage_lock(Path("
            + repr(str(tmp_path))
            + ")): pass"
        )
        child = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True
        )
        assert child.returncode != 0 and "already running" in child.stderr


def test_remaining_eta_prices_verified_sparse_obligations_and_fresh_phase_rates():
    from smartsom.telemetry.estimates import PhaseMeter, remaining_service

    plan = {
        "mode": "evaluation",
        "evaluation_horizons": [10, 20, 30, 40, 50],
        "control_names": [],
    }
    values = dict(
        evaluation_finished=3,
        evaluation_requested=5,
        evaluation_remaining_ticks=40,
        evaluation_historical_completed=3,
        evaluation_tick=0,
        evaluation_tick_limit=10,
        evaluation_case_active=True,
    )
    row = {"status": "running", "stage": "evaluation", "values": values}
    meter = PhaseMeter(minimum=1)
    meter.observe(row, now=0, wall_time=0)
    assert meter.snapshot()["execution_ticks"] == 0
    values["evaluation_tick"] = 10
    meter.observe(row, now=2, wall_time=2)
    observation = meter.snapshot()
    assert observation["execution_ticks"] == 10
    result = remaining_service(
        row, plan, observation, reference={"finalization": {"mean_seconds": 0}}
    )
    assert result["seconds"] == 6  # 30 remaining ticks / 5 ticks per second


def test_sparse_commits_retain_original_case_ordinals_seeds_and_order(
    tmp_path, monkeypatch
):
    frozen = prepared(tmp_path)
    target = tmp_path / "evaluation"
    reference = composable.evaluate(frozen, resume_dir=target)
    receipts = target / "evidence/.case-resume"
    missing = next(
        p
        for p in receipts.glob("case-*.json")
        if json.loads(p.read_text())["index"] == 1
    )
    missing.unlink()  # No receipt for original ordinal 1; later ordinals remain verified.
    calls = watch_calls(monkeypatch)
    result = api.resume(target)
    assert len(calls) == 1 and normalized(result.results) == normalized(
        reference.results
    )
    assert [row["replication"] for row in result.results] == list(range(5))


def test_atomic_replace_failure_preserves_prior_result_and_never_leaves_committed_half_json(
    tmp_path, monkeypatch
):
    destination = tmp_path / "receipt.json"
    evaluation_resume.durable_json(destination, {"value": "prior"})

    def fail(*_):
        raise OSError("replace interrupted")

    monkeypatch.setattr(evaluation_resume, "atomic_replace", fail)
    with pytest.raises(OSError, match="replace interrupted"):
        evaluation_resume.durable_json(destination, {"value": "replacement"})
    assert json.loads(destination.read_text()) == {"value": "prior"}
    assert set(tmp_path.iterdir()) == {destination}


def test_standalone_initial_eta_uses_only_initial_obligations_and_rate():
    from smartsom.telemetry.estimates import remaining_service

    row = {
        "status": "running",
        "stage": "evaluation",
        "values": {
            "evaluation_kind": "initial control",
            "evaluation_finished": 1,
            "evaluation_remaining_ticks": 12,
            "evaluation_case_active": False,
        },
    }
    plan = {
        "mode": "evaluation",
        "evaluation_phase": "initial",
        "evaluation_horizons": [12, 12, 12, 12, 12],
        "initial_control_horizons": [12, 12],
    }
    observation = {"phases": {"initial": {"rate": 6}}}
    result = remaining_service(
        row, plan, observation, reference={"finalization": {"mean_seconds": 0}}
    )
    assert result["phase_work"] == {"initial": 12}
    assert result["seconds"] == 2


def test_initial_plan_publication_interruption_can_resume_zero_committed_cases(
    tmp_path, monkeypatch
):
    frozen = prepared(tmp_path, 1)
    target = tmp_path / "evaluation"
    original = evaluation_resume.durable_json

    def interrupted(path, value):
        if Path(path).name == "plan.json":
            raise KeyboardInterrupt("before first case plan publish")
        return original(path, value)

    with monkeypatch.context() as patch:
        patch.setattr(evaluation_resume, "durable_json", interrupted)
        with pytest.raises(KeyboardInterrupt):
            composable.evaluate(frozen, resume_dir=target)
    assert not (target / "evidence/.case-resume/plan.json").exists()
    result = api.resume(target)
    assert len(result.results) == 1 and result.engineering_failures == 0


def test_successful_history_is_counted_even_with_sparse_cached_ordinals(
    tmp_path, monkeypatch
):
    frozen = prepared(tmp_path)
    cases = json.loads(frozen.evaluation_json)
    for case in cases:
        case["scenario"]["demands"] = []
    frozen = replace(frozen, evaluation_json=canonical_json(cases))
    target = tmp_path / "evaluation"
    composable.evaluate(frozen, resume_dir=target)
    receipts = target / "evidence/.case-resume"
    next(
        p
        for p in receipts.glob("case-*.json")
        if json.loads(p.read_text())["index"] == 1
    ).unlink()
    view = RuntimeDisplay(kind="evaluation", quiet=True)
    token = CURRENT.set(view)
    try:
        result = composable.resume(target)
        values = view.tasks["evaluation"]["values"]
        assert result.completed == 5
        assert values["evaluation_completed"] == 5
        assert values["evaluation_historical_completed"] == 4
        assert values["evaluation_current_completed"] == 1
    finally:
        CURRENT.reset(token)


def test_initial_only_target_has_its_own_display_total_when_final_is_zero(tmp_path):
    frozen = prepared(tmp_path, 0, initial=2)
    view = RuntimeDisplay(kind="evaluation", quiet=True)
    token = CURRENT.set(view)
    try:
        result = composable.evaluate(frozen, purpose="initial_control")
        assert len(result.results) == 2
        assert view.workflow["evaluation_cases"] == 2
        assert view.workflow["evaluation_horizons"] == [12, 12]
        assert view.workflow["evaluation_phase"] == "initial"
        assert view.tasks["evaluation"]["values"]["evaluation_target"] == 2
    finally:
        CURRENT.reset(token)


@pytest.mark.parametrize("purpose,count", [(None, 0), ("initial_control", 2)])
def test_zero_final_and_nonzero_initial_resume_and_rebuild_each_stage_without_simulation(
    tmp_path, monkeypatch, purpose, count
):
    frozen = prepared(tmp_path, 0, initial=2)
    target = tmp_path / "stage"
    calls = watch_calls(monkeypatch)
    first = composable.evaluate(frozen, purpose=purpose, resume_dir=target)
    assert len(first.results) == len(calls) == count
    calls.clear()
    repeated = api.resume(target)
    assert not calls and normalized(repeated.results) == normalized(first.results)
    summary = (target / "summary.json").read_bytes()
    (target / "summary.json").unlink()
    restored = api.resume(target)
    assert not calls and normalized(restored.results) == normalized(first.results)
    assert (target / "summary.json").read_bytes() == summary
    assert json.loads((target / "run.json").read_text()).get("purpose") == purpose
    assert len(list((target / "evidence/.case-resume").glob("case-*.json"))) == count


def tune_control_session(frozen, root, controls):
    from smartsom.experiments.tuning_session import AdaptiveSession

    session = object.__new__(AdaptiveSession)
    session.root = root
    session.prepared = frozen
    session.experiment_id = "entry"
    session.finalizer = None
    session.record = {"tuning": {"control_spec": controls}}
    session._runtime = lambda *args, **kwargs: None
    return session


def test_tune_cached_initial_control_verifies_receipts_and_publishes_history(
    tmp_path, monkeypatch
):
    frozen = prepared(tmp_path, 5, initial=2)
    session = tune_control_session(
        frozen,
        tmp_path / "attempt",
        {"names": ["initial"], "directory": str(tmp_path / "controls")},
    )
    monkeypatch.setattr(
        composable, "checkpoint_path", lambda *args: tmp_path / "unused-rule-checkpoint"
    )
    session._finish_evaluation()
    calls = watch_calls(monkeypatch)
    view = RuntimeDisplay(kind="train-evaluate", quiet=True)
    token = CURRENT.set(view)
    try:
        session._finish_evaluation()
        assert not calls
        progress = view.tasks["evaluation"]["values"]["evaluation_case_progress"][
            "initial_control"
        ]
        assert progress["evaluation_historical_completed"] == 2
        assert progress["evaluation_current_completed"] == 0
        assert progress["evaluation_remaining"] == 0
        controls = json.loads(
            (session.root / "evaluation/controls/initial.json").read_text()
        )
        assert all(
            row["ranking_eligible"] is False and "run_dir" not in row
            for row in controls
        )
        assert (
            tmp_path / "controls/.entry_initial.cases/.case-resume/plan.json"
        ).is_file()
    finally:
        CURRENT.reset(token)


@pytest.mark.parametrize("local_cached", [False, True])
@pytest.mark.parametrize("mismatch", [None, "input", "policy_seed"])
def test_waiting_initial_control_rechecks_new_shared_plan_under_lock(
    tmp_path, monkeypatch, mismatch, local_cached
):
    from smartsom.experiments import composable_study

    frozen = prepared(tmp_path, 0, initial=2)
    cases = json.loads(frozen.evaluation_json)
    config = json.loads(frozen.config_json)
    if mismatch == "input":
        cases[0]["scenario"]["tick_limit"] += 1
    elif mismatch == "policy_seed":
        config["seed"] += 1
    changed = replace(
        frozen,
        evaluation_json=canonical_json(cases),
        config_json=canonical_json(config),
    )
    controls = {"names": ["initial"], "directory": str(tmp_path / "controls")}
    publisher = tune_control_session(frozen, tmp_path / "publisher", controls)
    waiter = tune_control_session(changed, tmp_path / "waiter", controls)
    local = waiter.root / "evaluation/controls/initial.json"
    if local_cached:
        # Legacy local rows pass the old weak check even when world/policy differ.
        local.parent.mkdir(parents=True)
        local.write_text(
            json.dumps(
                [
                    dict(
                        case_id=case["case"],
                        replication=case["replication"],
                        seed=case["seed"],
                        engineering_failure=False,
                    )
                    for case in cases
                ]
            )
        )
    monkeypatch.setattr(
        composable, "checkpoint_path", lambda *args: tmp_path / "unused-rule-checkpoint"
    )
    shared = tmp_path / "controls/entry_initial.json"
    plan = tmp_path / "controls/.entry_initial.cases/.case-resume/plan.json"
    original_lock = composable_study._control_lock
    published = []

    @contextmanager
    def completes_before_waiter_acquires_lock(path):
        if not published:
            # Deterministic ordering: waiter reaches acquisition with no shared
            # cache, publisher finishes, then waiter obtains the actual OS lock.
            assert not plan.exists() and not shared.exists()
            published.append(True)
            publisher._finish_evaluation()
            assert plan.is_file()
            published.append(shared.read_bytes())
        with original_lock(path):
            yield

    monkeypatch.setattr(
        composable_study, "_control_lock", completes_before_waiter_acquires_lock
    )
    calls = watch_calls(monkeypatch)
    view = RuntimeDisplay(kind="train-evaluate", quiet=True)
    token = CURRENT.set(view)
    try:
        if mismatch:
            with pytest.raises(ValueError, match="plan/identity changed"):
                waiter._finish_evaluation()
        else:
            waiter._finish_evaluation()
            assert local.read_bytes() == shared.read_bytes()
            progress = view.tasks["evaluation"]["values"]["evaluation_case_progress"][
                "initial_control"
            ]
            assert progress["evaluation_historical_completed"] == 2
            assert progress["evaluation_current_completed"] == 0
            assert progress["evaluation_remaining"] == 0
        assert len(published) == 2 and shared.read_bytes() == published[1]
        assert len(calls) == 2  # Publisher only; waiter never simulates a case.
    finally:
        CURRENT.reset(token)
