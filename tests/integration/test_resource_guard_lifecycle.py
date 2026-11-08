"""Tiny official native runs retain committed work on cooperative guard failure."""

import hashlib
import json

import pytest

from smartsom.config.reservation import compile_reservation
from smartsom.experiments import reservation
from smartsom.experiments import resource_guard as guard
from tests.unit.test_algorithm_components import author, recipe
from tests.unit.test_resource_guard import LIMITS, SAFE


def prepare(tmp_path, *, task="train", recovery=False):
    path = author(tmp_path, recipe("dqn" if task == "train" else "rule"))
    changes = [
        "scenario.jobs.orders.total=2",
        "scenario.jobs.orders.initial=1",
        "scenario.jobs.arrivals.mean_interval_ticks=1",
        "scenario.jobs.processing.base_ticks={A: 1, B: 1, C: 1, D: 1}",
        "experiment.drain_frozen=true",
        "experiment.drain_ticks=12",
        "experiment.train_episodes=2",
        "experiment.training_conditions=[normal]",
        "experiment.periodic_evaluation.milestones=[]",
        "experiment.periodic_evaluation.modes=[argmax]",
        "experiment.run.methods=[configured]",
        "experiment.run.selection=argmax",
        "experiment.run.split=test",
        "experiment.run.action_seeds=1",
        "experiment.run.seconds=30",
        "experiment.run.max_failures=3",
        "experiment.test_cases=4",
    ]
    if task == "train":
        changes += [
            "experiment.final_evaluation=true",
            "experiment.checkpoint_every_episodes=1",
            "experiment.evaluation.checkpoints=[{episode: 2, modes: [argmax]}]",
            "algorithm.production.policies.dqn.learner.params.epochs=1",
            "algorithm.production.policies.dqn.learner.params.width=8",
            "algorithm.production.policies.dqn.learner.params.batch_size=4",
            "algorithm.production.policies.dqn.learner.params.replay_capacity=16",
        ]
    if recovery:
        candidate = dict(
            name="tiny",
            load="normal",
            orders=2,
            initial_orders=1,
            mean_interarrival=1,
            horizon_ticks=8,
            window_ticks=1,
            baseline_windows=3,
            minimum_window_deliveries=1,
            maximum_cv=1,
            maximum_relative_trend=1,
            faults=[dict(machine_id="machine_00", start_tick=3, duration_ticks=1)],
        )
        changes += [
            "experiment.evaluation.recovery.action=calibrate",
            f"experiment.evaluation.recovery.candidates={json.dumps([candidate])}",
        ]
    plan = compile_reservation(
        path,
        task=task,
        sets=changes,
        operations={"resources": {"guard": LIMITS.model_dump()}},
    )
    return reservation.allocate(plan, root=tmp_path / "run")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize(
    "failure_stage", ["checkpoint_publish", "checkpoint_committed"]
)
def test_checkpoint_guard_preserves_prior_commit_and_requires_explicit_retry(
    tmp_path, monkeypatch, failure_stage
):
    root = prepare(tmp_path)
    publishes = []
    prior = {}
    stages = []

    def observe(directory):
        stage = guard.CURRENT.get().stage
        stages.append(stage)
        if stage == failure_stage:
            publishes.append(stage)
            if len(publishes) == 3:  # Initial state, episode1, uncommitted episode2.
                pointer = root / "continuation/current.json"
                marker = json.loads(pointer.read_text())
                prior.update(
                    {pointer: sha(pointer), root / "last.pt": sha(root / "last.pt")}
                )
                prior.update(
                    {
                        root / "continuation" / marker["generation"] / name: digest
                        for name, digest in marker["files"].items()
                    }
                )
                return SAFE | {"worker_rss": 8 * guard.GIB}
        return SAFE

    monkeypatch.setattr(guard, "observe", observe)
    with pytest.raises(guard.ResourceGuardExceeded, match=failure_stage):
        reservation.execute_saved(root)
    assert len(json.loads((root / "attempts.json").read_text())) == 1
    assert json.loads((root / "progress.json").read_text())["status"] == "failed"
    assert all(sha(path) == digest for path, digest in prior.items())
    assert len((root / "episodes.jsonl").read_text().splitlines()) == (
        1 if failure_stage == "checkpoint_publish" else 2
    )
    assert {
        "model_build",
        "model_built",
        "train_rollout",
        "learner_update",
        "learner_updated",
    } <= set(stages)
    monkeypatch.setattr(
        guard, "observe", lambda root: stages.append(guard.CURRENT.get().stage) or SAFE
    )
    result = reservation.execute_saved(root)
    assert result["status"] == "completed"
    assert len((root / "episodes.jsonl").read_text().splitlines()) == 2
    assert all(
        sha(path) == digest
        for path, digest in prior.items()
        if path.name not in {"current.json", "last.pt"}
    )
    assert {
        "model_load",
        "model_loaded",
        "evaluation_model_load",
        "evaluation_model_loaded",
    } <= set(stages)
    assert len(list((root / "evaluation").rglob("case-*.json"))) == 4


@pytest.mark.parametrize("recovery", [False, True])
def test_evaluation_and_recovery_keep_committed_case_on_guard_failure(
    tmp_path, monkeypatch, recovery
):
    root = prepare(
        tmp_path, task="calibrate" if recovery else "evaluate", recovery=recovery
    )
    stages = []

    def observe(directory):
        stage = guard.CURRENT.get().stage
        stages.append(stage)
        return SAFE | ({"host_available": 0} if stage == "case_committed" else {})

    monkeypatch.setattr(guard, "observe", observe)
    with pytest.raises(guard.ResourceGuardExceeded, match="case_committed"):
        reservation.execute_saved(root)
    receipts = {path: sha(path) for path in (root / "case-commits").glob("case-*.json")}
    assert len(receipts) == 1
    assert len(json.loads((root / "attempts.json").read_text())) == 1
    assert ("recovery_model_build" if recovery else "evaluation_model_build") in stages
    assert ("recovery_rollout" if recovery else "evaluation_rollout") in stages
    monkeypatch.setattr(guard, "observe", lambda root: SAFE)
    assert reservation.execute_saved(root)["status"] == "completed"
    assert all(sha(path) == digest for path, digest in receipts.items())
    assert len(list((root / "case-commits").glob("case-*.json"))) == (
        1 if recovery else 4
    )


def test_training_rollout_guard_uses_existing_control_boundary(tmp_path, monkeypatch):
    from smartsom.experiments import reservation_runtime

    root = prepare(tmp_path)
    original = reservation_runtime.boundary

    def boundary(*args, **kwargs):
        worker = guard.CURRENT.get()
        if worker.stage == "train_rollout":
            worker.last = None  # Deterministic next periodic observation, no sleep.
            monkeypatch.setattr(guard, "observe", lambda root: SAFE | {"disk_free": 0})
        return original(*args, **kwargs)

    monkeypatch.setattr(guard, "observe", lambda root: SAFE)
    monkeypatch.setattr(reservation_runtime, "boundary", boundary)
    with pytest.raises(guard.ResourceGuardExceeded, match="train_rollout"):
        reservation.execute_saved(root)
    assert not (root / "episodes.jsonl").read_text().strip()
    assert (root / "continuation/current.json").is_file()
