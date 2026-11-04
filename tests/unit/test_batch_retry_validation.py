"""Isolated preflight for failed task recovery; no actual workers are started."""

import hashlib
import json
import random

import pytest

from smartsom.config.codec import digest
from smartsom.experiments import (
    author_batch,
    batch_retry,
    queue_admission,
    reservation,
)


@pytest.fixture
def failed_task(tmp_path, monkeypatch):
    root = tmp_path / "batch"
    child = root / "trials" / "trial" / "seed-4103"
    generation = child / "continuation" / "00000162-test"
    generation.mkdir(parents=True)
    (child / "inputs").mkdir()
    (child / "inputs/experiment.yaml").write_text("run:\n  seed: 4103\n")
    state = {
        "rows": [{}] * 162,
        "ticks": 817568,
        "condition_rng": random.Random(7).getstate(),
    }
    (generation / "state.json").write_text(json.dumps(state))
    (generation / "model.pt").write_bytes(b"model")
    files = {
        name: hashlib.sha256((generation / name).read_bytes()).hexdigest()
        for name in ("state.json", "model.pt")
    }
    (child / "continuation/current.json").write_text(
        json.dumps({"generation": generation.name, "files": files})
    )
    child_plan = {"seed": 4103}
    row = {
        "id": "trial--seed-4103",
        "child": str(child.relative_to(root)),
        "task": "train",
        "child_plan_sha256": digest(child_plan),
    }
    plan = {"files": [row]}
    monkeypatch.setattr(author_batch, "load", lambda owner: (owner, plan, {}))
    monkeypatch.setattr(reservation, "load", lambda path: child_plan)
    monkeypatch.setattr(batch_retry, "_retry_settled", lambda path, job: None)
    state = {
        "tasks": {row["id"]: {"root": str(child), "status": "failed"}},
        "additions": {},
    }
    return root, child, row, state


def test_inspect_failed_recovery_preserves_checkpoint(failed_task):
    root, child, row, state = failed_task
    before = (child / "continuation/current.json").read_bytes()
    result = batch_retry.inspect_target(root, row["id"], state)
    assert (result["episodes"], result["ticks"]) == (162, 817568)
    assert (child / "continuation/current.json").read_bytes() == before


@pytest.mark.parametrize(
    "status", ["running", "paused", "deferred", "completed", "removed"]
)
def test_inspect_rejects_non_failed(failed_task, status):
    root, _, row, state = failed_task
    state["tasks"][row["id"]]["status"] = status
    with pytest.raises(ValueError, match="exact failed task"):
        batch_retry.inspect_target(root, row["id"], state)


@pytest.mark.parametrize("hold", ["paused_tasks", "stopped_tasks", "removed_tasks"])
def test_inspect_never_clears_task_holds(failed_task, hold):
    root, _, row, state = failed_task
    state[hold] = [row["id"]]
    with pytest.raises(ValueError, match="explicit stop, pause or removal hold"):
        batch_retry.inspect_target(root, row["id"], state)
    assert state[hold] == [row["id"]]


def test_inspect_rejects_corrupt_checkpoint_and_cleanup(failed_task):
    root, child, row, state = failed_task
    (child / "continuation/00000162-test/model.pt").write_bytes(b"bad")
    with pytest.raises(ValueError, match="checkpoint digest mismatch"):
        batch_retry.inspect_target(root, row["id"], state)
    state["cleanup_active"] = {"task_id": row["id"]}
    with pytest.raises(ValueError, match="cleanup in progress"):
        batch_retry.inspect_target(root, row["id"], state)


def test_inspect_does_not_reclaim_detached_owner(failed_task, tmp_path, monkeypatch):
    root, _, row, state = failed_task
    state["tasks"][row["id"]]["owner_root"] = str(tmp_path / "detached")
    monkeypatch.setattr(queue_admission, "snapshot", lambda _: {})
    with pytest.raises(ValueError, match="external owner is detached"):
        batch_retry.inspect_target(root, row["id"], state)


def test_owner_exit_check_is_required(failed_task, monkeypatch):
    root, _, row, state = failed_task

    def reject_unverified(child, job):
        raise ValueError("worker exit not verified")

    monkeypatch.setattr(batch_retry, "_retry_settled", reject_unverified)
    with pytest.raises(ValueError, match="exit not verified"):
        batch_retry.inspect_target(root, row["id"], state)


def test_real_settled_rejects_unfinalized_owner(tmp_path, monkeypatch):
    from smartsom.experiments import control

    child = tmp_path / "task"
    child.mkdir()
    monkeypatch.setattr(
        control, "read", lambda path: {"root": str(path), "finished_at": None}
    )
    monkeypatch.setattr(control, "processes", lambda: {})
    with pytest.raises(ValueError, match="ownership was not finalized"):
        batch_retry._retry_settled(child, {"status": "failed"})


def test_ended_batch_requires_official_driver(failed_task):
    root, _, row, state = failed_task
    state["status"] = "failed"
    with pytest.raises(ValueError, match="batch has ended"):
        batch_retry.inspect_target(root, row["id"], state)


def test_invalid_rng_is_rejected_even_when_hash_matches(failed_task):
    root, child, row, state = failed_task
    pointer_path = child / "continuation/current.json"
    pointer = json.loads(pointer_path.read_text())
    state_path = child / "continuation" / pointer["generation"] / "state.json"
    saved = json.loads(state_path.read_text())
    saved["condition_rng"] = [1, 2, 3]
    state_path.write_text(json.dumps(saved))
    pointer["files"]["state.json"] = hashlib.sha256(state_path.read_bytes()).hexdigest()
    pointer_path.write_text(json.dumps(pointer))
    with pytest.raises(ValueError, match="condition RNG state cannot be restored"):
        batch_retry.inspect_target(root, row["id"], state)


def test_wrong_child_output_is_rejected(failed_task, tmp_path):
    root, _, row, state = failed_task
    state["tasks"][row["id"]]["root"] = str(tmp_path / "other")
    with pytest.raises(ValueError, match="root differs"):
        batch_retry.inspect_target(root, row["id"], state)


def test_frozen_child_digest_mismatch_is_rejected(failed_task, monkeypatch):
    root, _, row, state = failed_task
    monkeypatch.setattr(reservation, "load", lambda path: {"seed": 4102})
    with pytest.raises(ValueError, match="frozen child plan differs"):
        batch_retry.inspect_target(root, row["id"], state)


def test_missing_pointer_never_constructs_new_model(failed_task):
    root, child, row, state = failed_task
    (child / "continuation/current.json").unlink()
    with pytest.raises(FileNotFoundError):
        batch_retry.inspect_target(root, row["id"], state)
    assert not (child / "continuation/current.json").exists()


def test_retry_accepts_verified_foreground_but_not_live_child(tmp_path, monkeypatch):
    from smartsom.experiments import control

    child = tmp_path / "task"
    child.mkdir()
    owner = {
        "root": str(child),
        "finished_at": 1,
        "ownership_finalized": True,
        "session_context": "host",
        "owner": {"pid": 20, "created": "a"},
        "members": [{"pid": 21, "created": "b"}],
        "deletion_blockers": ["run has no isolated driver session"],
    }
    monkeypatch.setattr(control, "read", lambda _: owner)
    monkeypatch.setattr(control, "session_context", lambda: "host")
    monkeypatch.setattr(control, "session_members", lambda owner, table: [])
    monkeypatch.setattr(control, "processes", lambda: {})
    batch_retry._retry_settled(child, {"pid": 20})
    monkeypatch.setattr(
        control, "processes", lambda: {21: {"pid": 21, "created": "b", "state": "S"}}
    )
    with pytest.raises(ValueError, match="still running"):
        batch_retry._retry_settled(child, {"pid": 20})


def test_batch_preview_is_read_only_and_reports_each_failure(failed_task, monkeypatch):
    root, child, row, state = failed_task
    other = "other--seed-4101"
    state["tasks"][other] = {"root": str(child), "status": "failed", "exit_code": 1}
    state["tasks"]["active"] = {"status": "running"}
    (child / "attempts.json").write_text(
        json.dumps([{"type": "OSError", "message": "disk"}])
    )
    monkeypatch.setattr(batch_retry.runtime_control, "snapshot", lambda root_: state)
    before = json.dumps(state, sort_keys=True)
    result = batch_retry.preview_failed(root)
    assert [item["task_id"] for item in result["targets"]] == [other, row["id"]]
    assert result["targets"][0]["check"] == "blocked"
    assert result["targets"][1]["check"] == "eligible_for_coordinator_revalidation"
    assert result["targets"][1]["last_recorded_attempt"]["message"] == "disk"
    assert json.dumps(state, sort_keys=True) == before
    with pytest.raises(ValueError, match="duplicate"):
        batch_retry.preview_failed(root, [row["id"], row["id"]])


@pytest.mark.parametrize(
    "task_status, addition_status", [("running", "failed"), ("failed", "running")]
)
def test_batch_preview_never_hides_conflicting_observation(
    failed_task, monkeypatch, task_status, addition_status
):
    root, _, row, state = failed_task
    task = state["tasks"][row["id"]]
    addition = dict(task)
    task["status"] = task_status
    addition["status"] = addition_status
    state["additions"][row["id"]] = addition
    monkeypatch.setattr(batch_retry.runtime_control, "snapshot", lambda _: state)
    item = batch_retry.preview_failed(root)["targets"][0]
    assert item["task_id"] == row["id"]
    assert item["check"] == "blocked"
    assert "state is changing" in item["reason"]


def test_batch_preview_bad_root_does_not_hide_other_task(failed_task, monkeypatch):
    root, _, row, state = failed_task
    state["tasks"]["first"] = {"status": "failed", "root": 123}
    monkeypatch.setattr(batch_retry.runtime_control, "snapshot", lambda _: state)
    targets = batch_retry.preview_failed(root)["targets"]
    assert targets[0]["task_id"] == "first" and targets[0]["check"] == "blocked"
    assert (
        targets[1]["task_id"] == row["id"]
        and targets[1]["check"] == "eligible_for_coordinator_revalidation"
    )


def test_one_process_lookup_failure_does_not_abort_batch_preview(
    failed_task, monkeypatch
):
    import subprocess

    root, child, row, state = failed_task
    state["tasks"]["first"] = {"status": "failed", "root": str(child)}
    monkeypatch.setattr(batch_retry.runtime_control, "snapshot", lambda _: state)
    original = batch_retry.inspect_target

    def inspect(root_, key, state_, **kwargs):
        if key == "first":
            raise subprocess.CalledProcessError(1, "ps")
        return original(root_, key, state_, **kwargs)

    monkeypatch.setattr(batch_retry, "inspect_target", inspect)
    targets = batch_retry.preview_failed(root)["targets"]
    assert targets[0]["check"] == "blocked"
    assert targets[1]["task_id"] == row["id"]
    assert targets[1]["check"] == "eligible_for_coordinator_revalidation"


def test_preview_includes_paused_when_selected_and_not_running_or_deferred(
    failed_task, monkeypatch
):
    root, _, row, state = failed_task
    job = state["tasks"][row["id"]]
    job["status"] = "paused"
    state["paused_tasks"] = [row["id"]]
    state["tasks"]["running"] = {"status": "running"}
    state["tasks"]["deferred"] = {"status": "deferred"}
    monkeypatch.setattr(batch_retry.runtime_control, "snapshot", lambda _: state)
    assert batch_retry.preview_failed(root)["targets"] == []
    targets = batch_retry.preview_failed(root, include_paused=True)["targets"]
    assert [item["task_id"] for item in targets] == [row["id"]]
    assert targets[0]["check"] == "eligible_for_coordinator_revalidation"
    assert targets[0]["action"] == "resume"
    assert state["paused_tasks"] == [row["id"]]
