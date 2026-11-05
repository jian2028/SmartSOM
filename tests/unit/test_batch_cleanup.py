"""Single-task cleanup never removes shared batch state or sibling results."""

import json

import pytest

from smartsom.config.codec import digest
from smartsom.experiments import batch_cleanup, control


def setup_batch(tmp_path, monkeypatch):
    queue = tmp_path / "queue"
    queue.mkdir()
    tasks, rows = {}, []
    for name in ("base--seed-1", "sl--seed-1"):
        child = queue / "trials" / name
        (child / "control").mkdir(parents=True)
        (child / "inputs").mkdir()
        (child / "inputs" / "experiment.yaml").write_text("original: true\n")
        child_plan = {"id": name}
        (child / "plan.json").write_text(json.dumps(child_plan))
        (child / "trial.lock").touch()
        (child / "last.pt").write_bytes(name.encode())
        (child / "logs").mkdir()
        (child / "logs" / "run.log").write_text(name)
        rows.append(
            {
                "id": name,
                "child": str(child.relative_to(queue)),
                "child_plan_sha256": digest(child_plan),
            }
        )
        tasks[name] = {
            "root": str(child),
            "status": "removed" if name.startswith("base") else "running",
            "pid": 54321,
        }
    plan = {"files": rows, "inventory": {}}
    (queue / "plan.json").write_text(json.dumps(plan))
    (queue / "batch.json").write_text(
        json.dumps(
            {
                "plan_sha256": digest(plan),
                "status": "running",
                "files": {r["id"]: {"status": tasks[r["id"]]["status"]} for r in rows},
            }
        )
    )
    (queue / "control").mkdir()
    (queue / "control" / "runtime.json").write_text(
        json.dumps(
            {
                "tasks": tasks,
                "additions": {},
                "removed_tasks": ["base--seed-1"],
                "revision": 1,
            }
        )
    )
    monkeypatch.setattr(
        control,
        "read",
        lambda child: {
            "root": str(child),
            "finished_at": 1,
            "ownership_finalized": True,
            "session_context": "test",
            "owner": {"pid": 54321, "created": "old"},
            "members": [],
        },
    )
    monkeypatch.setattr(control, "processes", lambda: {})
    monkeypatch.setattr(control, "session_context", lambda: "test")
    return queue, queue / rows[0]["child"], queue / rows[1]["child"]


def test_preview_and_delete_one_task_preserves_batch_and_sibling(tmp_path, monkeypatch):
    queue, child, sibling = setup_batch(tmp_path, monkeypatch)
    old_sibling = (sibling / "last.pt").read_bytes()
    preview = batch_cleanup.preview(queue, "base--seed-1")
    assert {i["name"] for i in preview["items"]} == {"last.pt", "logs"}
    assert preview["operation"] == "permanent_delete"
    done = batch_cleanup.delete(queue, "base--seed-1", preview["confirmation"])
    assert done["status"] == "complete"
    assert not (child / "last.pt").exists()
    assert (child / "plan.json").exists() and (
        child / "inputs/experiment.yaml"
    ).exists()
    assert (sibling / "last.pt").read_bytes() == old_sibling
    assert (queue / "plan.json").exists()
    assert (
        batch_cleanup.delete(queue, "base--seed-1", preview["confirmation"])["status"]
        == "complete"
    )


def test_reject_active_symlink_hardlink_and_stale_confirmation(tmp_path, monkeypatch):
    queue, child, sibling = setup_batch(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="first be confirmed removed"):
        batch_cleanup.preview(queue, "sl--seed-1")
    before = batch_cleanup.preview(queue, "base--seed-1")
    (child / "last.pt").write_text("changed")
    with pytest.raises(ValueError, match="preview changed"):
        batch_cleanup.delete(queue, "base--seed-1", before["confirmation"])
    (child / "logs" / "link").symlink_to(sibling / "last.pt")
    with pytest.raises(ValueError, match="linked or mounted"):
        batch_cleanup.preview(queue, "base--seed-1")
    (child / "logs" / "link").unlink()
    (child / "last.pt").unlink()
    (child / "last.pt").hardlink_to(sibling / "last.pt")
    with pytest.raises(ValueError, match="hard-linked"):
        batch_cleanup.preview(queue, "base--seed-1")


def test_known_peer_reference_and_unresolved_relative_reference_block_draft(
    tmp_path, monkeypatch
):
    queue, child, sibling = setup_batch(tmp_path, monkeypatch)
    config = sibling / "inputs/experiment.yaml"
    config.write_text(f"run:\n  checkpoint: {child / 'last.pt'}\n")
    view = batch_cleanup.preview(queue, "base--seed-1")
    assert view["references"]["known"][0]["field"] == "run.checkpoint"
    with pytest.raises(ValueError, match="references require preservation"):
        batch_cleanup.delete(queue, "base--seed-1", view["confirmation"])
    assert (child / "last.pt").exists()
    config.write_text("run:\n  checkpoint: ../unknown.pt\n")
    view = batch_cleanup.preview(queue, "base--seed-1")
    assert (
        view["references"]["unknown"][0]["reason"]
        == "relative_reference_resolution_unverified"
    )
    with pytest.raises(ValueError, match="references require preservation"):
        batch_cleanup.delete(queue, "base--seed-1", view["confirmation"])
    assert (child / "last.pt").exists()


@pytest.mark.parametrize("body", ["", "null", "[]", "[1]", "run: []", "case_sets: [1]"])
def test_invalid_peer_configuration_is_unknown(tmp_path, monkeypatch, body):
    queue, _, sibling = setup_batch(tmp_path, monkeypatch)
    (sibling / "inputs/experiment.yaml").write_text(body)
    view = batch_cleanup.preview(queue, "base--seed-1")
    assert (
        view["references"]["unknown"][0]["reason"] == "frozen_configuration_unreadable"
    )


@pytest.mark.parametrize("value", ["[]", "false", "0"])
def test_non_string_falsey_references_are_unknown(tmp_path, monkeypatch, value):
    queue, _, sibling = setup_batch(tmp_path, monkeypatch)
    (sibling / "inputs/experiment.yaml").write_text(f"run:\n  checkpoint: {value}\n")
    view = batch_cleanup.preview(queue, "base--seed-1")
    assert view["references"]["unknown"][0]["reason"] == "non_string_reference"


def test_interrupted_cleanup_retries_from_receipt(tmp_path, monkeypatch):
    queue, child, _ = setup_batch(tmp_path, monkeypatch)
    view = batch_cleanup.preview(queue, "base--seed-1")
    original = batch_cleanup._write_receipt
    calls = []

    def interrupted(path, payload):
        calls.append(1)
        if len(calls) == 3:
            raise OSError("simulated interruption")
        original(path, payload)

    monkeypatch.setattr(batch_cleanup, "_write_receipt", interrupted)
    with pytest.raises(OSError, match="simulated interruption"):
        batch_cleanup.delete(queue, "base--seed-1", view["confirmation"])
    monkeypatch.setattr(batch_cleanup, "_write_receipt", original)
    result = batch_cleanup.delete(queue, "base--seed-1", view["confirmation"])
    assert result["status"] == "complete"
    assert (child / "plan.json").exists()


def test_partial_tree_failure_can_be_repreviewed(tmp_path, monkeypatch):
    queue, child, sibling = setup_batch(tmp_path, monkeypatch)
    view = batch_cleanup.preview(queue, "base--seed-1")
    original = batch_cleanup.shutil.rmtree

    def interrupted(name, *, dir_fd):
        (child / "logs" / "run.log").unlink()
        raise OSError("partial removal")

    interrupted.avoids_symlink_attacks = True
    monkeypatch.setattr(batch_cleanup.shutil, "rmtree", interrupted)
    with pytest.raises(OSError, match="partial removal"):
        batch_cleanup.delete(queue, "base--seed-1", view["confirmation"])
    monkeypatch.setattr(batch_cleanup.shutil, "rmtree", original)
    new_view = batch_cleanup.preview(queue, "base--seed-1")
    assert new_view["confirmation"] != view["confirmation"]
    done = batch_cleanup.delete(queue, "base--seed-1", new_view["confirmation"])
    assert done["status"] == "complete"
    assert (child / "plan.json").exists() and (sibling / "last.pt").exists()


def test_cleanup_serializes_queue_admission(tmp_path, monkeypatch):
    from smartsom.experiments import runtime_control

    queue, _, _ = setup_batch(tmp_path, monkeypatch)
    view = batch_cleanup.preview(queue, "base--seed-1")
    original = batch_cleanup.shutil.rmtree
    observed = []

    def check_queue_lock(name, *, dir_fd):
        with runtime_control.transaction(queue, wait=False) as state:
            assert state["cleanup_active"]["task_id"] == "base--seed-1"
            with pytest.raises(ValueError, match="admission paused"):
                runtime_control.require_admission_available(state)
            state["test_scheduler_progress"] = True
        observed.append(True)
        return original(name, dir_fd=dir_fd)

    check_queue_lock.avoids_symlink_attacks = True
    monkeypatch.setattr(batch_cleanup.shutil, "rmtree", check_queue_lock)
    batch_cleanup.delete(queue, "base--seed-1", view["confirmation"])
    assert observed
    assert runtime_control.snapshot(queue)["test_scheduler_progress"]
    assert "cleanup_active" not in runtime_control.snapshot(queue)


def test_abandoned_lease_can_resume_same_task(tmp_path, monkeypatch):
    from smartsom.experiments import runtime_control

    queue, child, _ = setup_batch(tmp_path, monkeypatch)
    with runtime_control.transaction(queue) as state:
        state["cleanup_active"] = {"task_id": "base--seed-1", "token": "abandoned"}
    view = batch_cleanup.preview(queue, "base--seed-1")
    assert (
        batch_cleanup.delete(queue, "base--seed-1", view["confirmation"])["status"]
        == "complete"
    )
    assert "cleanup_active" not in runtime_control.snapshot(queue)
    assert (child / "plan.json").exists()


def test_both_official_append_paths_reject_active_lease(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from smartsom.config import launch
    from smartsom.experiments import author_batch, batch_append, runtime_control

    queue, _, _ = setup_batch(tmp_path, monkeypatch)
    with runtime_control.transaction(queue) as state:
        state["cleanup_active"] = {"task_id": "base--seed-1", "token": "active"}
    source = tmp_path / "source"
    source.mkdir()
    monkeypatch.setattr(launch, "resolve_experiment", lambda value: source)
    monkeypatch.setattr(
        author_batch,
        "compile_directory",
        lambda *a, **kw: SimpleNamespace(contract="v4", files=[]),
    )
    with pytest.raises(ValueError, match="admission paused"):
        runtime_control.append(queue, source)
    monkeypatch.setattr(author_batch, "load", lambda root: (None, {}, None))
    monkeypatch.setattr(batch_append, "compile_sources", lambda *a, **kw: [])
    with pytest.raises(ValueError, match="admission paused"):
        batch_append.append(queue, source)


def test_new_reference_after_partial_failure_blocks_new_cleanup(tmp_path, monkeypatch):
    queue, child, sibling = setup_batch(tmp_path, monkeypatch)
    view = batch_cleanup.preview(queue, "base--seed-1")
    original = batch_cleanup.shutil.rmtree

    def interrupted(name, *, dir_fd):
        raise OSError("partial removal")

    interrupted.avoids_symlink_attacks = True
    monkeypatch.setattr(batch_cleanup.shutil, "rmtree", interrupted)
    with pytest.raises(OSError, match="partial removal"):
        batch_cleanup.delete(queue, "base--seed-1", view["confirmation"])
    monkeypatch.setattr(batch_cleanup.shutil, "rmtree", original)
    (sibling / "inputs/experiment.yaml").write_text(
        f"run:\n  checkpoint: {child / 'last.pt'}\n"
    )
    view = batch_cleanup.preview(queue, "base--seed-1")
    with pytest.raises(ValueError, match="references require preservation"):
        batch_cleanup.delete(queue, "base--seed-1", view["confirmation"])
    assert (child / "last.pt").exists()


def test_killed_lease_holder_can_be_reclaimed(tmp_path, monkeypatch):
    import os
    import signal

    from smartsom.experiments import runtime_control

    if not hasattr(os, "fork"):
        pytest.skip("requires POSIX fork")
    queue, child, _ = setup_batch(tmp_path, monkeypatch)
    read_fd, write_fd = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(read_fd)
        try:
            with batch_cleanup._cleanup_lease(queue, child, "base--seed-1"):
                os.write(write_fd, b"ready")
                signal.pause()
        finally:
            os._exit(1)
    os.close(write_fd)
    try:
        assert os.read(read_fd, 5) == b"ready"
        assert (
            runtime_control.snapshot(queue)["cleanup_active"]["task_id"]
            == "base--seed-1"
        )
        os.kill(pid, signal.SIGKILL)
        os.waitpid(pid, 0)
        pid = None
        view = batch_cleanup.preview(queue, "base--seed-1")
        assert (
            batch_cleanup.delete(queue, "base--seed-1", view["confirmation"])["status"]
            == "complete"
        )
        assert "cleanup_active" not in runtime_control.snapshot(queue)
    finally:
        os.close(read_fd)
        if pid is not None:
            os.kill(pid, signal.SIGKILL)
            os.waitpid(pid, 0)


def test_cli_preview_and_confirm(tmp_path, monkeypatch, capsys):
    from smartsom.experiments import catalog, cli

    queue, child, sibling = setup_batch(tmp_path, monkeypatch)
    monkeypatch.setattr(catalog, "resolve_run", lambda path: queue)
    monkeypatch.setattr(catalog, "require_current", lambda path: None)
    assert (
        cli.main(["batch-clean", str(queue), "--task", "base--seed-1", "--show-target"])
        == 0
    )
    view = json.loads(capsys.readouterr().out)
    assert (
        cli.main(
            [
                "batch-clean",
                str(queue),
                "--task",
                "base--seed-1",
                "--confirm-target",
                view["confirmation"],
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["status"] == "complete"
    assert not (child / "last.pt").exists() and (sibling / "last.pt").exists()
