"""A Tune recovery owns both the model and the evidence used to skip work."""

import json

import pytest

from smartsom.experiments.tune_checkpoint import restore, save


def evidence(root, identity="same"):
    root.mkdir()
    (root / "plan.json").write_text(json.dumps({"run_id": identity}))
    (root / "last.pt").write_bytes(b"first complete model")
    return root


def test_unpublished_status_can_disappear_during_snapshot(tmp_path, monkeypatch):
    from smartsom.experiments import tune_checkpoint

    root = evidence(tmp_path / "run")
    temporary = root / "tune-status.json.tmp"
    temporary.write_text('{"status":"training"}')
    for directory in ("source", "inputs"):
        (root / directory).mkdir()
        (root / directory / temporary.name).write_text("authored fixture")
    original = tune_checkpoint._owned_files

    def publishing_walk(path):
        for file in original(path):
            if file == temporary:
                temporary.replace(root / "tune-status.json")
            yield file

    monkeypatch.setattr(tune_checkpoint, "_owned_files", publishing_walk)
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    save(root, snapshot, {"phase": "training"})
    files = tune_checkpoint.read_metadata(snapshot)["files"]
    assert temporary.name not in files
    assert all(
        f"{directory}/{temporary.name}" in files for directory in ("source", "inputs")
    )
    monkeypatch.setattr(tune_checkpoint, "_owned_files", original)
    later = tmp_path / "later"
    later.mkdir()
    save(root, later, {"phase": "training"})
    assert "tune-status.json" in tune_checkpoint.read_metadata(later)["files"]
    restore(later, root)
    assert json.loads((root / "tune-status.json").read_text())["status"] == "training"


def test_restore_rolls_back_receipts_and_preserves_control(tmp_path):
    root = evidence(tmp_path / "run")
    checkpoint = tmp_path / "snapshot"
    checkpoint.mkdir()
    save(root, checkpoint, {"phase": "evaluation"})
    # Model publishers replace immutable payloads; never mutate a hardlink.
    (root / "next.pt").write_bytes(b"later model")
    (root / "next.pt").replace(root / "last.pt")
    panel = root / "periodic_evaluation/episode-00000002"
    panel.mkdir(parents=True)
    (panel / "summary.json").write_text('{"completed": true}')
    control = root / "control"
    control.mkdir()
    (control / "owner.json").write_text("live owner")
    assert restore(checkpoint, root) == {"phase": "evaluation"}
    assert (root / "last.pt").read_bytes() == b"first complete model"
    assert not (panel / "summary.json").exists()
    assert (control / "owner.json").read_text() == "live owner"


def test_corrupt_snapshot_does_not_modify_destination(tmp_path):
    root = evidence(tmp_path / "run")
    checkpoint = tmp_path / "snapshot"
    checkpoint.mkdir()
    save(root, checkpoint, {"phase": "training"})
    (checkpoint / "native/plan.json").write_text("corrupt")
    original = (root / "plan.json").read_bytes()
    with pytest.raises(ValueError, match="digest mismatch"):
        restore(checkpoint, root)
    assert (root / "plan.json").read_bytes() == original


def test_snapshot_cannot_replace_another_run(tmp_path):
    root = evidence(tmp_path / "run")
    other = evidence(tmp_path / "other", identity="other")
    checkpoint = tmp_path / "snapshot"
    checkpoint.mkdir()
    save(root, checkpoint, {"phase": "training"})
    with pytest.raises(ValueError, match="different run"):
        restore(checkpoint, other)
    assert json.loads((other / "plan.json").read_text())["run_id"] == "other"


def test_snapshot_rejects_nested_links(tmp_path):
    root = evidence(tmp_path / "run")
    folder = root / "continuation"
    folder.mkdir()
    (folder / "outside").symlink_to(tmp_path)
    checkpoint = tmp_path / "snapshot"
    checkpoint.mkdir()
    with pytest.raises(ValueError, match="symlinks"):
        save(root, checkpoint, {"phase": "training"})


@pytest.mark.parametrize("change", ["state", "files"])
def test_corrupt_metadata_cannot_remove_live_evidence(tmp_path, change):
    root = evidence(tmp_path / "run")
    checkpoint = tmp_path / "snapshot"
    checkpoint.mkdir()
    save(root, checkpoint, {"phase": "training"})
    (root / "episodes.jsonl").write_text("later committed evidence")
    path = checkpoint / "phase.json"
    metadata = json.loads(path.read_text())
    metadata.pop(change)
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="metadata digest mismatch"):
        restore(checkpoint, root)
    assert (root / "episodes.jsonl").read_text() == "later committed evidence"
    assert (root / "last.pt").read_bytes() == b"first complete model"


def test_bootstrap_cannot_silently_discard_existing_training(tmp_path):
    root = evidence(tmp_path / "run")
    checkpoint = tmp_path / "snapshot"
    checkpoint.mkdir()
    save(root, checkpoint, {"phase": "training"})
    (root / "continuation").mkdir()
    marker = root / "continuation/current.json"
    marker.write_text('{"generation": "first"}')
    with pytest.raises(ValueError, match="bootstrap-only"):
        restore(checkpoint, root)
    assert marker.is_file()


def test_restore_preserves_allowed_execution_changes(tmp_path):
    root = evidence(tmp_path / "run")
    execution = root / "execution.json"
    execution.write_text(
        json.dumps(
            {
                "settings": {"resources": {"cpus": 2}, "tracking": {"mode": "online"}},
                "history": [],
            }
        )
    )
    checkpoint = tmp_path / "snapshot"
    checkpoint.mkdir()
    save(root, checkpoint, {"phase": "training"})
    changed = {
        "settings": {"resources": {"cpus": 2}, "tracking": {"mode": "offline"}},
        "history": ["explicit change"],
    }
    execution.write_text(json.dumps(changed))
    restore(checkpoint, root)
    assert json.loads(execution.read_text()) == changed
    changed["settings"]["resources"]["cpus"] = 4
    execution.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="resource settings differ"):
        restore(checkpoint, root)


def test_immutable_hash_cache_keeps_full_payload_and_revalidates_restore(tmp_path):
    root = evidence(tmp_path / "run")
    generation = root / "continuation" / ("00000001-" + "a" * 32)
    generation.mkdir(parents=True)
    (generation / "model.pt").write_bytes(b"complete optimizer and model")
    (generation / "state.json").write_text('{"rng": 123}')
    cache = {}
    for index in range(2):
        destination = tmp_path / f"snapshot-{index}"
        destination.mkdir()
        save(root, destination, {"phase": "training"}, immutable_digests=cache)
    assert len(cache) == 2
    import shutil

    shutil.rmtree(root)
    shutil.rmtree(tmp_path / "snapshot-0")
    restore(destination, root)
    assert (generation / "model.pt").read_bytes() == b"complete optimizer and model"
    payload = destination / "native" / generation.relative_to(root) / "state.json"
    payload.write_text('{"rng": 999}')
    with pytest.raises(ValueError, match="digest mismatch"):
        restore(destination, root)
    assert (generation / "state.json").read_text() == '{"rng": 123}'


def test_replaced_generation_and_mutable_alias_are_not_hash_cached(tmp_path):
    root = evidence(tmp_path / "run")
    generation = root / "continuation" / ("00000001-" + "b" * 32)
    generation.mkdir(parents=True)
    model = generation / "model.pt"
    model.write_bytes(b"old")
    cache = {}
    first = tmp_path / "first"
    first.mkdir()
    save(root, first, {"phase": "training"}, immutable_digests=cache)
    replacement = root / "replacement.pt"
    replacement.write_bytes(b"new")
    replacement.replace(model)
    (root / "last.pt").unlink()
    (root / "last.pt").write_bytes(b"new alias")
    second = tmp_path / "second"
    second.mkdir()
    save(root, second, {"phase": "training"}, immutable_digests=cache)
    target = tmp_path / "restored"
    restore(second, target)
    assert (target / model.relative_to(root)).read_bytes() == b"new"
    assert (target / "last.pt").read_bytes() == b"new alias"
    assert (first / "native" / model.relative_to(root)).read_bytes() == b"old"


def test_sdk_symlinks_are_excluded_and_published_tracking_never_rolls_back(tmp_path):
    root = evidence(tmp_path / "run")
    tracking = root / "tracking"
    sdk = tracking / "wandb"
    sdk.mkdir(parents=True)
    segment = sdk / "offline-run-first"
    segment.mkdir()
    (sdk / "latest-run").symlink_to(segment, target_is_directory=True)
    journal = tracking / "events.jsonl"
    status = tracking / "status.json"
    journal.write_text('{"event_sequence":4}\n')
    status.write_text('{"sequence":4}')
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    save(root, checkpoint, {"phase": "training"})
    assert not (checkpoint / "native/tracking/wandb").exists()
    journal.write_text('{"event_sequence":4}\n{"event_sequence":9}\n')
    status.write_text('{"sequence":9}')
    (tracking / "later-attempt.json").write_text("published history")
    restore(checkpoint, root)
    assert json.loads(status.read_text())["sequence"] == 9
    assert len(journal.read_text().splitlines()) == 2
    assert (tracking / "later-attempt.json").exists()
    assert (sdk / "latest-run").is_symlink()
    restored = tmp_path / "relocated"
    restore(checkpoint, restored)
    assert json.loads((restored / "tracking/status.json").read_text())["sequence"] == 4


def test_snapshot_keeps_archived_dependency_lockfile(tmp_path):
    root = evidence(tmp_path / "run")
    (root / "source").mkdir()
    (root / "source/uv.lock").write_text("frozen dependency identity")
    (root / "driver.lock").write_text("runtime lock")
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    save(root, checkpoint, {"phase": "training"})
    destination = tmp_path / "restored"
    restore(checkpoint, destination)
    assert (destination / "source/uv.lock").read_text() == "frozen dependency identity"
    assert not (destination / "driver.lock").exists()


def test_execution_history_survives_rollback_and_fresh_destination(tmp_path):
    import shutil

    from smartsom.experiments.tune_provenance import merge, record, snapshot

    root = evidence(tmp_path / "run")
    record(root, "trial", "actor_started", attempt_id="A")
    first = tmp_path / "snapshot-A"
    first.mkdir()
    save(root, first, {"phase": "training"}, execution=snapshot(root, "trial", "A"))
    record(root, "trial", "actor_started", attempt_id="B")
    restore(first, root, trial_id="trial")
    metadata = json.loads((first / "phase.json").read_text())
    merge(root, "trial", metadata["execution"]["events"])
    record(
        root,
        "trial",
        "checkpoint_restored",
        attempt_id="B",
        source_attempt="A",
        source_checkpoint_digest=(first / "phase.sha256").read_text(),
    )
    second = tmp_path / "snapshot-B"
    second.mkdir()
    saved = snapshot(root, "trial", "B")
    save(root, second, {"phase": "evaluation"}, execution=saved)
    # A rollback cannot discard B's already observed execution history.
    restore(first, root, trial_id="trial")
    assert snapshot(root, "trial", "B") == saved
    shutil.rmtree(first)
    shutil.rmtree(root)
    restore(second, root, trial_id="trial")
    metadata = json.loads((second / "phase.json").read_text())
    merge(root, "trial", metadata["execution"]["events"])
    assert snapshot(root, "trial", "B") == saved


def test_invalid_execution_origin_fails_before_restoring_payload(tmp_path):
    from smartsom.experiments.tune_provenance import record, snapshot

    root = evidence(tmp_path / "run")
    record(root, "trial", "actor_started", attempt_id="A")
    execution = snapshot(root, "trial", "A")
    execution["producer_attempt"] = "missing-actor"
    destination = tmp_path / "snapshot"
    destination.mkdir()
    save(root, destination, {"phase": "training"}, execution=execution)
    (root / "episodes.jsonl").write_text("new local evidence")
    with pytest.raises(ValueError, match="producer has no execution origin"):
        restore(destination, root, trial_id="trial")
    assert (root / "episodes.jsonl").read_text() == "new local evidence"


def test_execution_event_tampering_is_rejected(tmp_path):
    from smartsom.experiments.tune_provenance import directory, record, snapshot

    record(tmp_path, "trial", "actor_started", attempt_id="A")
    path = next(directory(tmp_path, "trial").glob("*.json"))
    event = json.loads(path.read_text())
    event["attempt_id"] = "tampered"
    path.write_text(json.dumps(event))
    with pytest.raises(ValueError, match="provenance digest mismatch"):
        snapshot(tmp_path, "trial", "A")


def test_failure_before_first_report_does_not_serialize_algorithm_config(tmp_path):
    from types import SimpleNamespace

    from smartsom.experiments.tune_batch import _result_report

    result = SimpleNamespace(
        metrics={
            "config": {"algorithm_config": object()},
            "smartsom_checkpoint_score": float("inf"),
        },
        error=RuntimeError("original initialization failure"),
    )
    report = _result_report(tmp_path, result, control_root=tmp_path)
    assert report["status"] == "failed"
    assert report["error"] == "original initialization failure"
    assert "config" not in report
    assert json.loads((tmp_path / "batch.json").read_text()) == report
