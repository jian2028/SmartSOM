"""Atomic pointer faults and ownership boundaries are exercised without learners."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from smartsom.experiments import checkpoint_retention as retention


class Session:
    def __init__(self, root):
        self.root = root
        (root / "checkpoints").mkdir()
        self.config = SimpleNamespace(checkpointing=SimpleNamespace(retention=True))
        self.prepared = SimpleNamespace(scientific_sha256="frozen")
        self.settings = SimpleNamespace(record_initial=False)
        self.updates = self.ticks = 0
        self.best_update = None
        self.record = {"status": "running"}

    def _write_snapshot(self, directory, *, full, retention_owner):
        metadata = dict(
            update=self.updates,
            physical_ticks=self.ticks,
            retention_owner=retention_owner,
        )
        if full:
            (directory / "continuation.pkl").write_bytes(str(self.updates).encode())
            metadata["continuation_sha256"] = retention._sha(
                directory / "continuation.pkl"
            )
        else:
            metadata["inference_only"] = True
        (directory / "snapshot.json").write_text(json.dumps(metadata), encoding="utf-8")

    def _step_update(self, callback):
        self.updates += 1
        self.ticks += 256
        retention.save(self, best=self.updates == 1)
        if self.updates == 1:
            self.best_update = 1
        self.record = {"update": self.updates}
        return self.record


def test_latest_full_and_selected_inference_are_bounded(tmp_path):
    session = Session(tmp_path)
    for _ in range(4):
        retention.step(session, None)
    value = retention.manifest(tmp_path)
    assert value["updates"] == 4 and value["best_update"] == 1
    paths = list((tmp_path / "checkpoints").glob("update-*"))
    assert len(paths) == 2
    assert sum((path / "continuation.pkl").exists() for path in paths) == 1
    assert retention.resolve(tmp_path, "last") / "continuation.pkl" in [
        path / "continuation.pkl" for path in paths
    ]
    assert not (retention.resolve(tmp_path, "best") / "continuation.pkl").exists()
    best = json.loads((tmp_path / "checkpoints/best.json").read_text(encoding="utf-8"))
    assert best["actual_update"] == 1 and best["physical_ticks"] == 256


@pytest.mark.parametrize("after", [False, True])
def test_fault_at_atomic_pointer_uses_only_published_boundary(
    tmp_path, monkeypatch, after
):
    session = Session(tmp_path)
    retention.step(session, None)
    original = retention._write

    def fault(path, value):
        if Path(path).name == "retention.json":
            if after:
                original(path, value)
            raise RuntimeError("publication fault")
        original(path, value)

    with monkeypatch.context() as context:
        context.setattr(retention, "_write", fault)
        with pytest.raises(RuntimeError, match="publication fault"):
            retention.step(session, None)
    value = retention.manifest(tmp_path)
    assert value["updates"] == (2 if after else 1)
    # An alias may still point at update1 after atomic update2 publication.
    assert retention.resolve(tmp_path, "last").name == value["latest_full"]["path"]
    retention.collect(tmp_path)
    assert len(list((tmp_path / "checkpoints").glob("update-*"))) == 2


def test_foreign_historical_directory_is_never_collected(tmp_path):
    session = Session(tmp_path)
    retention.step(session, None)
    foreign = tmp_path / "checkpoints/update-000009"
    foreign.mkdir()
    (foreign / "snapshot.json").write_text(
        '{"retention_owner":"someone-else"}', encoding="utf-8"
    )
    retention.step(session, None)
    assert foreign.is_dir()


def test_cannot_adopt_old_checkpoint_tree(tmp_path):
    session = Session(tmp_path)
    (tmp_path / "checkpoints/update-000001").mkdir()
    with pytest.raises(ValueError, match="cannot adopt historical"):
        retention.step(session, None)


def test_corruption_blocks_resolution_and_collection(tmp_path):
    session = Session(tmp_path)
    retention.step(session, None)
    current = retention.resolve(tmp_path, "last")
    (current / "continuation.pkl").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="continuation hash"):
        retention.resolve(tmp_path, "last")
    with pytest.raises(ValueError, match="continuation hash"):
        retention.collect(tmp_path)
    assert current.is_dir()


def test_reader_lease_is_reentrant(tmp_path):
    session = Session(tmp_path)
    retention.step(session, None)

    @retention.leased
    def read(source):
        with retention.lease(source):
            return retention.resolve(source, "last")

    assert read(tmp_path).is_dir()


@pytest.mark.parametrize("algorithm", ["ppo", "dqn"])
def test_real_native_retained_full_boundary_restores_exactly(tmp_path, algorithm):
    pytest.importorskip("torch")
    pytest.importorskip("ray")
    import copy
    import pickle
    from dataclasses import replace

    from smartsom import api
    from smartsom.config.codec import canonical_json
    from smartsom.config.experiment import RetentionOptions, prepare
    from smartsom.config.scientific_contract import identity
    from smartsom.experiments.composable import (
        TrainingSession,
        allocate,
        checkpoint_path,
    )

    root = Path(__file__).resolve().parents[2]
    config = api.load_config(root / f"configs/test/runs/train_all_{algorithm}.yaml")
    config.training.total_ticks = 8
    config.training.ticks_per_update = 4
    config.validation.enabled = False
    config.output.root = str(tmp_path)
    config.checkpointing.retention = RetentionOptions()
    prepared = prepare(config)
    parameters = json.loads(prepared.parameters_json)
    parameters.update(batch_size=2)
    if algorithm == "ppo":
        parameters.update(n_epochs=1)
    else:
        parameters.update(warmup_ticks=0, target_update_ticks=4)
    prepared = replace(prepared, parameters_json=canonical_json(parameters))
    prepared = replace(prepared, scientific_sha256=identity(prepared))
    directory, record, prepared = allocate(prepared, "training")
    session = TrainingSession(prepared, directory, record)
    restored = None
    try:
        session.step_update()
        path = checkpoint_path(directory, "last")
        with (path / "continuation.pkl").open("rb") as stream:
            state = pickle.load(stream)
        session.step_update()
        expected = copy.deepcopy(session.actions)
        fingerprints = {
            key: value.fingerprint()
            for key, value in session.policies.items()
            if hasattr(value, "fingerprint")
        }
        restored = TrainingSession(prepared, directory, record)
        restored.restore(state)
        restored.step_update()
        assert restored.actions == expected
        assert {
            key: value.fingerprint()
            for key, value in restored.policies.items()
            if hasattr(value, "fingerprint")
        } == fingerprints
        assert retention.manifest(directory)["updates"] == 2
        assert len(list((directory / "checkpoints").glob("update-*"))) == 1
        # A crash after manifest publication may leave secondary aliases absent.
        (directory / "checkpoints/last.json").unlink()
        assert restored.finish().last_checkpoint == checkpoint_path(directory, "last")
    finally:
        session.close()
        if restored is not None:
            restored.close()
