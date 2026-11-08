"""Atomic pointer faults and ownership boundaries are exercised without learners."""

import json
import multiprocessing
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from smartsom.experiments import checkpoint_retention as retention


def _reservation_generations(root):
    import hashlib

    directory = root / "continuation"
    directory.mkdir()
    names = [f"{i:08d}-{'a' * 32}" for i in range(4)]
    for i, name in enumerate(names):
        p = directory / name
        p.mkdir()
        (p / "model.pt").write_bytes(str(i).encode())
        (p / "state.json").write_text("{}")
    latest = directory / names[-1]
    files = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in latest.iterdir()
    }
    (directory / "current.json").write_text(
        json.dumps({"generation": names[-1], "files": files})
    )
    (root / "last.pt").write_bytes((latest / "model.pt").read_bytes())
    (root / "last.pt.json").write_text(
        json.dumps({"checkpoint_sha256": files["model.pt"]})
    )
    (root / "best.pt").write_bytes(b"selected validation")
    return directory, names


def test_reservation_retention_preserves_current_previous_and_best(tmp_path):
    directory, names = _reservation_generations(tmp_path)
    retention.prune_continuations(tmp_path, 2)
    assert {p.name for p in directory.iterdir()} == {*names[-2:], "current.json"}
    assert (tmp_path / "best.pt").read_bytes() == b"selected validation"
    retention.prune_continuations(tmp_path, 2)
    assert (tmp_path / "last.pt").read_bytes() == b"3"


def test_reservation_retention_fail_closed_and_partial_trash(tmp_path, monkeypatch):
    directory, names = _reservation_generations(tmp_path)
    (tmp_path / "last.pt").write_bytes(b"wrong")
    with pytest.raises(ValueError, match="differs"):
        retention.prune_continuations(tmp_path, 2)
    assert all((directory / n).is_dir() for n in names)
    (tmp_path / "last.pt").write_bytes(b"3")
    original = retention.shutil.rmtree

    def interrupted(path):
        (path / "model.pt").unlink()
        raise RuntimeError("power loss")

    monkeypatch.setattr(retention.shutil, "rmtree", interrupted)
    with pytest.raises(RuntimeError, match="power loss"):
        retention.prune_continuations(tmp_path, 2)
    monkeypatch.setattr(retention.shutil, "rmtree", original)
    retention.prune_continuations(tmp_path, 2)
    assert {p.name for p in directory.iterdir()} == {*names[-2:], "current.json"}
    assert not list((tmp_path / ".continuation-trash").iterdir())


def test_reservation_retention_preserves_unpublished_generations(tmp_path):
    directory, names = _reservation_generations(tmp_path)
    duplicate = directory / f"00000001-{'b' * 32}"
    duplicate.mkdir()
    (duplicate / "model.pt").write_bytes(b"unpublished model")
    (duplicate / "state.json").write_text("{}")
    incomplete = directory / f"00000000-{'c' * 32}"
    incomplete.mkdir()
    (incomplete / "model.pt").write_bytes(b"partial checkpoint")
    retention.prune_continuations(tmp_path, 2)
    assert all((directory / n).exists() for n in names[1:])
    assert duplicate.is_dir() and incomplete.is_dir()
    report = json.loads((tmp_path / "continuation-retention.json").read_text())
    assert set(report["preserved_for_inspection"]) == {
        duplicate.name,
        incomplete.name,
        names[1],
    }


def _input_payload(root):
    (root / "checkpoints").mkdir()
    (root / "checkpoints/retention-owner.json").write_text("{}")
    source = root / "checkpoints/update-000001/groups/machine"
    source.mkdir(parents=True)
    for name in ("model.json", "weights.pt", "encoder.json"):
        (source / name).write_bytes(name.encode())
    return source


def test_pinned_inputs_reuse_payload_and_clean_up_at_process_exit(tmp_path):
    source = _input_payload(tmp_path)
    temporary = tmp_path / "temporary"
    temporary.mkdir()
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from smartsom.experiments.checkpoint_retention import pin_input; "
            "import sys; "
            "paths = [pin_input(sys.argv[1]) for _ in range(12)]; "
            "assert len(set(paths)) == 1; "
            "assert len(list(paths[0].parent.iterdir())) == 1; "
            "print(paths[0].parent)",
            str(source),
        ],
        env={**os.environ, "TMPDIR": str(temporary)},
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    assert not Path(result.stdout.strip()).exists()
    assert not list(temporary.iterdir())
    assert (source / "weights.pt").read_bytes() == b"weights.pt"


def _pin_in_spawn_worker(source, result):
    pins = [retention.pin_input(source) for _ in range(12)]
    assert len(set(pins)) == 1
    Path(result).write_text(str(pins[0].parent))


def test_spawn_worker_releases_its_pinned_pool_on_normal_return(tmp_path):
    source = _input_payload(tmp_path)
    result = tmp_path / "worker-result"
    child = multiprocessing.get_context("spawn").Process(
        target=_pin_in_spawn_worker, args=(str(source), str(result))
    )
    child.start()
    child.join(timeout=30)
    assert child.exitcode == 0
    assert not Path(result.read_text()).exists()
    assert (source / "weights.pt").read_bytes() == b"weights.pt"


def test_process_cleanup_waits_for_live_prepared_child_reader(tmp_path):
    source = _input_payload(tmp_path)
    script = tmp_path / "reader.py"
    result = tmp_path / "reader-result"
    script.write_text(
        "import multiprocessing, multiprocessing.util, sys, time\n"
        "from pathlib import Path\n"
        "from smartsom.experiments.checkpoint_retention import pin_input\n"
        "def reader(source, output):\n"
        "    time.sleep(.2)\n"
        "    Path(output).write_bytes((Path(source)/'weights.pt').read_bytes())\n"
        "if __name__ == '__main__':\n"
        "    pinned = pin_input(sys.argv[1])\n"
        "    child = multiprocessing.get_context('spawn').Process(\n"
        "        target=reader, args=(str(pinned), sys.argv[2]))\n"
        "    child.start()\n"
        "    print(pinned.parent)\n"
        # Intentionally let normal interpreter shutdown join the live reader.
    )
    parent = subprocess.run(
        [sys.executable, str(script), str(source), str(result)],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    assert result.read_bytes() == b"weights.pt", parent.stderr
    assert not Path(parent.stdout.strip()).exists()


def test_pin_copy_failure_keeps_existing_readers_and_removes_partial_payload(
    tmp_path, monkeypatch
):
    source = _input_payload(tmp_path)
    pool = retention._PinnedInputs()
    monkeypatch.setattr(retention, "_INPUT_POOL", pool)
    try:
        first = retention.pin_input(source)
        (source / "weights.pt").write_bytes(b"later generation")
        copy = retention.shutil.copyfile

        def fail(origin, target):
            if Path(origin).name == "weights.pt":
                raise OSError("injected model copy failure")
            return copy(origin, target)

        with monkeypatch.context() as context:
            context.setattr(retention.shutil, "copyfile", fail)
            with pytest.raises(OSError, match="injected model copy failure"):
                retention.pin_input(source)
        assert list(pool.root.iterdir()) == [first]
        assert len(pool.snapshots) == 1
        assert (first / "weights.pt").read_bytes() == b"weights.pt"
        second = retention.pin_input(source)
        assert second != first
        assert (second / "weights.pt").read_bytes() == b"later generation"
        assert (first / "weights.pt").read_bytes() == b"weights.pt"
    finally:
        pool.cleanup()


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


@pytest.mark.parametrize(
    "extra", ["user-notes.txt", "user/notes.txt", "empty-directory"]
)
def test_user_additions_survive_owned_generation_gc(tmp_path, extra):
    session = Session(tmp_path)
    retention.step(session, None)
    retention.step(session, None)
    old = retention.resolve(tmp_path, "last")
    notes = old / extra
    notes.parent.mkdir(parents=True, exist_ok=True)
    if extra == "empty-directory":
        notes.mkdir()
    else:
        notes.write_text("unique user evidence", encoding="utf-8")
    retention.step(session, None)
    assert notes.exists()
    if notes.is_file():
        assert notes.read_text(encoding="utf-8") == "unique user evidence"
    report = json.loads(
        (tmp_path / "checkpoints/retention-gc-preserved.json").read_text()
    )
    assert report["preserved"] == [
        {"path": old.name, "reason": "checkpoint contains unrecognized user content"}
    ]
    assert retention.manifest(tmp_path)["updates"] == 3


def test_recursive_namespace_sync_precedes_manifest_and_failed_sync_preserves_old(
    tmp_path, monkeypatch
):
    session = Session(tmp_path)
    original_snapshot = session._write_snapshot

    def nested(directory, **kwargs):
        original_snapshot(directory, **kwargs)
        nested_dir = directory / "groups/machine"
        nested_dir.mkdir(parents=True)
        (nested_dir / "payload.bin").write_bytes(b"native payload")

    session._write_snapshot = nested
    retention.step(session, None)
    previous = retention.manifest(tmp_path)
    syncs = []
    sync = retention._sync_directory
    write = retention._write

    def synced(path):
        syncs.append(Path(path))
        return sync(path)

    def published(path, value):
        if Path(path).name == "retention.json":
            for item in ("latest_full", "selected_best"):
                if value[item] is None:
                    continue
                directory = tmp_path / "checkpoints" / value[item]["path"]
                marker = json.loads(
                    (directory / "retention-generation.json").read_text()
                )
                staging = directory.with_name(marker["staging"])
                if staging in syncs:
                    assert syncs.index(staging / "groups/machine") < max(
                        i for i, p in enumerate(syncs) if p == staging / "groups"
                    )
                    assert syncs.index(staging / "groups") < max(
                        i for i, p in enumerate(syncs) if p == staging
                    )
        return write(path, value)

    with monkeypatch.context() as context:
        context.setattr(retention, "_sync_directory", synced)
        context.setattr(retention, "_write", published)
        retention.step(session, None)
    current = retention.manifest(tmp_path)
    assert current["updates"] == previous["updates"] + 1

    def failed(path):
        if Path(path).name == "machine":
            raise OSError("injected nested directory sync failure")
        return sync(path)

    with monkeypatch.context() as context:
        context.setattr(retention, "_sync_directory", failed)
        with pytest.raises(OSError, match="nested directory sync"):
            retention.step(session, None)
    assert retention.manifest(tmp_path) == current
    assert retention.resolve(tmp_path, "last").exists()
    assert retention.resolve(tmp_path, "best").exists()
