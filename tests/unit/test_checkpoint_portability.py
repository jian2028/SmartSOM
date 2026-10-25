"""UTF-8 snapshots and Windows long checkpoint paths keep verification intact."""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import pytest

from smartsom import api
from smartsom._filesystem import _windows_extended_name, native_path
from smartsom.experiments.composable import prepared_from_run
from smartsom.experiments.evidence import write_json
from smartsom.experiments.tuning_session import _copy, file_digests

ROOT = Path(__file__).resolve().parents[2]


def test_prepared_snapshot_reads_utf8_under_gbk_locale(tmp_path, monkeypatch):
    prepared = api.prepare(
        api.load_config(ROOT / "configs/test/runs/small_rules_auto.yaml"),
        training=False,
    )
    data = asdict(prepared)
    data["origins_json"] = json.dumps({"note": "调度 café"}, ensure_ascii=False)
    (tmp_path / "config").mkdir()
    write_json(tmp_path / "config/prepared.json", data)
    original = Path.read_text

    def gbk_default(path, encoding=None, errors=None):
        return original(path, encoding=encoding or "gbk", errors=errors)

    monkeypatch.setattr(Path, "read_text", gbk_default)
    restored = prepared_from_run(tmp_path)
    assert json.loads(restored.origins_json)["note"] == "调度 café"


@pytest.mark.parametrize("length", [240, 259, 260, 280, 320])
def test_checkpoint_copy_ordinary_long_paths(tmp_path, length):
    source = tmp_path / "source.json"
    source.write_text("调度 café", encoding="utf-8")
    parent = tmp_path / "destination"
    leaf = "payload.json"
    remaining = length - len(str(parent / leaf))
    assert remaining >= 0
    while remaining >= 3:
        count = min(remaining - 1, 50)
        parent /= "d" * count
        remaining -= count + 1
    leaf = "x" * remaining + leaf
    target = parent / leaf
    assert len(str(target)) == length
    _copy(source, target)
    assert native_path(target).read_text(encoding="utf-8") == "调度 café"
    # Public checkpoint references retain ordinary spelling.
    assert not str(target).startswith("\\\\?\\")


def test_nested_copy_retains_digests_and_unicode(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    write_json(source / "marker.json", {"name": "调度 café"})
    target = tmp_path / ("nested-" * 20) / ("nested-" * 20) / "copy"
    _copy(source, target)
    assert file_digests(target) == file_digests(source)
    write_json(target / "marker.json", {"name": "changed"})
    assert file_digests(target) != file_digests(source)
    assert (
        file_digests(source)["marker.json"]
        == hashlib.sha256((source / "marker.json").read_bytes()).hexdigest()
    )


@pytest.mark.parametrize(
    "path,expected",
    [
        (r"C:\runs\checkpoint", "\\\\?\\C:\\runs\\checkpoint"),
        (r"\\server\share\checkpoint", "\\\\?\\UNC\\server\\share\\checkpoint"),
        ("\\\\?\\C:\\runs\\checkpoint", "\\\\?\\C:\\runs\\checkpoint"),
    ],
)
def test_windows_extended_path_spelling(path, expected):
    assert _windows_extended_name(path) == expected


def test_actual_native_long_save_copy_model_resolution_and_resume(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("ray")
    import os
    import subprocess
    import sys

    # Each component stays below the filesystem limit; complete paths exceed MAX_PATH.
    output = tmp_path / ("roundtrip-" * 12) / ("roundtrip-" * 12) / ("roundtrip-" * 12)
    assert (
        len(str(output / "copied-ppo/checkpoints/update-000001/continuation.pkl")) > 320
    )
    script = ROOT / "tests/helpers/physical_job_contract_probe.py"
    env = dict(
        os.environ,
        SMARTSOM_TEST_ROOT=str(ROOT),
        SMARTSOM_TEST_OUTPUT=str(output),
        PYTHONPATH=str(ROOT / "src"),
        PYTHONUTF8="1",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        VECLIB_MAXIMUM_THREADS="1",
        RAY_ENABLE_UV_RUN_RUNTIME_ENV="0",
    )
    result = subprocess.run(
        [sys.executable, str(script), "--disk-resume"],
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "REAL_NATIVE_LONG" in result.stdout


def test_public_resume_reads_utf8_manifest_and_display_config(tmp_path, monkeypatch):
    from smartsom.experiments import composable
    from smartsom.telemetry.runtime import _options

    root = tmp_path / ("entrypoint-" * 12) / ("entrypoint-" * 12)
    native_path(root / "config").mkdir(parents=True)
    write_json(
        root / "run.json", {"schema": "smartsom.experiment/v4", "note": "调度 café"}
    )
    write_json(
        root / "config/experiment.json",
        {"logging": {"title": "调度 café", "progress": "off"}},
    )
    original = Path.read_text

    def legacy_locale(path, encoding=None, errors=None):
        return original(path, encoding=encoding or "gbk", errors=errors)

    monkeypatch.setattr(Path, "read_text", legacy_locale)
    assert _options({"source": root}).title == "调度 café"
    observed = []
    monkeypatch.setattr(
        composable,
        "resume",
        lambda source, **kwargs: observed.append(source) or "native",
    )
    assert api.resume(root) == "native"
    assert observed == [root]


def test_telemetry_utf8_progress_roundtrip_under_cp936(tmp_path, monkeypatch):
    import io

    from rich.console import Console

    from smartsom.telemetry.monitor import read_snapshot
    from smartsom.telemetry.runtime import DisplayOptions, RuntimeDisplay

    original_write, original_read = Path.write_text, Path.read_text

    def legacy_write(path, data, encoding=None, errors=None, newline=None):
        return original_write(
            path, data, encoding=encoding or "cp936", errors=errors, newline=newline
        )

    def legacy_read(path, encoding=None, errors=None):
        return original_read(path, encoding=encoding or "cp936", errors=errors)

    monkeypatch.setattr(Path, "write_text", legacy_write)
    monkeypatch.setattr(Path, "read_text", legacy_read)
    root = tmp_path / ("telemetry-" * 12) / ("telemetry-" * 12)
    view = RuntimeDisplay(
        DisplayOptions(progress="off", title="调度 café"),
        console=Console(file=io.StringIO(), force_terminal=False),
    )
    view.bind(root, name="调度 café")
    data = native_path(root / "logs/progress.json").read_bytes()
    assert "调度 café" in data.decode("utf-8")
    assert read_snapshot(root)["name"] == "调度 café"


def test_cli_resume_routes_utf8_long_manifest(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from smartsom.experiments.commands import resume

    root = tmp_path / ("cli-resume-" * 12) / ("cli-resume-" * 12)
    native_path(root).mkdir(parents=True)
    write_json(
        root / "run.json", {"schema": "smartsom.experiment/v4", "note": "调度 café"}
    )
    original = Path.read_text
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda path, encoding=None, errors=None: original(
            path, encoding=encoding or "cp936", errors=errors
        ),
    )
    monkeypatch.setattr(api, "resume", lambda path: {"resumed": str(path)})
    args = SimpleNamespace(
        source=root,
        retry_failed=False,
        background=None,
        max_concurrent=None,
        extension_module=None,
    )
    assert resume(args) == {"resumed": str(root)}


@pytest.mark.parametrize("primary_failure", [False, True])
def test_control_finish_failure_always_resets_runtime_context(
    primary_failure, monkeypatch
):
    import logging

    from smartsom.experiments import control
    from smartsom.telemetry import runtime

    old_level = logging.getLogger("optuna").level

    def fail_finish(self, status):
        self.closed.set()
        raise OSError("control publication failed")

    monkeypatch.setattr(control.Scope, "finish", fail_finish)

    @runtime.operation("training")
    def run():
        if primary_failure:
            raise ValueError("original training failure")
        return {"status": "completed"}

    with pytest.raises(ValueError if primary_failure else OSError) as caught:
        run()
    assert runtime.CURRENT.get() is None
    assert control.CURRENT.get() is None
    assert logging.getLogger("optuna").level == old_level
    if primary_failure:
        assert str(caught.value) == "original training failure"
        assert any(
            "control cleanup also failed" in note for note in caught.value.__notes__
        )


@pytest.mark.parametrize("explicit_filesystem", [False, True])
def test_actual_ray_pyarrow_deep_checkpoint_restore_and_artifact_sync(
    tmp_path, monkeypatch, explicit_filesystem
):
    pytest.importorskip("ray")
    fs = pytest.importorskip("pyarrow.fs")
    from ray.train import Checkpoint
    from ray.train._internal import storage
    from ray.train._internal.syncer import SyncConfig

    # Real Ray/PyArrow copy entrypoints, not a filesystem mock. Every component
    # stays under 255 chars while complete checkpoint members exceed MAX_PATH.
    root = tmp_path / ("arrow-" * 25) / ("arrow-" * 25)
    source = tmp_path / "arrow-source"
    native_path(source / "native/groups/dispatcher").mkdir(parents=True)
    payload = b"exact checkpoint fixture\x00\xff"
    native_path(source / "native/groups/dispatcher/payload.bin").write_bytes(payload)
    kwargs = {"storage_filesystem": fs.LocalFileSystem()} if explicit_filesystem else {}
    context = storage.StorageContext(
        storage_path=str(native_path(root)),
        experiment_dir_name="segment",
        trial_dir_name="trial",
        current_checkpoint_index=0,
        sync_config=SyncConfig(sync_artifacts=True),
        **kwargs,
    )
    persisted = context.persist_current_checkpoint(
        Checkpoint.from_directory(str(native_path(source)))
    )
    member = Path(context.checkpoint_fs_path) / "native/groups/dispatcher/payload.bin"
    assert len(str(member)) > 320
    assert native_path(member).read_bytes() == payload
    restored = tmp_path / "arrow-restored"
    persisted.to_directory(str(native_path(restored)))
    assert (
        native_path(restored / "native/groups/dispatcher/payload.bin").read_bytes()
        == payload
    )
    # Persist the already-deep source again, testing both sides of Arrow copying.
    context.current_checkpoint_index = 1
    second = context.persist_current_checkpoint(persisted)
    assert (
        native_path(
            Path(second.path) / "native/groups/dispatcher/payload.bin"
        ).read_bytes()
        == payload
    )
    # Exercise the same destination filesystem through Ray's artifact sync path.
    monkeypatch.setattr(
        storage, "_get_ray_train_session_dir", lambda: str(tmp_path / "ray-stage")
    )
    staging = Path(context.trial_working_directory)
    native_path(staging).mkdir(parents=True)
    native_path(staging / "artifact.txt").write_text("调度 café", encoding="utf-8")
    context.persist_artifacts(force=True)
    assert (
        native_path(Path(context.trial_fs_path) / "artifact.txt").read_text(
            encoding="utf-8"
        )
        == "调度 café"
    )


@pytest.mark.parametrize("payload_size", [0, 64, 200000])
def test_snapshot_hash_writer_preserves_protocol_five_bytes(payload_size):
    import io
    import pickle

    from smartsom.experiments.composable import _HashingSnapshotWriter

    shared = [
        "调度",
        bytes(range(256)) * payload_size,
        pickle.PickleBuffer(bytearray(b"protocol-five-buffer")),
    ]
    state = {"shared": shared, "alias": shared, "cycle": []}
    state["cycle"].append(state)
    expected = pickle.dumps(state, protocol=5)
    stream = io.BytesIO()
    writer = _HashingSnapshotWriter(stream)
    pickle.dump(state, writer, protocol=5)
    assert stream.getvalue() == expected
    assert writer.digest.hexdigest() == hashlib.sha256(expected).hexdigest()
    restored = pickle.loads(stream.getvalue())
    assert restored["shared"] is restored["alias"]
    assert restored["cycle"][0] is restored


def test_snapshot_hash_writer_propagates_failed_or_short_writes():
    from smartsom.experiments.composable import _HashingSnapshotWriter

    class Failed:
        def write(self, data):
            raise OSError("disk failure")

    class Short:
        def write(self, data):
            return len(data) - 1

    for stream, message in [(Failed(), "disk failure"), (Short(), "incomplete")]:
        writer = _HashingSnapshotWriter(stream)
        with pytest.raises(OSError, match=message):
            writer.write(b"snapshot")
        assert writer.digest.hexdigest() == hashlib.sha256(b"").hexdigest()


@pytest.mark.parametrize("name", ["train_all_ppo", "train_all_dqn"])
@pytest.mark.parametrize("mode", ["matrix", "grid"])
def test_native_snapshot_hash_profile_by_algorithm_and_transport(
    name, mode, tmp_path, monkeypatch
):
    import pickle

    pytest.importorskip("ray")
    from smartsom.experiments import composable

    config = api.load_config(ROOT / "configs/test/runs" / (name + ".yaml"))
    config.training.total_ticks = 8
    config.training.ticks_per_update = 4
    config.validation.enabled = False
    config.output.root = str(tmp_path)
    config.scenario_overrides["tick_limit"] = 48
    if mode == "matrix":
        config.composition = str(
            ROOT
            / "configs/test/compositions"
            / ("small_train_" + config.training.algorithm + ".yaml")
        )
        config.training.groups = tuple(
            g for g in config.training.groups if g != "mover"
        )
        config.scenario_overrides["transport"] = {
            "mode": "travel_time_matrix",
            "resolved": {"schema": "smartsom.travel-time-matrix/v1", "source": "auto"},
        }
    prepared = api.prepare(config)
    root, record, prepared = composable.allocate(prepared, "training")
    session = composable.TrainingSession(prepared, root, record)
    original_read = Path.read_bytes
    try:
        target = root / "streamed-snapshot"
        target.mkdir()

        assert (session.sims[0].protocol.matrix is None) == (mode == "grid")
        rereads = []

        def observe_continuation_read(path):
            if path.name == "continuation.pkl":
                rereads.append(path)
                assert session.settings.algorithm == "ppo", (
                    "DQN reread its continuation"
                )
            return original_read(path)

        with monkeypatch.context() as guard:
            guard.setattr(Path, "read_bytes", observe_continuation_read)
            session._write_snapshot(target)
        assert len(rereads) == (1 if session.settings.algorithm == "ppo" else 0)
        payload = (target / "continuation.pkl").read_bytes()
        metadata = json.loads((target / "snapshot.json").read_text())
        assert metadata["continuation_sha256"] == hashlib.sha256(payload).hexdigest()
        assert pickle.loads(payload)["ticks"] == session.ticks
        assert not (target / "continuation.pkl.tmp").exists()

        inference = root / "streamed-inference"
        inference.mkdir()
        session._write_snapshot(inference, full=False)
        assert json.loads((inference / "snapshot.json").read_text())["inference_only"]
        assert not (inference / "continuation.pkl").exists()

        failed = root / "failed-snapshot"
        failed.mkdir()

        def fail_dump(*args, **kwargs):
            raise OSError("serialization failed")

        with monkeypatch.context() as guard:
            guard.setattr(composable.pickle, "dump", fail_dump)
            with pytest.raises(OSError, match="serialization failed"):
                session._write_snapshot(failed)
        assert not (failed / "snapshot.json").exists()
        assert not (failed / "continuation.pkl").exists()
    finally:
        session.close()
