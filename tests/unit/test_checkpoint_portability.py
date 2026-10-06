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
