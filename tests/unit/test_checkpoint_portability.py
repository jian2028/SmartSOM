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
