"""Retired control journals remain readable without applying old requests."""

import json

from smartsom.experiments.runtime_control import snapshot


def test_snapshot_does_not_create_or_apply_control_files(tmp_path):
    assert snapshot(tmp_path) == {}
    assert not list(tmp_path.iterdir())
    directory = tmp_path / "control"
    directory.mkdir()
    path = directory / "runtime.json"
    record = {"status": "paused", "commands": {"old": {"status": "received"}}}
    path.write_text(json.dumps(record))
    before = path.read_bytes()
    assert snapshot(tmp_path) == record
    assert path.read_bytes() == before
