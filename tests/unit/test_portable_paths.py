"""Native extended paths retain relative-reference and fail-closed semantics."""

import hashlib
import os
from pathlib import Path

import pytest
from pydantic import BaseModel

from smartsom._filesystem import read_text
from smartsom.config.codec import read_model


def test_read_model_normalizes_parent_reference_on_extended_windows_path(tmp_path):
    class Settings(BaseModel):
        value: int

    (tmp_path / "nested").mkdir()
    target = tmp_path / "settings.yaml"
    target.write_bytes(b"value: 7\n")
    root = tmp_path.resolve()
    if os.name == "nt" and not str(root).startswith("\\\\?\\"):
        root = Path("\\\\?\\" + str(root))
    value, fingerprint = read_model(root / "nested/../settings.yaml", Settings)
    assert value.value == 7
    assert fingerprint == hashlib.sha256(target.read_bytes()).hexdigest()


def test_persistent_read_denial_is_not_treated_as_missing(monkeypatch):
    class Denied:
        def read_text(self, **kwargs):
            raise PermissionError("still denied")

    clock = iter([0.0, 3.0])
    monkeypatch.setattr("smartsom._filesystem.time.monotonic", lambda: next(clock))
    with pytest.raises(PermissionError, match="still denied"):
        read_text(Denied())
