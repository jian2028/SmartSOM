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

    root = tmp_path.resolve()
    if os.name == "nt":
        if not str(root).startswith("\\\\?\\"):
            root = Path("\\\\?\\" + str(root))
        root = root / ("extended-component-" * 6) / ("extended-component-" * 6)
        root.mkdir(parents=True)
        assert len(str(root)) > 260
    (root / "nested").mkdir()
    target = root / "settings.yaml"
    target.write_bytes(b"value: 7\n")
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


@pytest.mark.skipif(os.name == "nt", reason="POSIX component traversal semantics")
@pytest.mark.parametrize("parent_kind", ["missing", "file"])
def test_read_model_rejects_invalid_parent_before_dotdot(tmp_path, parent_kind):
    from smartsom.config.codec import ConfigurationError

    class Settings(BaseModel):
        value: int

    (tmp_path / "settings.json").write_text('{"value": 7}')
    parent = tmp_path / parent_kind
    if parent_kind == "file":
        parent.write_text("not a directory")
    with pytest.raises(ConfigurationError):
        read_model(parent / ".." / "settings.json", Settings)


@pytest.mark.skipif(os.name == "nt", reason="POSIX symlinks require no privilege")
def test_read_model_wraps_symlink_loop(tmp_path):
    from smartsom.config.codec import ConfigurationError

    class Settings(BaseModel):
        value: int

    target = tmp_path / "loop.json"
    target.symlink_to(target.name)
    with pytest.raises(ConfigurationError):
        read_model(target, Settings)


def test_posix_read_preserves_direct_open_failure(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from smartsom.config import codec

    class Settings(BaseModel):
        value: int

    target = tmp_path / "settings.json"
    target.write_text('{"value": 7}')

    class InvalidParent:
        suffix = ".json"

        def resolve(self):
            return target

        def read_bytes(self):
            raise NotADirectoryError("parent is not a directory")

    monkeypatch.setattr(codec, "os", SimpleNamespace(name="posix"), raising=False)
    with pytest.raises(codec.ConfigurationError, match="not a directory"):
        codec.read_model(InvalidParent(), Settings)


def test_windows_resolution_loop_is_configuration_error(monkeypatch):
    from types import SimpleNamespace

    from smartsom.config import codec

    class Settings(BaseModel):
        value: int

    class Loop:
        def resolve(self):
            raise RuntimeError("Symlink loop")

    monkeypatch.setattr(codec, "os", SimpleNamespace(name="nt"), raising=False)
    with pytest.raises(codec.ConfigurationError, match="Symlink loop"):
        codec.read_model(Loop(), Settings)
